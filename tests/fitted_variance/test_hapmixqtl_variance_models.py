"""The DEPRECATED variance models: `additive`, `two_component`, `library_scaled`.

Moved here from `tests/test_hapmixqtl.py` on 2026-09-23 to finish the
segregation begun by the quarantine. All of this machinery lives in
`tensorqtl/fitted_variance.py`; every test below names its deprecated
configuration explicitly (`variance_model=...`, `tau_mode='estimate'`), which
is the rule for this directory -- relying on defaults is what silently broke
the four files already here when the defaults flipped on 2026-09-21.

What these models computed, and why none of them ships: each fits a variance
FUNCTION per gene from that gene's own squared residuals -- `tau_g` alone under
`additive`, `(c_g, tau_g)` jointly under `two_component` and `library_scaled`
-- and then weights those same residuals by the fit. Under the two free
parameters, rescaling every `v_ig` in a gene by `k` returns `c_g / k` with
`tau_g` unchanged, so the weights cannot feel the absolute scale of the Gibbs
draws at all, only their within-gene shape. Default mode instead uses
`Var(eps_i) = sigma^2 v_i`: the Gibbs across-draw variance as a fixed shape,
with one residual scale fitted per variant.

`TestTotalVarianceModel` covers the separate `total_variance_model` argument,
which controls the TOTAL channel's variance function; it was `v_t + tau_t`
under every allelic `variance_model` until 2026-09-20.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import tensorqtl.hapmixqtl as hapmixqtl
from test_hapmixqtl import _make_dataset, _make_gaussian_seed
from tensorqtl.hapmixqtl import map_cis, map_nominal, _prepare_channels


# ---------------------------------------------------------------------------
#  Variance models: additive, two-component, library-scaled
# ---------------------------------------------------------------------------

class TestVarianceModels:

    @staticmethod
    def _simulate_channel(rng, N, c, tau, d=None):
        v = 10 ** rng.uniform(-3, -1, N)
        d = np.ones(N) if d is None else d
        y = rng.normal(0, np.sqrt(d * (c * v + tau)), N)
        return y, v

    def test_c_tau_recovers_simulated_values(self, device):
        """The joint (c, tau) fit recovers a planted two-component variance
        on a through-origin channel, with and without a per-sample factor."""
        rng = _make_gaussian_seed(300)
        N, c, tau = 4000, 3.0, 0.01
        y, v = self._simulate_channel(rng, N, c, tau)
        T = lambda x: torch.tensor(x, dtype=torch.float64, device=device)
        fit = hapmixqtl._estimate_c_tau(T(y), T(v), None, device, intercept=False)
        c_hat, tau_hat, conv = fit['c'], fit['tau'], fit['converged']
        assert conv
        assert abs(c_hat - c) < 0.4, c_hat
        assert abs(tau_hat - tau) < 0.003, tau_hat
        d = np.exp(rng.normal(0, 0.3, N))
        y2, v2 = self._simulate_channel(rng, N, c, tau, d)
        fit2 = hapmixqtl._estimate_c_tau(T(y2), T(v2), None, device, intercept=False, d_t=T(d))
        c2, tau2, conv2 = fit2['c'], fit2['tau'], fit2['converged']
        assert conv2
        assert abs(c2 - c) < 0.4, c2
        assert abs(tau2 - tau) < 0.003, tau2
        # with the factor in the weights, the standardized squared residuals
        # carry no per-sample structure; without it they track d
        base = c2 * v2 + tau2
        e2_with = y2 ** 2 / (d * base)
        fit3 = hapmixqtl._estimate_c_tau(T(y2), T(v2), None, device, intercept=False)
        c3, tau3 = fit3['c'], fit3['tau']
        e2_without = y2 ** 2 / (c3 * v2 + tau3)
        lo, hi = d <= np.quantile(d, 0.2), d >= np.quantile(d, 0.8)
        ratio_with = e2_with[hi].mean() / e2_with[lo].mean()
        ratio_without = e2_without[hi].mean() / e2_without[lo].mean()
        assert abs(ratio_with - 1) < 0.2, ratio_with
        assert ratio_without > 1.5, ratio_without

    def test_additive_is_the_default_and_unchanged(self):
        """variance_model='additive' reproduces the pre-existing map_cis output
        exactly and reports c_a = 1."""
        d = _make_dataset(seed=120)
        kw = dict(xL_df=d['xL_df'], xR_df=d['xR_df'], nperm=500, window=1000000,
                  seed=3, verbose=False, tau_mode='estimate')
        base = map_cis(d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'],
                       d['Va_df'], d['Vt_df'], d['pos_df'], **kw)
        add = map_cis(d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'],
                      d['Va_df'], d['Vt_df'], d['pos_df'], variance_model='additive', **kw)
        pd.testing.assert_frame_equal(base, add)
        assert (base['variance_model'] == 'additive').all()
        assert (base['c_a'] == 1.0).all() and (base['c_a_null'] == 1.0).all()

    def test_library_scaled_with_unit_factor_equals_two_component(self):
        d = _make_dataset(seed=121)
        kw = dict(xL_df=d['xL_df'], xR_df=d['xR_df'], nperm=500, window=1000000,
                  seed=5, verbose=False, tau_mode='estimate')
        two = map_cis(d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'],
                      d['Va_df'], d['Vt_df'], d['pos_df'], variance_model='two_component', **kw)
        ones = pd.Series(1.0, index=d['A_df'].columns)
        lib = map_cis(d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'],
                      d['Va_df'], d['Vt_df'], d['pos_df'], variance_model='library_scaled',
                      library_factor=ones, **kw)
        for col in ('variant_id',):
            assert (two[col] == lib[col]).all()
        for col in ('slope', 'slope_se', 'pval_perm', 'c_a', 'tau_a'):
            np.testing.assert_allclose(two[col].astype(float), lib[col].astype(float), rtol=1e-5)
        assert (two['variance_model'] == 'two_component').all()
        assert (lib['variance_model'] == 'library_scaled').all()
        assert (two['c_a'] > 0).all()

    def test_all_models_recover_the_causal_variant(self):
        d = _make_dataset(seed=122)
        lf = pd.Series(np.exp(_make_gaussian_seed(9).normal(0, 0.2, len(d['A_df'].columns))),
                       index=d['A_df'].columns)
        lf /= lf.mean()
        for model, factor in (('additive', None), ('two_component', None), ('library_scaled', lf)):
            res = map_cis(d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'],
                          d['Va_df'], d['Vt_df'], d['pos_df'], xL_df=d['xL_df'], xR_df=d['xR_df'],
                          nperm=500, window=1000000, seed=8, verbose=False,
                          variance_model=model, library_factor=factor, tau_refit=True,
                          tau_mode='estimate')
            row = res.loc[d['causal_pheno']]
            assert row['variant_id'] == d['causal_variant'], model
            assert row['pval_perm'] < 0.05, model
            assert set(['c_a', 'c_a_null', 'variance_model']) <= set(res.columns)
            assert np.isfinite(row['c_a']) and row['c_a'] >= 0

    def test_estimate_library_factors_recovers_planted_pattern(self):
        rng = _make_gaussian_seed(301)
        G, N = 400, 60
        samples = [f'S{i:03d}' for i in range(N)]
        d_true = np.exp(rng.normal(0, 0.3, N)); d_true /= d_true.mean()
        c = rng.uniform(0.5, 3.0, G); tau = rng.uniform(0.002, 0.02, G)
        v = 10 ** rng.uniform(-3, -1, (G, N))
        a = rng.normal(0, np.sqrt(d_true[None, :] * (c[:, None] * v + tau[:, None])))
        # a few samples uninformative for some genes
        v[rng.random((G, N)) < 0.1] = 0.0
        A_df = pd.DataFrame(a, index=[f'g{i}' for i in range(G)], columns=samples)
        Va_df = pd.DataFrame(v, index=A_df.index, columns=samples)
        d_hat = hapmixqtl.estimate_library_factors(A_df, Va_df, min_informative=30)
        assert list(d_hat.index) == samples
        assert abs(d_hat.mean() - 1.0) < 1e-9
        assert d_hat.attrs['n_genes'] > 100
        rho = pd.Series(d_hat.values).corr(pd.Series(d_true), method='spearman')
        assert rho > 0.9, rho
        assert np.abs(np.log(d_hat.values / d_true)).max() < 0.4
        sub = hapmixqtl.estimate_library_factors(A_df, Va_df, genes=A_df.index[:200], min_informative=30)
        assert pd.Series(sub.values).corr(pd.Series(d_true), method='spearman') > 0.85

    def test_estimate_library_factors_falls_back_without_a_floor(self):
        """Data with no between-sample floor has every tau clamped at zero;
        the estimator then uses the non-degenerate fits and says so."""
        rng = _make_gaussian_seed(302)
        G, N = 60, 50
        d_true = np.exp(rng.normal(0, 0.3, N)); d_true /= d_true.mean()
        v = 10 ** rng.uniform(-3, -1, (G, N))
        a = rng.normal(0, np.sqrt(d_true[None, :] * 2.0 * v))
        A_df = pd.DataFrame(a, columns=[f'S{i}' for i in range(N)]); Va_df = pd.DataFrame(v, columns=A_df.columns)
        # noise makes about half of these fits interior; require more than
        # the data set can supply so the fallback is exercised
        with pytest.warns(RuntimeWarning, match='interior'):
            d_hat = hapmixqtl.estimate_library_factors(A_df, Va_df, min_informative=30, min_genes=G + 1)
        assert not d_hat.attrs['interior_only']
        assert d_hat.attrs['n_genes'] >= 0.9 * G
        assert pd.Series(d_hat.values).corr(pd.Series(d_true), method='spearman') > 0.8

    def test_variance_prior_reduces_error_and_removes_the_clamp(self, device):
        """Empirical-Bayes priors toward expression bins: lower error on (c, tau)
        than the clamped fit, and the tau = 0 clamp all but disappears."""
        rng = _make_gaussian_seed(303)
        G, N = 200, 60
        samples = [f'S{i:03d}' for i in range(N)]
        # a floor small relative to the measurement term, as on well-expressed
        # BrainVar genes, so that the per-gene intercept is often unsupported
        c_true = np.exp(rng.normal(np.log(1.5), 0.4, G))
        tau_true = np.exp(rng.normal(np.log(0.002), 0.5, G))
        v = 10 ** rng.uniform(-2.5, -0.5, (G, N))
        a = rng.normal(0, np.sqrt(c_true[:, None] * v + tau_true[:, None]))
        A_df = pd.DataFrame(a, index=[f'g{i}' for i in range(G)], columns=samples)
        Va_df = pd.DataFrame(v, index=A_df.index, columns=samples)
        priors = hapmixqtl.estimate_variance_priors(A_df, Va_df, n_bins=4, min_informative=30)
        assert set(['prior_c', 'prior_tau', 'prior_sd_c', 'prior_sd_tau', 'bin']) <= set(priors.columns)
        assert list(priors.index) == list(A_df.index)
        assert priors.attrs['kappa'] >= 2.0 and priors.attrs['n_genes'] == G
        assert (priors['prior_tau'] > 0).all() and (priors['prior_sd_tau'] > 0).all()
        T = lambda x: torch.tensor(x, dtype=torch.float64, device=device)
        e_clamp, e_prior, tau0_clamp, floored = [], [], 0, 0
        for g in range(G):
            f0 = hapmixqtl._estimate_c_tau(T(a[g]), T(v[g]), None, device, intercept=False)
            f1 = hapmixqtl._estimate_c_tau(T(a[g]), T(v[g]), None, device, intercept=False,
                                           prior=hapmixqtl._prior_tuple(priors, A_df.index[g]))
            e_clamp.append(((f0['c'] - c_true[g]) / c_true[g]) ** 2 + ((f0['tau'] - tau_true[g]) / tau_true[g]) ** 2)
            e_prior.append(((f1['c'] - c_true[g]) / c_true[g]) ** 2 + ((f1['tau'] - tau_true[g]) / tau_true[g]) ** 2)
            tau0_clamp += f0['tau'] < 1e-6
            floored += f1['floored']
            assert f1['c'] > 0 and f1['tau'] > 0                # log-scale posterior: positivity is automatic
            assert np.isfinite(f1['c_raw']) and np.isfinite(f1['tau_raw'])
        assert np.mean(e_prior) < 0.7 * np.mean(e_clamp), (np.mean(e_prior), np.mean(e_clamp))
        assert tau0_clamp >= 0.05 * G, tau0_clamp          # the clamp does bite on this design
        assert floored <= 0.05 * G and floored <= tau0_clamp / 5, (floored, tau0_clamp)   # the prior all but removes it

    def test_map_cis_with_variance_prior(self):
        d = _make_dataset(seed=124)
        rng = _make_gaussian_seed(304)
        samples = list(d['A_df'].columns)
        extra = 60
        v = 10 ** rng.uniform(-2, -0.5, (extra, len(samples)))
        a = rng.normal(0, np.sqrt(1.5 * v + 0.02))
        A_big = pd.concat([d['A_df'], pd.DataFrame(a, index=[f'x{i}' for i in range(extra)], columns=samples)])
        Va_big = pd.concat([d['Va_df'], pd.DataFrame(v, index=[f'x{i}' for i in range(extra)], columns=samples)])
        kw = dict(xL_df=d['xL_df'], xR_df=d['xR_df'], nperm=500, window=1000000, seed=9, verbose=False, tau_mode='estimate')
        args = (d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'], d['Va_df'], d['Vt_df'], d['pos_df'])
        priors = hapmixqtl.estimate_variance_priors(A_big, Va_big, n_bins=2, min_informative=30)
        res = map_cis(*args, variance_model='two_component', variance_prior=priors, **kw)
        row = res.loc[d['causal_pheno']]
        assert row['variant_id'] == d['causal_variant'] and row['pval_perm'] < 0.05
        assert res['variance_prior'].astype(bool).all()
        assert set(['c_a_raw', 'tau_a_raw', 'c_a_floored']) <= set(res.columns)
        assert (res['tau_a'].astype(float) > 0).all()
        lf = pd.Series(np.exp(rng.normal(0, 0.2, len(samples))), index=samples); lf /= lf.mean()
        with pytest.raises(ValueError, match='different model'):
            map_cis(*args, variance_model='library_scaled', library_factor=lf, variance_prior=priors, **kw)
        priors_lf = hapmixqtl.estimate_variance_priors(A_big, Va_big, n_bins=2, min_informative=30, library_factor=lf)
        res2 = map_cis(*args, variance_model='library_scaled', library_factor=lf, variance_prior=priors_lf, **kw)
        assert res2.loc[d['causal_pheno'], 'variant_id'] == d['causal_variant']
        with pytest.raises(ValueError, match='two-component'):
            map_cis(*args, variance_model='additive', variance_prior=priors, **kw)

    def test_variance_model_argument_errors(self):
        d = _make_dataset(seed=123)
        kw = dict(xL_df=d['xL_df'], xR_df=d['xR_df'], nperm=100, window=1000000, verbose=False, tau_mode='estimate')
        args = (d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'], d['Va_df'], d['Vt_df'], d['pos_df'])
        ones = pd.Series(1.0, index=d['A_df'].columns)
        with pytest.raises(ValueError, match='library_factor'):
            map_cis(*args, variance_model='library_scaled', **kw)
        with pytest.raises(ValueError, match='library_factor'):
            map_cis(*args, variance_model='additive', library_factor=ones, **kw)
        with pytest.raises(ValueError, match='variance_model'):
            map_cis(*args, variance_model='multiplicative', **kw)
        kw_no_tau = {k: v for k, v in kw.items() if k != 'tau_mode'}
        with pytest.raises(ValueError, match="tau_mode='estimate'"):
            map_cis(*args, variance_model='two_component', tau_mode='zero',
                    **kw_no_tau)
        with pytest.raises(ValueError, match='positive'):
            map_cis(*args, variance_model='library_scaled', library_factor=ones * 0, **kw)


class TestTotalVarianceModel:
    """The TOTAL channel's variance function.

    It was v_t + tau_t under every allelic variance_model until 2026-09-20.
    The draws do carry per-donor information there -- Vt spans 2 to 3.5-fold
    across donors within a gene -- but with c fixed at 1, tau_t swamps it
    roughly 19 to 1 and the weights come out nearly equal. Fitting c_t is
    the lever.
    """

    def _fix(self, device, n=70, seed=321):
        rng = _make_gaussian_seed(seed)
        g = rng.choice([0.0, 1.0, 2.0], n)
        s = rng.choice([-1.0, 0.0, 1.0], n)
        t = 1.2 + 0.4 * (g / 2) + rng.normal(0, 0.2, n)
        a = 0.3 * s + rng.normal(0, 0.2, n)
        va = rng.uniform(0.05, 0.4, n)
        # heterogeneous total draw variance, the thing c_t can act on
        vt = rng.uniform(0.001, 0.02, n)
        return g, s, a, t, va, vt

    def _prep(self, device, a, t, va, vt, **kw):
        T = lambda x: torch.tensor(x, dtype=torch.float64, device=device)
        return _prepare_channels(T(a), T(t), T(va), T(vt), None, 'estimate',
                                 device, ase_covariates_t=None,
                                 return_info=True, **kw)

    def test_default_is_the_shipped_additive_total_channel(self, device):
        g, s, a, t, va, vt = self._fix(device)
        base = self._prep(device, a, t, va, vt)
        expl = self._prep(device, a, t, va, vt, total_variance_model='additive')
        assert torch.allclose(base[1], expl[1], atol=0, rtol=0)
        assert base[4]['c_t'] == 1.0, 'additive pins c_t at 1'

    def test_two_component_fits_c_t_and_changes_only_the_total_weights(self, device):
        g, s, a, t, va, vt = self._fix(device)
        add = self._prep(device, a, t, va, vt, total_variance_model='additive')
        two = self._prep(device, a, t, va, vt, total_variance_model='two_component')
        # allelic channel untouched
        assert torch.allclose(add[0], two[0], atol=0, rtol=0)
        # total channel moved, and c_t was actually fitted
        assert not torch.allclose(add[1], two[1])
        assert two[4]['c_t'] != 1.0
        assert two[4]['total_variance_model'] == 'two_component'

    def test_it_is_independent_of_the_allelic_variance_model(self, device):
        """The two channels' variance functions must not be entangled."""
        g, s, a, t, va, vt = self._fix(device, seed=654)
        x = self._prep(device, a, t, va, vt, total_variance_model='two_component')
        y = self._prep(device, a, t, va, vt, variance_model='two_component',
                       total_variance_model='two_component')
        # changing the ALLELIC model must leave the TOTAL weights alone
        assert torch.allclose(x[1], y[1], atol=0, rtol=0)

    def test_rejects_unknown_model_and_tau_mode_zero(self, device):
        g, s, a, t, va, vt = self._fix(device)
        with pytest.raises(ValueError, match='total_variance_model'):
            self._prep(device, a, t, va, vt, total_variance_model='library_scaled')
        T = lambda x: torch.tensor(x, dtype=torch.float64, device=device)
        with pytest.raises(ValueError, match='requires'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                _prepare_channels(T(a), T(t), T(va), T(vt), None, 'zero', device,
                                  ase_covariates_t=None,
                                  total_variance_model='two_component')

    def test_map_nominal_accepts_it(self, tmp_path):
        d = _make_dataset(seed=100)
        common = dict(genotype_df=d['genotype_df'], variant_df=d['variant_df'],
                      A_df=d['A_df'], T_df=d['T_df'], Va_df=d['Va_df'],
                      Vt_df=d['Vt_df'], phenotype_pos_df=d['pos_df'],
                      xL_df=d['xL_df'], xR_df=d['xR_df'], verbose=False, tau_mode='estimate')
        read = lambda p: pd.concat([pd.read_parquet(f) for f in
                                    sorted(Path(p).glob('*.parquet'))],
                                   ignore_index=True)
        da, db = tmp_path / 'a', tmp_path / 'b'
        da.mkdir(); db.mkdir()
        map_nominal(prefix='a', output_dir=str(da), **common)
        map_nominal(prefix='b', output_dir=str(db),
                    total_variance_model='two_component', **common)
        A, B = read(da), read(db)
        key = ['phenotype_id', 'variant_id']
        mg = A[key + ['slope_a', 'slope_t']].merge(
            B[key + ['slope_a', 'slope_t']], on=key, suffixes=('_a', '_b'))
        assert np.allclose(mg.slope_a_a, mg.slope_a_b, atol=0, rtol=0), \
            'the allelic channel must not move'
        assert not np.allclose(mg.slope_t_a, mg.slope_t_b), \
            'the total channel must move'
