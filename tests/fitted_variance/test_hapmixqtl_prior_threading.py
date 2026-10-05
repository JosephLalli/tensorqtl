"""The prior branch of _estimate_c_tau is tied to the unpenalized fit, and
map_cis threads the prior into both the scan and the lead refit."""
import sys, numpy as np, pandas as pd, pytest, torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_hapmixqtl import _make_dataset, _make_gaussian_seed
import tensorqtl.hapmixqtl as hapmixqtl
from tensorqtl.hapmixqtl import map_cis


class TestPriorReductions:
    """The prior path (damped Fisher scoring on (log c, log tau),
    hapmixqtl.py:754-784) is a SEPARATE iteration from the clamped path
    (hapmixqtl.py:727-752). Nothing in the shipped tests connects them, so a
    sign error in the score, a wrong Jacobian in ``I_th = J @ (Am/kappa) @ J``,
    or a mis-scaled tempering would pass every existing assertion as long as
    the posterior mode landed somewhere plausible. The docstring's own claim
    is the pin: 'The unpenalized stationary point of that likelihood is the
    same weighted regression of e^2 on [v, 1] as the clamped fit ... so the
    two estimators agree away from the boundary.'"""

    @staticmethod
    def _interior_channel(seed=907, N=2000, c=2.0, tau=0.01):
        rng = _make_gaussian_seed(seed)
        v = 10 ** rng.uniform(-3, -1, N)
        y = rng.normal(0, np.sqrt(c * v + tau), N)
        return y, v

    def test_prior_reduces_to_the_unpenalized_fit_as_it_widens(self, device):
        """With the prior sd taken to infinity the posterior mode must converge
        to the clamped fit's interior solution, and to its own (c_raw, tau_raw):
        the penalty is the only thing that separates the two iterations."""
        y, v = self._interior_channel()
        T = lambda x: torch.tensor(x, dtype=torch.float64, device=device)
        clamped = hapmixqtl._estimate_c_tau(T(y), T(v), None, device, intercept=False)
        assert clamped['converged'] and clamped['c'] > 0 and clamped['tau'] > 0, clamped
        # the reference must be an interior fit, or 'away from the boundary'
        # is vacuous and the comparison tests nothing
        assert np.isclose(clamped['c'], clamped['c_raw'], rtol=1e-6)
        assert np.isclose(clamped['tau'], clamped['tau_raw'], rtol=1e-6)
        prev = None
        for sd in (1e1, 1e2, 1e3, 1e6):
            # prior centred away from the truth on purpose: a wide prior must
            # be ignored regardless of where it points
            fit = hapmixqtl._estimate_c_tau(T(y), T(v), None, device, intercept=False,
                                            prior=(np.log(50.0), sd, np.log(5.0), sd, 2.0))
            assert fit['converged'], sd
            err = max(abs(fit['c'] / clamped['c'] - 1), abs(fit['tau'] / clamped['tau'] - 1))
            if prev is not None:
                # monotone while the penalty still dominates; below ~1e-6 the
                # residual is the fit's own step tolerance (tol=1e-7), not the prior
                assert err <= max(prev, 1e-6), (sd, err, prev)
            prev = err
            if sd >= 1e3:
                assert err < 1e-5, (sd, err, fit['c'], clamped['c'])
                # and the fit reports the same unpenalized solution it sat on
                assert np.isclose(fit['c_raw'], clamped['c_raw'], rtol=1e-5)
                assert np.isclose(fit['tau_raw'], clamped['tau_raw'], rtol=1e-5)

    def test_a_tight_prior_is_obeyed_and_the_tempering_is_a_real_lever(self, device):
        """The other end of the same lever: a prior sd taken to zero must pin
        the parameter at its prior mean, and kappa must actually temper the
        likelihood (larger kappa = the prior wins more). Without this, the
        kappa that estimate_variance_priors computes could be ignored -- and
        note kappa is floored at 2.0 in estimate_variance_priors
        (hapmixqtl.py:922), so `attrs['kappa'] >= 2.0` cannot detect it."""
        y, v = self._interior_channel()
        T = lambda x: torch.tensor(x, dtype=torch.float64, device=device)
        clamped = hapmixqtl._estimate_c_tau(T(y), T(v), None, device, intercept=False)
        pinned = hapmixqtl._estimate_c_tau(T(y), T(v), None, device, intercept=False,
                                           prior=(np.log(7.0), 1e-6, np.log(0.5), 1e-6, 2.0))
        assert abs(pinned['c'] - 7.0) < 1e-4, pinned['c']
        assert abs(pinned['tau'] - 0.5) < 1e-4, pinned['tau']
        m_c = np.log(20.0)
        pull = {}
        for kappa in (2.0, 20.0, 200.0):
            f = hapmixqtl._estimate_c_tau(T(y), T(v), None, device, intercept=False,
                                          prior=(m_c, 0.5, np.log(0.01), 0.5, kappa))
            pull[kappa] = abs(np.log(f['c']) - np.log(clamped['c'])) / (m_c - np.log(clamped['c']))
        assert 0.0 <= pull[2.0] < pull[20.0] < pull[200.0] <= 1.0, pull
        assert pull[200.0] - pull[2.0] > 0.2, pull


class TestPriorThreadsThroughMapCis:

    @staticmethod
    def _dataset_with_prior_background(seed=124, extra=60):
        d = _make_dataset(seed=seed)
        rng = _make_gaussian_seed(4004)
        samples = list(d['A_df'].columns)
        v = 10 ** rng.uniform(-2, -0.5, (extra, len(samples)))
        a = rng.normal(0, np.sqrt(1.5 * v + 0.02))
        ix = [f'bg{i}' for i in range(extra)]
        A_big = pd.concat([d['A_df'], pd.DataFrame(a, index=ix, columns=samples)])
        Va_big = pd.concat([d['Va_df'], pd.DataFrame(v, index=ix, columns=samples)])
        priors = hapmixqtl.estimate_variance_priors(A_big, Va_big, n_bins=2, min_informative=30)
        return d, priors

    def test_map_cis_c_a_null_is_the_prior_fit_of_that_gene(self, device):
        """``res['variance_prior'].astype(bool).all()`` only echoes
        `prior is not None` back; it cannot tell a threaded prior from a
        dropped one. This checks the reported c_a_null IS the prior fit of that
        gene's own informative samples, and that it differs from the no-prior
        fit -- so dropping `prior=_prior_tuple(...)` from either
        _prepare_channels call site fails the test."""
        d, priors = self._dataset_with_prior_background()
        args = (d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'],
                d['Va_df'], d['Vt_df'], d['pos_df'])
        kw = dict(xL_df=d['xL_df'], xR_df=d['xR_df'], nperm=500, window=1000000,
                  seed=9, verbose=False)
        no_prior = map_cis(*args, variance_model='two_component',
                           tau_mode='estimate', **kw)
        with_prior = map_cis(*args, variance_model='two_component',
                             variance_prior=priors, tau_mode='estimate', **kw)
        T = lambda x: torch.tensor(np.asarray(x, dtype=float), dtype=torch.float64, device=device)
        moved = 0
        for gene in d['A_df'].index:
            a = d['A_df'].loc[gene].values.astype(float)
            v = d['Va_df'].loc[gene].values.astype(float)
            keep = v > 1e-12
            direct = hapmixqtl._estimate_c_tau(
                T(a[keep]), T(v[keep]), None, device, intercept=False,
                prior=hapmixqtl._prior_tuple(priors, gene))
            assert np.isclose(float(with_prior.loc[gene, 'c_a_null']), direct['c'], rtol=1e-5), gene
            assert np.isclose(float(with_prior.loc[gene, 'tau_a_null']), direct['tau'], rtol=1e-5), gene
            if not np.isclose(float(no_prior.loc[gene, 'c_a_null']), direct['c'], rtol=1e-3):
                moved += 1
        assert moved >= 1, 'the prior changed no gene; the comparison is vacuous'

    def test_map_cis_threads_the_prior_into_the_lead_refit(self, device):
        """tau_refit=True re-enters _prepare_channels with the lead in the tau
        design (hapmixqtl.py:2095). No shipped test combines tau_refit with a
        prior, so the prior could be dropped on that second call and only the
        lead's slope/SE/nominal p -- the numbers the runbook reports -- would be
        wrong. Pins that c_a comes from the refit-with-prior fit, not from the
        null fit and not from an unpenalized refit."""
        d, priors = self._dataset_with_prior_background()
        args = (d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'],
                d['Va_df'], d['Vt_df'], d['pos_df'])
        kw = dict(xL_df=d['xL_df'], xR_df=d['xR_df'], nperm=500, window=1000000,
                  seed=9, verbose=False)
        res = map_cis(*args, variance_model='two_component', variance_prior=priors,
                      tau_refit=True, tau_mode='estimate', **kw)
        gene = d['causal_pheno']
        row = res.loc[gene]
        assert row['variant_id'] == d['causal_variant']
        assert bool(row['tau_refit'])
        a = d['A_df'].loc[gene].values.astype(float)
        v = d['Va_df'].loc[gene].values.astype(float)
        keep = v > 1e-12
        s_lead = (d['xL_df'].loc[d['causal_variant']].values
                  - d['xR_df'].loc[d['causal_variant']].values).astype(float)
        T = lambda x: torch.tensor(np.asarray(x, dtype=float), dtype=torch.float64, device=device)
        refit = hapmixqtl._estimate_c_tau(
            T(a[keep]), T(v[keep]), T(s_lead[keep]).unsqueeze(1), device, intercept=False,
            prior=hapmixqtl._prior_tuple(priors, gene))
        assert np.isclose(float(row['c_a']), refit['c'], rtol=1e-4), (row['c_a'], refit['c'])
        assert np.isclose(float(row['tau_a']), refit['tau'], rtol=1e-4), (row['tau_a'], refit['tau'])
        # the refit must actually move: a planted effect inflates the null tau
        assert not np.isclose(float(row['c_a']), float(row['c_a_null']), rtol=1e-3)
        unpenalized = hapmixqtl._estimate_c_tau(
            T(a[keep]), T(v[keep]), T(s_lead[keep]).unsqueeze(1), device, intercept=False)
        assert not np.isclose(float(row['c_a']), unpenalized['c'], rtol=1e-3), \
            'the refit looks unpenalized; the prior is not reaching the lead refit'
