"""Per-channel t references and the allelic admission floor (default mode).

Under se_mode='fitted' each channel's standard error is scaled by a residual
variance fitted on that channel's own informative donors, with
_channel_dof = n_eff - 1 - (residualizer columns) degrees of freedom: n_a - 1
for the through-origin allelic channel, n_t - 2 - n_cov for the total one.
Until 2026-09-27 map_nominal and map_cis referred pval_nominal, pval_a and
pval_t to one shared N - 2 - max(n_cov, n_cov_a), so a gene with 3 allelic
donors had a t statistic with 2 degrees of freedom read against 73: at nominal
0.05 that rejects 2 * P(t_2 > t_73^{-1}(0.975)) = 0.184 of the time.

Pins, all known-answer and run through the shipped entry points:
  (a) 3 allelic donors, floor lowered to 2 so the channel enters, normal
      errors at known weights: pval_a rejects at nominal under t_2 (and the
      old reference, applied to the same statistics, at about 0.18);
  (b) dof_nominal is dof_a when the total channel is off (where the floor
      is waived), dof_t below the floor, NaN where no channel is on, and the
      Welch-Satterthwaite closed form on a two-channel gene whose channel SEs
      are recomputed here from the data, the combined SE and p corrected
      by Meier's factor from the same SEs;
  (c) below the floor the combined slope, SE and p ARE the total channel's,
      and map_cis gives exactly what it gives with the allelic channel absent
      (observed scan and every permutation), its lead on the fitted scale;
  (d) at the default floor, every donor informative: dof_a = N - 1 - n_cov_a.
"""
import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

from tensorqtl.core import get_t_pval
from tensorqtl.hapmixqtl import (MIN_ALLELIC_DONORS, _satterthwaite_dof,
                                 map_cis, map_nominal)
from tensorqtl.mixqtl_replication import META_N_CUTOFF

SEED = 42


def _bank(seed, N=75, V=4, R=1, n_a=None, n_cov=0, sigma_a=0.7, sigma_t=0.5):
    """R independent null genes on one shared cis window of V variants.

    The first n_a donors are the allelic channel's informative donors (va > 0)
    and are heterozygous at variant 0; every other donor has va = 0 and a = 0,
    the state of a donor without haplotype-informative reads. Errors are
    normal with Var = sigma^2 v on both channels, the model default mode
    fits, so every channel t statistic is exactly t on its _channel_dof.
    """
    rng = np.random.RandomState(seed)
    n_a = N if n_a is None else n_a
    samples = [f'D{i:03d}' for i in range(N)]
    variants = [f'chr1_{50000 + 10 * j}_A_G' for j in range(V)]
    genes = [f'G{r:05d}' for r in range(R)]
    g = rng.choice([0.0, 1.0, 2.0], size=(V, N), p=[0.25, 0.5, 0.25])
    g[0, :n_a] = 1.0
    alt_on_L = rng.rand(V, N) < 0.5
    xL = np.where(g == 2, 1.0, np.where((g == 1) & alt_on_L, 1.0, 0.0))
    xR = np.where(g == 2, 1.0, np.where((g == 1) & ~alt_on_L, 1.0, 0.0))
    va = np.zeros((R, N))
    va[:, :n_a] = rng.uniform(0.02, 0.5, size=(R, n_a))
    vt = rng.uniform(0.02, 0.5, size=(R, N))
    a = sigma_a * np.sqrt(va) * rng.normal(size=(R, N))
    C = rng.normal(size=(N, n_cov))
    t = 2.0 + sigma_t * np.sqrt(vt) * rng.normal(size=(R, N)) + (C @ rng.normal(0, 0.3, n_cov))[None, :]
    frame = lambda x, idx: pd.DataFrame(np.asarray(x, dtype=np.float32), index=idx, columns=samples)
    return dict(
        genotype_df=frame(g, variants),
        variant_df=pd.DataFrame({'chrom': 'chr1', 'pos': [50000 + 10 * j for j in range(V)]},
                                index=variants),
        xL_df=frame(xL, variants), xR_df=frame(xR, variants),
        A_df=frame(a, genes), T_df=frame(t, genes), Va_df=frame(va, genes), Vt_df=frame(vt, genes),
        pos_df=pd.DataFrame({'chr': 'chr1', 'pos': 50000}, index=genes),
        covariates_df=(pd.DataFrame(C, index=samples, columns=[f'c{k}' for k in range(n_cov)])
                       if n_cov else None),
        samples=samples, n_a=n_a, N=N, n_cov=n_cov)


def _nominal(d, tmp_path, **kw):
    map_nominal(d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'], d['Va_df'],
                d['Vt_df'], d['pos_df'], xL_df=d['xL_df'], xR_df=d['xR_df'], prefix='p',
                covariates_df=d['covariates_df'], output_dir=str(tmp_path), verbose=False, **kw)
    return pd.read_parquet(tmp_path / 'p.hapmixqtl_pairs.chr1.parquet')


def _cis(d, **kw):
    return map_cis(d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'], d['Va_df'],
                   d['Vt_df'], d['pos_df'], xL_df=d['xL_df'], xR_df=d['xR_df'],
                   covariates_df=d['covariates_df'], verbose=False, **kw)


def test_the_floor_is_mixqtl_meta_cutoff():
    """MIN_ALLELIC_DONORS is mixQTL's META_N_CUTOFF (defined independently in
    the port, which this module does not import)."""
    assert MIN_ALLELIC_DONORS == META_N_CUTOFF == 15


# (a) -----------------------------------------------------------------------

def test_three_allelic_donors_reject_at_nominal_on_two_df(tmp_path):
    """1,000 null genes with n_a = 3 through-origin allelic donors: dof_a is 2
    on every pair and pval_a rejects at the nominal rate within 4 Monte Carlo
    standard errors, at 0.05 and at 0.01. The same statistics referred to the
    old shared N - 2 = 73 reject at about 0.184 / 0.118, the closed-form t_2
    tail beyond the t_73 quantile, so the two references are separated by
    about 19 standard errors at 0.05 and 34 at 0.01."""
    R, N = 1000, 75
    d = _bank(SEED, N=N, V=1, R=R, n_a=3)
    res = _nominal(d, tmp_path, min_allelic_donors=2)
    assert len(res) == R
    assert res['allelic_admitted'].all()
    assert (res['dof_a'] == 2).all() and (res['dof_t'] == N - 2).all()
    t_a = res['slope_a'].astype(float) / res['slope_a_se'].astype(float)
    assert np.allclose(res['pval_a'], get_t_pval(t_a, 2), rtol=1e-5)
    for alpha in (0.05, 0.01):
        se = np.sqrt(alpha * (1 - alpha) / R)
        rate = float((res['pval_a'] < alpha).mean())
        assert abs(rate - alpha) < 4 * se, (alpha, rate, se)
        old_expected = 2 * stats.t.sf(stats.t.isf(alpha / 2, N - 2), 2)
        old_rate = float((get_t_pval(t_a, N - 2) < alpha).mean())
        assert abs(old_rate - old_expected) < 4 * np.sqrt(old_expected * (1 - old_expected) / R), \
            (alpha, old_rate, old_expected)
        assert old_rate - rate > 10 * se, (alpha, old_rate, rate)


# (b) -----------------------------------------------------------------------

def test_satterthwaite_closed_form_on_literal_weights():
    """(w_a + w_t)^2 / (w_a^2/nu_a + w_t^2/nu_t) on hand-computed values:
    w = (1, 3), nu = (2, 10) gives 16 / 1.4; equal weights and equal nu give
    2 nu; a zero weight on either side gives exactly the other channel's nu,
    and zero on both sides (no statistic) gives NaN."""
    T = lambda x: torch.tensor(x, dtype=torch.float32)
    nu = _satterthwaite_dof(T([1.0, 2.0, 0.0, 5.0, 0.0]), T([3.0, 2.0, 4.0, 0.0, 0.0]), 2, 10)
    assert nu.dtype == torch.float64
    assert np.isclose(float(nu[0]), 16.0 / 1.4, rtol=1e-12)
    assert np.isclose(float(_satterthwaite_dof(T([2.0]), T([2.0]), 5, 5)[0]), 10.0, rtol=1e-12)
    assert float(nu[2]) == 10.0 and float(nu[3]) == 2.0 and np.isnan(float(nu[4]))
    # large weights are handled on their shares, without overflow
    big = _satterthwaite_dof(T([3e30]), T([1e30]), 2, 10)
    assert np.isclose(float(big[0]), 16.0 / (9 / 2 + 1 / 10), rtol=1e-6)


def test_dof_nominal_is_dof_a_when_the_total_channel_is_off(tmp_path):
    """keep_t_df all False switches the total channel off, so the combination
    is the allelic channel alone: dof_nominal == dof_a == n_a - 1 on every
    pair and pval_nominal == pval_a (the old code referred both to N - 2)."""
    d = _bank(SEED + 1, N=40, V=5, n_a=25)
    keep_t = pd.DataFrame(False, index=d['A_df'].index, columns=d['A_df'].columns)
    res = _nominal(d, tmp_path, keep_t_df=keep_t)
    assert res['allelic_admitted'].all()
    assert (res['dof_a'] == 24).all()
    assert np.array_equal(res['dof_nominal'].values, res['dof_a'].values.astype(float))
    assert np.allclose(res['pval_nominal'], res['pval_a'], rtol=1e-5, atol=0)


def test_dof_nominal_is_dof_t_below_the_floor(tmp_path):
    """10 informative allelic donors against the default floor of 15: the
    allelic channel is not admitted and dof_nominal is dof_t exactly."""
    d = _bank(SEED + 2, N=50, V=5, n_a=10, n_cov=3)
    res = _nominal(d, tmp_path)
    assert (~res['allelic_admitted']).all()
    assert (res['dof_t'] == 50 - 2 - 3).all() and (res['dof_a'] == 9).all()
    assert np.array_equal(res['dof_nominal'].values, res['dof_t'].values.astype(float))


def test_allelic_only_run_below_the_floor_is_still_tested(tmp_path):
    """keep_t_df all False switches the total channel off, so there is no
    combination to protect and the floor is waived (mixQTL's meta_analyze
    falls back to the channel it has): a 10-donor allelic channel is the
    statistic on its own 9 df, in map_nominal and map_cis alike, instead of
    an untested gene reported as p = 1."""
    d = _bank(SEED + 10, N=40, V=5, R=2, n_a=10)
    keep_t = pd.DataFrame(False, index=d['A_df'].index, columns=d['A_df'].columns)
    res = _nominal(d, tmp_path, keep_t_df=keep_t)
    assert res['allelic_admitted'].all()
    assert (res['dof_a'] == 9).all() and res['dof_t'].isna().all()
    ok = np.isfinite(res['slope_a_se'].astype(float)).values
    assert ok.any()
    assert (res['dof_nominal'][ok] == 9).all() and res['dof_nominal'][~ok].isna().all()
    assert np.allclose(res['pval_nominal'][ok], res['pval_a'][ok], rtol=1e-5, atol=0)
    cis = _cis(d, nperm=200, seed=SEED, keep_t_df=keep_t)
    assert cis['allelic_admitted'].astype(bool).all() and (cis['dof_nominal'] == 9).all()
    assert np.isfinite(cis['pval_perm'].astype(float)).all()


def test_a_gene_with_no_channel_is_reported_untested(tmp_path):
    """One informative allelic donor (the sparse-channel rule switches the
    channel off) and no total channel: nothing is tested, so the dof columns,
    pval_nominal and map_cis's pval_perm are NaN, not the 1.0 of a zero
    statistic."""
    d = _bank(SEED + 11, N=40, V=5, R=2, n_a=1)
    keep_t = pd.DataFrame(False, index=d['A_df'].index, columns=d['A_df'].columns)
    res = _nominal(d, tmp_path, keep_t_df=keep_t)
    assert (~res['allelic_admitted']).all()
    for col in ('dof_a', 'dof_t', 'dof_nominal', 'pval_nominal', 'pval_a', 'pval_t'):
        assert res[col].isna().all(), col
    cis = _cis(d, nperm=100, seed=SEED, keep_t_df=keep_t)
    for col in ('dof_nominal', 'pval_nominal', 'pval_perm', 'pval_beta'):
        assert cis[col].astype(float).isna().all(), col


def test_dof_nominal_matches_channel_ses_recomputed_by_hand(tmp_path):
    """Both channels admitted, the total with covariates. Each channel's
    fitted SE is recomputed here in float64 from the data -- through-origin
    WLS on the n_a informative donors for the allelic channel, WLS with
    intercept and covariates for the total -- and the Welch-Satterthwaite
    closed form of those SEs is map_nominal's dof_nominal, which lies
    strictly between min(nu_a, nu_t) and nu_a + nu_t; pval_nominal is the
    combined t, divided by the square root of Meier's factor
    1 + 4 f_a f_t (1/nu_a + 1/nu_t) built from the same SEs, referred to it."""
    N, n_a, n_cov = 60, 20, 4
    d = _bank(SEED + 3, N=N, V=4, n_a=n_a, n_cov=n_cov)
    res = _nominal(d, tmp_path).set_index('variant_id')
    gene = d['A_df'].index[0]
    a = d['A_df'].loc[gene].values.astype(float); va = d['Va_df'].loc[gene].values.astype(float)
    t = d['T_df'].loc[gene].values.astype(float); vt = d['Vt_df'].loc[gene].values.astype(float)
    C = d['covariates_df'].values.astype(float)
    nu_a, nu_t = n_a - 1, N - 2 - n_cov
    for vid in res.index:
        s = (d['xL_df'].loc[vid].values - d['xR_df'].loc[vid].values).astype(float)
        g2 = d['genotype_df'].loc[vid].values.astype(float) / 2
        k = va > 0
        w, x, y = 1 / va[k], s[k], a[k]
        b_a = (w * x * y).sum() / (w * x * x).sum()
        se_a = np.sqrt((w * (y - b_a * x) ** 2).sum() / nu_a / (w * x * x).sum())
        X = np.column_stack([np.ones(N), C, g2])
        W = 1 / vt
        XtWX = X.T @ (W[:, None] * X)
        beta = np.linalg.solve(XtWX, X.T @ (W * t))
        s2_t = (W * (t - X @ beta) ** 2).sum() / nu_t
        se_t = np.sqrt(s2_t * np.linalg.inv(XtWX)[-1, -1])
        w_a, w_t = 1 / se_a ** 2, 1 / se_t ** 2
        nu_c = (w_a + w_t) ** 2 / (w_a ** 2 / nu_a + w_t ** 2 / nu_t)
        f_a = w_a / (w_a + w_t)
        meier = 1 + 4 * f_a * (1 - f_a) * (1 / nu_a + 1 / nu_t)
        row = res.loc[vid]
        assert np.isclose(row['slope_a_se'], se_a, rtol=1e-4) and np.isclose(row['slope_t_se'], se_t, rtol=1e-4)
        assert row['dof_a'] == nu_a and row['dof_t'] == nu_t
        assert np.isclose(row['dof_nominal'], nu_c, rtol=1e-4), (vid, row['dof_nominal'], nu_c)
        assert min(nu_a, nu_t) < row['dof_nominal'] < nu_a + nu_t
        b_c = (w_a * b_a + w_t * beta[-1]) / (w_a + w_t)
        assert np.isclose(row['slope_se'], np.sqrt(meier / (w_a + w_t)), rtol=1e-4)
        p_c = 2 * stats.t.sf(abs(b_c) * np.sqrt((w_a + w_t) / meier), nu_c)
        assert np.isclose(row['pval_nominal'], p_c, rtol=1e-3), (vid, row['pval_nominal'], p_c)


# (c) -----------------------------------------------------------------------

def test_below_the_floor_the_combination_is_the_total_channel(tmp_path):
    """n_a = 8: combined slope, SE, p and dof are the total channel's to the
    bit, while pval_a is still reported, on its own n_a - 1 = 7 df."""
    d = _bank(SEED + 4, N=50, V=6, n_a=8, n_cov=2)
    res = _nominal(d, tmp_path)
    assert (~res['allelic_admitted']).all()
    assert np.array_equal(res['slope'].values, res['slope_t'].values)
    assert np.array_equal(res['slope_se'].values, res['slope_t_se'].values)
    assert np.array_equal(res['pval_nominal'].values, res['pval_t'].values)
    assert np.array_equal(res['dof_nominal'].values, res['dof_t'].values.astype(float))
    ok = np.isfinite(res['slope_a_se'].values)
    assert ok[0] and (res['dof_a'] == 7).all()
    t_a = res['slope_a'].astype(float) / res['slope_a_se'].astype(float)
    assert np.allclose(res['pval_a'][ok], get_t_pval(t_a[ok], 7), rtol=1e-5)


def test_map_cis_below_the_floor_equals_the_allelic_channel_absent():
    """The floor applies to the observed scan and to every permutation: a
    below-floor gene gives map_cis exactly what the same gene gives with its
    allelic channel switched off by keep_a_df (same seed, same records and
    sign flips), lead, statistic, pval_perm and pval_beta alike. Its
    pval_nominal is the total channel's t on dof_t."""
    d = _bank(SEED + 5, N=60, V=8, R=3, n_a=12, n_cov=3)
    floored = _cis(d, nperm=500, seed=SEED)
    keep_a = pd.DataFrame(False, index=d['A_df'].index, columns=d['A_df'].columns)
    absent = _cis(d, nperm=500, seed=SEED, keep_a_df=keep_a)
    assert (~floored['allelic_admitted'].astype(bool)).all()
    assert (floored['variant_id'] == absent['variant_id']).all()
    for col in ('slope', 'slope_se', 'pval_nominal', 'pval_perm', 'pval_beta', 'true_df',
                'beta_shape1', 'beta_shape2', 'dof_nominal'):
        assert np.array_equal(floored[col].astype(float).values, absent[col].astype(float).values,
                              equal_nan=True), col
    assert (floored['dof_nominal'] == 60 - 2 - 3).all()
    # the lead's reported channels are on the scan's fitted scale, so below
    # the floor the combined SE is the total channel's (the scan's SE comes
    # back through the float32 correlation scale, hence not bit-equal)
    assert np.allclose(floored['slope_se'].astype(float), floored['slope_t_se'].astype(float),
                       rtol=1e-4, atol=0)
    # the floor is what does it: admitted, the same gene's statistic moves
    admitted = _cis(d, nperm=500, seed=SEED, min_allelic_donors=2)
    assert admitted['allelic_admitted'].astype(bool).all()
    assert not np.array_equal(admitted['slope'].astype(float).values, floored['slope'].astype(float).values)


def test_map_cis_lead_reference_is_map_nominal_s(tmp_path):
    """A gene with both channels admitted: map_cis's lead carries the
    dof_nominal, pval_nominal, fitted per-channel SEs and pval_cis_trans that
    map_nominal reports for that pair (the per-channel SEs were known-variance
    in map_cis until 2026-09-27)."""
    d = _bank(SEED + 6, N=60, V=8, R=3, n_a=40, n_cov=3)
    cis = _cis(d, nperm=200, seed=SEED)
    pairs = _nominal(d, tmp_path)
    assert cis['allelic_admitted'].astype(bool).all()
    for pid, row in cis.iterrows():
        pair = pairs[(pairs['phenotype_id'] == pid) & (pairs['variant_id'] == row['variant_id'])].iloc[0]
        for col in ('dof_nominal', 'pval_nominal', 'slope_a_se', 'slope_t_se', 'pval_cis_trans'):
            assert np.isclose(float(row[col]), float(pair[col]), rtol=1e-4), (pid, col, row[col], pair[col])
        assert min(pair['dof_a'], pair['dof_t']) < row['dof_nominal'] < pair['dof_a'] + pair['dof_t']


def test_pval_cis_trans_is_on_the_difference_s_welch_satterthwaite_dof(tmp_path):
    """In default mode z = (slope_a - slope_t) / sqrt(se_a^2 + se_t^2) is
    referred to (se_a^2 + se_t^2)^2 / (se_a^4/dof_a + se_t^4/dof_t), not the
    shared N - 2 - n_cov; with 4 allelic donors (reported below the floor)
    the two differ."""
    d = _bank(SEED + 9, N=50, V=6, n_a=4, n_cov=2)
    res = _nominal(d, tmp_path)
    se_a, se_t = res['slope_a_se'].astype(float), res['slope_t_se'].astype(float)
    ok = np.isfinite(se_a)
    assert ok.any() and (res['dof_a'] == 3).all()
    z = (res['slope_a'].astype(float) - res['slope_t'].astype(float)) / np.sqrt(se_a ** 2 + se_t ** 2)
    nu = (se_a ** 2 + se_t ** 2) ** 2 / (se_a ** 4 / 3 + se_t ** 4 / (50 - 2 - 2))
    assert np.allclose(res['pval_cis_trans'][ok], 2 * stats.t.sf(np.abs(z[ok]), nu[ok]), rtol=1e-4)
    assert not np.allclose(res['pval_cis_trans'][ok], 2 * stats.t.sf(np.abs(z[ok]), 50 - 2 - 2), rtol=1e-3)


# (d) -----------------------------------------------------------------------

@pytest.mark.parametrize('n_ase_cov', [0, 2])
def test_every_donor_informative_gives_dof_a_n_minus_one_minus_ase_design(tmp_path, n_ase_cov):
    """Default floor, n_a = N: dof_a = N - 1 - n_cov_a (through-origin, so
    N - 1 with no allelic covariates) and dof_t = N - 2 - n_cov."""
    N, n_cov = 45, 3
    d = _bank(SEED + 7, N=N, V=5, n_cov=n_cov)
    ase_cov = (pd.DataFrame(np.random.RandomState(SEED).normal(size=(N, n_ase_cov)),
                            index=d['samples'], columns=[f'e{k}' for k in range(n_ase_cov)])
               if n_ase_cov else None)
    res = _nominal(d, tmp_path, ase_covariates_df=ase_cov)
    assert res['allelic_admitted'].all()
    assert (res['dof_a'] == N - 1 - n_ase_cov).all()
    assert (res['dof_t'] == N - 2 - n_cov).all()


def test_known_variance_path_keeps_the_shared_reference(tmp_path):
    """se_mode='model' (deprecated, kept to reproduce historical results) has
    no fitted scale and so no per-channel dof: every reference is the shared
    N - 2 - max(n_cov, n_cov_a), and a 5-donor allelic channel still enters."""
    N, n_cov = 50, 3
    d = _bank(SEED + 8, N=N, V=4, n_a=5, n_cov=n_cov)
    with pytest.warns(RuntimeWarning):
        res = _nominal(d, tmp_path, se_mode='model')
    for col in ('dof_a', 'dof_t', 'dof_nominal'):
        assert (res[col] == N - 2 - n_cov).all(), col
    assert res['allelic_admitted'].all()
