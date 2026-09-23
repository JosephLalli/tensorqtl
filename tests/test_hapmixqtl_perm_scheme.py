"""The permutation null of hapmixQTL (calculate_hapmixqtl_permutations).

perm_scheme='records' permutes donor records (whitened phenotype value,
weight, covariate row together; genotypes fixed) and is, by relabeling,
exactly the distribution of the statistic under a permutation of the
genotype columns. perm_scheme='residuals' permutes leverage-standardized
whitened residuals at fixed weights and is exact only when those residuals
are exchangeable across donors; on BrainVar they are not (weights spanning
two decades on well-expressed genes) and it ran a third short at nominal
5% (docstring of calculate_hapmixqtl_permutations).

Two pins: (1) the record scheme equals the genotype permutation to float
precision, on both channels with nuisance columns on each; (2) on a design
whose residual magnitude is correlated with the weights, null data made by
redrawing the noise with the genotypes fixed, the record scheme is at
nominal and the residual scheme is far below it.
"""
import numpy as np
import pytest
import torch

import tensorqtl.hapmixqtl as hapmixqtl
from tensorqtl.hapmixqtl import WeightedResidualizer, calculate_hapmixqtl_permutations


def _design(rng, N, V, device, dtype=torch.float64):
    T = lambda x: torch.tensor(np.asarray(x, dtype=float), dtype=dtype, device=device)
    g = rng.choice([0, 1, 2], size=(V, N), p=[0.25, 0.5, 0.25]).astype(float)
    sign = np.where(g == 1, rng.choice([-1.0, 1.0], size=(V, N)), 0.0)
    return T(g), T(sign), T


def test_record_permutation_equals_the_genotype_permutation(device):
    """For each permutation s the record scheme's maximum must equal the
    observed maximum with genotype columns indexed by argsort(s) and the
    records fixed, with covariates on the total channel and a nuisance
    column on the allelic one, so both branches of the routine are used."""
    rng = np.random.RandomState(21)
    N, V, nperm = 70, 120, 40
    g, sign, T = _design(rng, N, V, device)
    va = 10 ** rng.uniform(-2, 0, N); va[rng.choice(N, 8, replace=False)] = 0.0   # some non-informative donors
    vt = 10 ** rng.uniform(-3, -1.5, N)
    tau_a, tau_t = 0.02, 0.01
    sqrt_wa = T(np.where(va > 0, 1 / np.sqrt(va + tau_a), 0.0))
    sqrt_wt = T(1 / np.sqrt(vt + tau_t))
    a = T(rng.normal(0, 1, N) * np.sqrt(va + tau_a) * (va > 0))
    C_t = T(rng.normal(size=(N, 5)))
    C_a = T(rng.normal(size=(N, 1)))
    t = T(rng.normal(0, 1, N) * np.sqrt(vt + tau_t) + C_t.cpu().numpy() @ np.array([0.3, -0.2, 0.1, 0.0, 0.4]))
    res_a = WeightedResidualizer(C_a, sqrt_wa, intercept=False)
    res_t = WeightedResidualizer(C_t, sqrt_wt, intercept=True)
    perm = torch.tensor(np.array([rng.permutation(N) for _ in range(nperm)]), dtype=torch.long, device=device)
    dof = N - 2 - 5
    _, _, _, max_r2_rec, _ = calculate_hapmixqtl_permutations(
        g, sign, a, t, sqrt_wa, sqrt_wt, res_a, res_t, perm, dof=dof, perm_scheme='records')
    for k in range(nperm):
        inv = torch.argsort(perm[k])
        r_nom, _, _, _, _ = calculate_hapmixqtl_permutations(
            g[:, inv], sign[:, inv], a, t, sqrt_wa, sqrt_wt, res_a, res_t, perm[:1], dof=dof, perm_scheme='records')
        assert abs(float(r_nom) ** 2 - float(max_r2_rec[k])) < 1e-9, (k, float(r_nom) ** 2, float(max_r2_rec[k]))


def test_record_scheme_is_calibrated_where_the_residual_scheme_is_not(device):
    """Weights 1/(v + tau) with v spanning two decades; the true noise variance
    is (v + tau) f(v) with f rising in v, so the lowest-weight donors carry the
    largest standardized residuals, as on BrainVar's well-expressed genes.
    Null data: genotypes fixed, noise redrawn. The residual scheme's null has
    the scale of the unweighted mean of the standardized residuals and rejects
    far too rarely; the record scheme, whose null has the weight-weighted
    scale, stays near nominal (not exactly: it weights by w where the true
    null weights by w s^2)."""
    rng = np.random.RandomState(22)
    N, V, nperm, draws = 80, 150, 400, 300
    g, sign, T = _design(rng, N, V, device)
    va = 10 ** rng.uniform(-2, 0, N)
    tau_a, tau_t = 0.005, 0.01
    f = (va / np.median(va)) ** 0.45
    sqrt_wa = T(1 / np.sqrt(va + tau_a))
    vt = np.full(N, 0.02)
    sqrt_wt = T(1 / np.sqrt(vt + tau_t))
    res_a = WeightedResidualizer(None, sqrt_wa, intercept=False)
    res_t = WeightedResidualizer(None, sqrt_wt, intercept=True)
    perm = torch.tensor(np.array([rng.permutation(N) for _ in range(nperm)]), dtype=torch.long, device=device)
    w = 1 / (va + tau_a)
    z2 = f
    R = float((w * z2).sum() / w.sum() / z2.mean())
    assert R < 0.8, R                                   # the design does correlate residual size with weight
    rej = {'records': 0, 'residuals': 0}
    for d in range(draws):
        a = T(rng.normal(0, 1, N) * np.sqrt((va + tau_a) * f))
        t = T(rng.normal(0, 1, N) * np.sqrt(vt + tau_t))
        for scheme in rej:
            r_nom, _, _, max_r2, _ = calculate_hapmixqtl_permutations(
                g, sign, a, t, sqrt_wa, sqrt_wt, res_a, res_t, perm, dof=N - 2, perm_scheme=scheme)
            p = (1 + int((max_r2 >= float(r_nom) ** 2).sum())) / (nperm + 1)
            rej[scheme] += p < 0.05
    rate = {k: v / draws for k, v in rej.items()}
    assert 0.025 <= rate['records'] <= 0.085, rate      # 300 draws: s.e. 0.013
    assert rate['residuals'] < 0.03, rate
    assert rate['residuals'] < rate['records'] - 0.02, rate


def test_residual_scheme_still_available_and_map_cis_records_the_scheme():
    from tests.test_hapmixqtl import _make_dataset
    d = _make_dataset(seed=125)
    kw = dict(xL_df=d['xL_df'], xR_df=d['xR_df'], nperm=200, window=1000000, seed=5, verbose=False)
    args = (d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'], d['Va_df'], d['Vt_df'], d['pos_df'])
    r1 = hapmixqtl.map_cis(*args, **kw)
    r2 = hapmixqtl.map_cis(*args, perm_scheme='residuals', **kw)
    assert (r1['perm_scheme'] == 'records').all() and (r2['perm_scheme'] == 'residuals').all()
    # The scheme touches only the permutation columns. Compared with .equals()
    # rather than ==, because under the shipped default tau_mode='zero' there is
    # no tau parameter in the model and tau_a/tau_t come back as None, which ==
    # reports as unequal to itself.
    for col in ('variant_id', 'pval_nominal', 'slope', 'slope_se', 'tau_a', 'tau_t'):
        assert r1[col].equals(r2[col]), col
    with pytest.raises(ValueError):
        hapmixqtl.map_cis(*args, perm_scheme='shuffle', **kw)
