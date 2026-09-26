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

perm_scheme='records_signflip', the default since 2026-09-25, adds a random
swap of each permuted record's haplotype labels (negating its allelic log
ratio). Its pins: the swap negates the allelic numerator exactly and touches
nothing else; it centres the permuted allelic slope where 'records' does not;
it refuses to run without its signs; and a run without phase genotypes is
identical to 'records', which also pins that the signs are drawn after the
permutation indices.
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


def test_every_scheme_available_and_map_cis_records_the_scheme():
    from tests.test_hapmixqtl import _make_dataset
    d = _make_dataset(seed=125)
    kw = dict(xL_df=d['xL_df'], xR_df=d['xR_df'], nperm=200, window=1000000, seed=5, verbose=False)
    args = (d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'], d['Va_df'], d['Vt_df'], d['pos_df'])
    r0 = hapmixqtl.map_cis(*args, **kw)
    r1 = hapmixqtl.map_cis(*args, perm_scheme='records', **kw)
    r2 = hapmixqtl.map_cis(*args, perm_scheme='residuals', **kw)
    assert (r0['perm_scheme'] == 'records_signflip').all()        # the default since 2026-09-25
    assert (r1['perm_scheme'] == 'records').all() and (r2['perm_scheme'] == 'residuals').all()
    # The scheme touches only the permutation columns. Compared with .equals()
    # rather than ==, because under the shipped default tau_mode='zero' there is
    # no tau parameter in the model and tau_a/tau_t come back as None, which ==
    # reports as unequal to itself.
    for col in ('variant_id', 'pval_nominal', 'slope', 'slope_se', 'tau_a', 'tau_t'):
        assert r0[col].equals(r1[col]), col
        assert r1[col].equals(r2[col]), col
    with pytest.raises(ValueError):
        hapmixqtl.map_cis(*args, perm_scheme='shuffle', **kw)


# ---------------------------------------------------------------------------
# perm_scheme='records_signflip': the record permutation plus a random swap of
# each permuted record's haplotype labels, which negates its allelic log ratio.
# ---------------------------------------------------------------------------

def _allelic_setup(rng, N, device, nuisance):
    T = lambda x: torch.tensor(np.asarray(x, dtype=float), dtype=torch.float64, device=device)
    va = 10 ** rng.uniform(-2, 0, N)
    sqrt_wa = T(1 / np.sqrt(va))
    a = T(rng.normal(0, 1, N) * np.sqrt(va))
    C = T(rng.normal(size=(N, 2))) if nuisance else None
    res = WeightedResidualizer(C, sqrt_wa, intercept=False)
    g, sign, _ = _design(rng, N, 50, device)
    return sign, a, sqrt_wa, res, T


@pytest.mark.parametrize('nuisance', [False, True])
def test_signflip_negates_the_allelic_numerator_exactly(device, nuisance):
    """Swapping every sign negates xy exactly and leaves xx and yy untouched,
    on the through-origin branch and on the batched-QR branch with nuisance
    columns. This pins that the swap lands on the phenotype value before the
    projection and nowhere else (not on the weights, not on the covariates)."""
    rng = np.random.RandomState(31)
    N, nperm = 60, 30
    sign, a, sqrt_wa, res, T = _allelic_setup(rng, N, device, nuisance)
    perm = torch.tensor(np.array([rng.permutation(N) for _ in range(nperm)]), dtype=torch.long, device=device)
    flip = T(rng.choice([-1.0, 1.0], size=(nperm, N)))
    xy0, xx0, yy0 = hapmixqtl._record_permutation_channel(sign, a, sqrt_wa, res, perm)
    xy1, xx1, yy1 = hapmixqtl._record_permutation_channel(sign, a, sqrt_wa, res, perm, flip_t=flip)
    xy2, xx2, yy2 = hapmixqtl._record_permutation_channel(sign, a, sqrt_wa, res, perm, flip_t=-flip)
    assert torch.allclose(xy1, -xy2, rtol=0, atol=1e-12)
    assert torch.equal(xx0, xx1) and torch.equal(xx1, xx2)
    assert torch.allclose(yy1, yy2, rtol=1e-12, atol=1e-12)
    if not nuisance:
        # through the origin the residual is the value itself, so a swap leaves
        # its square alone; with nuisance columns the swapped values are
        # projected afresh and the residual sum of squares legitimately moves
        assert torch.allclose(yy0, yy1, rtol=1e-12, atol=1e-12)
    ones = torch.ones_like(flip)
    xy3, _, _ = hapmixqtl._record_permutation_channel(sign, a, sqrt_wa, res, perm, flip_t=ones)
    assert torch.allclose(xy3, xy0, rtol=0, atol=1e-12)     # all +1 is the unswapped scheme
    with pytest.raises(ValueError):
        hapmixqtl._record_permutation_channel(sign, a, sqrt_wa, res, perm, flip_t=flip[:, :-1])


def test_signflip_centres_the_permuted_allelic_slope(device):
    """A gene with a net allelic imbalance (every record's log ratio near +0.3)
    tested at a variant whose heterozygotes all carry ALT on haplotype L. The
    through-origin permuted slope under 'records' is centred on the imbalance,
    because every permutation lands positive ratios on s = +1; with the swap it
    is centred on zero within Monte Carlo error. On 46 BrainVar genes the
    'records' offset reached 0.21 standard errors (2026-09-25)."""
    rng = np.random.RandomState(32)
    N, nperm = 80, 4000
    T = lambda x: torch.tensor(np.asarray(x, dtype=float), dtype=torch.float64, device=device)
    va = 10 ** rng.uniform(-2, -0.5, N)
    sqrt_wa = T(1 / np.sqrt(va))
    a = T(0.3 + rng.normal(0, 1, N) * np.sqrt(va))
    sign = T(np.where(np.arange(N) < 30, 1.0, 0.0)[None, :])
    res = WeightedResidualizer(None, sqrt_wa, intercept=False)
    perm = torch.tensor(np.array([rng.permutation(N) for _ in range(nperm)]), dtype=torch.long, device=device)
    flip = T(rng.choice([-1.0, 1.0], size=(nperm, N)))
    for flip_t, centred in ((None, False), (flip, True)):
        xy, xx, _ = hapmixqtl._record_permutation_channel(sign, a, sqrt_wa, res, perm, flip_t=flip_t)
        b = (xy / xx).flatten().cpu().numpy()
        z = b.mean() / (b.std(ddof=1) / np.sqrt(nperm))
        if centred:
            assert abs(z) < 4, z
        else:
            assert b.mean() > 0.2 and z > 20, (b.mean(), z)


def test_signflip_needs_its_signs(device):
    rng = np.random.RandomState(33)
    N, V, nperm = 40, 20, 10
    g, sign, T = _design(rng, N, V, device)
    sqrt_wa = T(np.full(N, 2.0)); sqrt_wt = T(np.full(N, 3.0))
    a = T(rng.normal(size=N)); t = T(rng.normal(size=N))
    res_a = WeightedResidualizer(None, sqrt_wa, intercept=False)
    res_t = WeightedResidualizer(None, sqrt_wt, intercept=True)
    perm = torch.tensor(np.array([rng.permutation(N) for _ in range(nperm)]), dtype=torch.long, device=device)
    with pytest.raises(ValueError, match='flip_t'):
        calculate_hapmixqtl_permutations(g, sign, a, t, sqrt_wa, sqrt_wt, res_a, res_t, perm,
                                         dof=N - 2, perm_scheme='records_signflip')
    # 'records' ignores any flips it is handed, so a stray argument cannot change it
    flip = T(rng.choice([-1.0, 1.0], size=(nperm, N)))
    out0 = calculate_hapmixqtl_permutations(g, sign, a, t, sqrt_wa, sqrt_wt, res_a, res_t, perm,
                                            dof=N - 2, perm_scheme='records')
    out1 = calculate_hapmixqtl_permutations(g, sign, a, t, sqrt_wa, sqrt_wt, res_a, res_t, perm,
                                            dof=N - 2, perm_scheme='records', flip_t=flip)
    assert torch.equal(out0[3], out1[3])


def test_signflip_leaves_a_total_only_run_identical():
    """Without phase genotypes the allelic predictor is zero and the swap has
    nothing to act on, so the permuted statistics, and pval_perm, must be
    identical to 'records' at the same seed. This also pins that the signs are
    drawn AFTER the permutation indices: drawing them first would change the
    indices and hence the total channel."""
    from tests.test_hapmixqtl import _make_dataset
    d = _make_dataset(seed=126)
    kw = dict(nperm=300, window=1000000, seed=7, verbose=False)
    args = (d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'], d['Va_df'], d['Vt_df'], d['pos_df'])
    r_rec = hapmixqtl.map_cis(*args, perm_scheme='records', **kw)
    r_flip = hapmixqtl.map_cis(*args, perm_scheme='records_signflip', **kw)
    for col in ('variant_id', 'pval_nominal', 'slope', 'pval_perm'):
        assert r_rec[col].equals(r_flip[col]), col
