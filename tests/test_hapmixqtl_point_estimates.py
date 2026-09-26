"""The phenotype since 2026-09-25 (summaries_from_point_estimates) and
genotype-tied covariates in map_cis.

User rules pinned here: every VALUE comes from Salmon's point estimates and
the Gibbs draws only give its measurement VARIANCE, through the identical
transform; the unit is log2(CPM + 1) with CPM from edgeR's effective library
size, except the allelic ratio log2((L + 1/2) / (R + 1/2)); a zero-read total
donor keeps the counting variance at count + 1/2; genotype PCs stay with the
genotypes under permutation.
"""
import numpy as np
import pandas as pd
import pytest

import tensorqtl.hapmixqtl as hapmixqtl
from tensorqtl.hapmixqtl import summaries_from_point_estimates, LN2


def _toy(seed=0, F=3, N=6, D=200):
    rng = np.random.RandomState(seed)
    pL = rng.gamma(2, 40, (F, N)); pR = rng.gamma(2, 40, (F, N))
    pT = pL + pR + rng.gamma(2, 30, (F, N))
    yL = np.maximum(pL[..., None] + rng.normal(0, 3, (F, N, D)), 0)
    yR = np.maximum(pR[..., None] + rng.normal(0, 3, (F, N, D)), 0)
    yT = np.maximum(pT[..., None] + rng.normal(0, 4, (F, N, D)), 0)
    L = rng.uniform(1e7, 3e7, N)
    return pL, pR, pT, yL, yR, yT, L


def test_values_are_the_point_estimates_not_the_draws():
    pL, pR, pT, yL, yR, yT, L = _toy()
    A, T, Va, Vt, _ = summaries_from_point_estimates(pL, pR, pT, L, yL, yR, yT, count_noise=False)
    assert np.allclose(A, np.log2((pL + 0.5) / (pR + 0.5)), rtol=0, atol=1e-12)
    assert np.allclose(T, np.log2(pT / L[None, :] * 1e6 + 1), rtol=0, atol=1e-12)
    # shifting every draw leaves the VALUES untouched: they never read the draws
    A2, T2, _, _, _ = summaries_from_point_estimates(pL, pR, pT, L, yL + 50, yR, yT * 3, count_noise=False)
    assert np.array_equal(A, A2) and np.array_equal(T, T2)


def test_variance_is_the_same_transform_of_the_draws():
    pL, pR, pT, yL, yR, yT, L = _toy(1)
    _, _, Va, Vt, _ = summaries_from_point_estimates(pL, pR, pT, L, yL, yR, yT, count_noise=False)
    a_d = np.log2((yL + 0.5) / (yR + 0.5))
    t_d = np.log2(yT / L[None, :, None] * 1e6 + 1)
    assert np.allclose(Va, a_d.var(2), rtol=1e-12, atol=0)
    assert np.allclose(Vt, t_d.var(2), rtol=1e-12, atol=0)
    # the draws' absolute scale reaches the variance (no free rescaling)
    _, _, _, Vt2, _ = summaries_from_point_estimates(pL, pR, pT, L, yL, yR, yT * 2, count_noise=False)
    assert not np.allclose(Vt2, Vt)


def test_library_size_enters_cpm_and_the_counting_term():
    pL, pR, pT, yL, yR, yT, L = _toy(2)
    _, T1, _, Vt1, _ = summaries_from_point_estimates(pL, pR, pT, L, yL, yR, yT)
    _, T2, _, Vt2, _ = summaries_from_point_estimates(pL, pR, pT, L * 2, yL, yR, yT)
    assert (T2 < T1).all()                     # a larger library means lower CPM
    k = 1e6 / L
    y = pT + 0.5
    q_t = k[None, :] ** 2 * y / ((k[None, :] * y + 1) ** 2 * LN2 ** 2)
    _, _, _, Vt0, _ = summaries_from_point_estimates(pL, pR, pT, L, yL, yR, yT, count_noise=False)
    assert np.allclose(Vt1 - Vt0, q_t, rtol=1e-10, atol=0)


def test_zero_read_total_donor_keeps_a_positive_variance():
    pL, pR, pT, yL, yR, yT, L = _toy(3)
    pT[0, 0] = 0.0; yT[0, 0, :] = 0.0
    pL[0, 0] = pR[0, 0] = 0.0; yL[0, 0, :] = yR[0, 0, :] = 0.0
    A, T, Va, Vt, _ = summaries_from_point_estimates(pL, pR, pT, L, yL, yR, yT)
    assert T[0, 0] == 0.0                      # log2(0 + 1)
    assert Vt[0, 0] > 0                        # the count + 1/2 floor
    k = 1e6 / L[0]
    assert np.isclose(Vt[0, 0], k * k * 0.5 / ((k * 0.5 + 1) ** 2 * LN2 ** 2), rtol=1e-12)
    assert Va[0, 0] == 0.0                     # no haplotype-informative reads: excluded
    assert A[0, 0] == 0.0


def test_inputs_are_validated():
    pL, pR, pT, yL, yR, yT, L = _toy(4)
    with pytest.raises(ValueError):
        summaries_from_point_estimates(pL, pR, pT, L[:-1], yL, yR, yT)
    with pytest.raises(ValueError):
        summaries_from_point_estimates(pL, pR, pT, -L, yL, yR, yT)
    with pytest.raises(ValueError):
        summaries_from_point_estimates(pL, pR, pT[:, :-1], L, yL, yR, yT)


def test_map_cis_genotype_covariates_same_fit_different_null():
    """Splitting covariates into RNA-tied and genotype-tied changes nothing in
    the nominal fit (same design columns) and changes the permutation null."""
    from tests.test_hapmixqtl import _make_dataset
    d = _make_dataset(seed=127)
    N = d['A_df'].shape[1]
    rng = np.random.RandomState(9)
    cov = pd.DataFrame(rng.normal(size=(N, 3)), index=d['A_df'].columns, columns=['rin', 'e1', 'e2'])
    gcov = pd.DataFrame(rng.normal(size=(N, 2)), index=d['A_df'].columns, columns=['geno_pc1', 'geno_pc2'])
    args = (d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'], d['Va_df'], d['Vt_df'], d['pos_df'])
    kw = dict(xL_df=d['xL_df'], xR_df=d['xR_df'], nperm=300, window=1000000, seed=11, verbose=False)
    r_split = hapmixqtl.map_cis(*args, covariates_df=cov, genotype_covariates_df=gcov, **kw)
    r_joint = hapmixqtl.map_cis(*args, covariates_df=pd.concat([cov, gcov], axis=1), **kw)
    assert (r_split['n_genotype_covariates'] == 2).all() and (r_joint['n_genotype_covariates'] == 0).all()
    for col in ('variant_id', 'pval_nominal', 'slope', 'slope_se'):
        assert r_split[col].equals(r_joint[col]), col
    assert not r_split['pval_perm'].equals(r_joint['pval_perm'])
    with pytest.raises(ValueError):
        hapmixqtl.map_cis(*args, covariates_df=cov, genotype_covariates_df=cov, **kw)
