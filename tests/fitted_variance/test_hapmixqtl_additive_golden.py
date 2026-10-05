"""The additive path is pinned against the numbers it produced before the
variance-model change (golden values from commit 99921fd)."""
import sys, numpy as np, pandas as pd, pytest, torch
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_hapmixqtl import _make_dataset
from tensorqtl.hapmixqtl import map_cis


# Produced by map_cis at commit 99921fd (the tree before variance_model was
# added), on _make_dataset(seed=120) with nperm=500, seed=3, no covariates.
# Regenerate ONLY from a tree that predates the variance-model change, never
# from the current one: a golden refreshed from the code it is meant to
# constrain pins nothing.
_COLS = ['variant_id', 'pval_nominal', 'slope', 'slope_se', 'slope_a', 'slope_a_se',
         'slope_t', 'slope_t_se', 'tau_a', 'tau_t', 'tau_a_null', 'tau_t_null',
         'pval_perm', 'pval_beta', 'beta_shape1', 'beta_shape2', 'alpha_cis',
         'pval_cis_trans', 'true_df', 'pval_true_df', 'num_var', 'ma_count', 'af']
GOLDEN_ADDITIVE = {
    'ENSG00000000.1': ['chr1_10000_A_G', 2.302973671566499e-19, 1.006585955619812, 0.08400656282901764, 1.009158730506897, 0.11643827706575394, 1.0037930011749268, 0.12131837755441666, 0.4860413670539856, 0.08416609466075897, 0.4860413670539856, 0.08416609466075897, 0.001996007984031936, 1.1121299365130662e-21, 1.1558247804641724, 31.861724853515625, 1.005345454018597, 0.9746257565271272, 82.21986389160156, 2.4795298319066812e-20, 30.0, 70.0, 0.4375],
    'ENSG00000001.1': ['chr1_33000_A_G', 0.053750500714178584, 0.07312767952680588, 0.03733929246664047, 0.09944505244493484, 0.04094909131526947, -0.05670815706253052, 0.09095367044210434, 0.01914081536233425, 0.02368924580514431, 0.01914081536233425, 0.02368924580514431, 0.7864271457085829, 0.78095115523142, 1.0704216957092285, 34.91982650756836, -1.7536287122729017, 0.12151448660306408, 84.00511169433594, 0.04526629110016961, 30.0, 80.0, 0.5],
    'ENSG00000002.1': ['chr1_21000_A_G', 0.005370254148702639, 0.13178227841854095, 0.046011775732040405, 0.1398134082555771, 0.051629528403282166, 0.10078535974025726, 0.10143061727285385, 0.051224756985902786, 0.04802493751049042, 0.051224756985902786, 0.04802493751049042, 0.09780439121756487, 0.09095944983252949, 0.9592214822769165, 30.798154830932617, 1.3872392638762456, 0.732589523418743, 90.26998138427734, 0.002733581592468465, 30.0, 74.0, 0.5375000238418579],
}
GOLDEN_ADDITIVE_REFIT = {
    'ENSG00000000.1': ['chr1_10000_A_G', 3.4205600129593813e-47, 1.0161983966827393, 0.03117518685758114, 1.0160465240478516, 0.03451821208000183, 1.0168707370758057, 0.07261504977941513, 0.0, 0.0, 0.4860413670539856, 0.08416609466075897, 0.001996007984031936, 1.1121299365130662e-21, 1.1558247804641724, 31.861724853515625, 0.999189461356392, 0.9918470640678694, 82.21986389160156, 2.4795298319066812e-20, 30.0, 70.0, 0.4375],
    'ENSG00000001.1': ['chr1_33000_A_G', 0.04002989479743862, 0.07536864280700684, 0.036090198904275894, 0.09982312470674515, 0.03929077088832855, -0.0566537119448185, 0.09129250049591064, 0.014101069420576096, 0.024237757548689842, 0.01914081536233425, 0.02368924580514431, 0.7864271457085829, 0.78095115523142, 1.0704216957092285, 34.91982650756836, -1.761987366405475, 0.11944383545090703, 84.00511169433594, 0.04526629110016961, 30.0, 80.0, 0.5],
    'ENSG00000002.1': ['chr1_21000_A_G', 0.002626538006199987, 0.13571792840957642, 0.043664298951625824, 0.14360257983207703, 0.04835052788257599, 0.10085585713386536, 0.10166861861944199, 0.03871450200676918, 0.04847912862896919, 0.051224756985902786, 0.04802493751049042, 0.09780439121756487, 0.09095944983252949, 0.9592214822769165, 30.798154830932617, 1.4238397641246974, 0.7051996173849491, 90.26998138427734, 0.002733581592468465, 30.0, 74.0, 0.5375000238418579],
}
# Measured GPU-vs-CPU spread of this same call on this machine (float32 kernels
# plus the beta fit): 2e-7 on the nominal columns, 1.4e-5 on pval_beta and
# pval_true_df. pval_perm, variant_id, num_var, ma_count and af are bit-equal
# across devices, so those are pinned exactly.
_RTOL = {'pval_beta': 1e-4, 'pval_true_df': 1e-4, 'beta_shape1': 1e-4,
         'beta_shape2': 1e-4, 'true_df': 1e-4}
_EXACT = ('pval_perm', 'num_var', 'ma_count', 'af')


_PERM_COLS = ('pval_perm', 'pval_beta', 'beta_shape1', 'beta_shape2', 'true_df', 'pval_true_df')


@pytest.mark.parametrize('scheme', ['residuals', 'records'])
@pytest.mark.parametrize('refit,golden', [(False, GOLDEN_ADDITIVE), (True, GOLDEN_ADDITIVE_REFIT)])
def test_additive_reproduces_the_pre_change_numbers(refit, golden, scheme):
    """variance_model='additive' must reproduce the numbers the shipped model
    produced BEFORE the two-component refactor, on both the null-fit and the
    tau_refit lead path.

    TestVarianceModels.test_additive_is_the_default_and_unchanged compares
    map_cis(...) with map_cis(..., variance_model='additive'): both run the new
    code, so it pins the default argument value and nothing about the additive
    arithmetic. This pins the arithmetic. It is the test that fails if
    _channel_weights' additive branch, _estimate_tau_informative's design, the
    order of the tau/weight computation or the dtype of the whitening ever
    moves the default path.

    The golden values were produced with the whitened-residual permutation
    (perm_scheme='residuals'), so under that scheme every column is pinned.
    The donor-record permutation (the default since 2026-09-17) changes the
    permutation null by design; under it the nominal columns are pinned
    exactly and the permutation-derived columns only loosely.
    """
    d = _make_dataset(seed=120)
    res = map_cis(d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'],
                  d['Va_df'], d['Vt_df'], d['pos_df'], xL_df=d['xL_df'], xR_df=d['xR_df'],
                  nperm=500, window=1000000, seed=3, verbose=False, tau_refit=refit,
                  perm_scheme=scheme,
                  # The goldens come from commit 99921fd, whose DEFAULTS these were.
                  # Stated explicitly because the defaults moved on 2026-09-21 to
                  # tau_mode='zero' + se_mode='fitted'; leaving them implicit is what
                  # silently broke this test then.
                  tau_mode='estimate', se_mode='model')
    assert list(res.index) == list(golden), list(res.index)
    exp = pd.DataFrame(list(golden.values()), index=list(golden), columns=_COLS)
    assert (res['variant_id'] == exp['variant_id']).all()
    for col in _COLS[1:]:
        got = res[col].astype(float).values
        want = exp[col].astype(float).values
        if scheme == 'records' and col in _PERM_COLS:
            if col.startswith('pval'):
                assert np.all(np.abs(got - want) < 0.15), (col, got, want)
            continue
        if col in _EXACT:
            np.testing.assert_array_equal(got, want, err_msg=col)
        else:
            np.testing.assert_allclose(got, want, rtol=_RTOL.get(col, 1e-5),
                                       atol=1e-12, err_msg=col)
    # the additive path must not have acquired a fitted c, and the columns the
    # two-component models add must be inert here
    assert (res['c_a'].astype(float) == 1.0).all()
    assert (res['c_a_null'].astype(float) == 1.0).all()
    assert res['c_a_converged'].astype(bool).all()
    assert not res['variance_prior'].astype(bool).any()
    assert (res['variance_model'] == 'additive').all()
