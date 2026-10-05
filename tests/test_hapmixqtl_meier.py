"""Portable checks for Meier's correction of the combined standard error.

The combined slope is the inverse-variance weighted mean of the two channel
slopes with weights w_k = 1/se_k^2 estimated from the same residuals it
combines (a Graybill-Deal mean), so its plug-in variance 1/(w_a + w_t) is too
small. Meier (1953) gives, to first order in 1/nu_k, true over expected
reported variance M = 1 + 4 f_a f_t (1/nu_a + 1/nu_t), f_k = w_k/(w_a + w_t)
the weight shares and nu_k the channels' residual df (_meier_factor). The
combined SE becomes se_c sqrt(M) and the combined t falls by sqrt(M) on the
unchanged Welch-Satterthwaite dof, in map_nominal, map_cis's observed scan,
every permutation and the lead alike.

The tests use fixed tensors and fabricated phenotypes. They check the formula,
its use by the nominal fit, and the cases where fewer than two fitted channels
carry weight or no residual scale is fitted.
"""
import contextlib

import numpy as np
import pandas as pd
import pytest
import torch

from tensorqtl.hapmixqtl import _meier_factor, _satterthwaite_dof
from tests.test_hapmixqtl_allelic_df import _bank, _nominal

SEED = 42


def test_meier_and_satterthwaite_known_answers():
    """Fixed weights exercise two-channel, one-channel and empty cases."""
    w_a = torch.tensor([4.0, 1.0, 0.0, 0.0])
    w_t = torch.tensor([1.0, 4.0, 9.0, 0.0])
    dof_a, dof_t = 9.0, 19.0

    total = w_a.double() + w_t.double()
    f_a = torch.where(total > 0, w_a.double() / total.clamp(min=1e-300), 0.0)
    expected_m = 1.0 + 4.0 * f_a * (1.0 - f_a) * (1.0 / dof_a + 1.0 / dof_t)
    expected_m[(w_a == 0) | (w_t == 0)] = 1.0
    torch.testing.assert_close(_meier_factor(w_a, w_t, dof_a, dof_t), expected_m)

    expected_nu = total.square() / (
        w_a.double().square() / dof_a + w_t.double().square() / dof_t
    ).clamp(min=1e-300)
    expected_nu[w_t == 0] = dof_a
    expected_nu[w_a == 0] = dof_t
    expected_nu[total == 0] = float('nan')
    torch.testing.assert_close(
        _satterthwaite_dof(w_a, w_t, dof_a, dof_t), expected_nu,
        equal_nan=True,
    )


def test_fabricated_nominal_fit_applies_meier_factor(tmp_path):
    """The reported combined SE follows the formula on fabricated inputs."""
    d = _bank(SEED, N=50, V=5, R=2, n_a=30, n_cov=2)
    res = _nominal(d, tmp_path)
    se_a = res['slope_a_se'].to_numpy(dtype=np.float64)
    se_t = res['slope_t_se'].to_numpy(dtype=np.float64)
    se = res['slope_se'].to_numpy(dtype=np.float64)
    dof_a = res['dof_a'].to_numpy(dtype=np.float64)
    dof_t = res['dof_t'].to_numpy(dtype=np.float64)
    w_a = np.where(np.isfinite(se_a) & (se_a > 0), 1.0 / se_a**2, 0.0)
    w_t = np.where(np.isfinite(se_t) & (se_t > 0), 1.0 / se_t**2, 0.0)
    both = (w_a > 0) & (w_t > 0)
    assert both.any()
    f_a = w_a[both] / (w_a[both] + w_t[both])
    factor = 1.0 + 4.0 * f_a * (1.0 - f_a) * (
        1.0 / dof_a[both] + 1.0 / dof_t[both]
    )
    expected = np.sqrt(factor / (w_a[both] + w_t[both]))
    np.testing.assert_allclose(se[both], expected, rtol=2e-6, atol=0)


@pytest.mark.parametrize('case', ['allelic_off', 'no_phase', 'below_floor', 'allelic_only', 'model', 'robust'])
def test_no_factor_without_two_fitted_channels(tmp_path, case):
    """allelic_off (keep_a_df all False) and below_floor (n_a = 10 against
    the floor of 15): the combination is the total channel, taken verbatim.
    no_phase: the allelic channel is admitted but carries no weight.
    allelic_only (keep_t_df all False): the combination is the allelic
    channel. se_mode model / robust: no fitted scale, so the combined SE is
    the inverse-variance 1/sqrt(1/se_a^2 + 1/se_t^2) of the reported
    channel SEs, computed here; with two channels M would be above 1."""
    d = _bank(SEED + 21, N=50, V=5, R=2, n_a=10 if case == 'below_floor' else 30, n_cov=2)
    off = pd.DataFrame(False, index=d['A_df'].index, columns=d['A_df'].columns)
    if case == 'no_phase':
        d = dict(d, xL_df=None, xR_df=None)
    kw = dict(allelic_off=dict(keep_a_df=off), allelic_only=dict(keep_t_df=off),
              model=dict(se_mode='model'), robust=dict(se_mode='robust')).get(case, {})
    with pytest.warns(RuntimeWarning) if case in ('model', 'robust') else contextlib.nullcontext():
        res = _nominal(d, tmp_path, **kw)
    se, se_a, se_t = (res[c].astype(np.float64).values for c in ('slope_se', 'slope_a_se', 'slope_t_se'))
    if case in ('allelic_off', 'below_floor'):
        assert not res['allelic_admitted'].any()
        assert np.array_equal(se, se_t) and np.array_equal(res['pval_nominal'], res['pval_t'])
    elif case == 'no_phase':
        assert res['allelic_admitted'].all()
        assert np.allclose(se, se_t, rtol=1e-6, atol=0)
    elif case == 'allelic_only':
        ok = np.isfinite(se_a)
        assert ok.any() and np.allclose(se[ok], se_a[ok], rtol=1e-6, atol=0)
    else:
        w_a = np.where(np.isfinite(se_a) & (se_a > 0), 1 / se_a ** 2, 0.0)
        assert (w_a > 0).any()
        assert np.allclose(se, 1 / np.sqrt(w_a + 1 / se_t ** 2), rtol=1e-6, atol=0)
