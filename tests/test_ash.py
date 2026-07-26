"""Property tests for SuSiE-ash (susie(unmappable_effects="ash")).

SuSiE-ash adds a Mr.ASH adaptive-shrinkage polygenic background `theta`, refit
between IBSS iterations on residuals with confident credible-set variants masked
(port of susieR 2.0). These tests need no R: they check invariants (shapes,
determinism, GPU==CPU), signal recovery under a polygenic background, that the
background is actually absorbed by `theta`, and that the default (non-ash) path
is untouched. Bit-close agreement with susieR's own susie(unmappable_effects=
"ash") is checked separately in the oracle harness (needs susieR master).

The Mr.ASH coordinate-ascent core is verified numerically identical to susieR's
compiled caisa_cpp in tests/test_mrash.py; these tests exercise the IBSS
integration on top of it.
"""
import sys
import numpy as np
import pytest
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tensorqtl'))
import types
if 'pandas_plink' not in sys.modules:
    _f = types.ModuleType('pandas_plink')
    _f.read_plink = lambda *a, **k: None
    _f.read_plink1_bin = lambda *a, **k: None
    sys.modules['pandas_plink'] = _f
import susie as sm

torch.set_num_threads(2)


def _sim(seed=0, n=300, p=60, signal_cols=(10,), signal_beta=3.0,
         bg_scale=0.05, noise=0.5):
    """Polygenic background (all p variants ~N(0, bg_scale^2)) plus optional
    sparse signals. Returns (Xt [n,p] float32, yt [n,1] centered float32,
    theta_true [p], signal_cols)."""
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n, p)).astype(np.float32)
    theta_true = (rng.standard_normal(p) * bg_scale).astype(np.float32)
    y = (G @ theta_true).astype(np.float32)
    for c in signal_cols:
        y = y + signal_beta * G[:, c]
    y = y + noise * rng.standard_normal(n).astype(np.float32)
    Xt = torch.tensor(G)
    yt = torch.tensor(y - y.mean()).reshape(-1, 1)
    return Xt, yt, theta_true, signal_cols


def _fit_ash(Xt, yt, **kw):
    return sm.susie(Xt, yt, L=5, unmappable_effects='ash', max_iter=100, **kw)


def test_ash_reports_theta_tau2_and_valid_pi():
    """The ash path reports a dense theta (p-vector), a scalar tau2, and an
    ash_pi mixture that is a valid probability vector."""
    Xt, yt, _, _ = _sim()
    s = _fit_ash(Xt, yt)
    p = Xt.shape[1]
    assert s['theta'].shape == (p,)
    assert np.isfinite(s['theta']).all()
    assert np.isscalar(s['tau2']) or np.ndim(s['tau2']) == 0
    assert float(s['tau2']) >= 0.0
    assert np.isclose(s['ash_pi'].sum(), 1.0, atol=1e-8)
    assert (s['ash_pi'] >= 0).all()
    assert float(s['sigma2']) > 0.0


def test_ash_recovers_sparse_signal_under_polygenic_background():
    """A strong sparse effect embedded in a polygenic background is recovered as
    a credible set at the true variant with high PIP."""
    Xt, yt, _, cols = _sim(signal_cols=(10,), signal_beta=3.0)
    s = _fit_ash(Xt, yt)
    assert s.get('converged') is True
    assert int(np.argmax(s['pip'])) == cols[0]
    assert float(s['pip'][cols[0]]) > 0.9
    assert s['sets'] is not None and s['sets']['cs'] is not None
    members = [i for cs in s['sets']['cs'].values() for i in cs]
    assert cols[0] in members


def test_ash_absorbs_polygenic_background_into_theta():
    """With a polygenic background present, theta carries non-trivial mass; with
    a pure-noise (null) outcome, theta and tau2 collapse toward zero."""
    Xt_bg, yt_bg, _, _ = _sim(signal_cols=(), bg_scale=0.08, noise=0.5)
    s_bg = _fit_ash(Xt_bg, yt_bg)

    # null: same design, outcome is pure noise (no genetic component)
    rng = np.random.default_rng(1)
    n, p = Xt_bg.shape
    ynull = (0.5 * rng.standard_normal(n)).astype(np.float32)
    yt_null = torch.tensor(ynull - ynull.mean()).reshape(-1, 1)
    s_null = _fit_ash(Xt_bg, yt_null)

    mean_abs_theta_bg = float(np.abs(s_bg['theta']).mean())
    mean_abs_theta_null = float(np.abs(s_null['theta']).mean())
    assert mean_abs_theta_bg > mean_abs_theta_null
    assert float(s_bg['tau2']) > float(s_null['tau2'])
    # null truly has ~no background
    assert mean_abs_theta_null < 1e-2
    assert float(s_null['tau2']) < 1e-3


def test_ash_is_deterministic():
    """Re-fitting identical inputs yields identical PIPs and theta."""
    Xt, yt, _, _ = _sim(seed=7)
    s1 = _fit_ash(Xt, yt)
    s2 = _fit_ash(Xt, yt)
    assert np.array_equal(s1['pip'], s2['pip'])
    assert np.array_equal(s1['theta'], s2['theta'])


def test_ash_requires_estimate_residual_variance():
    """susieR gates the in-loop Mr.ASH refit on estimate_residual_variance=TRUE;
    the port raises rather than silently leaving theta at zero."""
    Xt, yt, _, _ = _sim()
    try:
        _fit_ash(Xt, yt, estimate_residual_variance=False)
    except ValueError as e:
        assert 'estimate_residual_variance' in str(e)
    else:
        raise AssertionError("expected ValueError for estimate_residual_variance=False")


def test_ash_rejects_unsupported_public_option_combinations():
    """Unknown unmappable-effects values and the incompatible NIG mode fail
    explicitly instead of silently taking a different fitting path."""
    Xt, yt, _, _ = _sim()
    with pytest.raises(ValueError, match='unmappable_effects'):
        sm.susie(Xt, yt, unmappable_effects='foo')
    with pytest.raises(ValueError, match='incompatible'):
        sm.susie(Xt, yt, unmappable_effects='ash', estimate_residual_method='NIG')


def test_ash_multi_sweep_fitted_value_decomposition():
    """After multiple IBSS/Mr.ASH sweeps, the reported fitted values equal the
    sparse fit plus the final unmasked Mr.ASH background and intercept."""
    Xt, yt, _, _ = _sim(seed=21, signal_cols=(4, 17), signal_beta=2.0,
                         bg_scale=0.08, noise=0.4)
    s = _fit_ash(Xt, yt)
    assert s['niter'] >= 2
    fit_device = s['Xr'].device
    X_fit = Xt.to(fit_device)
    xattr = sm.get_x_attributes(X_fit, center=True, scale=True)
    theta_t = torch.as_tensor(s['theta'], dtype=X_fit.dtype, device=fit_device)
    background = sm.compute_Xb(X_fit, theta_t,
                               xattr['scaled_center'], xattr['scaled_scale'])
    expected = s['Xr'].squeeze() + yt.to(fit_device).mean() + background
    assert torch.allclose(s['fitted'], expected, rtol=1e-5, atol=1e-5)


def test_default_path_unaffected_by_unmappable_effects_param():
    """unmappable_effects=None must reproduce the plain susie() fit exactly and
    must not attach ash-only fields."""
    Xt, yt, _, _ = _sim(signal_cols=(10,))
    s_default = sm.susie(Xt, yt, L=5, max_iter=100)
    s_none = sm.susie(Xt, yt, L=5, max_iter=100, unmappable_effects=None)
    assert np.array_equal(s_default['pip'], s_none['pip'])
    assert 'theta' not in s_default and 'theta' not in s_none
    assert 'tau2' not in s_default


def test_ash_gpu_cpu_equivalence():
    """When CUDA is available, the ash fit must match the CPU fit (PIPs identical,
    theta within tight float tolerance). Skipped if no GPU is visible."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device visible")
    Xt, yt, _, _ = _sim(signal_cols=(10,))
    s_gpu = _fit_ash(Xt.cuda(), yt.cuda())
    s_cpu = _fit_ash(Xt.cpu(), yt.cpu())
    assert np.abs(s_gpu['pip'] - s_cpu['pip']).max() < 1e-5
    assert np.abs(s_gpu['theta'] - s_cpu['theta']).max() < 1e-4
