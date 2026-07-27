"""Property tests for full SuSiE-ash (susie(unmappable_effects="ash")).

SuSiE-ash adds a Mr.ASH adaptive-shrinkage polygenic background `theta`, refit
between IBSS iterations on residuals with confident credible-set variants masked
(port of pinned susieR 2.0). These tests check invariants (shapes, determinism,
GPU==CPU), signal recovery under a polygenic background, that the background is
actually absorbed by `theta`, and that the default (non-ash) path is untouched.
Pinned state-transition and end-to-end agreement are checked in the retained
oracle harness.

The Mr.ASH coordinate-ascent core is verified numerically identical to susieR's
compiled caisa_cpp in tests/test_mrash.py; these tests exercise the IBSS
integration on top of it.
"""
import sys
import numpy as np
import pytest
import torch
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tensorqtl'))
import types
if 'pandas_plink' not in sys.modules:
    _f = types.ModuleType('pandas_plink')
    _f.read_plink = lambda *a, **k: None
    _f.read_plink1_bin = lambda *a, **k: None
    sys.modules['pandas_plink'] = _f
import susie as sm
import susieash

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
    assert s['c_hat'].shape == (5,)
    assert np.all((s['c_hat'] >= 0) & (s['c_hat'] <= 1))
    assert s['C_hat'] == pytest.approx(s['c_hat'].sum())
    assert s['ash_final_pass'] is True


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
    """Unknown unmappable-effects values fail instead of silently fitting SuSiE."""
    Xt, yt, _, _ = _sim()
    with pytest.raises(ValueError, match='unmappable_effects'):
        sm.susie(Xt, yt, unmappable_effects='foo')
    with pytest.raises(ValueError, match="estimate_prior_method='EM'"):
        sm.susie(
            Xt, yt, unmappable_effects='ash',
            estimate_prior_method='EM',
        )
    with pytest.raises(ValueError, match='intercept=True'):
        sm.susie(Xt, yt, unmappable_effects='ash', intercept=False)
    with pytest.raises(ValueError, match='standardize=True'):
        sm.susie(Xt, yt, unmappable_effects='ash', standardize=False)


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


def test_ash_intercept_reconstructs_predictions_on_raw_design():
    """Raw-scale coefficients and intercept reproduce the reported fitted values."""
    Xt, yt, _, _ = _sim(seed=22, signal_cols=(4, 17), signal_beta=2.0)
    Xt = Xt + torch.linspace(-2, 2, Xt.shape[1])
    yt = yt + 3.0
    s = _fit_ash(Xt, yt)

    fit_device = s['Xr'].device
    X_fit = Xt.to(fit_device)
    xattr = sm.get_x_attributes(X_fit, center=True, scale=True)
    sparse_t = (
        s['slot_weights'][:, None] * s['alpha'] * s['mu']
    ).sum(0)
    theta_t = torch.as_tensor(s['theta'], dtype=X_fit.dtype, device=fit_device)
    raw_coef_t = (sparse_t + theta_t) / xattr['scaled_scale']
    expected = X_fit @ raw_coef_t + s['intercept']
    assert torch.allclose(s['fitted'], expected, rtol=1e-5, atol=1e-5)


def test_ash_is_invariant_to_raw_column_affine_transform():
    """Positive rescaling and shifting of raw columns preserves the fitted model."""
    Xt, yt, _, _ = _sim(
        seed=23, n=120, p=24, signal_cols=(4, 17),
        signal_beta=1.8, bg_scale=0.06, noise=0.5,
    )
    scales = torch.logspace(-1, 1, Xt.shape[1])
    shifts = torch.linspace(-7, 9, Xt.shape[1])
    transformed = Xt * scales + shifts

    base = _fit_ash(Xt, yt, coverage=None)
    affine = _fit_ash(transformed, yt, coverage=None)

    np.testing.assert_allclose(base['pip'], affine['pip'], rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(
        base['theta'], affine['theta'], rtol=3e-4, atol=3e-5
    )
    torch.testing.assert_close(
        base['fitted'].cpu(), affine['fitted'].cpu(),
        rtol=3e-4, atol=3e-5,
    )
    assert float(base['sigma2']) == pytest.approx(
        float(affine['sigma2']), rel=3e-4, abs=3e-5
    )
    assert base['tau2'] == pytest.approx(
        affine['tau2'], rel=3e-4, abs=3e-6
    )


def test_ash_forwards_check_null_threshold(monkeypatch):
    """The public null threshold reaches every ash-mode SER update."""
    Xt, yt, _, _ = _sim(seed=24, n=60, p=10, signal_cols=(3,))
    seen = []
    original = sm.single_effect_regression

    def recording_ser(*args, **kwargs):
        seen.append(kwargs['check_null_threshold'])
        return original(*args, **kwargs)

    monkeypatch.setattr(sm, 'single_effect_regression', recording_ser)
    sm.susie(
        Xt, yt, L=3, unmappable_effects='ash', coverage=None,
        check_null_threshold=2.5, max_iter=1,
    )
    assert seen and set(seen) == {2.5}


def test_short_cycle_averages_alpha_and_reconciles_sparse_fitted():
    """Cycle convergence follows pinned alpha averaging without stale Xr."""
    X = torch.tensor([
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 1.0],
        [2.0, 1.0, 0.0],
        [3.0, 2.0, 1.0],
    ])
    xattr = sm.get_x_attributes(X)
    current = torch.tensor([
        [0.80, 0.15, 0.05],
        [0.10, 0.20, 0.70],
    ])
    other = torch.tensor([
        [0.10, 0.80, 0.10],
        [0.65, 0.25, 0.10],
    ])
    state = {
        'alpha': current.clone(),
        'mu': torch.tensor([
            [0.5, -0.3, 0.2],
            [-0.4, 0.1, 0.6],
        ]),
        'V': torch.ones(2),
        'null_index': 0,
        'slot_weights': torch.tensor([0.4, 0.8]),
    }

    def snapshot(alpha):
        tmp = dict(state)
        tmp['alpha'] = alpha
        return alpha.clone(), sm.susie_get_pip(tmp).clone()

    history = [snapshot(current), snapshot(other)]
    converged, _, lag = sm._pip_state_converged(
        state, history, tol=1e-8, cycle_window=5
    )
    assert converged and lag == 2
    expected_alpha = (current + other) / 2
    torch.testing.assert_close(state['alpha'], expected_alpha)

    sm._recompute_weighted_fitted(X, xattr, state)
    expected_effect = (
        state['slot_weights'][:, None] * expected_alpha * state['mu']
    ).sum(0)
    expected_fitted = sm.compute_Xb(
        X, expected_effect,
        xattr['scaled_center'], xattr['scaled_scale'],
    )
    torch.testing.assert_close(state['Xr'], expected_fitted)


def test_finite_sigma2_bound_returns_one_consistent_ash_fit():
    """A bound constrains the refit, and tau2 uses that same residual variance."""
    Xt, yt, _, _ = _sim(
        seed=25, n=100, p=20, signal_cols=(3,),
        signal_beta=1.0, bg_scale=0.08, noise=1.0,
    )
    bound = 0.05
    fit = sm.susie(
        Xt, yt, L=4, unmappable_effects='ash', coverage=None,
        max_iter=50, residual_variance_upperbound=bound,
    )
    assert float(fit['sigma2']) <= bound * (1 + 1e-6)

    fit_device = fit['alpha'].device
    xattr = sm.get_x_attributes(Xt.to(fit_device))
    sa2 = susieash.default_sa2_grid(
        xattr['d'].detach().cpu().numpy().astype(np.float64),
        Xt.shape[0],
    )
    expected_tau2 = float(
        (sa2 * fit['ash_pi']).sum() * float(fit['sigma2'])
    )
    assert fit['tau2'] == pytest.approx(expected_tau2, rel=2e-6, abs=1e-10)


def test_refit_reoptimizes_beta_and_pi_at_sigma2_bound(monkeypatch):
    """A hit bound triggers a fixed-sigma polish instead of a post-hoc clip."""
    calls = []

    def fake_mr_ash(X, y, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {
                'beta': np.array([0.4, -0.2]),
                'sigma2': 2.0,
                'pi': np.array([0.25, 0.75]),
            }
        assert kwargs['sigma2'] == 0.5
        assert kwargs['update_sigma'] is False
        np.testing.assert_array_equal(
            kwargs['beta_init'], np.array([0.4, -0.2])
        )
        np.testing.assert_array_equal(
            kwargs['pi'], np.array([0.25, 0.75])
        )
        return {
            'beta': np.array([0.3, -0.1]),
            'sigma2': 0.5,
            'pi': np.array([0.4, 0.6]),
        }

    monkeypatch.setattr(susieash.mrash, 'mr_ash', fake_mr_ash)
    beta, sigma2, pi, tau2 = susieash.refit(
        np.eye(2), np.ones(2), 1.0, np.zeros(2), None,
        np.array([0.0, 1.0]), 1e-4, True,
        sigma2_upperbound=0.5,
    )
    assert len(calls) == 2
    np.testing.assert_array_equal(beta, np.array([0.3, -0.1]))
    np.testing.assert_array_equal(pi, np.array([0.4, 0.6]))
    assert sigma2 == 0.5
    assert tau2 == pytest.approx(0.3)


def test_final_unmasked_pass_requires_convergence():
    """A max-iteration exit retains the last masked ash update, matching the
    upstream workhorse ordering, and does not claim a final unmasked refit."""
    Xt, yt, _, _ = _sim(seed=31, n=100, p=20)
    s = sm.susie(
        Xt, yt, L=4, unmappable_effects='ash',
        max_iter=1, tol=0, coverage=None,
    )
    assert s['converged'] is False
    assert s['ash_final_pass'] is False
    assert s['ash_iter'] == 1
    assert np.all(s['theta'][s['masked']] == 0)


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
    with mock.patch.object(torch.cuda, 'is_available', return_value=False):
        s_cpu = _fit_ash(Xt.cpu(), yt.cpu())
    s_gpu = _fit_ash(Xt.cuda(), yt.cuda())
    assert np.abs(s_gpu['pip'] - s_cpu['pip']).max() < 1e-5
    assert np.abs(s_gpu['theta'] - s_cpu['theta']).max() < 1e-4
    assert np.array_equal(s_gpu['masked'], s_cpu['masked'])
    torch.testing.assert_close(
        s_gpu['fitted'].cpu(), s_cpu['fitted'].cpu(),
        rtol=1e-5, atol=1e-5,
    )
