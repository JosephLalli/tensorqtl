import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import minimize_scalar


stub = types.ModuleType('pandas_plink')
stub.read_plink = lambda *args, **kwargs: None
sys.modules.setdefault('pandas_plink', stub)
sys.path.insert(0, str(Path(__file__).parents[1] / 'tensorqtl'))

import susie


def _optimize_prior_variance_brent_reference(
    V_init, betahat, shat2, prior_weights, check_null_threshold=0,
):
    """Pre-cache implementation reference for exact optimizer comparisons."""
    device = betahat.device
    dtype = betahat.dtype
    betahat64 = betahat.detach().to(torch.float64)
    shat264 = shat2.detach().to(torch.float64)
    weights64 = prior_weights.detach().to(torch.float64)

    def objective(log_variance):
        value = susie.neg_loglik_logscale(
            torch.as_tensor(log_variance, dtype=torch.float64, device=device),
            betahat64, shat264, weights64,
        )
        return float(value.detach().cpu())

    result = minimize_scalar(
        objective, bounds=(-30.0, 15.0), method='bounded',
        options={'xatol': 1e-8},
    )
    current = float(V_init)
    candidate = float(np.exp(result.x)) if result.success else current
    current_log = -np.inf if current == 0 else float(np.log(current))
    if objective(result.x) > objective(current_log):
        candidate = current

    V = torch.as_tensor(candidate, dtype=dtype, device=device)
    if (
        float(susie.loglik(0, betahat, shat2, prior_weights))
        + check_null_threshold
        >= float(susie.loglik(V, betahat, shat2, prior_weights))
    ):
        V = torch.zeros((), dtype=dtype, device=device)
    return V


def test_center_and_scale_flags_are_honored():
    X = torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    attrs = susie.get_x_attributes(X, center=False, scale=False)

    torch.testing.assert_close(attrs['scaled_center'], torch.zeros(2))
    torch.testing.assert_close(attrs['scaled_scale'], torch.ones(2))
    torch.testing.assert_close(attrs['d'], (X * X).sum(0))


def test_susie_accepts_cpu_inputs_and_list_prior_weights():
    generator = torch.Generator().manual_seed(2)
    X = torch.randn((40, 4), generator=generator)
    y = X[:, 1] + 0.25 * torch.randn(40, generator=generator)

    result = susie.susie(
        X, y.reshape(-1, 1), L=2, prior_weights=[0.1, 0.2, 0.3, 0.4],
        max_iter=20, coverage=None,
    )

    assert result['fitted'].shape == (40,)
    assert np.isfinite(result['elbo']).all()


def test_large_credible_set_purity_is_independent_of_global_rng_state():
    rng = np.random.default_rng(4)
    corr = rng.uniform(-1, 1, size=(130, 130))
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)
    corr = torch.tensor(corr)
    members = torch.arange(130)

    np.random.seed(1)
    torch.manual_seed(1)
    first = susie.get_purity(members, None, corr)
    np.random.seed(999)
    torch.manual_seed(999)
    second = susie.get_purity(members, None, corr)

    assert first == second


@pytest.mark.parametrize('name', ['min_abs_corr', 'median_abs_corr', 'cs_extension_corr'])
def test_invalid_correlation_threshold_is_rejected(name):
    result = {
        'alpha': torch.tensor([[1.0]]),
        'V': torch.tensor([1.0]),
        'null_index': 0,
    }

    with pytest.raises(ValueError, match=name):
        susie.susie_get_cs(result, Xcorr=torch.ones((1, 1)), **{name: 1.1})


def test_correlation_extension_adds_tight_proxies():
    corr = torch.tensor([
        [1.0, 0.995, 0.1],
        [0.995, 1.0, 0.2],
        [0.1, 0.2, 1.0],
    ])

    extended = susie.extend_cs_by_correlation(
        [torch.tensor([0])], threshold=0.99, null_index=0, Xcorr=corr,
    )

    torch.testing.assert_close(extended[0], torch.tensor([0, 1]))


def test_median_only_credible_set_filter_is_available_from_susie():
    generator = torch.Generator().manual_seed(8)
    X = torch.randn((80, 5), generator=generator)
    y = 2 * X[:, 2] + 0.2 * torch.randn(80, generator=generator)

    result = susie.susie(
        X, y.reshape(-1, 1), L=1, max_iter=30,
        min_abs_corr=None, median_abs_corr=0.5,
    )

    assert 'sets' in result
    assert 'pip' in result


def test_objective_helpers_can_reuse_expected_residual_sum_of_squares():
    generator = torch.Generator().manual_seed(11)
    X = torch.randn((50, 4), generator=generator)
    y = X[:, 0] + 0.3 * torch.randn(50, generator=generator)
    result = susie.susie(X, y.reshape(-1, 1), L=2, max_iter=20)

    device = result['alpha'].device
    X = X.to(device)
    y = (y.to(device) - y.mean().to(device)).reshape(-1, 1)
    xattr = susie.get_x_attributes(X)
    er2 = susie.get_ER2(X, xattr, y, result)

    torch.testing.assert_close(
        susie.get_objective(X, xattr, y, result),
        susie.get_objective(X, xattr, y, result, er2=er2),
    )
    torch.testing.assert_close(
        susie.estimate_residual_variance_fct(X, xattr, y, result),
        susie.estimate_residual_variance_fct(X, xattr, y, result, er2=er2),
    )


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param(
            'cuda', marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason='CUDA unavailable'
            ), id='cuda',
        ),
    ],
)
def test_cached_prior_variance_optimizer_matches_original_arithmetic(device):
    betahat = torch.tensor([0.4, -0.2, 0.0, 1.1], device=device)
    shat2 = torch.tensor([0.3, float('inf'), 0.5, 0.2], device=device)
    prior_weights = torch.tensor([0.1, 0.2, 0.3, 0.4], device=device)

    expected = _optimize_prior_variance_brent_reference(
        0.7, betahat, shat2, prior_weights, check_null_threshold=0.05,
    )
    observed = susie.optimize_prior_variance_brent(
        0.7, betahat, shat2, prior_weights, check_null_threshold=0.05,
    )

    assert torch.equal(observed, expected)


def test_ser_expected_loglik_accepts_precomputed_fitted_effect():
    generator = torch.Generator().manual_seed(29)
    X = torch.randn((16, 5), generator=generator)
    y = torch.randn((16, 1), generator=generator)
    xattr = susie.get_x_attributes(X)
    Eb = torch.randn(5, generator=generator)
    Eb2 = Eb.square() + 0.1
    fitted = susie.compute_Xb(
        X, Eb, xattr['scaled_center'], xattr['scaled_scale']
    )

    s2 = torch.tensor(0.8)
    ordinary = susie.SER_posterior_e_loglik(X, xattr, y, s2, Eb, Eb2)
    precomputed = susie.SER_posterior_e_loglik(
        X, xattr, y, s2, Eb, Eb2, fitted=fitted
    )

    assert torch.equal(precomputed, ordinary)


def _assert_nested_exact(observed, expected):
    assert type(observed) is type(expected)
    if torch.is_tensor(observed):
        assert observed.dtype == expected.dtype
        assert observed.device == expected.device
        assert torch.equal(observed, expected)
    elif isinstance(observed, np.ndarray):
        assert observed.dtype == expected.dtype
        assert np.array_equal(observed, expected, equal_nan=True)
    elif isinstance(observed, dict):
        assert observed.keys() == expected.keys()
        for key in observed:
            _assert_nested_exact(observed[key], expected[key])
    elif isinstance(observed, (list, tuple)):
        assert len(observed) == len(expected)
        for observed_item, expected_item in zip(observed, expected):
            _assert_nested_exact(observed_item, expected_item)
    elif isinstance(observed, float) and np.isnan(observed):
        assert np.isnan(expected)
    else:
        assert observed == expected


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA graphs require CUDA'
)
def test_compiled_ibss_cuda_graph_matches_eager_exactly():
    generator = torch.Generator().manual_seed(41)
    common = {
        'L': 2,
        'max_iter': 4,
        'tol': 0.0,
        'coverage': None,
        'estimate_prior_variance': True,
        'estimate_prior_method': 'EM',
    }
    previous_compiled_sweep = susie._compiled_update_each_effect
    susie._compiled_update_each_effect = None
    try:
        retained_result = None
        retained_tensors = None
        for signal_variant, prior_weights in (
            (1, torch.arange(1, 8, dtype=torch.float32)),
            (3, torch.arange(7, 0, -1, dtype=torch.float32)),
        ):
            X = torch.randn((48, 7), generator=generator)
            y = (
                0.7 * X[:, signal_variant]
                + 0.4 * torch.randn(48, generator=generator)
            )
            eager = susie.susie(
                X, y.reshape(-1, 1), prior_weights=prior_weights,
                compile_ibss=False, **common
            )
            captured = susie.susie(
                X, y.reshape(-1, 1), prior_weights=prior_weights,
                compile_ibss=True, **common
            )
            _assert_nested_exact(captured, eager)
            if retained_result is None:
                retained_result = captured
                retained_tensors = {
                    key: value.clone()
                    for key, value in captured.items()
                    if torch.is_tensor(value)
                }

        # A later replay must not overwrite tensors returned by an earlier fit.
        for key, expected in retained_tensors.items():
            assert torch.equal(retained_result[key], expected)

        # Capture and exercise the distinct fixed-zero-prior graph separately.
        susie._compiled_update_each_effect = None
        zero_prior_common = {
            **common,
            'scaled_prior_variance': 0.0,
            'estimate_prior_variance': False,
        }
        eager = susie.susie(
            X, y.reshape(-1, 1), compile_ibss=False, **zero_prior_common
        )
        captured = susie.susie(
            X, y.reshape(-1, 1), compile_ibss=True, **zero_prior_common
        )
        _assert_nested_exact(captured, eager)
        assert torch.equal(captured['V'], torch.zeros_like(captured['V']))

        # More than eight heterogeneous shapes must fall back to eager rather
        # than exhausting torch.compile's per-frame recompile limit.
        for p in range(8, 17):
            X_other = torch.randn((48, p), generator=generator)
            y_other = torch.randn(48, generator=generator)
            with pytest.warns(RuntimeWarning, match='different sweep signature'):
                result = susie.susie(
                    X_other, y_other.reshape(-1, 1), L=2, max_iter=1,
                    coverage=None, compile_ibss=True,
                    estimate_prior_variance=False,
                    scaled_prior_variance=0.0,
                )
            assert result['alpha'].shape == (2, p)
    finally:
        susie._compiled_update_each_effect = previous_compiled_sweep


def test_elbo_convergence_does_not_initialize_unused_pip_history(monkeypatch):
    calls = 0
    original = susie.susie_get_pip

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(susie, 'susie_get_pip', counted)
    generator = torch.Generator().manual_seed(43)
    X = torch.randn((32, 5), generator=generator)
    y = torch.randn(32, generator=generator)
    susie.susie(
        X, y.reshape(-1, 1), L=2, max_iter=1, tol=0.0, coverage=None
    )

    # The only call is the final reported PIP; ELBO convergence needs no
    # alpha/PIP history snapshot before the first sweep.
    assert calls == 1


def test_compile_ibss_rejects_non_boolean_flag():
    with pytest.raises(ValueError, match='True or False'):
        susie.susie(
            torch.randn(8, 2), torch.randn(8, 1),
            compile_ibss='yes', coverage=None,
        )


def test_compile_ibss_requires_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='requires a CUDA device'):
        susie.susie(
            torch.randn(8, 2), torch.randn(8, 1),
            compile_ibss=True, coverage=None,
        )
