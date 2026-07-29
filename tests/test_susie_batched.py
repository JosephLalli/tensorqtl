import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch


stub = types.ModuleType('pandas_plink')
stub.read_plink = lambda *args, **kwargs: None
sys.modules.setdefault('pandas_plink', stub)
sys.path.insert(0, str(Path(__file__).parents[1] / 'tensorqtl'))

import susie


def _heterogeneous_fixture():
    generator = torch.Generator().manual_seed(123)
    X_list = []
    y_list = []
    for p in (3, 7, 5):
        X = torch.randn((64, p), generator=generator)
        y = (
            0.8 * X[:, min(1, p - 1)]
            + 0.3 * torch.randn(64, generator=generator)
        )
        X_list.append(X)
        y_list.append(y.reshape(-1, 1))
    return X_list, y_list


def _assert_fit_close(
        observed, expected, *, compare_elbo=True, compare_niter=True):
    assert observed['alpha'].shape == expected['alpha'].shape
    if compare_niter:
        assert observed['niter'] == expected['niter']
    assert observed['converged'] == expected['converged']
    for key in (
        'alpha', 'mu', 'mu2', 'V', 'KL', 'lbf', 'Xr', 'sigma2',
        'sparse_effects', 'intercept', 'fitted',
    ):
        observed_value = observed[key]
        expected_value = expected[key]
        if not torch.is_tensor(observed_value):
            assert observed_value == expected_value
        else:
            torch.testing.assert_close(
                observed_value, expected_value, rtol=2e-5, atol=2e-4
            )
    np.testing.assert_allclose(
        observed['lbf_variable'], expected['lbf_variable'],
        rtol=2e-5, atol=2e-4,
    )
    np.testing.assert_allclose(
        observed['pip'], expected['pip'], rtol=2e-5, atol=2e-6
    )
    if compare_elbo:
        np.testing.assert_allclose(
            observed['elbo'], expected['elbo'], rtol=2e-5, atol=2e-4
        )


def test_batched_solver_matches_independent_heterogeneous_fits():
    X_list, y_list = _heterogeneous_fixture()
    common = {
        'L': 4,
        'max_iter': 30,
        'tol': 1e-3,
        'coverage': None,
    }

    observed = susie.susie_batched(
        X_list, y_list, batch_size=3, **common
    )
    expected = [
        susie.susie(X, y, **common)
        for X, y in zip(X_list, y_list)
    ]

    assert [fit['alpha'].shape for fit in observed] == [
        (3, 3), (4, 7), (4, 5)
    ]
    for batched_fit, scalar_fit in zip(observed, expected):
        _assert_fit_close(
            batched_fit, scalar_fit,
            compare_elbo=False, compare_niter=False,
        )
        np.testing.assert_allclose(
            batched_fit['elbo'][-1], scalar_fit['elbo'][-1],
            rtol=2e-5, atol=2e-4,
        )


def test_batch_companions_padding_and_order_do_not_change_fit():
    X_list, y_list = _heterogeneous_fixture()
    common = {
        'L': 4,
        'max_iter': 4,
        'tol': 0,
        'coverage': None,
    }
    together = susie.susie_batched(
        X_list, y_list, batch_size=3, **common
    )
    alone = susie.susie_batched(
        X_list, y_list, batch_size=1, **common
    )
    order = [2, 0, 1]
    reordered = susie.susie_batched(
        [X_list[i] for i in order],
        [y_list[i] for i in order],
        batch_size=3,
        **common,
    )
    restored = [None] * len(order)
    for fit, original_i in zip(reordered, order):
        restored[original_i] = fit

    for fit_together, fit_alone, fit_reordered in zip(
            together, alone, restored):
        _assert_fit_close(fit_together, fit_alone)
        _assert_fit_close(fit_together, fit_reordered)


def test_batched_solver_supports_per_gene_priors_and_variances():
    X_list, y_list = _heterogeneous_fixture()
    priors = [
        torch.arange(1, X.shape[1] + 1, dtype=torch.float32)
        for X in X_list
    ]
    residual_variances = torch.tensor([0.7, 0.8, 0.9])
    common = {
        'L': 2,
        'max_iter': 3,
        'tol': 0,
        'coverage': None,
        'estimate_prior_variance': False,
        'prior_weights': priors,
        'residual_variance': residual_variances,
    }

    observed = susie.susie_batched(
        X_list, y_list, batch_size=2, **common
    )
    expected = [
        susie.susie(
            X, y, prior_weights=prior,
            residual_variance=residual_variances[i],
            **{k: v for k, v in common.items()
               if k not in {'prior_weights', 'residual_variance'}},
        )
        for i, (X, y, prior) in enumerate(
            zip(X_list, y_list, priors)
        )
    ]

    for batched_fit, scalar_fit in zip(observed, expected):
        _assert_fit_close(batched_fit, scalar_fit)


def test_single_gene_accepts_a_flat_prior_vector():
    X_list, y_list = _heterogeneous_fixture()
    prior = torch.tensor([0.2, 0.3, 0.5])
    observed = susie.susie_batched(
        X_list[:1], y_list[:1], L=2, prior_weights=prior,
        max_iter=2, tol=0, coverage=None,
    )[0]
    expected = susie.susie(
        X_list[0], y_list[0], L=2, prior_weights=prior,
        max_iter=2, tol=0, coverage=None,
    )

    _assert_fit_close(observed, expected)


def test_batched_solver_returns_scalar_credible_set_contract():
    X_list, y_list = _heterogeneous_fixture()
    fits = susie.susie_batched(
        X_list, y_list, L=2, max_iter=30, batch_size=3
    )

    for fit, p in zip(fits, (3, 7, 5)):
        assert 'sets' in fit
        assert fit['pip'].shape == (p,)
        if fit['sets']['cs'] is not None:
            for members in fit['sets']['cs'].values():
                assert (members < p).all()


def test_batched_solver_rejects_unsupported_prior_optimizer():
    X_list, y_list = _heterogeneous_fixture()
    with pytest.raises(ValueError, match='none.*EM'):
        susie.susie_batched(
            X_list, y_list, estimate_prior_method='optim'
        )


def test_batched_solver_rejects_implicit_dtype_conversion():
    X_list, y_list = _heterogeneous_fixture()
    with pytest.raises(ValueError, match='torch.float32'):
        susie.susie_batched(
            [X_list[0].double()], [y_list[0].double()]
        )


@pytest.mark.parametrize('residual_variance', [[0.5], [0.5, 0.6, 0.7, 0.8]])
def test_batched_solver_validates_residual_variance_length(
        residual_variance):
    X_list, y_list = _heterogeneous_fixture()
    with pytest.raises(ValueError, match='one value per gene'):
        susie.susie_batched(
            X_list, y_list, residual_variance=residual_variance
        )


def test_batched_solver_honors_no_center_or_scale():
    X_list, y_list = _heterogeneous_fixture()
    common = {
        'L': 2,
        'max_iter': 3,
        'tol': 0,
        'coverage': None,
        'standardize': False,
        'intercept': False,
    }
    observed = susie.susie_batched(
        X_list, y_list, batch_size=3, **common
    )
    expected = [
        susie.susie(X, y, **common)
        for X, y in zip(X_list, y_list)
    ]

    for batched_fit, scalar_fit in zip(observed, expected):
        _assert_fit_close(batched_fit, scalar_fit)


def test_batched_solver_runs_on_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    X_list, y_list = _heterogeneous_fixture()
    fits = susie.susie_batched(
        X_list[:2], y_list[:2], L=2, max_iter=2, tol=0,
        coverage=None, batch_size=2,
    )

    assert len(fits) == 2
    assert all(fit['alpha'].device.type == 'cpu' for fit in fits)


def test_batched_solver_uses_batched_matrix_multiplication(monkeypatch):
    X_list, y_list = _heterogeneous_fixture()
    calls = 0
    original = torch.bmm

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, 'bmm', counted)
    susie.susie_batched(
        X_list, y_list, L=2, max_iter=1, tol=0,
        coverage=None, batch_size=3,
    )

    assert calls > 0


def test_prepacked_variant_major_matches_list_wrapper():
    X_list, y_list = _heterogeneous_fixture()
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    p_max = max(X.shape[1] for X in X_list)
    packed_X = torch.zeros(
        (len(X_list), p_max, X_list[0].shape[0]),
        dtype=torch.float32,
        device=device,
    )
    for i, X in enumerate(X_list):
        packed_X[i, :X.shape[1]] = X.T.to(device)
    packed_y = torch.stack(
        [y.reshape(-1).to(device) for y in y_list]
    )
    common = {
        'L': 3,
        'max_iter': 4,
        'tol': 0,
        'coverage': 0.95,
    }

    observed = susie.susie_batched_packed(
        packed_X,
        packed_y,
        [X.shape[1] for X in X_list],
        **common,
    )
    expected = susie.susie_batched(
        X_list,
        y_list,
        batch_size=len(X_list),
        **common,
    )

    for packed_fit, list_fit in zip(observed, expected):
        _assert_fit_close(packed_fit, list_fit)
        assert packed_fit['sets']['coverage'] == list_fit['sets']['coverage']
        if packed_fit['sets']['cs'] is not None:
            assert (
                packed_fit['sets']['cs'].keys()
                == list_fit['sets']['cs'].keys()
            )
            for name in packed_fit['sets']['cs']:
                np.testing.assert_array_equal(
                    packed_fit['sets']['cs'][name],
                    list_fit['sets']['cs'][name],
                )


def test_prepacked_variant_major_validates_layout_and_padding():
    X_t = torch.zeros((2, 4, 8), dtype=torch.float32)
    y_t = torch.zeros((2, 8), dtype=torch.float32)

    with pytest.raises(ValueError, match='variant-major'):
        susie.susie_batched_packed(
            X_t.transpose(1, 2), y_t, [4, 4]
        )
    with pytest.raises(ValueError, match='one value per gene'):
        susie.susie_batched_packed(X_t, y_t, [4])

    X_t[0, 3] = 1
    with pytest.raises(ValueError, match='padding must be zero'):
        susie.susie_batched_packed(X_t, y_t, [3, 4])


def test_variant_major_matrix_helpers_match_direct_algebra():
    generator = torch.Generator().manual_seed(42)
    X_t = torch.randn((2, 5, 7), generator=generator)
    b_t = torch.randn((2, 5), generator=generator)
    y_t = torch.randn((2, 7), generator=generator)
    M_t = torch.randn((2, 3, 5), generator=generator)
    cm_t = X_t.mean(2)
    csd_t = X_t.std(2, unbiased=True)
    standardized_t = (
        X_t - cm_t[:, :, None]
    ) / csd_t[:, :, None]

    torch.testing.assert_close(
        susie._batched_compute_Xb(X_t, b_t, cm_t, csd_t),
        torch.bmm(
            standardized_t.transpose(1, 2), b_t.unsqueeze(2)
        ).squeeze(2),
    )
    torch.testing.assert_close(
        susie._batched_compute_Xty(X_t, y_t, cm_t, csd_t),
        torch.bmm(standardized_t, y_t.unsqueeze(2)).squeeze(2),
    )
    torch.testing.assert_close(
        susie._batched_compute_MXt(M_t, X_t, cm_t, csd_t),
        torch.bmm(M_t, standardized_t),
    )
