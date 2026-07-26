import math
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
import susieslot


def _data(seed=1, n=100, p=30, signals=()):
    generator = torch.Generator().manual_seed(seed)
    X = torch.randn((n, p), generator=generator)
    beta = torch.zeros(p)
    for index, value in signals:
        beta[index] = value
    y = X @ beta + 0.6 * torch.randn(n, generator=generator)
    return X, y


def test_slot_prior_constructors_validate_inputs():
    beta_binomial = susie.slot_prior_betabinom()
    poisson = susie.slot_prior_poisson(C=3)

    assert beta_binomial['prior_type'] == 'betabinom'
    assert beta_binomial['a_beta'] == 1
    assert beta_binomial['b_beta'] == 2
    assert poisson['prior_type'] == 'poisson'
    assert poisson['nu'] == 8

    with pytest.raises(ValueError, match='a_beta'):
        susie.slot_prior_betabinom(a_beta=0)
    with pytest.raises(ValueError, match='C'):
        susie.slot_prior_poisson(C=-1)
    with pytest.raises(ValueError, match='update_schedule'):
        susie.slot_prior_poisson(C=3, update_schedule='other')
    with pytest.raises(ValueError, match='between 0 and 1'):
        susie.slot_prior_betabinom(c_hat_init=[-0.1, 0.2])


def test_beta_binomial_initialization_and_warm_start():
    prior = susie.slot_prior_betabinom()
    c_hat, state = susieslot.initialize_slot_state(
        prior, 4, torch.float64, torch.device('cpu')
    )
    torch.testing.assert_close(c_hat, torch.full((4,), 1 / 3, dtype=torch.float64))
    assert state['skip_threshold'] == 0

    warm = susie.slot_prior_betabinom(c_hat_init=[0.1, 0.2, 0.3, 0.4])
    c_hat, _ = susieslot.initialize_slot_state(
        warm, 4, torch.float64, torch.device('cpu')
    )
    torch.testing.assert_close(
        c_hat, torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    )

    # Pinned susieR ignores a mismatched warm start and uses the prior mean.
    c_hat, _ = susieslot.initialize_slot_state(
        warm, 3, torch.float64, torch.device('cpu')
    )
    torch.testing.assert_close(
        c_hat, torch.full((3,), 1 / 3, dtype=torch.float64)
    )


def test_gamma_poisson_initialization_uses_upstream_shape_parameters():
    prior = susie.slot_prior_poisson(C=3, nu=8)
    c_hat, state = susieslot.initialize_slot_state(
        prior, 6, torch.float64, torch.device('cpu')
    )

    torch.testing.assert_close(c_hat, torch.full((6,), 0.5, dtype=torch.float64))
    assert state['a_g'] == 11
    assert state['b_g'] == pytest.approx(8 / 3 + 1)

    warm = susie.slot_prior_poisson(
        C=3, nu=8, c_hat_init=[0.1, 0.2, 0.3, 0.4]
    )
    _, warm_state = susieslot.initialize_slot_state(
        warm, 4, torch.float64, torch.device('cpu')
    )
    assert warm_state['a_g'] == pytest.approx(9)

    mismatched, mismatched_state = susieslot.initialize_slot_state(
        warm, 3, torch.float64, torch.device('cpu')
    )
    torch.testing.assert_close(
        mismatched, torch.full((3,), 1.0, dtype=torch.float64)
    )
    assert mismatched_state['a_g'] == pytest.approx(11)


def test_beta_binomial_coordinate_update_matches_closed_form():
    c_hat, state = susieslot.initialize_slot_state(
        susie.slot_prior_betabinom(a_beta=2, b_beta=3),
        3, torch.float64, torch.device('cpu')
    )
    model = {
        'alpha': torch.full((3, 2), 0.5, dtype=torch.float64),
        'lbf': torch.tensor([1.25, np.nan, np.nan], dtype=torch.float64),
        'slot_weights': c_hat,
        'c_hat_state': state,
    }
    old_c, new_c = susieslot.update_slot_weight(model, 0)
    k_others = 2 * (2 / 5)
    expected = 1 / (
        1 + math.exp(
            -(math.log(2 + k_others) - math.log(3 + 2 - k_others) + 1.25)
        )
    )

    assert old_c == pytest.approx(2 / 5)
    assert new_c == pytest.approx(expected)
    assert float(model['slot_weights'][0]) == pytest.approx(expected)


@pytest.mark.parametrize('schedule', ['sequential', 'batch'])
def test_gamma_poisson_shape_and_skip_threshold(schedule):
    c_hat, state = susieslot.initialize_slot_state(
        susie.slot_prior_poisson(
            C=2, nu=8, update_schedule=schedule,
            skip_threshold_multiplier=0.5,
        ),
        4, torch.float64, torch.device('cpu')
    )
    model = {
        'alpha': torch.full((4, 2), 0.5, dtype=torch.float64),
        'lbf': torch.tensor([2.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        'slot_weights': c_hat,
        'c_hat_state': state,
    }
    susieslot.update_slot_weight(model, 0)
    susieslot.finish_slot_sweep(model)

    assert state['a_g'] == pytest.approx(
        state['nu'] + float(model['slot_weights'].sum())
    )
    assert state['skip_threshold'] > 0


def test_skip_threshold_freezes_the_entire_slot():
    X, y = _data(seed=7, p=12, signals=((2, 1.4),))
    xattr = susie.get_x_attributes(X)
    state = susie.init_finalize(
        susie.init_setup(
            len(y), X.shape[1], 3, 0.2, y.var(unbiased=True)
        )
    )
    state['slot_weights'], state['c_hat_state'] = (
        susieslot.initialize_slot_state(
            susie.slot_prior_poisson(C=2), 3,
            state['alpha'].dtype, state['alpha'].device,
        )
    )
    state['slot_weights'][0] = 1e-12
    state['c_hat_state']['skip_threshold'] = 1e-3
    alpha_before = state['alpha'][0].clone()
    mu_before = state['mu'][0].clone()

    susie.update_each_effect(
        X, xattr, y[:, None], state, estimate_prior_variance=True
    )

    torch.testing.assert_close(state['alpha'][0], alpha_before)
    torch.testing.assert_close(state['mu'][0], mu_before)
    assert float(state['slot_weights'][0]) == pytest.approx(1e-12)


def test_weighted_residual_variance_identity_matches_direct_formula():
    X, y = _data(seed=8, n=70, p=10, signals=((1, 1.0), (7, -0.8)))
    fit = susie.susie(
        X, y[:, None], L=4,
        slot_prior=susie.slot_prior_betabinom(),
        max_iter=20, coverage=None,
    )
    centered_y = y - y.mean()
    xattr = susie.get_x_attributes(X)
    Xr_L = susie.compute_MXt(fit['alpha'] * fit['mu'], X, xattr)
    second_moment = torch.matmul(fit['alpha'] * fit['mu2'], xattr['d'])
    expected = (
        ((centered_y - fit['Xr'])**2).sum()
        + (
            fit['slot_weights'] * second_moment
            - fit['slot_weights']**2 * (Xr_L**2).sum(1)
        ).sum()
    )

    torch.testing.assert_close(
        susie.get_ER2(X, xattr, centered_y[:, None], fit), expected
    )


@pytest.mark.parametrize(
    'prior',
    [
        pytest.param(susie.slot_prior_betabinom(), id='beta-binomial'),
        pytest.param(susie.slot_prior_poisson(C=3), id='poisson-sequential'),
        pytest.param(
            susie.slot_prior_poisson(C=3, update_schedule='batch'),
            id='poisson-batch',
        ),
    ],
)
def test_end_to_end_slot_fit_reports_consistent_effects(prior):
    X, y = _data(
        seed=9, n=120, p=25, signals=((2, 1.3), (18, -0.9))
    )
    fit = susie.susie(X, y[:, None], L=8, slot_prior=prior, max_iter=100)

    assert fit['converged']
    assert fit['c_hat'].shape == (8,)
    assert np.all((fit['c_hat'] >= 0) & (fit['c_hat'] <= 1))
    assert fit['C_hat'] == pytest.approx(fit['c_hat'].sum())
    prediction = fit['intercept'] + X @ fit['sparse_effects']
    torch.testing.assert_close(prediction, fit['fitted'])
    active = fit['V'] > 1e-9
    expected_pip = 1 - (
        1 - fit['slot_weights'][active, None] * fit['alpha'][active]
    ).prod(0)
    np.testing.assert_allclose(fit['pip'], expected_pip.cpu().numpy())


def test_null_and_excess_slot_fit_is_finite_and_sparse():
    X, y = _data(seed=11, n=100, p=20)
    fit = susie.susie(
        X, y[:, None], L=15,
        slot_prior=susie.slot_prior_poisson(C=2),
        max_iter=100,
    )

    assert np.isfinite(fit['c_hat']).all()
    assert fit['C_hat'] < 5
    assert torch.isfinite(fit['fitted']).all()


def test_none_slot_prior_preserves_default_result():
    X, y = _data(seed=13, n=80, p=12, signals=((4, 1.1),))
    default = susie.susie(X, y[:, None], L=3, max_iter=40)
    explicit_none = susie.susie(
        X, y[:, None], L=3, slot_prior=None, max_iter=40
    )

    for key in ['alpha', 'mu', 'mu2', 'V', 'Xr', 'fitted']:
        torch.testing.assert_close(default[key], explicit_none[key])
    np.testing.assert_array_equal(default['pip'], explicit_none['pip'])
    assert 'c_hat' not in default
