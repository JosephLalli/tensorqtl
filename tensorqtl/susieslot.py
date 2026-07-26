"""Slot-activity priors for the individual-data SuSiE implementation.

This module ports the small, data-independent part of the slot-prior
implementation in susieR 2.0.  A slot's weight is the posterior probability
``c_hat[l]`` that its single effect is active.
"""

import math
import numbers

import numpy as np
import torch


def _positive_scalar(value, name):
    if not isinstance(value, numbers.Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a positive finite scalar.')
    return float(value)


def _nonnegative_scalar(value, name):
    if not isinstance(value, numbers.Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be a non-negative finite scalar.')
    return float(value)


def _validate_c_hat_init(c_hat_init):
    if c_hat_init is None:
        return None
    c_hat_init = np.asarray(c_hat_init, dtype=float)
    if c_hat_init.ndim != 1 or not np.isfinite(c_hat_init).all():
        raise ValueError('c_hat_init must be a finite one-dimensional array.')
    if ((c_hat_init < 0) | (c_hat_init > 1)).any():
        raise ValueError('c_hat_init values must be between 0 and 1.')
    return c_hat_init.copy()


def slot_prior_betabinom(a_beta=1, b_beta=2, c_hat_init=None,
                         skip_threshold_multiplier=0):
    """Construct the collapsed Beta-Binomial prior used for slot activity."""
    return {
        '_slot_prior': True,
        'prior_type': 'betabinom',
        'a_beta': _positive_scalar(a_beta, 'a_beta'),
        'b_beta': _positive_scalar(b_beta, 'b_beta'),
        'update_schedule': 'sequential',
        'c_hat_init': _validate_c_hat_init(c_hat_init),
        'skip_threshold_multiplier': _nonnegative_scalar(
            skip_threshold_multiplier, 'skip_threshold_multiplier'
        ),
    }


def slot_prior_poisson(C, nu=8, update_schedule='sequential',
                       c_hat_init=None, skip_threshold_multiplier=0):
    """Construct the Gamma-Poisson prior with Poisson slot approximation."""
    if update_schedule not in {'sequential', 'batch'}:
        raise ValueError("update_schedule must be 'sequential' or 'batch'.")
    return {
        '_slot_prior': True,
        'prior_type': 'poisson',
        'C': _positive_scalar(C, 'C'),
        'nu': _positive_scalar(nu, 'nu'),
        'update_schedule': update_schedule,
        'c_hat_init': _validate_c_hat_init(c_hat_init),
        'skip_threshold_multiplier': _nonnegative_scalar(
            skip_threshold_multiplier, 'skip_threshold_multiplier'
        ),
    }


def initialize_slot_state(slot_prior, L, dtype, device):
    """Return the initial L-vector of slot weights and mutable update state."""
    if not isinstance(slot_prior, dict) or not slot_prior.get('_slot_prior', False):
        raise ValueError(
            'slot_prior must be created by slot_prior_betabinom() or '
            'slot_prior_poisson().'
        )

    c_hat_init = slot_prior['c_hat_init']
    if c_hat_init is not None and len(c_hat_init) != L:
        raise ValueError(f'c_hat_init must have length L={L}.')

    if slot_prior['prior_type'] == 'betabinom':
        if c_hat_init is None:
            prior_mean = slot_prior['a_beta'] / (
                slot_prior['a_beta'] + slot_prior['b_beta']
            )
            c_hat = np.repeat(min(prior_mean, 1 - 1e-10), L)
        else:
            c_hat = c_hat_init
        state = {
            'prior_type': 'betabinom',
            'a_beta': slot_prior['a_beta'],
            'b_beta': slot_prior['b_beta'],
            'update_schedule': 'sequential',
            'skip_threshold_multiplier': slot_prior['skip_threshold_multiplier'],
            'skip_threshold': 0.0,
        }
    else:
        if c_hat_init is None:
            c_hat = np.repeat(min(slot_prior['C'] / L, 1 - 1e-10), L)
            a_g = slot_prior['nu'] + slot_prior['C']
        else:
            c_hat = c_hat_init
            a_g = slot_prior['nu'] + float(np.sum(c_hat))
        state = {
            'prior_type': 'poisson',
            'C': slot_prior['C'],
            'nu': slot_prior['nu'],
            'a_g': a_g,
            'b_g': slot_prior['nu'] / max(slot_prior['C'], 1e-6) + 1,
            'update_schedule': slot_prior['update_schedule'],
            'skip_threshold_multiplier': slot_prior['skip_threshold_multiplier'],
            'skip_threshold': 0.0,
        }

    return torch.as_tensor(c_hat, dtype=dtype, device=device), state


def _sigmoid(value):
    value = max(min(float(value), 20.0), -20.0)
    return 1.0 / (1.0 + math.exp(-value))


def update_slot_weight(model, l):
    """Update one ``c_hat`` coordinate after its SER Bayes-factor update."""
    state = model['c_hat_state']
    old_c = float(model['slot_weights'][l])
    L = model['alpha'].shape[0]
    lbf_l = float(model['lbf'][l])
    if not math.isfinite(lbf_l):
        lbf_l = 0.0

    if state['prior_type'] == 'betabinom':
        k_others = float(model['slot_weights'].sum() - model['slot_weights'][l])
        log_odds = (
            math.log(state['a_beta'] + k_others)
            - math.log(state['b_beta'] + L - 1 - k_others)
            + lbf_l
        )
    else:
        a_g = torch.tensor(
            state['a_g'], dtype=model['slot_weights'].dtype,
            device=model['slot_weights'].device,
        )
        log_odds = (
            float(torch.digamma(a_g))
            - math.log(state['b_g'])
            - math.log(L)
            + lbf_l
        )

    new_c = _sigmoid(log_odds)
    model['slot_weights'][l] = new_c

    if (
        state['prior_type'] == 'poisson'
        and state['update_schedule'] == 'sequential'
    ):
        state['a_g'] = state['nu'] + float(model['slot_weights'].sum())

    return old_c, new_c


def finish_slot_sweep(model):
    """Apply batch Gamma updates and the upstream adaptive skip threshold."""
    state = model['c_hat_state']
    if (
        state['prior_type'] == 'poisson'
        and state['update_schedule'] == 'batch'
    ):
        state['a_g'] = state['nu'] + float(model['slot_weights'].sum())

    if state['skip_threshold_multiplier'] <= 0:
        return

    L = model['alpha'].shape[0]
    if state['prior_type'] == 'betabinom':
        k_total = float(model['slot_weights'].sum())
        approx = (
            math.log(state['a_beta'] + k_total)
            - math.log(state['b_beta'] + L - 1 - k_total)
        )
        k_others = k_total - _sigmoid(approx)
        baseline_log_odds = (
            math.log(state['a_beta'] + k_others)
            - math.log(state['b_beta'] + L - 1 - k_others)
        )
    else:
        a_g = torch.tensor(
            state['a_g'], dtype=model['slot_weights'].dtype,
            device=model['slot_weights'].device,
        )
        baseline_log_odds = (
            float(torch.digamma(a_g))
            - math.log(state['b_g'])
            - math.log(L)
        )

    state['skip_threshold'] = (
        state['skip_threshold_multiplier'] * _sigmoid(baseline_log_odds)
    )


def slot_prior_elbo(model):
    """Return the slot-prior and Bernoulli-entropy ELBO contribution."""
    state = model['c_hat_state']
    c_hat = model['slot_weights']
    eps = torch.finfo(c_hat.dtype).eps
    c = c_hat.clamp(eps, 1 - eps)
    bernoulli_entropy = -(c * c.log() + (1 - c) * (1 - c).log()).sum()

    if state['prior_type'] == 'betabinom':
        k = c_hat.sum()
        a = torch.as_tensor(state['a_beta'], dtype=c.dtype, device=c.device)
        b = torch.as_tensor(state['b_beta'], dtype=c.dtype, device=c.device)

        def log_beta(x, y):
            return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)

        return log_beta(a + k, b + len(c_hat) - k) - log_beta(a, b) + bernoulli_entropy

    a_g = torch.as_tensor(state['a_g'], dtype=c.dtype, device=c.device)
    b_g = torch.as_tensor(state['b_g'], dtype=c.dtype, device=c.device)
    nu = torch.as_tensor(state['nu'], dtype=c.dtype, device=c.device)
    expected = torch.as_tensor(state['C'], dtype=c.dtype, device=c.device)
    expected_log_mu = torch.digamma(a_g) - torch.log(b_g)
    expected_mu = a_g / b_g
    gamma_prior = (
        (nu - 1) * expected_log_mu
        - (nu / expected.clamp_min(1e-10)) * expected_mu
        + nu * torch.log(nu / expected.clamp_min(1e-10))
        - torch.lgamma(nu)
    )
    gamma_entropy = (
        a_g - torch.log(b_g) + torch.lgamma(a_g)
        + (1 - a_g) * torch.digamma(a_g)
    )
    poisson_slots = c_hat.sum() * (expected_log_mu - math.log(len(c_hat)))
    return gamma_prior + gamma_entropy + poisson_slots + bernoulli_entropy
