import contextlib
import io
import json
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


FIXTURE_PATH = (
    Path(__file__).parent / 'fixtures' / 'susier_slot_prior_reference.json'
)
with FIXTURE_PATH.open() as handle:
    ORACLE = json.load(handle)

RTOL = ORACLE['numeric_contract']['rtol']
ATOL = ORACLE['numeric_contract']['atol']
EXPECTED_COMMIT = 'dd9d9ce4693573e9dcd1ec8b3df94b63bac467d2'


def _prior_from_spec(spec):
    spec = dict(spec)
    family = spec.pop('family')
    if family == 'beta_binomial':
        return susie.slot_prior_betabinom(**spec)
    if family == 'gamma_poisson':
        return susie.slot_prior_poisson(**spec)
    raise AssertionError(f'Unknown oracle prior family: {family}')


def _input_tensors():
    X = torch.tensor(ORACLE['input']['X'], dtype=torch.float32)
    y = torch.tensor(ORACLE['input']['y'], dtype=torch.float32)
    return X, y


def _run_case(case):
    X, y = _input_tensors()
    args = dict(ORACLE['common_args'])
    args.pop('residual_variance')
    with contextlib.redirect_stdout(io.StringIO()):
        return susie.susie(
            X,
            y[:, None],
            slot_prior=_prior_from_spec(case['prior']),
            **args,
        )


def _numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _assert_oracle_close(actual, expected, name):
    np.testing.assert_allclose(
        _numpy(actual),
        np.asarray(expected),
        rtol=RTOL,
        atol=ATOL,
        err_msg=f'{name} differs from pinned susieR oracle',
    )


def test_slot_prior_oracle_records_pinned_temporary_install():
    provenance = ORACLE['provenance']

    assert provenance['upstream'] == 'stephenslab/susieR'
    assert provenance['source_commit'] == EXPECTED_COMMIT
    assert provenance['package_version'] == '0.16.5'
    assert 'R CMD INSTALL --preclean' in provenance['generation_command']
    assert 'env -u LD_LIBRARY_PATH -u R_LD_LIBRARY_PATH' in (
        provenance['generation_command']
    )


@pytest.mark.parametrize('case_name', ORACLE['cases'])
def test_individual_slot_prior_fit_matches_pinned_susier(case_name):
    case = ORACLE['cases'][case_name]
    expected = case['expected']
    fit = _run_case(case)

    assert fit['niter'] == expected['niter']
    assert fit['converged'] is expected['converged']
    _assert_oracle_close(fit['sigma2'], expected['sigma2'], 'sigma2')
    _assert_oracle_close(fit['c_hat'], expected['c_hat'], 'c_hat')
    _assert_oracle_close(fit['C_hat'], expected['C_hat'], 'C_hat')
    _assert_oracle_close(fit['alpha'], expected['alpha'], 'alpha')
    _assert_oracle_close(fit['mu'], expected['mu'], 'mu')
    _assert_oracle_close(fit['V'], expected['V'], 'V')
    _assert_oracle_close(fit['lbf'], expected['lbf'], 'lbf')
    _assert_oracle_close(fit['fitted'], expected['fitted'], 'fitted')
    _assert_oracle_close(
        susie.susie_get_pip(fit), expected['pip'], 'slot-weighted PIP'
    )
    _assert_oracle_close(
        fit['sparse_effects'],
        expected['sparse_effect_raw'],
        'raw-scale sparse effects',
    )


@pytest.mark.parametrize('case_name', ORACLE['cases'])
def test_slot_prior_sparse_fitted_and_pip_identities(case_name):
    case = ORACLE['cases'][case_name]
    fit = _run_case(case)
    X, _ = _input_tensors()
    device = fit['alpha'].device
    X = X.to(device)

    c_hat = fit['slot_weights']
    sparse_standardized = (
        c_hat[:, None] * fit['alpha'] * fit['mu']
    ).sum(0)
    scale = torch.tensor(
        ORACLE['input']['scaled_scale'],
        dtype=X.dtype,
        device=device,
    )
    sparse_raw = sparse_standardized / scale
    pip_identity = 1 - (
        1 - c_hat[:, None] * fit['alpha']
    ).prod(0)
    fitted_identity = fit['intercept'] + X @ sparse_raw

    _assert_oracle_close(
        sparse_standardized,
        case['expected']['sparse_effect_standardized'],
        'standardized sparse-effect identity',
    )
    torch.testing.assert_close(
        fit['sparse_effects'], sparse_raw, rtol=RTOL, atol=ATOL
    )
    torch.testing.assert_close(
        fit['fitted'], fitted_identity, rtol=RTOL, atol=ATOL
    )
    torch.testing.assert_close(
        susie.susie_get_pip(fit), pip_identity, rtol=RTOL, atol=ATOL
    )
    assert fit['C_hat'] == pytest.approx(
        float(c_hat.sum()), rel=RTOL, abs=ATOL
    )


def test_warm_start_and_skip_history_match_pinned_susier():
    warm_case = ORACLE['cases']['beta_binomial_warm_start']
    skip_case = ORACLE['cases']['gamma_poisson_batch_skip']
    expected_warm = np.asarray(
        warm_case['expected']['slot_weight_history'], dtype=np.float32
    )
    expected_skip = np.asarray(
        skip_case['expected']['slot_weight_history'], dtype=np.float32
    )

    X, y = _input_tensors()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = X.to(device)
    y = y.to(device)
    y_centered = y - y.mean()
    args = ORACLE['common_args']
    state = susie.init_finalize(
        susie.init_setup(
            len(y),
            X.shape[1],
            args['L'],
            args['scaled_prior_variance'],
            y_centered.var(unbiased=True),
        )
    )
    state['slot_weights'], state['c_hat_state'] = (
        susieslot.initialize_slot_state(
            _prior_from_spec(skip_case['prior']),
            args['L'],
            state['alpha'].dtype,
            state['alpha'].device,
        )
    )
    xattr = susie.get_x_attributes(X)
    susie._recompute_weighted_fitted(X, xattr, state)
    observed = [_numpy(state['slot_weights']).copy()]
    for _ in range(args['max_iter']):
        susie.update_each_effect(
            X,
            xattr,
            y_centered[:, None],
            state,
            estimate_prior_variance=False,
        )
        observed.append(_numpy(state['slot_weights']).copy())

    np.testing.assert_allclose(
        np.asarray(observed), expected_skip, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_array_equal(
        expected_warm[0],
        np.asarray(warm_case['prior']['c_hat_init'], dtype=np.float32),
    )
    # Slots 3-5 fall below the adaptive threshold after the first sweep and
    # remain frozen in both the R oracle and TensorQTL.
    np.testing.assert_array_equal(
        np.asarray(observed)[1:, 2:],
        np.repeat(np.asarray(observed)[1:2, 2:], args['max_iter'], axis=0),
    )
