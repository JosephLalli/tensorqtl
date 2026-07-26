"""Pinned susieR oracle checks for the full SuSiE-ash state policy."""

import json
import sys
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).parents[1] / 'tensorqtl'))
import types
if 'pandas_plink' not in sys.modules:
    stub = types.ModuleType('pandas_plink')
    stub.read_plink = lambda *args, **kwargs: None
    stub.read_plink1_bin = lambda *args, **kwargs: None
    sys.modules['pandas_plink'] = stub
import susie
import susieash


FIXTURE = (
    Path(__file__).parent
    / 'fixtures'
    / 'susier_ash_state_reference.json'
)
PINNED_COMMIT = 'dd9d9ce4693573e9dcd1ec8b3df94b63bac467d2'


def _fixture():
    with FIXTURE.open() as handle:
        return json.load(handle)


def _as_numpy(value):
    return value.detach().cpu().numpy()


def test_ash_state_fixture_has_exact_pinned_provenance():
    fixture = _fixture()
    provenance = fixture['provenance']
    assert provenance['source_commit'] == PINNED_COMMIT
    assert provenance['upstream_function'] == (
        'update_ash_variance_components'
    )
    assert provenance['skip_mrash'] is True
    assert 'R CMD INSTALL --preclean' in provenance['generation_command']
    assert 'env -u LD_LIBRARY_PATH -u R_LD_LIBRARY_PATH' in (
        provenance['generation_command']
    )


def test_full_ash_policy_matches_pinned_state_oracle():
    fixture = _fixture()
    Xcorr = torch.tensor(fixture['input']['Xcorr'], dtype=torch.float64)
    mu = torch.tensor(fixture['input']['mu'], dtype=torch.float64)
    c_hat = torch.tensor(fixture['input']['c_hat'], dtype=torch.float64)
    L, p = mu.shape

    for case_name, case in fixture['cases'].items():
        state = susieash.initialize_state(L, p, torch.device('cpu'))
        if 'initial_masked' in case:
            state['masked'] = torch.tensor(
                case['initial_masked'], dtype=torch.bool
            )
        for alpha_values, expected in zip(
            case['alpha'], case['expected'], strict=True
        ):
            alpha = torch.tensor(alpha_values, dtype=torch.float64)
            result = susieash.update_policy(
                alpha, mu, Xcorr, c_hat, state
            )

            assert state['ash_iter'] == expected['ash_iter'], case_name
            np.testing.assert_array_equal(
                _as_numpy(result['current_case']),
                expected['current_case'],
                err_msg=case_name,
            )
            np.testing.assert_array_equal(
                _as_numpy(result['sentinels']),
                expected['sentinels'],
                err_msg=case_name,
            )
            np.testing.assert_allclose(
                _as_numpy(result['effect_purity']),
                expected['effect_purity'],
                rtol=0,
                atol=1e-15,
                err_msg=case_name,
            )
            np.testing.assert_array_equal(
                _as_numpy(result['current_collision']),
                expected['collision'],
                err_msg=case_name,
            )

            for field in [
                'ever_diffuse',
                'diffuse_iter_count',
                'masked',
                'ever_unmasked',
                'unmask_candidate_iters',
                'force_exposed_iter',
                'second_chance_used',
            ]:
                np.testing.assert_array_equal(
                    _as_numpy(state[field]),
                    expected[field],
                    err_msg=f'{case_name}: {field}',
                )

            b_confident = _as_numpy(result['b_confident'])
            np.testing.assert_allclose(
                np.square(b_confident).sum(),
                expected['b_confident_ss'],
                rtol=0,
                atol=1e-15,
                err_msg=case_name,
            )
            np.testing.assert_allclose(
                np.abs(b_confident).max(),
                expected['b_confident_max'],
                rtol=0,
                atol=1e-15,
                err_msg=case_name,
            )


def test_oracle_cases_activate_required_ash_branches():
    fixture = _fixture()
    cases = fixture['cases']

    wait = cases['wait_expose_second_chance']['expected']
    assert any(any(v > 0 for v in x['force_exposed_iter']) for x in wait)
    assert any(any(x['second_chance_used']) for x in wait)

    oscillation = cases['oscillation_reversal']['expected']
    assert oscillation[0]['current_case'][0] == 2
    assert oscillation[1]['current_case'][0] == 3
    # The CASE 2 -> CASE 3 flip is marked sticky-diffuse, and its confident
    # subtraction is reversed. Only the other confident slot remains.
    assert oscillation[1]['ever_diffuse'][0] > 0

    collision = cases['collision']['expected'][0]
    assert collision['collision'][:2] == [True, True]
    assert collision['diffuse_iter_count'][:2] == [0, 0]

    all_cases = cases['all_three_cases']['expected'][0]
    assert all_cases['current_case'] == [1, 2, 3]

    delayed = cases['delayed_unmask']['expected']
    assert delayed[0]['masked'][-1] is True
    assert delayed[0]['unmask_candidate_iters'][-1] == 1
    assert delayed[1]['masked'][-1] is False
    assert delayed[1]['ever_unmasked'][-1] is True


def test_end_to_end_ash_matches_pinned_standardized_design_oracle():
    fixture = _fixture()['end_to_end']
    X = torch.tensor(fixture['input']['X'], dtype=torch.float32)
    y = torch.tensor(fixture['input']['y'], dtype=torch.float32)[:, None]
    args = fixture['args']
    fit = susie.susie(
        X,
        y,
        L=args['L'],
        scaled_prior_variance=args['scaled_prior_variance'],
        estimate_prior_variance=args['estimate_prior_variance'],
        estimate_prior_method=args['estimate_prior_method'],
        estimate_residual_variance=args['estimate_residual_variance'],
        unmappable_effects='ash',
        coverage=args['coverage'],
        max_iter=args['max_iter'],
        tol=args['tol'],
        standardize=args['standardize'],
        intercept=args['intercept'],
    )
    expected = fixture['expected']

    assert fit['niter'] == expected['niter']
    assert fit['converged'] is expected['converged']
    assert fit['ash_final_pass'] is True
    np.testing.assert_allclose(
        _as_numpy(fit['alpha']), expected['alpha'], rtol=3e-5, atol=2e-7
    )
    np.testing.assert_allclose(
        _as_numpy(fit['mu']), expected['mu'], rtol=3e-5, atol=2e-6
    )
    np.testing.assert_allclose(
        _as_numpy(fit['mu2']), expected['mu2'], rtol=3e-5, atol=3e-6
    )
    np.testing.assert_allclose(
        _as_numpy(fit['V']), expected['V'], rtol=3e-5, atol=2e-6
    )
    np.testing.assert_allclose(
        fit['c_hat'], expected['c_hat'], rtol=3e-5, atol=2e-7
    )
    assert fit['C_hat'] == np.sum(fit['c_hat'])
    np.testing.assert_allclose(
        fit['theta'], expected['theta'], rtol=3e-5, atol=2e-6
    )
    np.testing.assert_allclose(
        fit['tau2'], expected['tau2'], rtol=3e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        fit['pip'], expected['pip'], rtol=3e-5, atol=2e-7
    )
    np.testing.assert_allclose(
        _as_numpy(fit['fitted']),
        expected['fitted'],
        rtol=3e-5,
        atol=2e-6,
    )
    np.testing.assert_array_equal(fit['masked'], expected['masked'])
    np.testing.assert_array_equal(
        fit['ever_diffuse'], expected['ever_diffuse']
    )
    np.testing.assert_array_equal(
        fit['second_chance_used'], expected['second_chance_used']
    )

    sparse_standardized = susie._weighted_sparse_effect(fit)
    np.testing.assert_allclose(
        _as_numpy(sparse_standardized),
        expected['sparse_effect_standardized'],
        rtol=3e-5,
        atol=2e-6,
    )
    total_raw = fit['sparse_effects'] + fit['theta_raw']
    reconstructed = X @ total_raw + fit['intercept']
    torch.testing.assert_close(
        reconstructed, fit['fitted'], rtol=3e-5, atol=2e-6
    )

    # Pinned run_final_ash_pass computes tau2 from the final Mr.ASH sigma2 but
    # leaves model$sigma2 at its pre-pass value. TensorQTL intentionally returns
    # the final optimizer's sigma2 so theta, tau2, and residual variance describe
    # one fit. Keep this sole expected scalar difference explicit.
    assert not np.isclose(float(fit['sigma2']), expected['sigma2'], atol=1e-3)
    assert float(fit['sigma2']) > 0
