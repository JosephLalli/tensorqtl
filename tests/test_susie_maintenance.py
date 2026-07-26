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
