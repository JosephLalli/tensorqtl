import importlib.util
from pathlib import Path

import torch
import numpy as np
import pytest

# Load the hapmixqtl module directly from the repository (same pattern used in other tests)
_MODULE_PATH = Path(__file__).parent.parent / "tensorqtl" / "hapmixqtl.py"
_SPEC = importlib.util.spec_from_file_location("hapmixqtl_test_module_alpha", _MODULE_PATH)
hapmixqtl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hapmixqtl)

def _compute_alpha_a_torch(yL, yR, tauL, tauR, kappa, eps):
    """
    Replicate the intended alpha_a computation from hapmixqtl using PyTorch tensors.
    Inputs may be scalars or 1D arrays; outputs torch tensor.
    """
    # convert to torch tensors
    yL_t = torch.tensor(yL, dtype=torch.float32)
    yR_t = torch.tensor(yR, dtype=torch.float32)
    tauL_t = torch.tensor(tauL, dtype=torch.float32)
    tauR_t = torch.tensor(tauR, dtype=torch.float32)

    # clamp / NaN handling like the implementation
    tauL_t = torch.nan_to_num(torch.clamp(tauL_t, min=1.0), nan=1.0)
    tauR_t = torch.nan_to_num(torch.clamp(tauR_t, min=1.0), nan=1.0)

    denL = torch.clamp(yL_t + kappa, min=eps)
    denR = torch.clamp(yR_t + kappa, min=eps)
    dL = 1.0 / (denL * denL)
    dR = 1.0 / (denR * denR)

    alpha_a_t = (tauL_t * dL + tauR_t * dR) / torch.clamp(dL + dR, min=eps)
    return alpha_a_t

def test_alpha_equals_tau_when_taus_equal():
    # If tauL == tauR the effective alpha_a should equal that tau for any yL, yR
    kappa = 1.0
    eps = hapmixqtl.EPSILON

    yL = [0.5, 2.0, 50.0]
    yR = [1.0, 2.0, 10.0]
    tau = 2.5
    alpha = _compute_alpha_a_torch(yL, yR, tau, tau, kappa, eps)

    assert torch.allclose(alpha, torch.full_like(alpha, float(tau)), atol=1e-6)

def test_alpha_reduces_to_mean_when_y_equal():
    # If yL == yR then dL == dR and alpha should reduce to mean(tauL, tauR)
    kappa = 1.0
    eps = hapmixqtl.EPSILON

    y = [0.1, 1.0, 10.0]
    tauL = torch.tensor([1.5, 2.0, 3.0])
    tauR = torch.tensor([2.5, 4.0, 5.0])

    alpha = _compute_alpha_a_torch(y, y, tauL, tauR, kappa, eps)
    expected = (tauL + tauR) / 2.0

    assert torch.allclose(alpha, expected, atol=1e-6)

def test_alpha_approaches_low_expression_tau():
    # If yL << yR (yL much smaller), dL >> dR so alpha should be close to tauL
    kappa = 1.0
    eps = hapmixqtl.EPSILON

    yL = [1e-6, 1e-8, 0.0]  # extremely small left haplotype means
    yR = [100.0, 50.0, 10.0]  # large right haplotype means
    tauL = torch.tensor([10.0, 5.0, 2.0])
    tauR = torch.tensor([1.0, 1.0, 1.0])

    alpha = _compute_alpha_a_torch(yL, yR, tauL, tauR, kappa, eps)

    # alpha should be close to tauL (dominant low-expression haplotype)
    assert torch.allclose(alpha, tauL, rtol=1e-2, atol=1e-4)

def test_weights_are_finite_and_no_nans_when_taus_nan_or_y_zero():
    # Ensure numeric stability: NaN taus -> 1, zero y values handled via kappa and EPSILON
    kappa = 1.0
    eps = hapmixqtl.EPSILON

    yL = [0.0, 0.0]
    yR = [0.0, 1e-6]
    tauL = [float("nan"), float("nan")]
    tauR = [float("nan"), 3.0]

    Va = torch.tensor([0.5, 0.2], dtype=torch.float32)

    alpha = _compute_alpha_a_torch(yL, yR, tauL, tauR, kappa, eps)

    # compute base ASE weights as used in code
    w_a_base = 1.0 / torch.clamp(alpha * Va, min=eps)

    # Check no NaNs or infs
    assert torch.isfinite(alpha).all()
    assert torch.isfinite(w_a_base).all()
    # weights should be positive
    assert (w_a_base > 0).all()
