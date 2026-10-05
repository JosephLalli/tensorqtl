"""Fabricated channel simulator shared by the hapmixQTL calibration tests."""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from tensorqtl.hapmixqtl import _prepare_channels, calculate_hapmixqtl_nominal


DTYPE = torch.float64


def simulate_channels(N, V, rng, beta=0.0, maf=0.35,
                      v_inf_lo=0.05, v_inf_hi=0.30,
                      sigma_bio=0.0, rho_at=0.0):
    """Simulate fabricated hapmixQTL channels for one gene and V variants."""
    g = rng.binomial(2, maf, size=(V, N)).astype(np.float64)
    sign = np.zeros((V, N))
    het = g == 1
    sign[het] = rng.choice([-1.0, 1.0], size=int(het.sum()))

    va = rng.uniform(v_inf_lo, v_inf_hi, N)
    vt = rng.uniform(v_inf_lo, v_inf_hi, N)
    z = rng.multivariate_normal([0, 0], [[1.0, rho_at], [rho_at, 1.0]], size=N)
    a = np.sqrt(va) * z[:, 0]
    t = 2.0 + np.sqrt(vt) * z[:, 1]

    if sigma_bio > 0:
        a = a + rng.normal(0, sigma_bio, N)
        t = t + rng.normal(0, sigma_bio, N)
    if beta != 0.0:
        a = a + beta * sign[0]
        t = t + beta * (g[0] / 2.0)
    return g, sign, a, t, va, vt


def run_association(g, sign, a, t, va, vt, tau_mode='zero', covariates=None,
                    ase_covariates='same'):
    """Run the production statistic on fabricated channels."""
    dev = 'cpu'
    g_t = torch.tensor(g, dtype=DTYPE, device=dev)
    s_t = torch.tensor(sign, dtype=DTYPE, device=dev)
    a_t = torch.tensor(a, dtype=DTYPE, device=dev)
    t_t = torch.tensor(t, dtype=DTYPE, device=dev)
    va_t = torch.tensor(va, dtype=DTYPE, device=dev)
    vt_t = torch.tensor(vt, dtype=DTYPE, device=dev)
    cov_t = None if covariates is None else torch.tensor(covariates, dtype=DTYPE, device=dev)
    if isinstance(ase_covariates, str):
        ase_t = ase_covariates
    else:
        ase_t = None if ase_covariates is None else torch.tensor(
            ase_covariates, dtype=DTYPE, device=dev)
    sqrt_wa, sqrt_wt, res_a, res_t = _prepare_channels(
        a_t, t_t, va_t, vt_t, cov_t, tau_mode, dev, ase_covariates_t=ase_t)
    tstat, slope, se, slope_a, se_a, slope_tc, se_tc = calculate_hapmixqtl_nominal(
        g_t, s_t, a_t, t_t, sqrt_wa, sqrt_wt, res_a, res_t)
    return (tstat.numpy(), slope.numpy(), se.numpy(),
            slope_tc.numpy(), se_tc.numpy(), slope_a.numpy(), se_a.numpy())


def pvals_from_t(tstat, N, n_cov=0):
    """Match the production t-reference used by the calibration test."""
    dof = N - 2 - n_cov
    return 2 * stats.t.sf(np.abs(tstat), dof)
