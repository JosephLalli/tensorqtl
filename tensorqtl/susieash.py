"""Helpers for the experimental SuSiE-ash integration.

This is a deliberately limited port: ``c_hat`` is fixed at 1 and the full
three-state masking logic from susieR is represented by a two-state rule.
"""

import numpy as np
import torch

import mrash


def default_sa2_grid(w, n):
    """Return the default Mr.ASH mixture-variance grid."""
    return mrash.default_sa2_grid(w, n)


def pip(alpha_t):
    """Compute PIPs from an L x p alpha matrix."""
    return 1.0 - torch.prod(1.0 - alpha_t, dim=0)


def variant_corr(x_std_t, d_t):
    """Compute correlations between columns of a standardized design."""
    gram = x_std_t.T @ x_std_t
    denom = torch.sqrt(torch.outer(d_t, d_t)).clamp_min(1e-30)
    return gram / denom


def confident_mask(alpha_t, mu_t, Xcorr_t, cs_threshold=0.9,
                   purity_threshold=0.5, active_tol=5e-5,
                   ld_threshold=0.5, pip_nbhd_thresh=0.4,
                   pip_self_thresh=0.1):
    """Return confident effects and variants protected from the dense fit.

    Active single-effect components protect their LD neighborhoods from being
    absorbed by the Mr.ASH background. Components whose working credible sets
    also pass the purity threshold contribute to the confident sparse effect.
    """
    L, p = alpha_t.shape
    device = alpha_t.device
    b_confident = torch.zeros(p, dtype=alpha_t.dtype, device=device)
    alpha_protected = torch.zeros_like(alpha_t)
    for l in range(L):
        a = alpha_t[l]
        if float(a.max() - a.min()) < active_tol:
            continue
        alpha_protected[l] = a
        order = torch.argsort(a, descending=True)
        k = min(
            int((torch.cumsum(a[order], dim=0) < cs_threshold).sum().item()) + 1,
            p,
        )
        cs_idx = order[:k]
        if k <= 1:
            purity = 1.0
        else:
            sub = Xcorr_t[cs_idx][:, cs_idx].abs()
            iu = torch.triu_indices(k, k, offset=1, device=device)
            purity = float(sub[iu[0], iu[1]].min().item())
        if purity >= purity_threshold:
            b_confident = b_confident + a * mu_t[l]

    pip_protected = pip(alpha_protected)
    ld_adj = (Xcorr_t.abs() > ld_threshold).to(pip_protected.dtype)
    neighborhood_pip = ld_adj @ pip_protected
    mask_t = (
        (neighborhood_pip > pip_nbhd_thresh)
        | (pip_protected > pip_self_thresh)
    )
    return b_confident, mask_t


def refit(x_std_np, target_np, sigma2, beta_init_np, ash_pi, sa2_np,
          convtol, update_sigma):
    """Refit the Mr.ASH background on a standardized design."""
    out = mrash.mr_ash(
        x_std_np,
        target_np,
        sa2=sa2_np,
        sigma2=float(sigma2),
        pi=ash_pi,
        beta_init=beta_init_np,
        update_pi=True,
        update_sigma=bool(update_sigma),
        method_q='sigma_dep_q',
        intercept=False,
        max_iter=1000,
        min_iter=1,
        convtol=convtol,
    )
    tau2 = float((sa2_np * out['pi']).sum() * out['sigma2'])
    return out['beta'], out['sigma2'], out['pi'], tau2
