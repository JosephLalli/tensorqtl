"""SuSiE-ash state handling and Mr.ASH refit helpers.

The masking policy is a direct procedural port of the individual-data
``update_ash_variance_components`` implementation in susieR 2.0.  It keeps
slot-level confidence history separate from variant-level mask history so the
policy can be tested independently of the numerical Mr.ASH solver.
"""

import numpy as np
import torch

import mrash


def default_sa2_grid(w, n):
    """Return the default Mr.ASH mixture-variance grid."""
    return mrash.default_sa2_grid(w, n)


def pip(alpha_t):
    """Compute ordinary PIPs from an L x p alpha matrix."""
    return 1.0 - torch.prod(1.0 - alpha_t, dim=0)


def variant_corr(x_std_t, d_t):
    """Compute correlations between columns of a standardized design."""
    gram = x_std_t.T @ x_std_t
    denom = torch.sqrt(torch.outer(d_t, d_t)).clamp_min(1e-30)
    return (gram / denom).clamp(-1, 1)


def initialize_state(L, p, device):
    """Initialize the persistent slot and variant state for SuSiE-ash.

    R uses zero as the uninitialized sentinel because its variant indices start
    at one.  The Python port uses -1 so variant zero remains a valid sentinel.
    """
    return {
        'ash_iter': 0,
        'prev_case': torch.zeros(L, dtype=torch.int64, device=device),
        'prev_sentinel': torch.full(
            (L,), -1, dtype=torch.int64, device=device
        ),
        'ever_diffuse': torch.zeros(L, dtype=torch.int64, device=device),
        'diffuse_iter_count': torch.zeros(
            L, dtype=torch.int64, device=device
        ),
        'masked': torch.zeros(p, dtype=torch.bool, device=device),
        'ever_unmasked': torch.zeros(p, dtype=torch.bool, device=device),
        'unmask_candidate_iters': torch.zeros(
            p, dtype=torch.int64, device=device
        ),
        'force_exposed_iter': torch.zeros(
            p, dtype=torch.int64, device=device
        ),
        'second_chance_used': torch.zeros(
            p, dtype=torch.bool, device=device
        ),
    }


def _working_cs_purity(alpha_l, Xcorr_t, cs_threshold):
    """Return sentinel and minimum-correlation purity of the working CS."""
    p = alpha_l.numel()
    # R's order(..., decreasing=TRUE) preserves index order for ties. The
    # default torch sort is unstable (and chooses different tied sentinels on
    # CPU and CUDA), so request stability explicitly.
    order = torch.argsort(alpha_l, descending=True, stable=True)
    cs_size = min(
        int((torch.cumsum(alpha_l[order], 0) <= cs_threshold).sum()) + 1,
        p,
    )
    sentinel = int(order[0])
    if cs_size <= 1:
        return sentinel, 1.0
    cs = order[:cs_size]
    corr = Xcorr_t[cs][:, cs].abs()
    upper = torch.triu_indices(
        cs_size, cs_size, offset=1, device=alpha_l.device
    )
    return sentinel, float(corr[upper[0], upper[1]].min())


def update_policy(alpha_t, mu_t, Xcorr_t, c_hat_t, state,
                  purity_threshold=0.5, pip_threshold=0.1,
                  ld_threshold=0.5):
    """Advance the full diffuse/uncertain/confident ash masking policy.

    The returned ``b_confident`` is on the standardized-X coefficient scale and
    includes slot activity weights. ``mask`` identifies variants whose retained
    background coefficients are zeroed before and after the corresponding
    in-loop Mr.ASH refit. As in the pinned implementation, the coordinate solver
    still visits all variants internally; this is protection of the retained
    background, not hard exclusion from optimization. Persistent arrays in
    ``state`` are updated in place and also returned for convenience.
    """
    diffuse_purity = 0.1
    cs_threshold = 0.9
    neighborhood_pip_threshold = 0.4
    collision_threshold = 0.9
    tight_ld_threshold = 0.95
    stable_diffuse_iters = 2
    second_chance_wait = 3
    delayed_unmask_iters = 2

    L, p = alpha_t.shape
    if mu_t.shape != (L, p) or Xcorr_t.shape != (p, p):
        raise ValueError('alpha, mu, and Xcorr dimensions are inconsistent.')
    if c_hat_t.shape != (L,):
        raise ValueError('c_hat must contain one activity weight per slot.')

    state['ash_iter'] += 1
    sentinels = torch.empty(L, dtype=torch.int64, device=alpha_t.device)
    effect_purity = torch.ones(
        L, dtype=alpha_t.dtype, device=alpha_t.device
    )
    is_active = torch.zeros(L, dtype=torch.bool, device=alpha_t.device)
    for l in range(L):
        sent, purity = _working_cs_purity(
            alpha_t[l], Xcorr_t, cs_threshold
        )
        sentinels[l] = sent
        effect_purity[l] = purity
        is_active[l] = alpha_t[l].max() - alpha_t[l].min() >= 5e-5

    current_collision = torch.zeros(
        L, dtype=torch.bool, device=alpha_t.device
    )
    active_slots = torch.where(is_active)[0]
    for l_t in active_slots:
        l = int(l_t)
        other_slots = active_slots[active_slots != l]
        if other_slots.numel() == 0:
            continue
        other_sentinels = sentinels[other_slots]
        if bool(
            (Xcorr_t[sentinels[l], other_sentinels].abs()
             > collision_threshold).any()
        ):
            current_collision[l] = True
            state['ever_diffuse'][l] += 1

    b_confident = torch.zeros(
        p, dtype=alpha_t.dtype, device=alpha_t.device
    )
    alpha_protected = torch.zeros_like(alpha_t)
    force_unmask = torch.zeros(
        p, dtype=torch.bool, device=alpha_t.device
    )
    force_mask = torch.zeros(
        p, dtype=torch.bool, device=alpha_t.device
    )
    current_case = torch.zeros(
        L, dtype=torch.int64, device=alpha_t.device
    )

    for l in range(L):
        purity = float(effect_purity[l])
        sentinel = int(sentinels[l])
        previous_sentinel = int(state['prev_sentinel'][l])
        if sentinel != previous_sentinel and previous_sentinel >= 0:
            if (
                float(Xcorr_t[sentinel, previous_sentinel].abs())
                < tight_ld_threshold
            ):
                state['diffuse_iter_count'][l] = 0

        can_be_confident = (
            purity >= purity_threshold
            and int(state['ever_diffuse'][l]) == 0
        )
        if purity < diffuse_purity:
            # CASE 1: diffuse. Protect only the sentinel neighborhood and
            # unusually large alpha entries, and force that neighborhood masked.
            current_case[l] = 1
            state['diffuse_iter_count'][l] = 0
            moderate_ld = Xcorr_t[sentinel].abs() > ld_threshold
            to_protect = moderate_ld | (alpha_t[l] > 5.0 / p)
            alpha_protected[l, to_protect] = alpha_t[l, to_protect]
            force_mask |= moderate_ld
        elif not can_be_confident:
            # CASE 2: uncertain. Collisions reset the stability counter. Stable
            # slots eventually expose their tight-LD region to Mr.ASH.
            current_case[l] = 2
            if bool(current_collision[l]):
                state['diffuse_iter_count'][l] = 0
            else:
                state['diffuse_iter_count'][l] += 1
                alpha_protected[l] = alpha_t[l]
                if (
                    int(state['diffuse_iter_count'][l])
                    >= stable_diffuse_iters
                ):
                    tight_ld = (
                        Xcorr_t[sentinel].abs() > tight_ld_threshold
                    )
                    expose = tight_ld & ~state['second_chance_used']
                    newly = expose & (state['force_exposed_iter'] == 0)
                    state['force_exposed_iter'][newly] = state['ash_iter']
                    if bool(newly.any()):
                        state['diffuse_iter_count'][l] = 0
                    alpha_protected[l, expose] = 0
                    force_unmask |= expose
        else:
            # CASE 3: confident. Mr.ASH sees the response after subtraction of
            # this slot's c_hat-weighted posterior mean.
            current_case[l] = 3
            state['diffuse_iter_count'][l] = 0
            alpha_protected[l] = alpha_t[l]
            b_confident += c_hat_t[l] * alpha_t[l] * mu_t[l]

    oscillated = (
        (state['prev_case'] != 0)
        & (current_case != 0)
        & (
            ((state['prev_case'] == 2) & (current_case == 3))
            | ((state['prev_case'] == 3) & (current_case == 2))
        )
    )
    unstable_case3 = oscillated & (current_case == 3)
    state['ever_diffuse'][oscillated] += 1
    for l_t in torch.where(unstable_case3)[0]:
        l = int(l_t)
        b_confident -= c_hat_t[l] * alpha_t[l] * mu_t[l]

    state['prev_case'] = current_case.clone()
    state['prev_sentinel'] = sentinels.clone()

    pip_protected = pip(alpha_protected)
    ld_adj = (Xcorr_t.abs() > ld_threshold).to(alpha_t.dtype)
    neighborhood_pip = ld_adj @ pip_protected
    want_masked = (
        (neighborhood_pip > neighborhood_pip_threshold)
        | (pip_protected > pip_threshold)
        | force_mask
    )

    previously_masked = state['masked']
    reset = want_masked | ~previously_masked
    state['unmask_candidate_iters'][~reset] += 1
    state['unmask_candidate_iters'][reset] = 0
    ready_to_unmask = previously_masked & (
        (
            (state['unmask_candidate_iters'] >= delayed_unmask_iters)
            & ~state['ever_unmasked']
        )
        | force_unmask
    )
    state['ever_unmasked'][ready_to_unmask] = True
    masked = (
        (previously_masked | want_masked)
        & ~ready_to_unmask
        & ~state['ever_unmasked']
    )

    should_restore = (
        (state['force_exposed_iter'] > 0)
        & (
            state['ash_iter'] - state['force_exposed_iter']
            >= second_chance_wait
        )
        & ~state['second_chance_used']
    )
    if bool(should_restore.any()):
        state['second_chance_used'][should_restore] = True
        state['force_exposed_iter'][should_restore] = 0
        state['ever_unmasked'][should_restore] = False
        masked[should_restore] = True
    state['masked'] = masked

    return {
        'state': state,
        'b_confident': b_confident,
        'mask': masked,
        'sentinels': sentinels,
        'effect_purity': effect_purity,
        'is_active': is_active,
        'current_collision': current_collision,
        'current_case': current_case,
        'oscillated': oscillated,
        'alpha_protected': alpha_protected,
        'pip_protected': pip_protected,
        'neighborhood_pip': neighborhood_pip,
        'force_mask': force_mask,
        'force_unmask': force_unmask,
    }


def refit(x_std_np, target_np, sigma2, beta_init_np, ash_pi, sa2_np,
          convtol, update_sigma, sigma2_upperbound=np.inf):
    """Refit the Mr.ASH background on a standardized design."""
    upperbound = float(sigma2_upperbound)
    if upperbound <= 0 or np.isnan(upperbound):
        raise ValueError('sigma2_upperbound must be positive.')

    def run(fixed_sigma2, beta_init, pi_init, update):
        return mrash.mr_ash(
            x_std_np,
            target_np,
            sa2=sa2_np,
            sigma2=float(fixed_sigma2),
            pi=pi_init,
            beta_init=np.asarray(beta_init, dtype=np.float64).copy(),
            update_pi=True,
            update_sigma=bool(update),
            method_q='sigma_dep_q',
            intercept=False,
            max_iter=1000,
            min_iter=1,
            convtol=convtol,
        )

    out = run(sigma2, beta_init_np, ash_pi, update_sigma)
    if out['sigma2'] > upperbound:
        # A post-hoc sigma2 clip leaves beta, pi, and tau2 describing the
        # unconstrained fit. Re-optimize beta/pi with sigma2 fixed at the bound
        # so every returned variance component belongs to one constrained fit.
        out = run(
            upperbound, out['beta'], out['pi'], False
        )
    tau2 = float((sa2_np * out['pi']).sum() * out['sigma2'])
    return out['beta'], out['sigma2'], out['pi'], tau2
