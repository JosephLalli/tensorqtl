"""Print the derived figures the report quotes, so they are never hand-copied."""

import json

import numpy as np

D = '/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_replication_20260919'


def main():
    s = json.load(open(f'{D}/summary.json'))
    arms = {a['arm']: a for a in s['weighting_ablation']}
    g = arms['gibbs_1_over_v']
    nq = arms['gibbs_draws_only']
    print(f"null draws: {s['n_null_draws']}   genes: {s['n_genes']}")
    print('\nderived figures quoted in REPORT.md:')
    print(f"  gibbs ratio vs OLS        {g['median_var_ratio_vs_ols']:.4f}")
    print(f"  -> SE factor sqrt(1/r)    {1 / np.sqrt(g['median_var_ratio_vs_ols']):.3f}")
    print(f"  gibbs calib, fitted sigma {g['median_calibration']:.4f}")
    print(f"  gibbs calib, shipped SE   {g['median_calib_knownvar']:.4f}")
    print(f"  -> SE understates by      {np.sqrt(g['median_calib_knownvar']):.3f}")
    print(f"  q-on / q-off ratios       {g['median_var_ratio_vs_ols']:.4f} / "
          f"{nq['median_var_ratio_vs_ols']:.4f}")
    print('\nablation table rows:')
    for k in ('equal_ols', 'gibbs_1_over_v', 'gibbs_draws_only', 'gibbs_capped',
              'harmonic_uncapped', 'harmonic_poisson_capped'):
        a = arms[k]
        kv = a['median_calib_knownvar']
        print(f"  {k:26s} var {a['median_var_beta']:.6f}  ratio "
              f"{a['median_var_ratio_vs_ols']:.3f}  calib {a['median_calibration']:.3f}  "
              f"knownvar {kv:.3f}  fold {a['median_weight_fold']:.1f}  "
              f"effn {a['median_eff_n']:.1f}")
    fp = s['residual_floor_profile']
    print('\nfloor profile (relative to tau=0):')
    for k in sorted(fp, key=float):
        print(f'  tau {float(k):>4} x v_med   {fp[k]:.4f}')
    print(f"  best tau_mult {s['residual_floor_best_tau_mult']}")
    print(f"  tau=2x penalty {100 * (fp['2.0'] - 1):.1f}% worse than tau=0")


if __name__ == '__main__':
    main()
