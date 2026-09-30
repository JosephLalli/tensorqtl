"""Arithmetic audit of sealed half-read causal-comparison MSE and reported SE."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from half_read_io import RESULTS, atomic_path

SOURCE = RESULTS / 'half_read_trial_20260929'
SETS = ('corrected_null_store_20260925', 'stratum30_100')
SCENARIOS = ('beta0.4', 'beta0.8')
OUTS = ('mse_audit.tsv', 'mse_examples.tsv', 'mse_audit.json')


def mean_or_none(x):
    return float(np.mean(x)) if len(x) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    occupied = [args.output / name for name in OUTS if (args.output / name).exists()]
    if occupied:
        raise SystemExit(f'refusing to overwrite: {occupied}')

    data = pd.read_parquet(SOURCE / 'causal_comparison.parquet')
    source = json.loads((SOURCE / 'causal_summary.json').read_text())
    rows, examples, checks = [], [], []
    for dataset_set in SETS:
        for scenario in SCENARIOS:
            q = data[(data.dataset_set == dataset_set) & (data.scenario == scenario) &
                     (data.arm == 'split')].copy()
            if len(q) != 150 or q[['observed_combined', 'voom_combined', 'allelic_truth',
                                   'observed_combined_se', 'voom_combined_se']].isna().any().any():
                raise AssertionError(f'{dataset_set} {scenario}: expected 150 finite split units')
            truth = q.allelic_truth.to_numpy(float)
            old = q.observed_combined.to_numpy(float)
            half = q.voom_combined.to_numpy(float)
            old_se = q.observed_combined_se.to_numpy(float)
            half_se = q.voom_combined_se.to_numpy(float)
            old_error, half_error = old - truth, half - truth
            old_square, half_square = old_error ** 2, half_error ** 2
            source_mse = source['sets'][dataset_set]['split'][scenario]['combined']['mse']
            source_se = source['sets'][dataset_set]['split'][scenario]['combined']['raw_se']
            mse_old, mse_half = float(old_square.mean()), float(half_square.mean())
            row = dict(
                dataset_set=dataset_set, scenario=scenario, n_units=int(len(q)),
                mean_signed_error_old=mean_or_none(old_error),
                mean_signed_error_half_read=mean_or_none(half_error),
                mean_direction_aligned_error_old=mean_or_none(np.sign(truth) * old_error),
                mean_direction_aligned_error_half_read=mean_or_none(np.sign(truth) * half_error),
                mean_se_old=mean_or_none(old_se), mean_se_half_read=mean_or_none(half_se),
                sqrt_mean_se_squared_old=float(np.sqrt(np.mean(old_se ** 2))),
                sqrt_mean_se_squared_half_read=float(np.sqrt(np.mean(half_se ** 2))),
                mse_old=mse_old, mse_half_read=mse_half,
                rmse_old=float(np.sqrt(mse_old)), rmse_half_read=float(np.sqrt(mse_half)),
                mse_ratio_half_over_old=float(mse_half / mse_old),
                mean_se_ratio_half_over_old=float(half_se.mean() / old_se.mean()),
                source_mse_old=float(source_mse['observed']),
                source_mse_half_read=float(source_mse['half_read']),
                source_mse_ratio_half_over_old=float(source_mse['half_over_observed']),
                source_mean_se_old=float(source_se['observed']),
                source_mean_se_half_read=float(source_se['half_read']),
                source_mse_old_difference=float(mse_old - source_mse['observed']),
                source_mse_half_read_difference=float(mse_half - source_mse['half_read']),
            )
            if abs(row['source_mse_old_difference']) > 1e-15 or abs(row['source_mse_half_read_difference']) > 1e-15:
                raise AssertionError(f'{dataset_set} {scenario}: source MSE does not match')
            rows.append(row)
            checks.append(dict(dataset_set=dataset_set, scenario=scenario,
                               exact_source_mse=True, n_units=int(len(q))))
            if scenario == 'beta0.4':
                e = q.sort_values(['rep', 'gene'], kind='stable').head(3).copy()
                e['truth'] = e.allelic_truth
                e['beta_old'] = e.observed_combined
                e['beta_half'] = e.voom_combined
                e['error_old'] = e.beta_old - e.truth
                e['error_half'] = e.beta_half - e.truth
                e['square_old'] = e.error_old ** 2
                e['square_half'] = e.error_half ** 2
                examples.append(e[['dataset_set', 'gene', 'rep', 'truth', 'beta_old', 'beta_half',
                                   'error_old', 'error_half', 'square_old', 'square_half']])

    audit = pd.DataFrame(rows)
    with atomic_path(args.output / 'mse_audit.tsv') as temporary:
        audit.to_csv(temporary, sep='\t', index=False, float_format='%.17g')
    with atomic_path(args.output / 'mse_examples.tsv') as temporary:
        pd.concat(examples, ignore_index=True).to_csv(temporary, sep='\t', index=False,
                                                    float_format='%.17g')
    receipt = {
        'source': str(SOURCE), 'arm': 'split', 'scope': 'sealed prior 150-unit non-null support only',
        'error': 'combined slope minus signed allelic_truth',
        'reported_se': 'arithmetic mean of reported combined slope SE',
        'sqrt_mean_se_squared': 'separate root-mean-square SE; it is not the arithmetic mean SE',
        'source_mse_verification': checks, 'rows': rows,
    }
    with atomic_path(args.output / 'mse_audit.json') as temporary:
        temporary.write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'wrote {args.output / OUTS[0]}')
    print(f'wrote {args.output / OUTS[1]}')
    print(f'wrote {args.output / OUTS[2]}')


if __name__ == '__main__':
    main()
