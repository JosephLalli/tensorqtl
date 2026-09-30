"""Recover full half-read split scans as compact fixed-unit and gene-lead caches.

Each invocation handles one gene stratum in its own process.  It preserves the
half-read phenotype and split arm used for the sealed causal-unit p-value cache,
but retains all causal/sentinel units and one deterministic nominal lead per
gene instead of writing the complete nominal scan.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEPLOY = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
SETS = {'deep': 'corrected_null_store_20260925', 'low': 'stratum30_100'}
KEY = ['gene', 'variant_id', 'beta_abs', 'rep']
FIXED_COLUMNS = ['stratum', 'gene', 'variant_id', 'beta_abs', 'rep', 'is_null',
                 'slope', 'slope_se', 'pval_nominal']
LEAD_COLUMNS = ['stratum', 'gene', 'beta_abs', 'rep', 'lead_variant', 'lead_p',
                'lead_absstat', 'is_null']
EXPECTED_MAPPER = '5bac2bf029a01272a5f829e9a1318eb6d6a1629fdee08eacaf8efa90af8990b6'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_arrays(dataset):
    h = hashlib.sha256()
    with np.load(dataset) as values:
        for key in sorted(values.files):
            h.update(key.encode())
            h.update(np.ascontiguousarray(values[key]).tobytes())
    return h.hexdigest()


def equal_with_nan(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    return np.array_equal(actual, expected, equal_nan=True)


def check_exact_cache(selected, cached, stratum, beta, rep):
    expected = cached.loc[selected.set_index(KEY).index].reset_index()
    expected = expected.sort_values(KEY, kind='stable').reset_index(drop=True)
    actual = selected.sort_values(KEY, kind='stable').reset_index(drop=True)
    for col in ('slope', 'slope_se', 'pval_nominal'):
        if not equal_with_nan(actual[col].to_numpy(float), expected[col].to_numpy(float)):
            raise AssertionError(f'{stratum} beta={beta:g} rep={rep}: {col} does not exactly match prior cache')


def leads(fitted, fixed):
    scan = fitted[['gene', 'variant_id', 'pval_nominal', 'slope', 'slope_se']].copy()
    scan['pval_nominal'] = pd.to_numeric(scan['pval_nominal'], errors='coerce')
    valid = np.isfinite(scan.pval_nominal) & scan.pval_nominal.between(0, 1)
    scan = scan.loc[valid].copy()
    scan['absstat'] = np.abs(scan.slope / scan.slope_se)
    scan['absstat'] = scan['absstat'].where(np.isfinite(scan.absstat), -np.inf)
    chosen = (scan.sort_values(['gene', 'pval_nominal', 'absstat', 'variant_id'],
                               ascending=[True, True, False, True], kind='stable')
                  .drop_duplicates('gene', keep='first')
                  .rename(columns={'variant_id': 'lead_variant', 'pval_nominal': 'lead_p',
                                   'absstat': 'lead_absstat'})
                  [['gene', 'lead_variant', 'lead_p', 'lead_absstat']])
    result = fixed[['gene', 'is_null']].merge(chosen, on='gene', how='left', validate='one_to_one')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stratum', choices=SETS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fixed_file = args.output / f'half_read_fixed_{args.stratum}.parquet'
    lead_file = args.output / f'half_read_leads_{args.stratum}.parquet'
    manifest_file = args.output / f'manifest_{args.stratum}.json'
    source_dir = args.output / f'source_{args.stratum}'
    scratch = args.output / f'scratch_{args.stratum}'
    targets = (fixed_file, lead_file, manifest_file, source_dir, scratch)
    if any(path.exists() for path in targets):
        raise SystemExit(f'refusing overwrite for {args.stratum} under {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)

    os.environ['PLASMODE_GENE_SET'] = SETS[args.stratum]
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'plasmode'))
    from half_read_trial import half_read
    import common as C
    import torch

    if C.GENE_SET != SETS[args.stratum]:
        raise AssertionError('stratum environment did not select requested set')
    mapper_hash = digest(Path(C.map_nominal.__code__.co_filename))
    if mapper_hash != EXPECTED_MAPPER:
        raise AssertionError(f'mapper hash {mapper_hash} differs from required {EXPECTED_MAPPER}')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    previous_file = (DEPLOY / 'half_read_pvalue_comparison_20260929' /
                     f'half_read_pvalues_{args.stratum}.parquet')
    previous = pd.read_parquet(previous_file)
    if len(previous) != 550 or previous.duplicated(KEY).any():
        raise AssertionError(f'{previous_file}: expected 550 unique rows, found {len(previous)}')
    previous = previous.set_index(KEY).sort_index()

    meta_path = C.DATASETS / 'meta.json'
    meta = json.loads(meta_path.read_text())
    I, _, _ = C.load()
    S = C.setup(I)
    fixed_rows, lead_rows, runs, inputs = [], [], [], []
    exact_rows = 0
    for beta in (0.0, 0.2, 0.4, 0.8):
        for rep in range(meta['n_datasets'][str(beta)]):
            dataset_path = C.DATASETS / f'beta{beta}' / f'rep{rep:03d}.npz'
            ds = C.load_dataset(C.DATASETS, f'beta{beta}', rep)
            stored_path = C.RESULTS / f'beta{beta}/split/nominal_rep{rep:03d}.parquet'
            actual_fingerprint = C.fingerprint(ds, 'split')
            baseline_fingerprint = C.stored_fingerprint(stored_path)
            if actual_fingerprint != baseline_fingerprint:
                raise AssertionError(f'{stored_path}: comparator fingerprint mismatch')
            fitted, _ = C.run_nominal(S, ds | {'T': half_read(ds['pT'], ds['eff_lib'])}, 'split', scratch)
            fitted = fitted.rename(columns={'phenotype_id': 'gene'})
            fixed = pd.DataFrame({'gene': meta['genes'],
                                  'variant_id': ds['causal_variant'].astype(str),
                                  'is_null': ds['is_null']})
            fixed = fixed.assign(beta_abs=beta, rep=rep)
            fixed = fixed.merge(fitted[['gene', 'variant_id', 'slope', 'slope_se', 'pval_nominal']],
                                on=['gene', 'variant_id'], how='left', validate='one_to_one', indicator=True)
            if not (fixed.pop('_merge') == 'both').all():
                raise AssertionError(f'{args.stratum} beta={beta:g} rep={rep}: scan omitted fixed unit')
            check = fixed if beta == 0 else fixed.loc[~fixed.is_null]
            check_exact_cache(check, previous, args.stratum, beta, rep)
            exact_rows += len(check)
            fixed_rows.append(fixed)
            lead_rows.append(leads(fitted, fixed).assign(beta_abs=beta, rep=rep))
            finite = fitted.pval_nominal[np.isfinite(fitted.pval_nominal)]
            if not finite.between(0, 1).all():
                raise AssertionError(f'{args.stratum} beta={beta:g} rep={rep}: invalid finite nominal p-value')
            inputs.append({'path': str(dataset_path), 'file_sha256': digest(dataset_path),
                           'array_sha256': digest_arrays(dataset_path),
                           'split_input_fingerprint': actual_fingerprint,
                           'baseline_split_input_fingerprint': baseline_fingerprint})
            runs.append({'beta_abs': beta, 'rep': rep, 'scan_pairs': len(fitted), 'fixed_rows': len(fixed),
                         'lead_rows': len(lead_rows[-1]), 'nonfinite_pval_nominal': int((~np.isfinite(fitted.pval_nominal)).sum())})
            print(f'{args.stratum} beta={beta:g} rep={rep}: {len(fitted):,} pairs; 100 fixed; 100 leads; exact cache match', flush=True)

    fixed = pd.concat(fixed_rows, ignore_index=True).assign(stratum=args.stratum)[FIXED_COLUMNS]
    lead = pd.concat(lead_rows, ignore_index=True).assign(stratum=args.stratum)[LEAD_COLUMNS]
    expected_truth = sum(100 * meta['n_datasets'][str(beta)] for beta in (0.0, 0.2, 0.4, 0.8))
    if len(fixed) != expected_truth or len(lead) != expected_truth:
        raise AssertionError(f'expected {expected_truth} fixed and lead rows, got {len(fixed)} and {len(lead)}')
    dataset_key = ['stratum', 'gene', 'beta_abs', 'rep']
    if fixed.duplicated(dataset_key).any() or lead.duplicated(dataset_key).any():
        raise AssertionError('fixed or lead cache has duplicate gene/dataset rows')
    if len(fixed.merge(lead[dataset_key], on=dataset_key, how='inner', validate='one_to_one')) != expected_truth:
        raise AssertionError('fixed and lead cache do not join uniquely')
    finite = fixed.pval_nominal[np.isfinite(fixed.pval_nominal)]
    if not finite.between(0, 1).all():
        raise AssertionError('fixed cache has invalid finite nominal p-value')
    fixed.to_parquet(fixed_file, index=False)
    lead.to_parquet(lead_file, index=False)
    source_dir.mkdir()
    source_paths = [Path(__file__), Path(half_read.__code__.co_filename), Path(C.__file__),
                    Path(C.map_nominal.__code__.co_filename)]
    source_hashes = {}
    for source in source_paths:
        target = source_dir / source.name
        shutil.copy2(source, target)
        source_hashes[source.name] = digest(source)
    manifest = {'stratum': args.stratum, 'gene_set': C.GENE_SET,
                'scope': 'unchanged half-read split fit; all fixed units and one nominal lead per gene',
                'outputs': {'fixed': fixed_file.name, 'leads': lead_file.name},
                'previous_causal_cache': str(previous_file), 'previous_causal_cache_sha256': digest(previous_file),
                'source_sha256': digest(Path(__file__)), 'source_snapshots_sha256': source_hashes,
                'mapper_sha256': mapper_hash, 'core_sha256': digest(Path(C.__file__)),
                'meta_sha256': digest(meta_path), 'input_fingerprints': inputs, 'runs': runs,
                'exact_cache_matches': {'rows': exact_rows, 'expected_rows': 550, 'columns': ['slope', 'slope_se', 'pval_nominal']},
                'nan_p_counts': {'fixed': int((~np.isfinite(fixed.pval_nominal)).sum()),
                                 'leads': int((~np.isfinite(lead.lead_p)).sum())},
                'truth_counts': {'rows': expected_truth, 'null': int(fixed.is_null.sum()),
                                 'nonnull': int((~fixed.is_null).sum())},
                'torch_version': torch.__version__, 'gpu': torch.cuda.get_device_name()}
    manifest_file.write_text(json.dumps(manifest, indent=2) + '\n')
    shutil.rmtree(scratch)


if __name__ == '__main__':
    main()
