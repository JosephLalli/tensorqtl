"""Recover nominal p-values for the sealed half-read causal-unit cache.

Run once per stratum in separate processes.  This repeats only the unchanged
half-read split fit and writes its prespecified causal units; it does not run
or alter any baseline method.
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

from half_read_io import DEPLOY, atomic_path, digest
SETS = {
    'deep': 'corrected_null_store_20260925',
    'low': 'stratum30_100',
}
KEY = ['gene', 'variant_id', 'beta_abs', 'rep']
OUT_COLUMNS = KEY + ['pval_nominal', 'dof_nominal', 'slope', 'slope_se',
                     'pval_a', 'pval_t', 'dof_a', 'dof_t', 'allelic_admitted']


def digest_arrays(dataset):
    h = hashlib.sha256()
    with np.load(dataset) as values:
        for key in sorted(values.files):
            h.update(key.encode())
            h.update(np.ascontiguousarray(values[key]).tobytes())
    return h.hexdigest()


def selected_units(ds, genes, beta, rep):
    units = pd.DataFrame({'gene': genes,
                          'variant_id': ds['causal_variant'].astype(str),
                          'is_null': ds['is_null']})
    if beta != 0:
        units = units.loc[~units.is_null].copy()
    return units.assign(beta_abs=beta, rep=rep).drop(columns='is_null').reset_index(drop=True)


def max_normalized_difference(actual, expected, scale):
    actual, expected, scale = (np.asarray(x, dtype=float) for x in (actual, expected, scale))
    finite = np.isfinite(actual) & np.isfinite(expected)
    if not np.array_equal(np.isfinite(actual), np.isfinite(expected)):
        raise AssertionError('cached and refit finite patterns differ')
    if not finite.any():
        return 0.0
    return float(np.max(np.abs(actual[finite] - expected[finite]) /
                        np.maximum(np.abs(scale[finite]), 1e-8)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stratum', choices=SETS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output_file = args.output/f'half_read_pvalues_{args.stratum}.parquet'
    manifest_file = args.output/f'manifest_{args.stratum}.json'
    source_dir = args.output/f'source_{args.stratum}'
    scratch = args.output/f'scratch_{args.stratum}'
    if any(path.exists() for path in (output_file, manifest_file, source_dir, scratch)):
        raise SystemExit(f'refusing overwrite for {args.stratum} under {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)

    os.environ['PLASMODE_GENE_SET'] = SETS[args.stratum]
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'plasmode'))
    from half_read_trial import half_read
    import common as C
    import torch

    if C.GENE_SET != SETS[args.stratum]:
        raise AssertionError('stratum environment did not select requested set')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    cache_path = DEPLOY/'half_read_se_comparison_20260929/comparison_units.parquet'
    cached = pd.read_parquet(cache_path)
    cached = cached[(cached.stratum == args.stratum) & (cached.method == 'half_read')]
    if len(cached) != 550 or cached.duplicated(KEY).any():
        raise AssertionError(f'{cache_path}: expected 550 unique half-read units, found {len(cached)}')
    cached = cached.set_index(KEY).sort_index()

    meta_path = C.DATASETS/'meta.json'
    meta = json.loads(meta_path.read_text())
    I, _, _ = C.load()
    S = C.setup(I)
    rows, receipts, input_hashes = [], [], []
    max_slope_delta, max_se_delta = 0.0, 0.0
    for beta in (0.0, 0.2, 0.4, 0.8):
        for rep in range(meta['n_datasets'][str(beta)]):
            dataset_path = C.DATASETS/f'beta{beta}/rep{rep:03d}.npz'
            ds = C.load_dataset(C.DATASETS, f'beta{beta}', rep)
            stored_path = C.RESULTS/f'beta{beta}/split/nominal_rep{rep:03d}.parquet'
            if C.stored_fingerprint(stored_path) != C.fingerprint(ds, 'split'):
                raise AssertionError(f'{stored_path}: comparator fingerprint mismatch')
            fitted, _ = C.run_nominal(S, ds | {'T': half_read(ds['pT'], ds['eff_lib'])}, 'split', scratch)
            fitted = fitted.rename(columns={'phenotype_id': 'gene'})
            units = selected_units(ds, meta['genes'], beta, rep)
            selected = units.merge(fitted, on=['gene', 'variant_id'], how='left',
                                   validate='one_to_one', indicator=True)
            if not (selected['_merge'] == 'both').all():
                raise AssertionError(f'beta {beta} rep {rep}: refit omitted selected unit')
            selected = selected.drop(columns='_merge')
            indexed = selected.set_index(KEY).sort_index()
            expected = cached.loc[indexed.index]
            slope_delta = max_normalized_difference(indexed.slope, expected.slope, expected.slope_se)
            se_delta = max_normalized_difference(indexed.slope_se, expected.slope_se, expected.slope_se)
            max_slope_delta, max_se_delta = max(max_slope_delta, slope_delta), max(max_se_delta, se_delta)
            if max(slope_delta, se_delta) > 1e-5:
                raise AssertionError(f'beta {beta} rep {rep}: selected slope/SE differs from cache by '
                                     f'{max(slope_delta, se_delta):.3g} SE units')
            rows.append(selected[OUT_COLUMNS])
            input_hashes.append({'path': str(dataset_path), 'file_sha256': digest(dataset_path),
                                 'array_sha256': digest_arrays(dataset_path)})
            receipts.append({'beta_abs': beta, 'rep': rep, 'scan_pairs': len(fitted),
                             'selected_units': len(selected), 'slope_max_normalized_difference': slope_delta,
                             'slope_se_max_normalized_difference': se_delta})
            print(f'{args.stratum} beta={beta:g} rep={rep}: {len(fitted):,} pairs; '
                  f'{len(selected)} units; max delta={max(slope_delta, se_delta):.3g} SE', flush=True)
    result = pd.concat(rows, ignore_index=True).assign(stratum=args.stratum)
    result = result[['stratum'] + OUT_COLUMNS]
    if len(result) != 550 or result.duplicated(['stratum'] + KEY).any():
        raise AssertionError(f'expected 550 unique output units, found {len(result)}')
    finite_p = result.pval_nominal[np.isfinite(result.pval_nominal)]
    diagnostics = {'rows': len(result), 'finite_pval_nominal': len(finite_p),
                   'nonfinite_pval_nominal': int(result.pval_nominal.isna().sum()),
                   'pval_nominal_zero': int((finite_p == 0).sum()),
                   'pval_nominal_min_positive': (float(finite_p[finite_p > 0].min())
                                                  if (finite_p > 0).any() else None)}
    with atomic_path(output_file) as temporary:
        result.to_parquet(temporary, index=False)
    source_dir.mkdir()
    source_paths = [Path(__file__), Path(__file__).with_name('half_read_io.py'),
                    Path(half_read.__code__.co_filename), Path(C.__file__),
                    Path(C.map_nominal.__code__.co_filename)]
    source_hashes = {}
    for source in source_paths:
        target = source_dir/source.name
        with atomic_path(target) as temporary:
            shutil.copy2(source, temporary)
        source_hashes[source.name] = digest(source)
    manifest = {'stratum': args.stratum, 'gene_set': C.GENE_SET,
                'scope': 'unchanged half-read split fit; prespecified causal units only',
                'output': output_file.name, 'cache': str(cache_path),
                'cache_sha256': digest(cache_path), 'source_sha256': digest(Path(__file__)),
                'source_snapshots_sha256': source_hashes,
                'mapper_sha256': digest(Path(C.map_nominal.__code__.co_filename)),
                'core_sha256': digest(Path(C.__file__)), 'meta_sha256': digest(meta_path),
                'dataset_hashes': input_hashes, 'runs': receipts,
                'max_slope_normalized_difference': max_slope_delta,
                'max_slope_se_normalized_difference': max_se_delta,
                'diagnostics': diagnostics, 'torch_version': torch.__version__,
                'gpu': torch.cuda.get_device_name()}
    with atomic_path(manifest_file) as temporary:
        temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    shutil.rmtree(scratch)


if __name__ == '__main__':
    main()
