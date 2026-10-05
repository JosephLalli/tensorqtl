"""Runner-level provenance for the adopted half-read default."""

import importlib.util
import gzip
from pathlib import Path

import numpy as np


def _runner_module():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'run_hapmixqtl_from_salmon.py'
    spec = importlib.util.spec_from_file_location('hapmix_salmon_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_input_provenance_reports_xor_admission_policy():
    runner = _runner_module()
    pL = np.array([[0.0, 0.4, 0.5], [1.0, 0.0, 1.0]])
    pR = np.array([[0.0, 0.6, 0.5], [0.0, 1.0, 1.0]])

    got = runner.default_input_provenance(pL, pR, count_noise=False)

    assert got['total_transform'] == 'log2((total+0.5)/(effective_library_size+1)*1e6)'
    assert got['total_working_variance'] == 'unit'
    assert got['ase_count_noise'] is False
    assert got['ase_one_sided_threshold'] == 0.5
    assert got['n_ase_one_sided_excluded'] == 3
    assert got['n_ase_donor_gene_pairs'] == 6


def test_load_counts_half_read_uses_only_paired_fractional_gibbs_draws(tmp_path):
    runner = _runner_module()
    salmon = tmp_path / 'sample'
    bootstrap = salmon / 'aux_info' / 'bootstrap'
    bootstrap.mkdir(parents=True)
    names = ['paired_hapA', 'paired_hapB', 'unpaired']
    draws_by_transcript = np.array([[1.25, 2.5], [3.75, 4.5], [9.0, 10.0]])
    with gzip.open(bootstrap / 'names.tsv.gz', 'wt') as fh:
        fh.write('\t'.join(names))
    with gzip.open(bootstrap / 'bootstraps.gz', 'wb') as fh:
        fh.write(draws_by_transcript.T.astype(np.float64).tobytes())
    (salmon / 'aux_info' / 'meta_info.json').write_text('{"num_bootstraps": 2}')
    manifest = tmp_path / 'manifest.tsv'
    manifest.write_text(f'S1\t{salmon}\n')
    tx2gene = tmp_path / 'tx2gene.tsv'
    tx2gene.write_text('paired\tG1\nunpaired\tG1\n')

    genes, samples, yl, yr = runner.load_counts(
        manifest, tx2gene, ('_hapA', '_hapB'), tmp_path, include_total=False)

    assert genes.tolist() == ['G1']
    assert samples == ['S1']
    np.testing.assert_allclose(yl[:, 0, :], [[1.25, 2.5]])
    np.testing.assert_allclose(yr[:, 0, :], [[3.75, 4.5]])

    genes, samples, yl, yr, yt = runner.load_counts(
        manifest, tx2gene, ('_hapA', '_hapB'), tmp_path)

    assert genes.tolist() == ['G1']
    assert samples == ['S1']
    np.testing.assert_allclose(yl[:, 0, :], [[1.25, 2.5]])
    np.testing.assert_allclose(yr[:, 0, :], [[3.75, 4.5]])
    np.testing.assert_allclose(yt[:, 0, :], [[14.0, 17.0]])
