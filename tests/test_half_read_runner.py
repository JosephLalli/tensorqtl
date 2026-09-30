"""Runner-level provenance for the adopted half-read default."""

import importlib.util
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
