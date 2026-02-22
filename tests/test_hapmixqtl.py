import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


_MODULE_PATH = Path(__file__).parent.parent / "tensorqtl" / "hapmixqtl.py"
_SPEC = importlib.util.spec_from_file_location("hapmixqtl_test_module", _MODULE_PATH)
hapmixqtl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hapmixqtl)


def test_ase_off_falls_back_to_total():
    torch.manual_seed(1)
    n = 120
    g = torch.randint(0, 3, (1, n), dtype=torch.float32)
    y = 0.4 * (g[0] / 2) + 0.2 * torch.randn(n)
    w = torch.ones(n)

    beta_t, se_t, _, _, _ = hapmixqtl._weighted_channel_ols_vectorized(g / 2, y, w)
    beta_comb, se_comb, _ = hapmixqtl._combine_channels(np.nan, np.nan, float(beta_t[0]), float(se_t[0]))
    assert np.isclose(beta_comb, float(beta_t[0]), atol=1e-6)
    assert np.isclose(se_comb, float(se_t[0]), atol=1e-6)


def test_recover_allelic_effect_and_se_shrinks():
    torch.manual_seed(2)
    beta_true = 0.6

    def fit_once(n):
        s = torch.randint(0, 2, (n,), dtype=torch.float32) * 2 - 1
        y = beta_true * s + 0.5 * torch.randn(n)
        w = torch.ones(n)
        b, bse, _, _, _ = hapmixqtl._weighted_channel_ols_single(s, y, w)
        return b, bse

    b_small, se_small = fit_once(80)
    b_large, se_large = fit_once(400)

    assert abs(b_small - beta_true) < 0.2
    assert abs(b_large - beta_true) < 0.1
    assert se_large < se_small


def test_tau_inflation_increases_uncertainty():
    torch.manual_seed(2)
    n = 120
    g = torch.randint(0, 3, (1, n), dtype=torch.float32) / 2
    idx = torch.randperm(n)[: n // 3]
    mask = torch.zeros(n, dtype=torch.bool)
    mask[idx] = True
    noise = 1.5 * torch.randn(n)
    noise[mask] = 0.01 * torch.randn(mask.sum())
    y = 0.8 * g[0] + noise
    v = torch.full((n,), 0.02)

    w_base = 1.0 / v
    _, se_base, _, _, _ = hapmixqtl._weighted_channel_ols_vectorized(g, y, w_base)

    tau = torch.ones(n)
    tau[mask] = 20.0
    w_tau = 1.0 / torch.clamp(v * tau, min=1e-12)
    _, se_tau, _, _, _ = hapmixqtl._weighted_channel_ols_vectorized(g, y, w_tau)

    assert float(w_tau.sum()) < float(w_base.sum())
    assert float(se_tau[0]) >= float(se_base[0])


def test_reader_enforces_sample_alignment(tmp_path):
    def write_bed(path, sample_cols):
        df = pd.DataFrame({
            '#chr': ['chr1'],
            'start': [0],
            'end': [1],
            'pid': ['f1'],
            sample_cols[0]: [1.0],
            sample_cols[1]: [2.0],
        })
        df.to_csv(path, sep='\t', index=False)

    a = tmp_path / 'A.bed'
    t = tmp_path / 'T.bed'
    va = tmp_path / 'Va.bed'
    vt = tmp_path / 'Vt.bed'
    tl = tmp_path / 'tauL.bed'
    tr = tmp_path / 'tauR.bed'
    write_bed(a, ['S1', 'S2'])
    write_bed(t, ['S1', 'S2'])
    write_bed(va, ['S1', 'S2'])
    write_bed(vt, ['S1', 'S2'])
    write_bed(tl, ['S1', 'S2'])
    write_bed(tr, ['S2', 'S1'])

    with pytest.raises(ValueError, match='Sample order mismatch'):
        hapmixqtl.read_hapmix_inputs(str(a), str(t), str(va), str(vt), str(tl), str(tr))
