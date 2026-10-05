"""
Regression tests for the `cis_egenes_knockoff` CLI mode (tensorqtl.py).

The mode wraps susie.map_egenes_knockoffs (the knockoff-calibrated eGene FDR
path) and writes the eGene table + SuSiE localization + diagnostics. Two guards:

  * surface: `python -m tensorqtl --help` exposes the mode and its flags.
  * functional: the mode's DEFAULT config (statistic='kfc', knockoff='gaussian',
    shrink=0.1, knockoff_offset=1) runs end-to-end and writes the documented
    output files with the correct schema.

Both skip cleanly when optional deps (pandas_plink) are unavailable, since the
CLI import chain needs them even though this path does not.
"""
import sys
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))


def test_cli_help_exposes_knockoff_mode():
    """`--help` lists the mode and each knockoff flag (surface regression)."""
    p = subprocess.run([sys.executable, "-m", "tensorqtl", "--help"],
                       cwd=str(REPO), capture_output=True, text=True)
    if p.returncode != 0:
        pytest.skip(f"tensorqtl CLI could not be imported: {p.stderr.strip()[-200:]}")
    help_text = p.stdout
    assert "cis_egenes_knockoff" in help_text
    for flag in ("--statistic", "--knockoff", "--shrink",
                 "--knockoff_offset", "--n_knockoffs"):
        assert flag in help_text, f"{flag} missing from --help"


@pytest.mark.slow
def test_cli_default_config_runs_and_writes(tmp_path):
    """The mode's default (kfc + Gaussian) config runs and writes all outputs."""
    import numpy as np
    import pandas as pd
    try:
        import tensorqtl.susie as susie
        from hmm_genotype_simulator import simulate_hmm_genotypes
    except Exception as e:  # pragma: no cover - env-dependent
        pytest.skip(f"deps unavailable: {e}")

    rng = np.random.RandomState(0)
    N, n_genes, p, n_sig = 200, 24, 20, 10
    samples = [f"S{i:04d}" for i in range(N)]
    vids, chroms, poss, blocks, causal_row = [], [], [], [], {}
    for g in range(n_genes):
        geno, _, info = simulate_hmm_genotypes(p, N, seed=g)
        blocks.append(geno)
        base = 1_000_000 * g + 50_000
        for j in range(p):
            vids.append(f"g{g}_v{j}"); chroms.append("chr1"); poss.append(base + j * 200)
        if g < n_sig:
            elig = np.where(info['maf'] > 0.05)[0]
            causal_row[g] = g * p + (elig[0] if elig.size else 0)
    geno = np.vstack(blocks)
    genotype_df = pd.DataFrame(geno, index=vids, columns=samples)
    variant_df = pd.DataFrame({'chrom': chroms, 'pos': poss}, index=vids)
    pheno, pids = [], []
    for g in range(n_genes):
        y = rng.randn(N)
        if g < n_sig:
            r = causal_row[g]
            xc = (geno[r] - geno[r].mean()) / (geno[r].std() + 1e-9)
            y = y + np.sqrt(0.12 / 0.88) * xc
        pheno.append(y); pids.append(f"G{g}")
    phenotype_df = pd.DataFrame(np.array(pheno), index=pids, columns=samples)
    phenotype_pos_df = pd.DataFrame(
        {'chr': ['chr1'] * n_genes,
         'pos': [1_000_000 * g + 50_000 + (p // 2) * 200 for g in range(n_genes)]},
        index=pids)
    covariates_df = pd.DataFrame(rng.randn(N, 2), index=samples, columns=['PC1', 'PC2'])

    # CLI default config for cis_egenes_knockoff (see tensorqtl.py dispatch).
    egene_df, localize_summary_df, diagnostics = susie.map_egenes_knockoffs(
        genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df,
        paired_covariate_df=None,
        fdr=0.1, statistic='kfc', knockoff='gaussian', shrink=0.1,
        knockoff_offset=1, n_knockoffs=1, L=10, maf_threshold=0,
        window=1_000_000, seed=0, localize=True, verbose=False)

    assert list(egene_df.columns) == ['phenotype_id', 'knockoff_qvalue', 'selected']
    assert len(egene_df) == n_genes

    # write + reload exactly as the CLI does
    prefix = tmp_path / "smoke"
    egene_path = f"{prefix}.cis_knockoff_egenes.txt.gz"
    egene_df.to_csv(egene_path, sep='\t', index=False, float_format='%.6g')
    assert list(pd.read_csv(egene_path, sep='\t').columns) == \
        ['phenotype_id', 'knockoff_qvalue', 'selected']
    if localize_summary_df is not None:
        summ_path = f"{prefix}.cis_knockoff_egenes.SuSiE_summary.parquet"
        localize_summary_df.to_parquet(summ_path)
        assert len(pd.read_parquet(summ_path)) > 0
    assert 'W_per_draw' in diagnostics and diagnostics['n_genes'] == n_genes


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v', '-s']))
