"""Salmon point estimates and edgeR effective library sizes, beside the Gibbs cache.

User rules, 2026-09-25: every value in the pipeline comes from Salmon's POINT
estimates (quant.sf NumReads); the Gibbs draws give only measurement variance;
the unit is log2(CPM + 1) with CPM from edgeR's effective library size
(lib.size x TMM norm.factors); the expression-PC gene filter is the eQTL gene
filter. This builds the inputs those rules need, once, aligned to the Gibbs
cache so a value and its variance index the same (gene, donor):

  point_estimates/pL.npy, pR.npy, pT.npy   [cache genes x cache samples]
      haplotype counts over PAIRED transcripts and the total over ALL
      transcripts, summed exactly as the Gibbs ingest (load_counts) sums draws
  point_estimates/totals_all.tsv.gz         every tx2gene gene x sample: the
      matrix edgeR normalizes
  point_estimates/restrict_calibration.txt  the calibration-phase gene
      restriction: protein-coding (at least one curated RefSeq NM_ transcript)
      and autosomal (chr1-22). User decision 2026-09-25: this restriction is
      for testing and calibrating the model only; the deployment filter may
      be very different.
  point_estimates/edger/                     edger_library_normalization.R:
      edger_samples.tsv (lib.size, TMM factor, effective library size per
      sample), calibration_genes.txt (filterByExpr AND the restriction: the
      eQTL gene set and the expression-PC gene set), and the filterByExpr
      result over all biotypes

GATES (abort on failure): pT for the cache genes equals the all-gene totals;
pL + pR never exceeds pT; edgeR's sample order equals the cache's; and the
point-estimate totals track the Gibbs posterior means (Pearson of
log(count + 1) above 0.99), which checks that the two readers sum the same
transcripts to the same genes for the same donors.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_hapmixqtl_from_salmon as H          # noqa: E402

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
CACHE = D / 'cache' / 'gibbs_56b63c3b37ed5df8'
OUT = CACHE / 'point_estimates'
MANIFEST, TX2GENE, GENES = D / 'cohort' / 'salmon.tsv', D / 'annot' / 'tx2gene.tsv', D / 'annot' / 'genes.tsv'
SUFFIXES = ('_L', '_R')
AUTOSOMES = {f'chr{i}' for i in range(1, 23)}
RSCRIPT = Path(__file__).resolve().parent / 'edger_library_normalization.R'


def main():
    OUT.mkdir(exist_ok=True)
    genes = open(CACHE / 'genes.txt').read().split()
    samples = open(CACHE / 'samples.txt').read().split()
    print(f'{len(genes)} cache genes, {len(samples)} samples; reading quant.sf', flush=True)
    pL, pR, pT, totals = H.load_point_estimates(MANIFEST, TX2GENE, SUFFIXES, genes, samples)

    # ---- gates on the readers -----------------------------------------------
    sub = totals.reindex(genes).fillna(0.0).to_numpy()
    g1 = float(np.max(np.abs(sub - pT)))
    if g1 > 1e-6:
        raise SystemExit(f'GATE 1 FAILED: pT differs from the all-gene totals by {g1}')
    g2 = float(np.max(pL + pR - pT))
    if g2 > 1e-4:
        raise SystemExit(f'GATE 2 FAILED: pL + pR exceeds pT by {g2}')
    YT = np.load(CACHE / 'YT.npy', mmap_mode='r')
    mT = np.empty(pT.shape)
    for s0 in range(0, len(genes), 2000):
        mT[s0:s0 + 2000] = np.asarray(YT[s0:s0 + 2000]).mean(2)
    r = float(np.corrcoef(np.log1p(pT).ravel(), np.log1p(mT).ravel())[0, 1])
    if not r > 0.99:
        raise SystemExit(f'GATE 3 FAILED: point totals vs Gibbs posterior means r = {r}')
    rel = np.abs(pT - mT) / np.maximum(mT, 1.0)
    YL = np.load(CACHE / 'YL.npy', mmap_mode='r'); YR = np.load(CACHE / 'YR.npy', mmap_mode='r')
    inf_point = (pL + pR) > 0
    inf_gibbs = np.zeros_like(inf_point)
    for s0 in range(0, len(genes), 2000):
        inf_gibbs[s0:s0 + 2000] = (np.asarray(YL[s0:s0 + 2000]).sum(2) + np.asarray(YR[s0:s0 + 2000]).sum(2)) > 0
    np.save(OUT / 'pL.npy', pL); np.save(OUT / 'pR.npy', pR); np.save(OUT / 'pT.npy', pT)
    totals.to_csv(OUT / 'totals_all.tsv.gz', sep='\t')

    # ---- calibration restriction and edgeR ----------------------------------
    tx = pd.read_csv(TX2GENE, sep='\t', header=None, names=['tx', 'gene'])
    pc = set(tx.loc[tx.tx.str.startswith('NM_'), 'gene'])
    gp = pd.read_csv(GENES, sep='\t', header=None, dtype={1: str}, names=['gene', 'chr', 'start', 'end', 'pos'])
    gp = gp[~gp.gene.duplicated(keep=False)]
    auto = set(gp.loc[gp.chr.isin(AUTOSOMES), 'gene'])
    restrict = sorted(set(totals.index) & pc & auto)
    (OUT / 'restrict_calibration.txt').write_text('\n'.join(restrict) + '\n')
    res = subprocess.run(['Rscript', str(RSCRIPT), str(OUT / 'totals_all.tsv.gz'),
                          str(OUT / 'restrict_calibration.txt'), str(OUT / 'edger')],
                         capture_output=True, text=True)
    print(res.stdout.strip(), res.stderr.strip()[-500:] if res.returncode else '', flush=True)
    if res.returncode:
        raise SystemExit('edgeR normalization failed')
    es = pd.read_csv(OUT / 'edger' / 'edger_samples.tsv', sep='\t')
    if es['sample'].astype(str).tolist() != samples:
        raise SystemExit('GATE 4 FAILED: edgeR sample order differs from the cache')
    cal = open(OUT / 'edger' / 'calibration_genes.txt').read().split()

    summary = dict(
        n_cache_genes=len(genes), n_samples=len(samples), n_all_genes=int(totals.shape[0]),
        gate_pT_vs_totals_max_abs=g1, gate_pLpR_minus_pT_max=g2,
        point_vs_gibbs_mean_pearson_log1p=r,
        point_vs_gibbs_mean_rel_diff_median=float(np.median(rel)),
        point_vs_gibbs_mean_rel_diff_q99=float(np.quantile(rel, 0.99)),
        informative_agreement=dict(both=int((inf_point & inf_gibbs).sum()),
                                   point_only=int((inf_point & ~inf_gibbs).sum()),
                                   gibbs_only=int((~inf_point & inf_gibbs).sum())),
        n_restrict_calibration=len(restrict), n_calibration_genes=len(cal),
        n_calibration_genes_in_cache=len(set(cal) & set(genes)),
        lib_size_raw_median=float(totals.sum(0).median()),
        edger_lib_size_median=float(es.lib_size.median()),
        tmm_norm_factor_range=[float(es.norm_factor.min()), float(es.norm_factor.max())],
        eff_lib_size_median=float(es.eff_lib_size.median()),
        suffixes=list(SUFFIXES),
        rule='values from point estimates; Gibbs draws only for variance; log2(CPM+1) with edgeR effective library size')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
