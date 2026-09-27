"""Select the plasmode benchmark's 30-100-read gene set (make_datasets.GENE_SETS['stratum30_100']).

WHY. Transcriptome-wide the 30-100-read stratum was the worst for nominal-p calibration
(0.075 at 0.05 on the pre-correction pipeline, scripts/coupling_reach.py), and the
committed 100-gene set (corrected_null_store_20260925) is 67% >= 700 reads, so the
benchmark is extended to a second gene set drawn from that stratum (user request
2026-09-27).

POOL. Cache genes (CACHE/genes.txt) that pass the eQTL gene filter the expression PCs
were built on (point_estimates/edger/calibration_genes.txt), the set
compare_mixqtl_replication.load_point_estimate_inputs accepts. Per gene, over the
cohort's donors, a donor is ADMITTED to the allelic channel when pL + pR > 0 and not
exactly one side is below EXPRESSIBLE_MIN reads (make_datasets.allelic_kept without its
Va > EPS term, which the Gibbs draws decide; the two counts are compared below). The
STRATUM statistic is the median of pL + pR over the admitted donors; a gene is in the
stratum when it lies in [STRATUM_LO, STRATUM_HI) and the gene has at least
MIN_ALLELIC_DONORS admitted donors (hapmixqtl's floor for the allelic channel to enter
the combined statistic). Candidates must also have a unique position in annot/genes.tsv
(corrected_null_store.select_genes' rule).

SELECTION. N_GENES candidates in the order of one permutation from SeedSequence(SEED,
spawn_key=(SELECT_KEY,)). VCF coverage of the cis window is checked as the pipeline
checks it: after loading through load_point_estimate_inputs, every gene must have at
least one tested variant (cis window, outside every selected gene's body, MAF >= 0.05),
else this script stops naming the gene; nothing is replaced silently.

OUTPUT, in the gene set's directory (corrected_null_store's files, so every plasmode
script reads them unchanged): genes.txt; regions.bed (chr, window start, window end,
gene; the window is min(start, pos) - WIN - 1000 to max(end, pos) + WIN + 1000);
gene_selection.tsv (gene, chr, start, end, pos, source); gene_design.tsv (gene,
n_tested_variants, n_allelic_keep, n_allelic_drop, n_zero_haplotype,
median_allele_resolved_reads over ALL donors, the band statistic score.py reads; plus
median_admitted_reads, the stratum statistic, and n_admitted); pool_stratum.tsv (every
pool gene's two medians and admitted count, for the record).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import compare_mixqtl_replication as CM                                   # noqa: E402
import make_datasets as MD                                                # noqa: E402
from tensorqtl.hapmixqtl import MIN_ALLELIC_DONORS, summaries_from_point_estimates   # noqa: E402

SET_NAME = 'stratum30_100'                      # the make_datasets.GENE_SETS entry this script fills
OUT = MD.GENE_SETS[SET_NAME]['gene_dir']
CACHE = Path(CM.CACHE)                          # the Gibbs cache and its point estimates
CAL = Path(CM.PE) / 'edger' / 'calibration_genes.txt'   # the eQTL gene filter load_point_estimate_inputs enforces
GENES_TSV = MD.D / 'annot' / 'genes.tsv'        # gene, chr, start, end, pos (TSS); no header
STRATUM_LO, STRATUM_HI = 30, 100                # reads; the transcriptome-wide stratum of coupling_reach.py (task, 2026-09-27)
N_GENES = 100                                   # as the committed set (task, 2026-09-27)
SEED, SELECT_KEY = 42, 6                        # spawn keys 1-3 are make_datasets', 4-5 run_arms'


def load_pool():
    genes = (CACHE / 'genes.txt').read_text().split()
    cal = set(CAL.read_text().split())
    gp = pd.read_csv(GENES_TSV, sep='\t', header=None, dtype={1: str}, names=['gene', 'chr', 'start', 'end', 'pos'])
    dup = set(gp.gene[gp.gene.duplicated(keep=False)])
    gp = gp[~gp.gene.duplicated(keep=False)].set_index('gene')
    pool = [g for g in genes if g in cal]
    gi = {g: i for i, g in enumerate(genes)}
    rows = np.array([gi[g] for g in pool])
    pL = np.asarray(np.load(Path(CM.PE) / 'pL.npy', mmap_mode='r')[rows])
    pR = np.asarray(np.load(Path(CM.PE) / 'pR.npy', mmap_mode='r')[rows])
    hap = pL + pR
    adm = (hap > 0) & ~((pL < MD.EXPRESSIBLE_MIN) ^ (pR < MD.EXPRESSIBLE_MIN))
    t = pd.DataFrame(dict(
        gene=pool, n_admitted=adm.sum(1), median_allele_resolved_reads=np.median(hap, axis=1),
        median_admitted_reads=[np.median(h[a]) if a.any() else np.nan for h, a in zip(hap, adm)]))
    t['in_stratum'] = (t.median_admitted_reads >= STRATUM_LO) & (t.median_admitted_reads < STRATUM_HI)
    t['enough_donors'] = t.n_admitted >= MIN_ALLELIC_DONORS
    t['unique_position'] = t.gene.isin(gp.index) & ~t.gene.isin(dup)
    print(f'{len(genes):,} cache genes x {pL.shape[1]} donors; {len(pool):,} pass the eQTL gene filter ({CAL}); '
          f'{int(t.in_stratum.sum()):,} with median haplotype-informative reads over admitted donors in '
          f'[{STRATUM_LO}, {STRATUM_HI}); {int((t.in_stratum & t.enough_donors).sum()):,} of them with >= '
          f'{MIN_ALLELIC_DONORS} admitted donors; {int((t.in_stratum & t.enough_donors & t.unique_position).sum()):,} '
          f'of those with a unique position in {GENES_TSV}', flush=True)
    return t, gp


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t, gp = load_pool()
    MD.write_atomic(OUT / 'pool_stratum.tsv', lambda fh: t.to_csv(fh, sep='\t', index=False), 'w')
    cand = t[t.in_stratum & t.enough_donors & t.unique_position].gene.to_numpy()
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(SELECT_KEY,)))
    genes = sorted(cand[rng.permutation(len(cand))[:N_GENES]].tolist())
    sel = gp.loc[genes].reset_index()
    sel['source'] = SET_NAME
    MD.write_atomic(OUT / 'gene_selection.tsv', lambda fh: sel.to_csv(fh, sep='\t', index=False), 'w')
    MD.write_atomic(OUT / 'regions.bed', lambda fh: fh.write(''.join(
        f'{r.chr}\t{max(0, min(r.start, r.pos) - CM.WIN - 1000)}\t{max(r.end, r.pos) + CM.WIN + 1000}\t{r.gene}\n'
        for r in sel.itertuples())), 'w')
    MD.write_atomic(OUT / 'genes.txt', lambda fh: fh.write('\n'.join(genes) + '\n'), 'w')
    print(f'{len(genes)} genes selected from {len(cand):,} candidates (SeedSequence({SEED}, spawn_key=({SELECT_KEY},))); '
          f'chromosomes {sel.chr.value_counts().sort_index().to_dict()}', flush=True)

    # the pipeline's own load: VCF coverage of every cis window, and the design as corrected_null_store writes it
    I = CM.load_point_estimate_inputs(gene_list=str(OUT / 'genes.txt'), regions=str(OUT / 'regions.bed'))
    keep = I['keep']
    if list(I['genes']) != genes or sorted(I['order']) != sorted((CACHE / 'samples.txt').read_text().split()):
        raise SystemExit('loader genes or donors differ from the selection or the cache')
    n_tested = pd.Series({g: len(CM.gene_variant_index(I, g)) for g in genes})
    none = sorted(n_tested.index[n_tested == 0])
    if none:
        raise SystemExit(f'{len(none)} genes have no tested variant (cis window not covered by the VCF): {none}')
    _, _, Va, _, _ = summaries_from_point_estimates(I['pL'], I['pR'], I['pT'], I['eff_lib'], I['YL'], I['YR'], I['YT'])
    Va = Va[:, keep]
    pL, pR = I['pL'][:, keep], I['pR'][:, keep]
    zero = (pL < MD.EXPRESSIBLE_MIN) ^ (pR < MD.EXPRESSIBLE_MIN)
    hap = pL + pR
    adm = (hap > 0) & ~zero
    design = pd.DataFrame(dict(
        gene=genes, n_tested_variants=n_tested.loc[genes].values,
        n_allelic_keep=(Va > MD.EPS).sum(1), n_allelic_drop=(np.where(zero, 0.0, Va) > MD.EPS).sum(1),
        n_zero_haplotype=zero.sum(1), median_allele_resolved_reads=np.median(hap, axis=1),
        median_admitted_reads=[np.median(h[a]) for h, a in zip(hap, adm)], n_admitted=adm.sum(1)))
    MD.write_atomic(OUT / 'gene_design.tsv', lambda fh: design.to_csv(fh, sep='\t', index=False), 'w')
    pool = t.set_index('gene').loc[genes]
    same = (np.allclose(pool.median_admitted_reads.values, design.median_admitted_reads.values)
            and np.array_equal(pool.n_admitted.values, design.n_admitted.values))
    if not same:
        raise SystemExit('stratum statistics differ between the cache-order pool and the VCF-order loader')
    mism = design[design.n_admitted != design.n_allelic_drop]
    print(f'tested variants per gene min {n_tested.min():,} / median {int(n_tested.median()):,} / max '
          f'{n_tested.max():,} ({int(n_tested.sum()):,} tested pairs; RASQUAL runs without --force only while '
          f'2 x max <= 30,000, main.c:582); median admitted reads per gene min {design.median_admitted_reads.min():.1f} '
          f'/ median {design.median_admitted_reads.median():.1f} / max {design.median_admitted_reads.max():.1f}; '
          f'median over all donors min {design.median_allele_resolved_reads.min():.1f} / median '
          f'{design.median_allele_resolved_reads.median():.1f} / max {design.median_allele_resolved_reads.max():.1f}; '
          f'admitted donors per gene min {design.n_admitted.min()} / median {int(design.n_admitted.median())} / max '
          f'{design.n_admitted.max()}; zero-haplotype pairs {int(zero.sum()):,} of {zero.size:,}; genes where '
          f'n_admitted != n_allelic_drop (Va <= EPS on an admitted record): {len(mism)}'
          + (f' {mism.gene.tolist()}' if len(mism) else ''), flush=True)
    print(f'wrote genes.txt, regions.bed, gene_selection.tsv, gene_design.tsv, pool_stratum.tsv to {OUT}')


if __name__ == '__main__':
    main()
