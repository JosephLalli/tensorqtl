"""Salmon's equivalence classes predict the allelic Gibbs variance: the premise
of the plasmode generator's allelic variance rule (make_datasets.py).

Salmon 1.10.3's Gibbs round (src/CollapsedGibbsSampler.cpp) draws each
transcript's rate from Gamma(count + prior, 1/(0.1 + effLen)) (line 149) and
reassigns the reads of every multi-transcript equivalence class (the set of
reads compatible with exactly the same transcripts) by a multinomial on those
rates (lines 257-265). For a gene's L/R haplotype pair, a read in a class
holding both haplotypes of the gene says nothing about the haplotype
fraction. By the delta method, the across-draw variance of
log2((YL + 0.5)/(YR + 0.5)) beyond Gamma shot noise is then

    excess ~ s^2 (1/u_L + 1/u_R) / ln(2)^2  =  s^2 H

with u_h the reads in single-gene classes holding only haplotype h of the
gene (among its paired transcripts), a the reads in classes holding both, and
s = a / (a + u_L + u_R) the ambiguous share. Observed excess = across-draw
variance (ddof=0) of the log2 ratio minus the Gamma shot-noise (counting)
term (1/(mean YL + 0.5) + 1/(mean YR + 0.5)) / ln(2)^2 at the draw means.

One donor, SAMPLE, whose Salmon directory is taken from the cohort manifest
the Gibbs cache was built from (keyed on the DNA library id, never on a name),
with its dumped equivalence classes (--dumpEq, no weights). Genes: cache genes
with u_L and u_R >= MIN_U in that donor.

Spearman = the Pearson correlation of the ranks. The counting-term comparator
below omits the 1/ln(2)^2 factor; a rank correlation is unchanged by it.

PASS if, over genes with positive observed excess and s > 0, the median of
excess / (s^2 H) is in RATIO_BAND, and Spearman(excess, s^2 H) exceeds
Spearman(excess, counting term). THE THRESHOLDS WERE SET AFTER THE FIRST
RESULT WAS SEEN (2026-09-26: median 0.989, IQR 0.795-1.169, over 3,589 of
3,595 genes, 4 dropped for non-positive excess and 2 for s = 0; Spearman
0.877 against 0.491; median haplotype-informative share 0.117), so this is a
regression guard of the derivation, not an independent test of it. The ratio
trends with s (median 1.99 at s < 0.5, 0.82 at s >= 0.95), so s^2 is not the
exact form; the generator's rule does not depend on it, because thinning
leaves s unchanged and scales only u.

Output: CHECKS/salmon_premise.json (atomic). Exits non-zero on FAIL.
"""
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_datasets as MD                    # noqa: E402
from tensorqtl.hapmixqtl import LN2           # noqa: E402

CACHE = MD.D / 'cache' / 'gibbs_56b63c3b37ed5df8'     # Gibbs draws, the pipeline's input
MANIFEST = MD.D / 'cohort' / 'salmon.tsv'              # DNA library id -> Salmon directory; the cache's input
TX2GENE = MD.D / 'annot' / 'tx2gene.tsv'               # the cache's transcript -> gene map
CHECKS = MD.ROOT / 'checks'
SAMPLE = '100_D1'                    # first manifest row; the donor whose classes were dumped
SALMON_VERSION = '1.10.3'            # the version whose CollapsedGibbsSampler.cpp is cited above
K = MD.KAPPA                         # pseudocount of the allelic log ratio (summaries_from_point_estimates)
MIN_U = 20                           # informative reads per haplotype for a stable 1/u (exploratory run)
RATIO_BAND = (0.8, 1.25)             # set after the first result (median 0.989), see docstring
SUFFIXES = ('_L', '_R')              # g2gtools haplotype suffixes


def spearman(x, y):
    return float(pd.Series(x).corr(pd.Series(y), method='spearman'))


def read_classes(qdir):
    meta = json.loads((qdir / 'aux_info' / 'meta_info.json').read_text())
    for k, want in (('salmon_version', SALMON_VERSION), ('samp_type', 'gibbs')):
        if meta[k] != want:
            raise SystemExit(f'{qdir}/aux_info/meta_info.json: {k} = {meta[k]!r}, expected {want!r}')
    f = qdir / 'aux_info' / 'eq_classes.txt.gz'
    with gzip.open(f, 'rt') as fh:
        n_t, n_e = int(fh.readline()), int(fh.readline())
        names = [fh.readline().rstrip('\n') for _ in range(n_t)]
        classes = [fh.readline().split() for _ in range(n_e)]
    bad = sum(len(c) != int(c[0]) + 2 for c in classes)
    reads = sum(int(c[-1]) for c in classes)
    print(f'{f}: {n_t:,} targets, {n_e:,} equivalence classes, {reads:,} reads; meta_info: '
          f'{meta["num_valid_targets"]:,} targets, {meta["num_mapped"]:,} mapped, '
          f'{meta["num_bootstraps"]} Gibbs draws', flush=True)
    if bad:
        raise SystemExit(f'{f}: {bad} class lines do not have k + 2 fields (weights dumped?)')
    if n_t != meta['num_valid_targets'] or reads != meta['num_mapped']:
        raise SystemExit(f'{f}: targets {n_t} / reads {reads} differ from meta_info '
                         f'{meta["num_valid_targets"]} / {meta["num_mapped"]}')
    return names, classes


def class_counts(names, classes, tx2gene):
    """Per gene: reads in single-gene classes holding only L, only R, or both, of its paired transcripts."""
    base = np.array([n[:-2] if n.endswith(SUFFIXES) else n for n in names])
    hap = np.array([n[-1] if n.endswith(SUFFIXES) else '' for n in names])
    paired_bases = set(base[hap == 'L']) & set(base[hap == 'R'])
    gene = tx2gene.reindex(base)
    if gene.isna().any():
        raise SystemExit(f'{int(gene.isna().sum())} Salmon targets have no gene in {TX2GENE}, '
                         f'e.g. {base[gene.isna().to_numpy()][:3]}')
    gene = gene.to_numpy()
    paired = np.isin(base, list(paired_bases))
    print(f'{len(paired_bases):,} paired transcripts; every target has a gene', flush=True)
    uL, uR, amb = Counter(), Counter(), Counter()
    multi_gene, unpaired = 0, 0
    for c in classes:
        t = np.array(c[1:-1], dtype=int)
        g = set(gene[t])
        if len(g) != 1:
            multi_gene += int(c[-1])
            continue
        g = g.pop()
        hp = set(hap[t[paired[t]]])
        if hp == {'L'}:
            uL[g] += int(c[-1])
        elif hp == {'R'}:
            uR[g] += int(c[-1])
        elif hp == {'L', 'R'}:
            amb[g] += int(c[-1])
        else:
            unpaired += int(c[-1])
    print(f'reads in multi-gene classes, not counted: {multi_gene:,}; reads in single-gene classes holding '
          f'no paired transcript, not counted: {unpaired:,}', flush=True)
    return uL, uR, amb


def main():
    manifest = pd.read_csv(MANIFEST, sep='\t', header=None, names=['sample', 'dir']).set_index('sample')['dir']
    if SAMPLE not in manifest.index:
        raise SystemExit(f'{SAMPLE} not in {MANIFEST}')
    qdir = Path(manifest[SAMPLE])
    print(f'{SAMPLE}: Salmon directory {qdir} (from {MANIFEST})', flush=True)
    tx2gene = pd.read_csv(TX2GENE, sep='\t', header=None, names=['tx', 'gene']).set_index('tx')['gene']
    names, classes = read_classes(qdir)
    uL, uR, amb = class_counts(names, classes, tx2gene)

    genes = (CACHE / 'genes.txt').read_text().split()
    samples = (CACHE / 'samples.txt').read_text().split()
    j = samples.index(SAMPLE)
    rows = [i for i, g in enumerate(genes) if uL[g] >= MIN_U and uR[g] >= MIN_U]
    print(f'{len(genes):,} cache genes; {len(rows):,} with u_L and u_R >= {MIN_U} in {SAMPLE}', flush=True)
    yl = np.asarray(np.load(CACHE / 'YL.npy', mmap_mode='r')[rows, j])
    yr = np.asarray(np.load(CACHE / 'YR.npy', mmap_mode='r')[rows, j])
    dv = np.log2((yl + K) / (yr + K)).var(axis=1)
    counting = 1 / (yl.mean(1) + K) + 1 / (yr.mean(1) + K)
    excess = dv - counting / LN2 ** 2
    u_l = np.array([uL[genes[i]] for i in rows], float)
    u_r = np.array([uR[genes[i]] for i in rows], float)
    a = np.array([amb[genes[i]] for i in rows], float)
    H = (1 / u_l + 1 / u_r) / LN2 ** 2
    s = a / (a + u_l + u_r)
    pos, amb_pos = excess > 0, s > 0
    keep = pos & amb_pos
    print(f'excess <= 0: {int((~pos).sum())} genes; s = 0 (no ambiguous reads): {int((~amb_pos).sum())} '
          f'genes; retained {int(keep.sum())} of {len(rows)}', flush=True)
    ratio = excess[keep] / (s[keep] ** 2 * H[keep])
    ratio_all = excess[amb_pos] / (s[amb_pos] ** 2 * H[amb_pos])
    res = dict(
        sample=SAMPLE, salmon_dir=str(qdir), min_u=MIN_U, genes_min_u=len(rows),
        genes_excess_le_0=int((~pos).sum()), genes_s_eq_0=int((~amb_pos).sum()), genes_retained=int(keep.sum()),
        ratio_median=float(np.median(ratio)), ratio_iqr=[float(np.quantile(ratio, .25)), float(np.quantile(ratio, .75))],
        ratio_median_incl_nonpositive_excess=float(np.median(ratio_all)),
        ratio_over_H_without_s2_median=float(np.median(excess[keep] / H[keep])),
        spearman_excess_s2H=spearman(excess[keep], (s ** 2 * H)[keep]),
        spearman_excess_H=spearman(excess[keep], H[keep]),
        spearman_excess_counting=spearman(excess[keep], counting[keep]),
        median_informative_share=float(np.median(1 - s)),
        by_ambiguous_share={f'[{lo}, {hi})': dict(genes=int(m.sum()), ratio_median=float(np.median(
            excess[m] / (s[m] ** 2 * H[m])))) for lo, hi in ((0, .5), (.5, .8), (.8, .95), (.95, 1.01))
            for m in [keep & (s >= lo) & (s < hi)] if m.any()},
        ratio_band=list(RATIO_BAND), thresholds_set_after_first_result=True)
    ok = (RATIO_BAND[0] <= res['ratio_median'] <= RATIO_BAND[1]
          and res['spearman_excess_s2H'] > res['spearman_excess_counting'])
    res['passed'] = bool(ok)
    print(f'excess / (s^2 (1/u_L + 1/u_R) / ln2^2): median {res["ratio_median"]:.3f} '
          f'IQR [{res["ratio_iqr"][0]:.3f}, {res["ratio_iqr"][1]:.3f}] over {int(keep.sum())} genes '
          f'(with the {int((amb_pos & ~pos).sum())} non-positive-excess genes included: '
          f'{res["ratio_median_incl_nonpositive_excess"]:.3f}); without s^2: {res["ratio_over_H_without_s2_median"]:.3f}')
    for k, v in res['by_ambiguous_share'].items():
        print(f'  ambiguous share s in {k}: {v["genes"]:5d} genes, median {v["ratio_median"]:.3f}')
    print(f'Spearman(excess, s^2 H) {res["spearman_excess_s2H"]:.3f}; Spearman(excess, H) '
          f'{res["spearman_excess_H"]:.3f}; Spearman(excess, counting term) {res["spearman_excess_counting"]:.3f}')
    print(f'median haplotype-informative share of paired reads (1 - s) {res["median_informative_share"]:.3f}')
    print(f'{"PASS" if ok else "FAIL"} (median in [{RATIO_BAND[0]}, {RATIO_BAND[1]}] and '
          f'Spearman(excess, s^2 H) > Spearman(excess, counting term); thresholds set after the first result)')
    CHECKS.mkdir(parents=True, exist_ok=True)
    out = CHECKS / 'salmon_premise.json'
    MD.write_atomic(out, lambda fh: fh.write(MD.dumps(res)), 'w')
    print(f'wrote {out}')
    if not ok:
        raise SystemExit('FAILED: salmon premise')


if __name__ == '__main__':
    main()
