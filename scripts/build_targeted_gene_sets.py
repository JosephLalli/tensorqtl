"""Two targeted gene sets for the weighting null (user request, 2026-09-26).

Neither set was saved by the earlier BrainVar reference-comparison analysis,
so both are rebuilt here from its surviving inputs, with its own definitions
where one exists.

REFERENCE-DEPENDENT EXPRESSION. The metric of
nf_stage/RNA_reference_comparison_results/gene_results_and_qtl_package_experiments.py
lines 933-945: per gene, the Spearman correlation across RNA samples between
T2T and GRCh38 expression (STAR+Salmon tximeta gene CPM, the same inputs as
its lines 51-59), threshold < 0.95, protein-coding (gene_biotype in the T2T
NCBI110 GTF). The notebook additionally restricted to genes overlapping
segmental duplications; that flag is recorded here, not applied. Genes must
reach 0.1 CPM in at least 20% of samples under both references (the
notebook's cpm_cutoff and sample_cutoff). Of the qualifying genes that pass
the calibration gene filter and have Gibbs draws, the 100 with the lowest
Spearman are taken.

HIGH-EFFECT, HIGH-SE HITS. No earlier definition exists. From the surviving
full cis permutation run on T2T
(brainvar_eqtl_native_full_gene_list_eqtl_20260729T094311Z/t2t): genes
significant at Benjamini-Hochberg FDR 0.05 on pval_beta, whose lead variant
has both |slope| and slope_se in the top 10% of significant genes. GeneIDs
map to symbols through the T2T GTF's db_xref. All that pass the calibration
filter with Gibbs draws are taken.

Outputs in brainvar_hapmix_deploy/targeted_gene_sets_20260926/:
reference_dependent.txt, high_beta_high_se.txt, and a metrics TSV for each.
"""
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
OUT = D / 'targeted_gene_sets_20260926'
CACHE = D / 'cache' / 'gibbs_56b63c3b37ed5df8'
BV2 = Path('/mnt/ssd/lalli/nf_stage/RNA_reference_comparison_results/reference_comparison_results/bv2')
T2T_CPM = BV2 / 'T2T_NCBI110_star_alignment_ENSEMBL_args' / 'tximeta' / 'whole_experiment.gene.cpm.tsv'
GRCH38_CPM = BV2 / 'GRCh38_p14_NCBI110_star_alignment' / 'tximeta' / 'whole_experiment.gene.cpm.tsv'
GTF = Path('/mnt/ssd/lalli/nf_stage/genome_refs/T2T-CHM13_v2_ncbi110/'
           'GCF_009914755.1_RS_2024_08-T2T-CHM13v2.0_genomic.UCSC_chr.with_GRCh38_rCDS_chrM.exon_ids.gtf')
SD_GENES = Path('/mnt/ssd/lalli/nf_stage/brainvar2/alignment_eval/sd_genes.txt')
PERM = Path('/mnt/ssd/lalli/nf_stage/brainvar_eqtl_native_full_gene_list_eqtl_20260729T094311Z/t2t/'
            'brainvar2_t2t_native_full_list_permutation.cis_qtl.txt.gz')
N_REF, SPEARMAN_MAX, CPM_MIN, SAMPLE_FRAC, TOP = 100, 0.95, 0.1, 0.2, 0.10


def gtf_genes():
    out = subprocess.run(['grep', '-P', '\tgene\t', str(GTF)], capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        a = line.split('\t')[8]
        gid = re.search(r'gene_id "([^"]+)"', a)
        xref = re.search(r'db_xref "GeneID:(\d+)"', a)
        bt = re.search(r'gene_biotype "([^"]+)"', a)
        if gid:
            rows.append((gid.group(1), xref.group(1) if xref else None, bt.group(1) if bt else None))
    return pd.DataFrame(rows, columns=['gene', 'geneid', 'biotype']).drop_duplicates('gene')


def runnable():
    cal = set((CACHE / 'point_estimates' / 'edger' / 'calibration_genes.txt').read_text().split())
    cache = set((CACHE / 'genes.txt').read_text().split())
    gp = pd.read_csv(D / 'annot' / 'genes.tsv', sep='\t', header=None, dtype={1: str},
                     names=['gene', 'chr', 'start', 'end', 'pos'])
    uniq = set(gp.gene[~gp.gene.duplicated(keep=False)])
    return cal & cache & uniq


def main():
    OUT.mkdir(exist_ok=True)
    gtf = gtf_genes()
    ok_run = runnable()
    sd = set(pd.read_csv(SD_GENES, sep='\t', header=None)[0])
    summ = {}

    # ---- reference-dependent expression ------------------------------------
    t2t = pd.read_csv(T2T_CPM, sep='\t', index_col=0)
    g38 = pd.read_csv(GRCH38_CPM, sep='\t', index_col=0)
    samples = sorted(set(t2t.columns) & set(g38.columns))
    genes = sorted(set(t2t.index) & set(g38.index))
    a, b = t2t.loc[genes, samples], g38.loc[genes, samples]
    expressed = ((a >= CPM_MIN).mean(1) >= SAMPLE_FRAC) & ((b >= CPM_MIN).mean(1) >= SAMPLE_FRAC)
    a, b = a[expressed], b[expressed]
    rho = pd.Series([stats.spearmanr(x, y).statistic for x, y in zip(a.values, b.values)],
                    index=a.index, name='spearman')
    pc = set(gtf.gene[gtf.biotype == 'protein_coding'])
    m = pd.DataFrame(dict(spearman=rho))
    m['protein_coding'] = m.index.isin(pc)
    m['segmental_duplication'] = m.index.isin(sd)
    m['runnable'] = m.index.isin(ok_run)
    m['median_cpm_t2t'] = a.median(1)
    m['median_cpm_grch38'] = b.median(1)
    qual = m[(m.spearman < SPEARMAN_MAX) & m.protein_coding]
    chosen = qual[qual.runnable].sort_values('spearman').head(N_REF)
    m.sort_values('spearman').to_csv(OUT / 'reference_dependent_metrics.tsv.gz', sep='\t')
    (OUT / 'reference_dependent.txt').write_text('\n'.join(chosen.index) + '\n')
    summ['reference_dependent'] = dict(
        samples=len(samples), genes_both=len(genes), expressed_both=int(expressed.sum()),
        protein_coding_expressed=int(m.protein_coding.sum()),
        qualifying=int(len(qual)), qualifying_segmental_duplication=int(qual.segmental_duplication.sum()),
        qualifying_runnable=int(qual.runnable.sum()), chosen=int(len(chosen)),
        chosen_spearman_range=[float(chosen.spearman.min()), float(chosen.spearman.max())],
        chosen_segmental_duplication=int(chosen.segmental_duplication.sum()))

    # ---- high-effect, high-se significant hits -----------------------------
    r = pd.read_csv(PERM, sep='\t')
    r['geneid'] = r.phenotype_id.str.replace('GeneID:', '', regex=False)
    r = r.merge(gtf[['gene', 'geneid']].dropna(), on='geneid', how='left')
    p = r.pval_beta.values
    order = np.argsort(p)
    bh = np.empty_like(p)
    bh[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    r['bh'] = np.minimum(bh, 1)
    sig = r[r.bh < 0.05].copy()
    hs, hb = sig.slope_se.quantile(1 - TOP), sig.slope.abs().quantile(1 - TOP)
    sig['high_beta_high_se'] = (sig.slope_se >= hs) & (sig.slope.abs() >= hb)
    sig['runnable'] = sig.gene.isin(ok_run)
    hits = sig[sig.high_beta_high_se]
    chosen_h = hits[hits.runnable].gene.dropna().drop_duplicates()
    sig.to_csv(OUT / 'high_beta_high_se_metrics.tsv.gz', sep='\t', index=False)
    (OUT / 'high_beta_high_se.txt').write_text('\n'.join(sorted(chosen_h)) + '\n')
    summ['high_beta_high_se'] = dict(
        genes_tested=int(len(r)), mapped_to_symbol=int(r.gene.notna().sum()), significant_bh05=int(len(sig)),
        slope_se_threshold=float(hs), abs_slope_threshold=float(hb), qualifying=int(len(hits)),
        qualifying_runnable=int(len(chosen_h)))
    (OUT / 'summary.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
