"""Real-data referee, inputs and discovery: the plasmode's arms mapped on the observed 92-donor BrainVar data, and a
replication scan in the BrainVar eQTL cohort's donors that none of the arms saw. TReCASE discovery and the scoring
are the next stage's; this writes everything they read (facts.json lists the paths and counts).

(1) HELD-OUT DONORS. The 225-donor cohort (SAMPLE_MANIFEST, DNA library in matchingDNALibrary; the run's
DATASET_MANIFEST names its genotype store, phenotypes and covariates, each checked against the manifest's sha256 or
size) minus the 92 discovery donors (DISCOVERY, DNA library ids). The manifest is the one bridge from the DNA library
id to the SubjectID labels of the 225 run's phenotype, covariate and expression files; every join is on the DNA
library id. One person can carry two DNA libraries (321_D1 / 321_D2), so each held-out donor's genotypes are also
compared with each discovery donor's on IDENTITY_VARIANTS tested variants at MAF >= IDENTITY_MAF (discovery dosages
from the analysis VCF, held-out from the store; integer store dosages only); a concordance above IDENTITY_MAX stops
the run. Phenotypes: the 225 run's rank-inverse-normal expression restricted to the held-out donors. Covariates: its
17 base covariates restricted to them plus its expression PCs (as many as its covariate profile carries) recomputed
on them by its construction (brainvar_eqtl.expression.calculate_expression_pca, residualized mode: each gene of its
PCA input residualized on [1, base], centered, scaled, PCs of the donor Gram matrix, each PC's sign set so its
largest-|score| donor is positive); the same code must first reproduce the 225 run's PCs within PCA_TOL. Leakage,
not removed: the phenotypes' and the PCA input's rank-inverse-normal transform, TMM factors and gene filter, the
genotype PCs and the base covariates' standardization, and the store's mean imputation of missing genotypes were all
computed over the 225 donors, 90 of them discovery donors; none of it fits held-out expression to discovery genotypes.

(2) REFEREE GENES AND VARIANTS. select_stratum_genes.py's pool (cache genes with Gibbs draws that pass the eQTL gene
filter) with a phenotype in the 225 run (our gene symbol is its universe's t2t_expression_id; our TSS equals its
phenotype BED end, checked), in one seeded order (SeedSequence(SEED, (ORDER_KEY,))), written in full so a later stage
can take a nested prefix. Tested variants: biallelic phased SNPs of the analysis VCF within CM.WIN of the TSS at
MAF >= CM.MAF over the 92 donors, present in the 225 genotype store under the same chr_pos_ref_alt. Two departures
from the plasmode's loader, both decided 2026-09-28: gene bodies are kept (the loader drops every selected gene's
body, a rule of the RASQUAL comparison's genotype-permutation null; over thousands of genes it would drop most
intragenic variants, each gene's own body included), and variants absent from the store are not tested, so every
discovery test has a held-out counterpart (the store's filters are quality filters: missingness <= 5%, MAF >= 1%
within the cohort's subgroups, HWE p >= 1e-6, RNA_reference_comparison_results/2_calculate_genotype_covariates.ipynb;
the match rate is printed by MAF band and chromosome). The id map is gated on the 90 donors in both: their store
dosages must equal their VCF dosages at >= MAP_AGREE_MIN of integer genotypes. Genes left with no tested variant are
dropped (printed). The loader (CM.load_point_estimate_inputs) is called with one gene for the gene-independent inputs
(VCF over the merged windows, donor order, covariates, library sizes), because its window loop costs genes x variants;
the gene rows of the point estimates and draws are read here as it reads them, and its invariants (common.load) are
re-checked.

(3) DISCOVERY on the observed data: records in place (perm the identity, no label swap), nothing thinned; A, T, Va, Vt
from summaries_from_point_estimates, as 02_make_datasets.py builds them at f = 1. Arms and settings are 03_run_arms.py's,
through its functions: hapmixQTL gibbs / split / unit / plus_one (map_nominal; map_cis, records_signflip, A3.NPERM
permutations, seed SEED), total-only tensorQTL (tensorqtl.cis on T, 17 covariates, seed SEED), mixQTL at the published
and permissive cutoffs (mixqtl_scan; mixqtl_permutation_scan with A3.MIXQTL_NPERM permutations from
SeedSequence(SEED, (MIX_PERM_KEY,)), shared by genes and both arms), eigenMT's M_eff per gene (A3.eigenmt_tests, GPU).
Units of BLOCK genes (GPU arms, in this process) and MIX_BLOCK genes (mixQTL, POOL forked worker processes) along the
order; map_cis draws its permutations once per call from its seed, so a unit's genes get what one call over every
gene would give them. A finished unit (its files exist) is not rerun. The first TIMING_GENES genes are run fresh and
timed; the discovery wall time of the whole order is projected as the larger of its GPU time and its mixQTL
CPU time / POOL; above BUDGET_H hours a prefix of the order (whole BLOCKs) that fits is run instead, the same genes
for every arm (genes/subset.json).

(4) REPLICATION: tensorqtl.cis.map_nominal on the held-out donors, total expression (the 225 run's phenotype), the
held-out covariates, window CM.WIN, maf_threshold 0 (a discovery variant at MAF >= 0.05 over the 92 donors can fall
below it over the held-out donors), over each discovered gene's tested variants (all of them are in the store).
Slopes are tensorQTL's own (per ALT allele, on the rank-inverse-normal scale).

Output (OUT): facts.json; cohort/heldout_donors.tsv, identity.tsv; genes/referee_order.tsv, regions.bed,
dropped_no_tested.txt, subset.json; variant_map.tsv.gz; heldout/genotypes.parquet, phenotypes.parquet,
covariates.tsv; discovery/<arm>/nominal_<lo>_<hi>.parquet and cis_<lo>_<hi>.parquet (lo, hi: ranks in the order;
03's columns and parquet metadata), discovery/eigenmt_m_eff.tsv; replication/replication.cis_qtl_pairs.<chr>.parquet.
"""
import concurrent.futures as cf
import gzip
import hashlib
import json
import multiprocessing
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark' / 'plasmode'))
import common as C                                               # noqa: E402
from select_stratum_genes import CACHE, CAL                      # noqa: E402  the pool: cache genes passing the eQTL gene filter
from tensorqtl import cis as TQ                                  # noqa: E402
from tensorqtl.hapmixqtl import summaries_from_point_estimates   # noqa: E402

A3 = C.module('03_run_arms')   # the plasmode's arm functions and settings
CM = C.CM

OUT = C.D / 'referee_replication_20260928'   # every output (task, 2026-09-28)
NF = Path('/mnt/ssd/lalli/nf_stage')
SAMPLE_MANIFEST = NF / 'brainvar_eqtl_pc_tuning_20260722T173803/shared_prepared/prepared/common/sample_manifest.tsv'   # 225 donors
DATASET_MANIFEST = NF / 'brainvar_eqtl_native_full_gene_list_preparation_20260729T094311Z/t2t/dataset_manifest.json'   # the 225 run
DISCOVERY = C.D / 'cohort' / 'samples.txt'   # the 92 discovery donors, DNA library ids
ANNOT = C.D / 'annot' / 'genes.tsv'          # gene, chr, start, end, TSS; the loader's gene positions
IDENTITY_VARIANTS, IDENTITY_MAF = 20000, 0.2
IDENTITY_MAX = 0.9     # measured here 2026-09-28: same donor 1.000, best unrelated match 0.43-0.49 on these variants
MAP_AGREE_MIN = 0.99   # the 90 shared donors' store vs VCF dosages; measured 1.00000 (2026-09-28); a wrong row or donor bridge pairs unrelated genotypes
PCA_TOL = 1e-6         # absolute, on PC scores up to ~177; the original code reproduced them to 9e-13 (2026-09-28)
ORDER_KEY, MIX_PERM_KEY = 7, 8   # spawn keys no plasmode script uses (02: 1-3; 03: 4, 5; select_stratum_genes: 6; 01: 10-13; 06: 30, 33)
TIMING_GENES = 100     # task
BUDGET_H = 6.0         # task: the full gene set if its discovery fits, else a prefix of the order
BLOCK, MIX_BLOCK = 100, 20   # genes per GPU unit (03's per-call cost ~3 s) and per mixQTL unit; TIMING_GENES is a multiple of both
POOL = 14              # mixQTL worker processes: with this process 15 of the 16 the task allows
MAIN_THREADS = 2       # BLAS and torch CPU threads in this process
TRECASE_PROCESS_H_PER_GENE = 0.15   # plasmode README: ~15 process-h per 100-gene dataset (median 4,695 tested variants per gene)
MAF_BANDS = (0.05, 0.1, 0.2, 0.3, 0.5)
PER_GENE = ('A', 'T', 'Va', 'Vt', 'pL', 'pR', 'pT')
ARMS = C.HAPMIX_ARMS + (C.TENSORQTL,) + tuple(C.MIXQTL_ARMS)
FACTS = {}
I = DS = WIN = GENES = MIX_PERM = GENE_TABLE = None   # set by main before the mixQTL workers fork, which inherit them


def verified(entry):
    """A manifest entry's path after its sha256, or its size where the manifest gives no sha256, is checked."""
    p = Path(entry['path'])
    if 'sha256' in entry:
        if hashlib.sha256(p.read_bytes()).hexdigest() != entry['sha256']:
            raise SystemExit(f'{p}: sha256 differs from the manifest')
    elif p.stat().st_size != entry['size_bytes']:
        raise SystemExit(f'{p}: {p.stat().st_size} bytes, the manifest says {entry["size_bytes"]}')
    return p


def write_tsv(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix != '.gz':
        C.write_atomic(path, lambda fh: df.to_csv(fh, sep='\t', index=False), 'w')
        return
    def write(fh):   # to_csv ignores compression= on a handle it is given, so compress through a binary gzip handle
        with gzip.GzipFile(fileobj=fh, mode='wb') as gz:
            df.to_csv(gz, sep='\t', index=False)
    C.write_atomic(path, write)


def write_frame_parquet(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    df.to_parquet(tmp, index=False, compression='zstd')
    tmp.rename(path)


def cohort(dm):
    """The 225-donor manifest, the discovery donors, the held-out donors (DNA library ids)."""
    man = pd.read_csv(verified(dm['files']['sample_manifest']), sep='\t', dtype=str)
    if len(man) != dm['samples'] or not (man.matchingDNALibrary.is_unique and man.SubjectID.is_unique):
        raise SystemExit(f'{SAMPLE_MANIFEST}: {len(man)} rows, DNA library or SubjectID not unique')
    disc = DISCOVERY.read_text().split()
    lib = set(man.matchingDNALibrary)
    overlap, absent = sorted(set(disc) & lib), sorted(set(disc) - lib)
    held = sorted(lib - set(disc))
    print(f'cohort: {len(man)} donors in the 225 run, {len(disc)} discovery donors; overlap by DNA library id '
          f'{len(overlap)}; discovery donors not in the 225 run {absent}; held-out {len(held)}', flush=True)
    if set(held) & set(disc) or len(set(disc)) != len(disc):
        raise SystemExit('a held-out donor is a discovery donor, or a discovery id repeats')
    FACTS['cohort'] = dict(donors_225=len(man), discovery=len(disc), overlap=len(overlap), discovery_not_in_225=absent,
                           heldout=len(held))
    h = man.set_index('matchingDNALibrary').loc[held].reset_index()[['matchingDNALibrary', 'SubjectID', 'LibraryID']]
    write_tsv(OUT / 'cohort' / 'heldout_donors.tsv', h.rename(columns={'matchingDNALibrary': 'dna_library'}))
    return man, disc, held


def referee_genes(dm):
    """The pool with a 225-run phenotype, in the seeded order: gene, gene_id, chr, tss."""
    genes = (CACHE / 'genes.txt').read_text().split()
    cal = set(CAL.read_text().split())
    pool = sorted(g for g in genes if g in cal)
    uni = pd.read_csv(verified(dm['lineage']['native_filtered_gene_universe']), sep='\t', dtype=str)
    bed = pd.read_csv(verified(dm['files']['phenotype_bed']), sep='\t', usecols=[0, 1, 2, 3],
                      dtype={'#chr': str, 'phenotype_id': str})
    bed.columns = ['chr', 'start', 'end', 'gene_id']
    if not (uni.t2t_expression_id.is_unique and uni.shared_gene_id.is_unique and bed.gene_id.is_unique
            and set(bed.gene_id) == set(uni.shared_gene_id)):
        raise SystemExit('225 run: gene universe ids not unique or not the phenotype BED ids')
    gp = pd.read_csv(ANNOT, sep='\t', header=None, names=['gene', 'chr', 'start', 'end', 'tss'], dtype={'chr': str})
    if not gp.gene.is_unique:
        raise SystemExit(f'{ANNOT}: gene ids repeat')
    t = pd.DataFrame({'gene': pool}).merge(uni.rename(columns={'t2t_expression_id': 'gene', 'shared_gene_id': 'gene_id'})
                                           [['gene', 'gene_id']], on='gene')
    t = t.merge(gp[['gene', 'chr', 'tss']], on='gene', validate='one_to_one').merge(
        bed[['gene_id', 'chr', 'end']].rename(columns={'chr': 'bed_chr'}), on='gene_id', validate='one_to_one')
    bad = t[(t.chr != t.bed_chr) | (t.tss != t.end)]
    print(f'referee genes: {len(genes):,} cache genes, {len(pool):,} pass the eQTL gene filter ({CAL}); {len(t):,} with a '
          f'225-run phenotype; TSS or chromosome differing from the phenotype BED: {len(bad)}', flush=True)
    if len(bad):
        raise SystemExit(f'positions differ from the 225 run, e.g. {bad.head(3).to_dict("records")}')
    rng = np.random.default_rng(np.random.SeedSequence(C.SEED, spawn_key=(ORDER_KEY,)))
    t = t.iloc[rng.permutation(len(t))].reset_index(drop=True)[['gene', 'gene_id', 'chr', 'tss']]
    FACTS['genes'] = dict(cache=len(genes), pool=len(pool), with_225_phenotype=len(t))
    return t


def write_regions(t, path):
    """The genes' cis windows (plus 1,000 bp, as select_stratum_genes.py), merged per chromosome, BED."""
    rows = []
    for c, g in t.groupby('chr', sort=False):
        iv = np.stack([np.maximum(0, g.tss.values - CM.WIN - 1000), g.tss.values + CM.WIN + 1000], 1)
        iv = iv[np.argsort(iv[:, 0])]
        s, e = iv[0]
        for a, b in iv[1:]:
            if a > e:
                rows.append((c, s, e))
                s, e = a, b
            else:
                e = max(e, b)
        rows.append((c, s, e))
    path.parent.mkdir(parents=True, exist_ok=True)
    C.write_atomic(path,lambda fh: fh.write(''.join(f'{c}\t{s}\t{e}\n' for c, s, e in rows)), 'w')
    print(f'regions: {len(rows):,} merged windows, {sum(e - s for _, s, e in rows) / 1e6:,.0f} Mb, {path}', flush=True)


def load(genes):
    """The loader's inputs for `genes` (module docstring (2)) with the point estimates and draws in gene order."""
    one = OUT / 'genes' / 'loader_gene.txt'
    C.write_atomic(one, lambda fh: fh.write(genes[0] + '\n'), 'w')
    t0 = time.perf_counter()
    I = CM.load_point_estimate_inputs(gene_list=str(one), regions=str(OUT / 'genes' / 'regions.bed'))
    gi = {g: i for i, g in enumerate((CACHE / 'genes.txt').read_text().split())}
    rows = np.array([gi[g] for g in genes])
    for k in ('pL', 'pR', 'pT'):
        I[k] = np.asarray(np.load(Path(CM.PE) / f'{k}.npy', mmap_mode='r')[rows])
    for k in ('YL', 'YR', 'YT'):
        I[k] = np.asarray(np.load(CACHE / f'{k}.npy', mmap_mode='r')[rows])
    I['genes'] = list(genes)
    if not (I['gp'].loc[genes, 'pos'].values == GENE_TABLE.set_index('gene').loc[genes, 'tss'].values).all():
        raise SystemExit('loader gene positions differ from the annotation read here')
    keep = I['keep']
    R = {k: I[k][:, keep] for k in ('pL', 'pR', 'pT', 'YL', 'YR', 'YT')}
    for k in ('pL', 'pR', 'YL', 'YR'):
        if R[k].min() < 0:
            raise SystemExit(f'negative Salmon count in {k}: {R[k].min()}')
    for what, (l, r, tt) in (('point estimates', ('pL', 'pR', 'pT')), ('Gibbs draws', ('YL', 'YR', 'YT'))):
        u = float((R[tt] - R[l] - R[r]).min())
        if u < -C.ROUNDING_TOL:
            raise SystemExit(f'{what}: total minus paired haplotypes is {u:.3g} < -{C.ROUNDING_TOL}')
    print(f'loaded {len(genes):,} genes x {len(I["order"])} donors x {R["YL"].shape[2]} Gibbs draws; {len(I["vdf"]):,} '
          f'phased biallelic SNPs in the regions; {time.perf_counter() - t0:.0f} s', flush=True)
    return I, R


def window_rows(vdf, cand, gp, genes):
    """{gene: the rows of cand (sorted vdf rows) within CM.WIN of its TSS}, the loader's |pos - TSS| <= WIN."""
    ch, pos = vdf.chrom.to_numpy()[cand], vdf.pos.to_numpy()[cand]
    out = {}
    for c, g in gp.loc[genes].groupby('chr', sort=False):
        on = ch == c
        rows, p = cand[on], pos[on]
        if (np.diff(p) < 0).any():
            raise SystemExit(f'{c}: VCF positions not sorted')
        lo = np.searchsorted(p, g.pos.values - CM.WIN, 'left')
        hi = np.searchsorted(p, g.pos.values + CM.WIN, 'right')
        out.update({gene: rows[a:b] for gene, a, b in zip(g.index, lo, hi)})
    return out


def band_rates(maf, mapped, chrom):
    by_maf = {f'[{a}, {b})': [int(((maf >= a) & (maf < b)).sum()), float(mapped[(maf >= a) & (maf < b)].mean())]
              for a, b in zip(MAF_BANDS[:-1], MAF_BANDS[1:])}
    by_chr = {c: [int((chrom == c).sum()), float(mapped[chrom == c].mean())] for c in pd.unique(chrom)}
    return by_maf, by_chr


def store_genotypes(pf, src):
    """float32 [len(src), store donors], rows in the order of src (positional store rows, as
    brainvar_eqtl.genotypes.read_genotype_row_subset reads them)."""
    o = np.argsort(src)
    s = src[o]
    sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
    starts = np.concatenate([[0], np.cumsum(sizes)])
    grp = np.searchsorted(starts[1:], s, side='right')
    out = np.empty((len(s), len(pf.schema_arrow.names)), np.float32)
    for g in np.unique(grp):
        m = grp == g
        out[o[m]] = pf.read_row_group(int(g)).to_pandas().to_numpy(np.float32)[s[m] - starts[g]]
    return out


def variants(dm, I, genes, held):
    """Tested variants (module docstring (2)): the per-gene rows, the tested union, the store genotypes over it."""
    vdf = I['vdf']
    af = I['dos'].mean(1) / 2.0
    maf = np.minimum(af, 1 - af)
    common = np.where(maf >= CM.MAF)[0]
    win = window_rows(vdf, common, I['gp'], genes)
    union = np.unique(np.concatenate(list(win.values())))
    vt = pd.read_csv(verified(dm['files']['variant_table']), sep='\t', usecols=['id', 'chrom', 'pos'],
                     dtype={'id': str, 'chrom': str, 'pos': np.int64})
    pf = pq.ParquetFile(verified(dm['files']['genotype_parquet']))
    if len(vt) != pf.metadata.num_rows or not vt.id.is_unique:
        raise SystemExit(f'store: {len(vt):,} variant ids for {pf.metadata.num_rows:,} rows, or ids repeat')
    u = vdf.iloc[union]
    key = u.chrom + '_' + u.pos.astype(str) + '_' + u.ref + '_' + u.alt
    src = pd.Index(vt.id).get_indexer(key)
    mapped = src >= 0
    if not ((vt.chrom.values[src[mapped]] == u.chrom.values[mapped]).all()
            and (vt.pos.values[src[mapped]] == u.pos.values[mapped]).all()):
        raise SystemExit('store chromosome or position differs from the VCF at a matched id')
    by_maf, by_chr = band_rates(maf[union], mapped, u.chrom.to_numpy())
    print(f'variant map: {len(vt):,} store variants; {len(union):,} VCF SNPs within {CM.WIN:,} bp of a referee TSS at MAF '
          f'>= {CM.MAF}, {int(mapped.sum()):,} in the store by chr_pos_ref_alt ({mapped.mean():.4f}); by MAF band '
          f'[n, rate] {by_maf}; by chromosome {by_chr}', flush=True)
    tested = union[mapped]
    G = store_genotypes(pf, src[mapped])
    cols = list(pf.schema_arrow.names)
    integer = G == np.round(G)
    print(f'store genotypes read: {G.shape[0]:,} variants x {G.shape[1]} donors; NaN {int(np.isnan(G).sum())}; '
          f'non-integer (mean-imputed missing) {int((~integer).sum()):,} of {G.size:,} ({(~integer).mean():.2e}), in '
          f'{int((~integer).any(1).sum()):,} variants', flush=True)
    if np.isnan(G).any() or set(held) - set(cols) or len(cols) != dm['genotype_samples']:
        raise SystemExit('store: NaN dosages, held-out donors missing, or a donor count other than the manifest\'s')
    order = list(I['order'])
    shared = [d for d in order if d in cols]
    V = I['dos'][tested][:, [order.index(d) for d in shared]]
    St = G[:, [cols.index(d) for d in shared]]
    ok = integer[:, [cols.index(d) for d in shared]]
    agree = float(((St == V) & ok).sum() / ok.sum())
    flip = float(((St == 2 - V) & ok).sum() / ok.sum())
    per = ((St == V) & ok).sum(1) / np.maximum(ok.sum(1), 1)
    print(f'map gate, {len(shared)} donors in both: store dosage = VCF dosage for {agree:.5f} of integer genotypes '
          f'(= 2 - VCF: {flip:.4f}); per-variant agreement min / 1% / 5% / median '
          f'{np.quantile(per, [0, 0.01, 0.05, 0.5]).round(4).tolist()}', flush=True)
    if agree < MAP_AGREE_MIN:
        raise SystemExit(f'GATE FAILED: shared donors agree at {agree:.4f} < {MAP_AGREE_MIN}')
    identity(G, cols, integer, I, tested, maf, held, shared)
    FACTS['variants'] = dict(vcf_snps_in_regions=len(vdf), vcf_tested_like=len(union), mapped=int(mapped.sum()),
                             mapped_rate=float(mapped.mean()), mapped_by_maf=by_maf, mapped_by_chromosome=by_chr,
                             store_variants=len(vt), store_noninteger_share=float((~integer).mean()),
                             gate_shared_donors=len(shared), gate_agreement=agree, gate_flip_agreement=flip)
    vm = pd.DataFrame(dict(variant_id=u.index.astype(str), chrom=u.chrom.values, pos=u.pos.values, ref=u.ref.values,
                           alt=u.alt.values, maf_discovery=maf[union], store_id=np.where(mapped, key.values, None),
                           store_row=pd.Series(src).where(mapped).astype('Int64').values))
    write_tsv(OUT / 'variant_map.tsv.gz', vm)
    hg = pd.DataFrame(G[:, [cols.index(d) for d in held]], columns=held)
    hg.insert(0, 'variant_id', vdf.index[tested].astype(str))
    write_frame_parquet(OUT / 'heldout' / 'genotypes.parquet', hg)
    print(f'wrote {OUT / "variant_map.tsv.gz"} ({len(vm):,} rows) and {OUT / "heldout" / "genotypes.parquet"} '
          f'({len(hg):,} variants x {len(held)} held-out donors)', flush=True)
    win = {g: r[np.isin(r, tested)] for g, r in win.items()}
    return win, tested, G[:, [cols.index(d) for d in held]]


def identity(G, cols, integer, I, tested, maf, held, shared):
    """Genotype concordance of every held-out donor with every discovery donor (module docstring (1))."""
    cand = np.where(maf[tested] >= IDENTITY_MAF)[0]
    pick = cand[np.linspace(0, len(cand) - 1, min(IDENTITY_VARIANTS, len(cand))).astype(int)]
    order = list(I['order'])
    V = I['dos'][tested[pick]].astype(np.float32)
    conc = {}
    for who, donors in (('heldout', held), ('shared', shared)):
        H = G[pick][:, [cols.index(d) for d in donors]]
        valid = integer[pick][:, [cols.index(d) for d in donors]].astype(np.float32)
        num = sum(((H == x) * valid).T @ (V == x).astype(np.float32) for x in (0.0, 1.0, 2.0))
        conc[who] = pd.DataFrame(num / valid.sum(0)[:, None], index=donors, columns=order)
    self_c = np.array([conc['shared'].loc[d, d] for d in shared])
    top = conc['heldout'].max(1)
    print(f'identity: {len(pick):,} variants at MAF >= {IDENTITY_MAF}; each held-out donor\'s highest concordance with a '
          f'discovery donor min / median / max {top.min():.3f} / {top.median():.3f} / {top.max():.3f} '
          f'({top.idxmax()} with {conc["heldout"].loc[top.idxmax()].idxmax()}); the {len(shared)} shared donors with '
          f'themselves (store vs VCF) min {self_c.min():.4f}', flush=True)
    rows = pd.DataFrame(dict(dna_library=top.index, best_discovery_donor=conc['heldout'].idxmax(1).values,
                             concordance=top.values))
    write_tsv(OUT / 'cohort' / 'identity.tsv', rows)
    FACTS['identity'] = dict(variants=len(pick), heldout_max=float(top.max()), heldout_median=float(top.median()),
                             shared_self_min=float(self_c.min()), threshold=IDENTITY_MAX)
    if top.max() > IDENTITY_MAX or self_c.min() <= IDENTITY_MAX:
        raise SystemExit(f'GATE FAILED: a held-out donor matches a discovery donor above {IDENTITY_MAX}, or a shared donor '
                         f'does not match itself')


def expression_pcs(E, base, n):
    """brainvar_eqtl.expression.calculate_expression_pca, residualized mode: the first n PCs of E (genes x donors)."""
    X = np.column_stack([np.ones(len(base)), base.to_numpy(float)])
    Y = E.T.to_numpy(float)
    Y = Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]
    Y -= Y.mean(0)
    sd = Y.std(0, ddof=1)
    if np.isclose(sd, 0).any():
        raise SystemExit(f'expression PCA: {int(np.isclose(sd, 0).sum())} genes constant after residualization')
    Y /= sd
    w, V = np.linalg.eigh(Y @ Y.T)
    o = np.argsort(w)[::-1][:n]
    S = V[:, o] * np.sqrt(np.maximum(w[o], 0.0))
    S *= np.where(S[np.abs(S).argmax(0), np.arange(n)] < 0, -1.0, 1.0)
    return pd.DataFrame(S, index=E.columns, columns=[f'expr_PC{i}' for i in range(1, n + 1)])


def heldout_inputs(dm, man, held, t):
    """Held-out phenotypes (referee genes x held-out donors) and covariates (module docstring (1))."""
    lib = dict(zip(man.SubjectID, man.matchingDNALibrary))
    e36 = json.loads(verified(dm['lineage']['template_e36_dataset_manifest']).read_text())
    n_pcs = int(e36['phenotype_summary']['expression_pca']['selected_pcs'])
    base = pd.read_csv(verified(dm['files']['covariates_main_base']), sep='\t', index_col=0).rename(index=lib)
    main = pd.read_csv(verified(dm['files']['covariates_main']), sep='\t', index_col=0).rename(index=lib)
    pcs225 = pd.read_csv(Path(dm['files']['covariates_main']['path']).parent / 'expression_pcs.tsv', sep='\t',
                         index_col=0).rename(index=lib)
    E = pd.read_csv(verified(e36['lineage']['prepared_normalized_expression']), sep='\t', index_col=0).rename(columns=lib)
    pc_cols = [f'expr_PC{i}' for i in range(1, n_pcs + 1)]
    if list(main.columns) != list(base.columns) + pc_cols or not (set(base.index) == set(E.columns) == set(lib.values())):
        raise SystemExit('225 covariate profile is not base + expression PCs, or donors differ between files')
    print(f'225 run covariates: {base.shape[1]} base + {n_pcs} expression PCs over {len(base)} donors; PCA input '
          f'{E.shape[0]:,} genes', flush=True)
    E, pcs225 = E[base.index], pcs225.loc[base.index]
    diff = float(np.abs(expression_pcs(E, base, n_pcs).values - pcs225[pc_cols].values).max())
    print(f'PCA gate: recomputed 225-donor PCs differ from the run\'s by at most {diff:.2e} (tolerance {PCA_TOL})', flush=True)
    if diff > PCA_TOL:
        raise SystemExit('GATE FAILED: the 225 run\'s expression PCs are not reproduced')
    bh = base.loc[held]
    const = [c for c in bh.columns if np.ptp(bh[c].values) == 0]
    bh = bh.drop(columns=const)
    cov = pd.concat([bh, expression_pcs(E[held], bh, n_pcs)], axis=1)
    X = np.column_stack([np.ones(len(cov)), cov.values])
    if np.linalg.matrix_rank(X) != X.shape[1]:
        raise SystemExit('held-out covariates are rank deficient')
    cov.index.name = 'dna_library'
    C.write_atomic(OUT / 'heldout' / 'covariates.tsv', lambda fh: cov.to_csv(fh, sep='\t'), 'w')
    bed = pd.read_csv(verified(dm['files']['phenotype_bed']), sep='\t', index_col=3).rename(columns=lib)
    ph = bed.loc[t.gene_id, held]
    ph.index = t.gene.values
    out = ph.reset_index().rename(columns={'index': 'gene'})
    out.insert(1, 'gene_id', t.gene_id.values)
    write_frame_parquet(OUT / 'heldout' / 'phenotypes.parquet', out)
    print(f'held-out covariates: {bh.shape[1]} base (constant over the held-out donors, dropped: {const}) + {n_pcs} '
          f'recomputed PCs = {cov.shape[1]} over {len(cov)} donors, residual df {len(cov) - 2 - cov.shape[1]}; phenotypes '
          f'{ph.shape[0]:,} genes x {ph.shape[1]} donors', flush=True)
    FACTS['heldout'] = dict(covariates=cov.shape[1], base=bh.shape[1], base_dropped_constant=const, expression_pcs=n_pcs,
                            pca_gate_max_abs_diff=diff, residual_df=len(cov) - 2 - cov.shape[1], phenotype_genes=ph.shape[0])
    return ph, cov


def block(lo, hi):
    """common.setup and the observed-data dataset for the genes of ranks lo:hi, over their tested variants only."""
    genes = GENES[lo:hi]
    rows = np.unique(np.concatenate([WIN[g] for g in genes]))
    Ib = {**I, 'genes': genes, 'vdf': I['vdf'].iloc[rows], 'dos': I['dos'][rows], 'xL': I['xL'][rows],
          'xR': I['xR'][rows], 'idx': np.arange(len(rows))}
    S = C.setup(Ib)
    if [int(S['n_tested'][g]) for g in genes] != [len(WIN[g]) for g in genes]:
        raise SystemExit(f'genes {lo}-{hi}: common.setup\'s tested counts differ from the windows computed here')
    return S, {k: (v[lo:hi] if k in PER_GENE else v) for k, v in DS.items()}


def paths(arm, lo, hi):
    d = OUT / 'discovery' / arm
    return d / f'nominal_{lo:05d}_{hi:05d}.parquet', d / f'cis_{lo:05d}_{hi:05d}.parquet'


def gpu_unit(lo, hi, device, fresh):
    """The hapmixQTL and tensorqtl arms and eigenMT's M_eff on genes lo:hi; seconds spent."""
    t0 = time.perf_counter()
    S, ds = block(lo, hi)
    scratch = OUT / 'discovery' / 'scratch'
    for arm in C.HAPMIX_ARMS + (C.TENSORQTL,):
        nom, cis = paths(arm, lo, hi)
        if not fresh and nom.exists() and cis.exists():
            continue
        sha = C.fingerprint(ds, arm)
        if arm == C.TENSORQTL:
            nominal, res = A3.run_tensorqtl(S, ds, C.SEED, scratch)
        else:
            nominal, _ = C.run_nominal(S, ds, arm, scratch)
            res = A3.run_cis(S, ds, arm, C.SEED)
        C.write_parquet(nominal, nom, sha, C.UNITS[arm])
        C.write_parquet(res, cis, sha, C.UNITS[arm])
        print(f'genes {lo}-{hi} {arm:9s} map_nominal {len(nominal):,} rows; map_cis NaN pval_beta '
              f'{int(res.pval_beta.isna().sum())}', flush=True)
    em = OUT / 'discovery' / 'eigenmt' / f'm_eff_{lo:05d}_{hi:05d}.tsv'
    if fresh or not em.exists():
        write_tsv(em, A3.eigenmt_tests(S, device))
    return time.perf_counter() - t0


def mix_unit(lo, hi, fresh):
    """Both mixQTL arms on genes lo:hi, in a worker process; seconds spent."""
    threadpool_limits(A3.WORKER_THREADS)
    t0 = time.perf_counter()
    S, ds = block(lo, hi)
    for arm, cutoffs in C.MIXQTL_ARMS.items():
        nom, cis = paths(arm, lo, hi)
        if not fresh and nom.exists() and cis.exists():
            continue
        sha = C.fingerprint(ds, arm)
        nominal, n_asc, n_trc = A3.run_mixqtl(S, ds, cutoffs)
        C.write_parquet(nominal, nom, sha, C.UNITS[arm])
        gl = A3.mixqtl_gene_level(S, ds, cutoffs, nominal, MIX_PERM)
        C.write_parquet(gl, cis, sha, C.UNITS[arm])
        print(f'genes {lo}-{hi} {arm:17s} {len(nominal):,} rows; genes with >= {C.MX.META_N_CUTOFF} allelic / total samples '
              f'{int((n_asc >= C.MX.META_N_CUTOFF).sum())} / {int((n_trc >= C.MX.META_N_CUTOFF).sum())}; NaN pval_perm '
              f'{int(gl.pval_perm.isna().sum())}', flush=True)
    return time.perf_counter() - t0


def discovery(n_all):
    """Module docstring (3); the number of genes run (a prefix of the order)."""
    sub = OUT / 'genes' / 'subset.json'
    pool = cf.ProcessPoolExecutor(POOL, mp_context=multiprocessing.get_context('fork'))
    units = lambda n: [(lo, min(lo + MIX_BLOCK, n)) for lo in range(0, n, MIX_BLOCK)]   # noqa: E731
    first = [pool.submit(mix_unit, lo, hi, True) for lo, hi in units(TIMING_GENES)] if not sub.exists() else []
    if not torch.cuda.is_available():
        raise SystemExit('no CUDA device: map_cis and eigenMT run on the GPU (03_run_arms.py)')
    device = torch.device('cuda')
    print(f'torch {torch.__version__}, {torch.cuda.get_device_name(device)}; mixQTL in {POOL} worker processes', flush=True)
    if first:
        gpu_s = gpu_unit(0, TIMING_GENES, device, True)
        mix_s = [j.result() for j in first]
        gpu_pg, mix_pg = gpu_s / TIMING_GENES, sum(mix_s) / TIMING_GENES
        rate = max(gpu_pg, mix_pg / POOL)
        proj_h = n_all * rate / 3600
        n_run = n_all if proj_h <= BUDGET_H else int(BUDGET_H * 3600 / rate) // BLOCK * BLOCK
        rec = dict(rule=f'the first {TIMING_GENES} genes of the order timed; all {n_all} genes if the projected discovery wall '
                        f'time max(GPU s/gene, mixQTL CPU s/gene / {POOL}) x genes is <= {BUDGET_H} h, else the longest '
                        f'prefix of whole {BLOCK}-gene units that fits',
                   timing_genes=TIMING_GENES, gpu_seconds=gpu_s, mixqtl_cpu_seconds=float(sum(mix_s)),
                   mixqtl_units_seconds=mix_s, gpu_seconds_per_gene=gpu_pg, mixqtl_cpu_seconds_per_gene=mix_pg,
                   workers=POOL, projected_hours_all=proj_h, genes_all=n_all, n_genes=n_run, subset=n_run < n_all,
                   genes_file=str(OUT / 'genes' / 'referee_order.tsv'),
                   trecase_projection=dict(note='projection from the plasmode README (~15 process-h per 100 genes at a '
                                                'median 4,695 tested variants per gene), not measured here',
                                           process_hours=TRECASE_PROCESS_H_PER_GENE * n_run))
        C.write_json(sub, rec)
        print(f'TIMING: {TIMING_GENES} genes: GPU arms + eigenMT {gpu_s:.0f} s ({gpu_pg:.2f} s/gene), mixQTL both arms '
              f'{sum(mix_s):.0f} CPU s ({mix_pg:.2f} s/gene, {len(mix_s)} units); projected discovery of all {n_all:,} genes '
              f'{proj_h:.2f} h (GPU {n_all * gpu_pg / 3600:.2f} h, mixQTL {n_all * mix_pg / POOL / 3600:.2f} h on {POOL} '
              f'workers); budget {BUDGET_H} h -> {n_run:,} genes{" (a prefix of the order)" if n_run < n_all else ""}. '
              f'TReCASE projection (plasmode README, not measured here): {TRECASE_PROCESS_H_PER_GENE * n_run:,.0f} '
              f'process-h for these genes', flush=True)
    n_run = json.loads(sub.read_text())['n_genes']
    jobs = {pool.submit(mix_unit, lo, hi, False): (lo, hi) for lo, hi in units(n_run) if hi > TIMING_GENES or not first}
    t0 = time.perf_counter()
    for lo in range(0, n_run, BLOCK):
        if lo < TIMING_GENES and first:
            continue
        gpu_unit(lo, min(lo + BLOCK, n_run), device, False)
        print(f'GPU arms through gene {min(lo + BLOCK, n_run):,} of {n_run:,}; {(time.perf_counter() - t0) / 3600:.2f} h', flush=True)
    for j in cf.as_completed(jobs):
        j.result()
    pool.shutdown()
    if (OUT / 'discovery' / 'scratch').exists():   # absent when every unit was already done
        shutil.rmtree(OUT / 'discovery' / 'scratch')
    em = pd.concat([pd.read_csv(OUT / 'discovery' / 'eigenmt' / f'm_eff_{lo:05d}_{min(lo + BLOCK, n_run):05d}.tsv', sep='\t')
                    for lo in range(0, n_run, BLOCK)], ignore_index=True)
    if list(em.gene) != GENES[:n_run]:
        raise SystemExit('eigenMT files do not cover the genes run in order')
    write_tsv(OUT / 'discovery' / 'eigenmt_m_eff.tsv', em)
    counts = {}
    for arm in ARMS:
        files = [paths(arm, lo, min(lo + step, n_run)) for step in ((MIX_BLOCK,) if arm in C.MIXQTL_ARMS else (BLOCK,))
                 for lo in range(0, n_run, step)]
        cis = pd.concat([pd.read_parquet(c) for _, c in files], ignore_index=True)
        rows = sum(pq.read_metadata(n).num_rows for n, _ in files)
        if list(cis.phenotype_id) != GENES[:n_run] or rows != sum(len(WIN[g]) for g in GENES[:n_run]):
            raise SystemExit(f'{arm}: cis files do not cover the genes in order, or nominal rows differ from tested pairs')
        counts[arm] = dict(nominal_rows=rows, genes=len(cis), nan_pval_perm=int(cis.pval_perm.isna().sum()))
        if 'pval_beta' in cis:
            counts[arm]['nan_pval_beta'] = int(cis.pval_beta.isna().sum())
    print(f'discovery: {n_run:,} genes, {sum(len(WIN[g]) for g in GENES[:n_run]):,} tested pairs; per arm {counts}; eigenMT '
          f'M_eff min / median / max {em.m_eff.min()} / {int(em.m_eff.median())} / {em.m_eff.max()}', flush=True)
    FACTS['discovery'] = dict(n_genes=n_run, tested_pairs=sum(len(WIN[g]) for g in GENES[:n_run]), arms=counts,
                              map_cis_seed=C.SEED, nperm=A3.NPERM, perm_scheme=A3.PERM_SCHEME,
                              mixqtl_nperm=A3.MIXQTL_NPERM, mixqtl_perm_spawn_key=MIX_PERM_KEY)
    return n_run


def replication(n_run, tested, Gh, held, ph, cov):
    """Module docstring (4) over the first n_run genes."""
    genes = GENES[:n_run]
    rows = np.unique(np.concatenate([WIN[g] for g in genes]))
    at = np.searchsorted(tested, rows)
    vdf = I['vdf'].iloc[rows]
    gdf = pd.DataFrame(Gh[at], index=vdf.index.astype(str), columns=held)
    out = OUT / 'replication'
    out.mkdir(parents=True, exist_ok=True)
    C.quiet(TQ.map_nominal, gdf, vdf[['chrom', 'pos']], ph.loc[genes, held], I['gp'].loc[genes][['chr', 'pos']],
            'replication', covariates_df=cov.loc[held], maf_threshold=0, window=CM.WIN, output_dir=str(out), verbose=False)
    files = sorted(out.glob('replication.cis_qtl_pairs.*.parquet'))
    df = pd.concat([pd.read_parquet(f, columns=['phenotype_id', 'variant_id', 'pval_nominal']) for f in files], ignore_index=True)
    per = df.groupby('phenotype_id').size().reindex(genes)
    if list(per.values) != [len(WIN[g]) for g in genes]:
        raise SystemExit(f'replication map_nominal returned {len(df):,} rows against {sum(len(WIN[g]) for g in genes):,} pairs')
    mono = int((np.ptp(Gh[at], axis=1) == 0).sum())
    print(f'replication: {len(held)} held-out donors, {cov.shape[1]} covariates, {n_run:,} genes, {len(df):,} pairs in '
          f'{len(files)} files; variants constant over the held-out donors {mono:,}; non-finite pval_nominal '
          f'{int((~np.isfinite(df.pval_nominal)).sum()):,}', flush=True)
    FACTS['replication'] = dict(genes=n_run, pairs=len(df), variants=len(rows), heldout_constant_variants=mono,
                                nonfinite_pval=int((~np.isfinite(df.pval_nominal)).sum()),
                                files=[str(f) for f in files], slope_unit='tensorQTL per ALT allele, rank-inverse-normal scale')


def main():
    global I, DS, WIN, GENES, MIX_PERM, GENE_TABLE
    threadpool_limits(MAIN_THREADS)
    torch.set_num_threads(MAIN_THREADS)
    OUT.mkdir(parents=True, exist_ok=True)
    dm = json.loads(DATASET_MANIFEST.read_text())
    man, disc, held = cohort(dm)
    GENE_TABLE = referee_genes(dm)
    write_regions(GENE_TABLE, OUT / 'genes' / 'regions.bed')
    I, R = load(list(GENE_TABLE.gene))
    if sorted(I['order']) != sorted(disc):
        raise SystemExit('loader donors differ from the discovery donors')
    eff_lib = I['eff_lib'][I['keep']]
    A, T, Va, Vt, _ = summaries_from_point_estimates(R['pL'], R['pR'], R['pT'], eff_lib, R['YL'], R['YR'], R['YT'])
    N = len(I['order'])
    DS = dict(A=A, T=T, Va=Va, Vt=Vt, pL=R['pL'], pR=R['pR'], pT=R['pT'], eff_lib=eff_lib,
              perm=np.arange(N), swap=np.ones(N, np.int8), causal_variant=np.array([], dtype=str))
    for k in ('YL', 'YR', 'YT'):
        del I[k]
    del R
    WIN, tested, Gh = variants(dm, I, list(GENE_TABLE.gene), held)
    none = [g for g in GENE_TABLE.gene if len(WIN[g]) == 0]
    C.write_atomic(OUT / 'genes' / 'dropped_no_tested.txt', lambda fh: fh.write(''.join(f'{g}\n' for g in none)), 'w')
    keep = ~GENE_TABLE.gene.isin(none).values
    rank = np.flatnonzero(keep)
    for k in PER_GENE:
        DS[k] = DS[k][rank]
    GENE_TABLE = GENE_TABLE[keep].reset_index(drop=True)
    GENES = list(GENE_TABLE.gene)
    nt = np.array([len(WIN[g]) for g in GENES])
    order_out = GENE_TABLE.assign(rank=np.arange(len(GENES)), n_tested=nt)[['rank', 'gene', 'gene_id', 'chr', 'tss', 'n_tested']]
    write_tsv(OUT / 'genes' / 'referee_order.tsv', order_out)
    print(f'referee genes: {len(GENES):,} in the order ({len(none)} without a tested variant dropped); tested variants per '
          f'gene min / median / max {nt.min()} / {int(np.median(nt))} / {nt.max()}, {int(nt.sum()):,} pairs; chromosomes '
          f'{GENE_TABLE.chr.value_counts().to_dict()}', flush=True)
    FACTS['genes'].update(dropped_no_tested=len(none), referee=len(GENES), tested_pairs=int(nt.sum()),
                          tested_per_gene=[int(nt.min()), int(np.median(nt)), int(nt.max())],
                          by_chromosome=GENE_TABLE.chr.value_counts().to_dict(), order_spawn_key=ORDER_KEY)
    ph, cov = heldout_inputs(dm, man, held, GENE_TABLE)
    ph = ph.loc[:, held]
    C.write_json(OUT / 'facts.json', FACTS)
    rng = np.random.default_rng(np.random.SeedSequence(C.SEED, spawn_key=(MIX_PERM_KEY,)))
    MIX_PERM = np.array([rng.permutation(N) for _ in range(A3.MIXQTL_NPERM)])
    n_run = discovery(len(GENES))
    C.write_json(OUT / 'facts.json', FACTS)
    replication(n_run, tested, Gh, held, ph, cov)
    FACTS['paths'] = dict(
        genes=str(OUT / 'genes' / 'referee_order.tsv'), subset=str(OUT / 'genes' / 'subset.json'),
        variant_map=str(OUT / 'variant_map.tsv.gz'), heldout_genotypes=str(OUT / 'heldout' / 'genotypes.parquet'),
        heldout_phenotypes=str(OUT / 'heldout' / 'phenotypes.parquet'), heldout_covariates=str(OUT / 'heldout' / 'covariates.tsv'),
        heldout_donors=str(OUT / 'cohort' / 'heldout_donors.tsv'), identity=str(OUT / 'cohort' / 'identity.tsv'),
        discovery=str(OUT / 'discovery'), eigenmt=str(OUT / 'discovery' / 'eigenmt_m_eff.tsv'),
        replication=str(OUT / 'replication'), regions=str(OUT / 'genes' / 'regions.bed'))
    C.write_json(OUT / 'facts.json', FACTS)
    print(f'wrote {OUT / "facts.json"}', flush=True)


if __name__ == '__main__':
    main()
