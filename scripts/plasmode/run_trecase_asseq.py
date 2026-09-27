"""TReCASE (asSeq 0.99.501, trecase) on every plasmode dataset, nominal only.

Datasets: make_datasets.py. Tested variants, genotype frames and covariates:
run_arms.setup, so asSeq sees the variants and record order the other arms see.
The per-gene asSeq call is run_trecase_asseq.R; this script writes its inputs,
runs it in at most JOBS Rscript processes at once (one gene per process) and
converts its output.

INPUTS, per dataset (column i is real record perm[i], already applied by the
dataset, as in run_arms):
  Y       thinned total point estimates pT [N x genes], doubles as they are
          (the TReC negative binomial goes through lgammafn / digamma on
          doubles, glm.c:867-918).
  Y1, Y2  haplotype 1 = L, haplotype 2 = R, on the records make_datasets.
          allelic_kept admits (as run_rasqual.py's het set and every hapmixQTL
          arm); both are 0 on every other record, which asSeq then leaves out
          of the ASE part (Y1 + Y2 < min.AS.reads, trecase.c:617 and :787).
          DEVIATION from the task's "Y1 = pL, Y2 = pR", pending the user's
          confirmation, made the same day as run_rasqual.py's: with every record,
          950-957 records per dataset with exactly one haplotype below 0.5 read
          (Salmon exact zeros) enter as maximal imbalance in a random direction,
          and on the 2026-09-26 smoke they turned ASPHD1's causal ASE_b / ln 2 to
          -0.547 against a beta of +0.8. Undo: Y1 = rint(pL), Y2 = rint(pR) in
          write_dataset.
          ROUNDED to the nearest integer (np.rint per side, as run_rasqual.py's
          AS field), a second deviation from the task's "no rounding": asSeq's
          beta-binomial adds lchoose(n, nA) (ase.c:43; trecase.c:101), R's
          lchoose rounds a non-integer nA while n stays fractional, H0 takes
          nA = Y1 (trecase.c:620) and H1 nA = Y2 for Zh 0, 1, 4 (:792-799), so
          the two likelihoods carry different constants and asSeq stops at
          "likelihood decreases for ASE model" (:1182-1185), as all three smoke
          genes did with fractional counts on 2026-09-26.
  X       the RNA-tied covariates (cov_df rows in the order perm) and the
          genotype PCs in place, 17 columns, NO constant column: glmFit centres
          every column and charges the intercept itself (glm.c:588-626; :702
          "assume there is an intercept", dfr = Nu - 1 - x_rank), and a constant
          column fails R/trecase.R's tiny-variance stop.
  offset  log(eff_lib), the edgeR effective library size in record order.
  Z       phased genotypes coded 3 xL + xR: 0 ref|ref, 1 ref|alt (haplotype 1
          = L carries REF), 3 alt|ref, 4 alt|alt, so Y1 is phased with it.
          trecase.c:792-801 takes the ALT haplotype's reads as nA and R/trecase.R
          maps 3 -> 1, 4 -> 2 for the TReC dosage, so every b is ln(kappa), kappa
          = ALT over REF. Built once per chromosome from every tested-set variant
          on it with varying ALT dosage (the 507 at constant dosage that map_cis
          drops are dropped); asSeq's window (same chromosome, |ePos - mPos| <=
          1,000,000, trecase.c:692-698) picks each gene's variants, and the rows
          it writes must be exactly run_arms.setup's tested pairs with varying
          dosage (checked per gene).
  ePos    run_arms' gene position (I['gp'].pos, the window centre CM uses).
asSeq defaults kept: min.AS.reads 5, min.AS.sample 5, min.n.het 5, transTestP
0.05, maxit 100. p.cut is 1000, not 1, a third deviation: trecase.c:1278 writes a
row only when a p is strictly below p.cut, and an unfitted model carries p = 995,
so p.cut = 1 would drop every test whose p's are all exactly 1 or all unfitted.
trace is 1 (run_trecase_asseq.R), so each gene's log (WORK/.../out/<gene>.log)
records why a joint fit failed; the failure classes are counted per dataset. On
the 2026-09-27 smoke (147 tests; logs WORK/smoke/beta0.8/rep000/out) all 34 joint
failures were the theta step's L-BFGS-B fail 52 (lbfgsb1.c:4194-4197,
ABNORMAL_TERMINATION_IN_LNSRCH, :927), at |gradient| <= 3.4e-3; asSeq runs
L-BFGS-B with pgtol 0 (trecase.c:459). The joint columns are then NA
(trecase.c:1269-1275) and final_Pvalue is the TReC p. A second, silent route to
a missing joint fit: where the TReC fit with asSeq's own dosage model (glmNBlog,
trecase.c:729-738) fails, asSeq refits TReC with the 0/1/2 dosage as a linear
covariate (glmNB, :740-752) and sets adjZ = 0 (:744), which skips the joint model
(:865); that TReC b is then a log fold change per ALT allele, not ln(kappa)
(counted as trec_linear_dosage; the row is not flagged).

OUTPUT: OUT/<scenario>/trecase/nominal_repNNN.parquet, one row per tested pair
with varying dosage (run_arms.write_parquet, fingerprint
run_arms.fingerprint(ds, 'trecase'), unit log2):
  pval_nominal   asSeq's printed final_Pvalue (%.2e, three significant digits).
                 Its rule (trecase.c:1311-1323): the TReC p when trans_Pvalue <
                 transTestP or trans_Pvalue is NA (joint model not fitted), the
                 joint p otherwise; NA where the chosen p is NA.
  final_stat     'joint' or 'trec': the rule applied to the printed trans_Pvalue
                 and checked by string equality of final_Pvalue with the chosen
                 column. A printed trans_Pvalue of 5.00e-02 lies on either side of
                 transTestP, so there final_stat is the column final_Pvalue
                 matches (counted; 'trec' where it matches both).
  slope          b / ln 2 of that statistic (log2 kappa, ALT over REF).
  slope_se       DERIVED, |slope| / sqrt(Chisq) of that statistic from its printed
                 Chisq (%.3f): asSeq reports no standard error; this is the Wald
                 back-derivation, a standard error only where the Wald
                 approximation holds, and not one where the statistic has 2 df
                 (theta at its boundary, trecase.c:1188, 1243). NaN where
                 Chisq <= 0 or NA.
  pval_a, slope_a, slope_a_se, chisq_a, df_a          ASE model (ASE_*)
  pval_t, slope_t, slope_t_se, chisq_t, df_t          TReC model (TReC_*)
  joint_ok (Joint_Pvalue printed), pval_joint, slope_joint, slope_joint_se,
  chisq_joint, df_joint; trans_chisq, pval_trans; n_trec, n_ase, n_ase_het
  (asSeq's n_TReC, n_ASE, n_ASE_Het); nb_od, bb_od = NBod / BBod where joint_ok,
  else NaN (only the joint iteration sets them, trecase.c:1014, 1057, 1293).
Every p is asSeq's printed value; every *_se is derived as above. The 507 tested
pairs at constant ALT dosage have no row (map_nominal writes them, with slope_t 0
and se inf), including TPPP's causal variant chr5_699683_A_G in rep 002 of every
beta > 0 scenario, listed as causal_not_run: score.py must skip it for this arm.

WHAT THIS CANNOT ANSWER. TReCASE's likelihood is for read counts; these are
Salmon's fractional point estimates, rounded per side for the ASE part (above)
and fractional in the TReC part.

SMOKE (SMOKE = True): beta 0.8 rep 000, SMOKE_GENES, VCF read over their regions
only (as run_rasqual.py), outputs under OUT/smoke and WORK/smoke. SMOKE_VARIANTS
caps the tested variants per gene (the causal variant, the split arm's lead and
the variants nearest the causal); SMOKE_VARIANTS = None runs the smoke genes
whole, under OUT/smoke_whole_genes and WORK/smoke_whole_genes: the timing run the
projection is based on, not the smoke. ASPHD1 / NISCH / ZNF420 (2,828 / 3,062 /
3,329 variants) took 77 / 610 / 238 s inside trecase before allelic_kept
(2026-09-26), then 101 / 764 / 316 s and 116 / 718 / 401 s in two runs with it
(2026-09-27, host load 170-185 of 256 cores): projected 5.35 and 5.60 h at 32
processes. Host load and the joint fits that now converge both changed between
the runs, and are not separated. Known-
answer check, reported and not a stop: at each smoke gene's causal variant the
sign of ASE_b equals the sign of beta and of the through-origin slope of the
dataset's A on xL - xR over allelic_kept records; each lead (smallest
pval_nominal, then largest chisq) against run_arms' split arm. Timing: seconds per
variant, and the full run's projection at JOBS processes against MAX_HOURS; the
smoke genes run 3 processes at once on a shared host, so the projection assumes
32 free cores. The full run is SMOKE = False.
"""
import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import make_datasets as MD                           # noqa: E402
import run_arms as RA                                # noqa: E402
import compare_mixqtl_replication as CM              # noqa: E402

DATASETS = MD.ROOT / 'datasets'
OUT = MD.ROOT / 'results_trecase_asseq'
WORK = MD.ROOT / 'trecase_asseq_work'
ARM = 'trecase'
R_RUNNER = Path(__file__).resolve().with_name('run_trecase_asseq.R')
R_ENV = {'R_LD_LIBRARY_PATH': '/usr/lib/R/lib:/usr/lib/x86_64-linux-gnu',
         'LD_LIBRARY_PATH': '/usr/local/cuda/lib64'}   # CLAUDE.md, "R's BLAS crash is an environment clash"
JOBS = 48                      # Rscript processes at once on the shared 256-core host: 48 alongside RASQUAL's 64 kept load near 215 on 2026-09-27; 32 projected 3.8-4.3 h
TRANS_TEST_P = 0.05            # asSeq transTestP default (R/trecase.R), the rule at trecase.c:1311
TRANS_BORDER = '5.00e-02'      # %.2e of a trans p in [0.04995, 0.05005): either side of TRANS_TEST_P
MIN_AS_READS = 5               # asSeq min.AS.reads default (on Y1 + Y2, trecase.c:615-617); here only for counts
MIN_N_HET = 5                  # asSeq min.n.het default; here only to count tests it leaves without ASE
STRONG_P = 1e-4                # counts only: joint-fit failures among tests with TReC p below it
MAX_HOURS = 2.0                # task rule: the full run only if projected under 2 h at JOBS processes
SMOKE = False                  # True: beta0.8 rep000, SMOKE_GENES, VCF read over their regions only
SMOKE_GENES = ('ASPHD1', 'NISCH', 'ZNF420')   # run_rasqual.py's smoke genes: non-null at beta 0.8 rep 000, both signs
SMOKE_VARIANTS = 50            # at most this many tested variants per smoke gene; None = whole genes (the timing run)
CHANNELS = {'t': 'TReC', 'a': 'ASE', 'joint': 'Joint'}
FAILS = {'joint_theta': 'fail to estimate theta in joint model',   # trecase.c:1003-1012, printed at trace >= 1
         'joint_bxj': 'fail to estimate bxj in b_ml',              # :923-930
         'joint_phi': 'fail to estimate phi in joint model',       # :1063-1069; maxit (:1111) prints only at trace > 1
         'ase': 'fail ASE model', 'trec': 'Fail TReC',             # :834-838, :756-760
         'trec_linear_dosage': 'convSNPj@glmNB ='}                 # :740-752, see the docstring


def load(work):
    """run_arms.setup on the loader inputs; SMOKE reads the VCF over the smoke genes' regions only.

    The gene list stays all 100 genes, so every gene body still excludes variants and the
    smoke genes' tested sets equal the full run's.
    """
    regions = MD.REGIONS
    if SMOKE:
        bed = pd.read_csv(MD.REGIONS, sep='\t', header=None)
        sub = bed[bed[3].isin(SMOKE_GENES)]
        if len(sub) != len(SMOKE_GENES):
            raise SystemExit(f'{MD.REGIONS}: {len(sub)} lines for {SMOKE_GENES}')
        regions = work / 'regions.bed'
        regions.parent.mkdir(parents=True, exist_ok=True)
        MD.write_atomic(regions, lambda fh: sub.to_csv(fh, sep='\t', header=False, index=False), 'w')
    return RA.setup(CM.load_point_estimate_inputs(gene_list=str(MD.GENES), regions=str(regions)))


def chrom_int(c):
    if not (c.startswith('chr') and c[3:].isdigit()):
        raise SystemExit(f'chromosome {c!r} is not chr<integer>; asSeq needs integer chromosomes')
    return int(c[3:])


def write_bin(path, a):
    """float64, C order: an [k, N] array is R's N x k matrix read column-major."""
    MD.write_atomic(path, lambda fh: fh.write(np.ascontiguousarray(a, np.float64).tobytes()))


def smoke_subset(S, ds, genes):
    """Per smoke gene: the causal variant, the split arm's lead and the variants nearest the causal, at most SMOKE_VARIANTS."""
    split = pd.read_parquet(RA.RESULTS / 'beta0.8' / 'split' / 'nominal_rep000.parquet',
                            columns=['phenotype_id', 'variant_id', 'pval_nominal'])
    keep = set()
    for g in genes:
        cv = str(ds['causal_variant'][S['genes'].index(g)])
        lead = split[split.phenotype_id == g].sort_values('pval_nominal', kind='stable').variant_id.iloc[0]
        pos = S['vdf'].pos.loc[sorted(S['scanned'][g])]
        near = (pos - pos.loc[cv]).abs().sort_values(kind='stable').index.astype(str)
        keep |= {cv, lead} | set([v for v in near if v not in (cv, lead)][:SMOKE_VARIANTS - 2])
    return keep


def write_genotypes(S, genes, d, keep):
    """chr<k>.Z.bin and chr<k>.markers.tsv for the chromosomes of `genes`; variant ids per chromosome.

    `keep`: variant ids to restrict to (the smoke subset), or None for every tested-set variant.
    """
    I, vdf = S['I'], S['vdf']
    idx = I['idx']
    d.mkdir(parents=True, exist_ok=True)
    chroms = sorted({chrom_int(S['gp'].loc[g, 'chr']) for g in genes})
    dos = I['dos'][idx]
    varying = ~(dos == dos[:, [0]]).all(1)
    if keep is not None:
        varying &= vdf.index.astype(str).isin(keep)
    ids = {}
    for c in chroms:
        on = (vdf.chrom.values == f'chr{c}') & varying
        xL, xR = I['xL'][idx[on]].astype(np.int64), I['xR'][idx[on]].astype(np.int64)
        if not (np.isin(xL, (0, 1)).all() and np.isin(xR, (0, 1)).all() and np.array_equal(xL + xR, dos[on])):
            raise SystemExit(f'chr{c}: phased alleles outside {{0, 1}} or xL + xR differs from ALT dosage')
        write_bin(d / f'chr{c}.Z.bin', 3 * xL + xR)
        m = pd.DataFrame(dict(variant_id=vdf.index[on].astype(str), chr=c, pos=vdf.pos.values[on].astype(int)))
        MD.write_atomic(d / f'chr{c}.markers.tsv', lambda fh, m=m: m.to_csv(fh, sep='\t', index=False), 'w')
        ids[c] = m.variant_id.tolist()
    print(f'genotypes: {len(chroms)} chromosomes, {sum(map(len, ids.values())):,} tested-set variants with '
          f'varying ALT dosage (of {len(vdf):,} in the tested set read, {len(vdf) - int(varying.sum())} dropped for constant '
          f'dosage{" or outside the smoke subset" if keep is not None else ""})', flush=True)
    return ids


def check_windows(S, genes, ids, expected):
    """Before any job: the markers asSeq's window will test for each gene (same chromosome,
    |ePos - mPos| <= CM.WIN, trecase.c:692-698) are exactly its expected tested variants.

    Catches a chromosome that carries several genes, where Z holds every gene's variants,
    before hours of asSeq rather than after the gene's Rscript ends (convert checks the rows too).
    """
    for g in genes:
        c = chrom_int(S['gp'].loc[g, 'chr'])
        pos = S['vdf'].pos.loc[ids[c]].values
        inwin = set(np.array(ids[c])[np.abs(pos - int(S['gp'].loc[g, 'pos'])) <= CM.WIN])
        if inwin != expected[g]:
            raise SystemExit(f'{g}: asSeq\'s window holds {len(inwin)} markers against {len(expected[g])} expected; '
                             f'{len(inwin - expected[g])} extra, {len(expected[g] - inwin)} missing')
    print(f'window check: every gene\'s markers within {CM.WIN:,} bp are its tested variants ({len(genes)} genes)',
          flush=True)


def allelic_counts(ds, kk):
    """Y1, Y2 as asSeq gets them: rint(pL), rint(pR) on allelic_kept records, 0 elsewhere; and the kept mask."""
    pL, pR = ds['pL'][kk], ds['pR'][kk]
    kept = MD.allelic_kept(pL, pR, ds['Va'][kk])
    return np.where(kept, np.rint(pL), 0.0), np.where(kept, np.rint(pR), 0.0), kept


def write_dataset(S, ds, d, genes):
    """Y, Y1, Y2, X, offset and genes.tsv of one dataset for the R runner."""
    I = S['I']
    kk = [S['genes'].index(g) for g in genes]
    X = np.column_stack([I['cov_df'].values[ds['perm']], I['geno_cov_df'].values])
    Y1, Y2, _ = allelic_counts(ds, kk)
    arrays = dict(Y=ds['pT'][kk], Y1=Y1, Y2=Y2, X=X.T, offset=np.log(ds['eff_lib']))
    for name, a in arrays.items():
        if not np.isfinite(a).all():
            raise SystemExit(f'{d}: non-finite values in {name}')
    for name in ('Y', 'Y1', 'Y2'):
        if arrays[name].min() < 0:
            raise SystemExit(f'{d}: negative count in {name}: {arrays[name].min()}')
    d.mkdir(parents=True, exist_ok=True)
    for name, a in arrays.items():
        write_bin(d / f'{name}.bin', a)
    tab = pd.DataFrame(dict(gene=genes, chr=[chrom_int(S['gp'].loc[g, 'chr']) for g in genes],
                            pos=[int(S['gp'].loc[g, 'pos']) for g in genes]))
    MD.write_atomic(d / 'genes.tsv', lambda fh: tab.to_csv(fh, sep='\t', index=False), 'w')
    return X.shape[1]


def run_gene(ddir, gdir, g, tag):
    """One Rscript process; wall seconds. A failed process stops the run with its log tail."""
    log = Path(f'{tag}.log')
    t0 = time.perf_counter()
    with open(log, 'w') as fh:
        rc = subprocess.run(['Rscript', str(R_RUNNER), str(ddir), str(gdir), g, str(tag)],
                            stdout=fh, stderr=subprocess.STDOUT, env={**os.environ, **R_ENV}).returncode
    if rc != 0:
        raise SystemExit(f'{g}: Rscript exited {rc}; {log}:\n' + ''.join(log.read_text().splitlines(True)[-15:]))
    return time.perf_counter() - t0


def colname(c, k):
    return f'slope_{k}_se' if c == 'se' else f'{c}_{k}'


def convert(g, tag, markers, expected):
    """asSeq's rows for gene g in the output layout, its status row, and the trace-log failure lines.

    `expected`: the gene's tested variants with varying dosage among `markers` (all of them in the full run).
    """
    raw = pd.read_csv(f'{tag}_eqtl.txt', sep='\t', dtype=str, keep_default_na=False)
    status = pd.read_csv(f'{tag}_status.tsv', sep='\t').iloc[0]
    if (raw.GeneRowID != '1').any():
        raise SystemExit(f'{g}: GeneRowID other than 1 in {tag}_eqtl.txt')
    vid = np.array(markers)[raw.MarkerRowID.astype(int).values - 1]
    if len(set(vid)) != len(vid) or set(vid) != expected:
        raise SystemExit(f'{g}: asSeq wrote {len(vid)} rows ({len(set(vid))} variants) against {len(expected)} '
                         f'tested variants with varying dosage; {len(set(vid) - expected)} outside the set, '
                         f'{len(expected - set(vid))} missing')
    num = lambda c: pd.to_numeric(raw[c].replace('NA', np.nan)).values.astype(float)
    ch = {}
    for k, name in CHANNELS.items():
        b, c2 = num(f'{name}_b'), num(f'{name}_Chisq')
        s = b / MD.LN2
        se = np.full(len(s), np.nan)
        pos = np.isfinite(c2) & (c2 > 0)
        se[pos] = np.abs(s[pos]) / np.sqrt(c2[pos])
        ch[k] = dict(pval=num(f'{name}_Pvalue'), slope=s, se=se, chisq=c2, df=num(f'{name}_df'))
    final, trec, joint = raw.final_Pvalue.values, raw.TReC_Pvalue.values, raw.Joint_Pvalue.values
    pt = num('trans_Pvalue')
    use_joint = np.isfinite(pt) & (pt >= TRANS_TEST_P)       # trecase.c:1311: TReC where trans p < transTestP or NA
    border = raw.trans_Pvalue.values == TRANS_BORDER
    use_joint[border] = final[border] != trec[border]
    wrong = np.where(use_joint, joint, trec) != final
    if wrong.any():
        i = np.where(wrong)[0][0]
        raise SystemExit(f'{g}: final_Pvalue {final[i]} is not the {"Joint" if use_joint[i] else "TReC"} p that '
                         f'trecase.c:1311-1323 selects (trans_Pvalue {raw.trans_Pvalue.values[i]}, TReC {trec[i]}, '
                         f'Joint {joint[i]})')
    joint_ok = np.isfinite(ch['joint']['pval'])
    pick = lambda c: np.where(use_joint, ch['joint'][c], ch['t'][c])
    cols = dict(phenotype_id=np.full(len(raw), g), variant_id=vid, pval_nominal=num('final_Pvalue'),
                slope=pick('slope'), slope_se=pick('se'), chisq=pick('chisq'), df=pick('df'),
                final_stat=np.where(use_joint, 'joint', 'trec'))
    for k in ('a', 't', 'joint'):
        cols.update({colname(c, k): ch[k][c] for c in ('pval', 'slope', 'se', 'chisq', 'df')})
    cols.update(joint_ok=joint_ok, trans_chisq=num('trans_Chisq'), pval_trans=pt, n_trec=num('n_TReC'),
                n_ase=num('n_ASE'), n_ase_het=num('n_ASE_Het'), nb_od=np.where(joint_ok, num('NBod'), np.nan),
                bb_od=np.where(joint_ok, num('BBod'), np.nan))
    border_n = dict(border=int(border.sum()), border_both=int((border & (final == trec) & (final == joint)).sum()))
    return pd.DataFrame(cols), status, border_n, Path(f'{tag}.log').read_text()


def trace_counts(text):
    """Failure lines of trace-1 logs by class, and the fail codes / gradients of the theta failures."""
    c = {k: text.count(s) for k, s in FAILS.items()}
    th = re.findall(r'theta=(\S+), gradience=(\S+), fail=(\d+)', text)
    c['theta_fail_codes'] = pd.Series([int(f) for _, _, f in th], dtype=int).value_counts().sort_index().to_dict()
    c['theta_fail_abs_gradient_max'] = max([abs(float(x)) for _, x, _ in th], default=None)
    return c


def dataset_counts(df, ds, S, genes, expected, status, secs, border, logs):
    """Counts printed per dataset; the causal-variant rows of non-null genes.

    A causal variant outside `expected` (constant ALT dosage: TPPP's chr5_699683_A_G in rep 002 of
    every beta > 0 scenario) has no asSeq row by construction and is counted; any other causal
    variant without a row stops the run.
    """
    kk = [S['genes'].index(g) for g in genes]
    cz = pd.DataFrame(dict(phenotype_id=genes, variant_id=ds['causal_variant'][kk].astype(str),
                           is_null=ds['is_null'][kk], beta=ds['beta'][kk]))
    nn = cz[~cz.is_null]
    run = np.array([v in expected[g] for g, v in zip(nn.phenotype_id, nn.variant_id)], dtype=bool)
    C = nn[run].merge(df, on=['phenotype_id', 'variant_id'], how='left')
    if C.final_stat.isna().any():
        raise SystemExit(f'{int(C.final_stat.isna().sum())} causal variants of non-null genes have no asSeq row')
    pL, pR = ds['pL'][kk], ds['pR'][kk]
    Y1, Y2, kept = allelic_counts(ds, kk)
    admitted = (Y1 + Y2) >= MIN_AS_READS
    strong = df.pval_t < STRONG_P
    yf = pd.Series([int(s.yFailBaselineModel) for s in status]).value_counts().sort_index()
    c = dict(
        informative_records=int((pL + pR > 0).sum()),
        informative_zeroed_not_allelic_kept=int(((pL + pR > 0) & ~kept).sum()),
        as_records_admitted=int(admitted.sum()),
        admitted_one_side_rounded_to_0=int((admitted & ((Y1 == 0) | (Y2 == 0))).sum()),
        kept_admission_changed_by_rounding=int((kept & ((pL + pR >= MIN_AS_READS) != admitted)).sum()),
        admitted_total_changed_over_half_read=int((admitted & (np.abs(Y1 + Y2 - pL - pR) > 0.5)).sum()),
        rows=len(df), genes=len(genes), tested_run=sum(len(expected[g]) for g in genes),
        tested_varying=sum(len(S['scanned'][g]) for g in genes),
        tested_constant_dosage=int(sum(S['n_tested'][g] - len(S['scanned'][g]) for g in genes)),
        yFailBaselineModel={int(k): int(v) for k, v in yf.items()},
        trec_na=int(df.pval_t.isna().sum()), ase_na=int(df.pval_a.isna().sum()),
        ase_na_few_het=int((df.pval_a.isna() & (df.n_ase_het < MIN_N_HET)).sum()),
        joint_na=int((~df.joint_ok).sum()),
        trec_p_below_strong=int(strong.sum()), joint_na_trec_p_below_strong=int((strong & ~df.joint_ok).sum()),
        joint_na_by_trace=trace_counts(''.join(logs)),
        final_joint=int((df.final_stat == 'joint').sum()), final_trec=int((df.final_stat == 'trec').sum()),
        final_trec_joint_na=int(((df.final_stat == 'trec') & ~df.joint_ok).sum()),
        final_trec_trans_rejects=int(((df.final_stat == 'trec') & df.joint_ok).sum()),
        final_na=int(df.pval_nominal.isna().sum()), slope_se_nan=int(df.slope_se.isna().sum()),
        final_df_not_1=int((df.df.notna() & (df.df != 1)).sum()),
        trans_p_at_border=sum(b['border'] for b in border),
        trans_p_at_border_final_matches_both=sum(b['border_both'] for b in border),
        trans_chisq_negative=int((df.joint_ok & df.trans_chisq.isna()).sum()),
        causal_not_run=[f'{g} {v}' for g, v in zip(nn.phenotype_id[~run], nn.variant_id[~run])],
        causal_nonnull=len(C), causal_joint_na=int((~C.joint_ok.astype(bool)).sum()),
        causal_final_joint=int((C.final_stat == 'joint').sum()), causal_final_trec=int((C.final_stat == 'trec').sum()),
        causal_final_na=int(C.pval_nominal.isna().sum()),
        trecase_warnings=int(sum(s.n_warnings for s in status)), wall_seconds=[round(s, 2) for s in secs],
        trecase_seconds=[round(float(s.seconds), 2) for s in status])
    return c, C


def smoke_checks(df, C, ds, S):
    """Known-answer check at the causal variants, and each lead against run_arms' split arm."""
    split = pd.read_parquet(RA.RESULTS / 'beta0.8' / 'split' / 'nominal_rep000.parquet',
                            columns=['phenotype_id', 'variant_id', 'slope', 'pval_nominal'])
    I, agree = S['I'], 0
    for r in C.itertuples():
        k, j = S['genes'].index(r.phenotype_id), S['rows'][r.variant_id]
        kept = MD.allelic_kept(ds['pL'][k], ds['pR'][k], ds['Va'][k])
        s = (I['xL'][j] - I['xR'][j]).astype(float)[kept]
        a_slope = (s * ds['A'][k][kept]).sum() / (s ** 2).sum()
        ok = np.sign(r.slope_a) == np.sign(r.beta) == np.sign(a_slope)
        agree += bool(ok)
        print(f'  {r.phenotype_id} causal {r.variant_id} beta {r.beta:+.1f}: ASE_b / ln2 {r.slope_a:+.3f} '
              f'(p {r.pval_a:.2e}, n_ASE_Het {r.n_ase_het:.0f}), through-origin slope of A on xL - xR over '
              f'allelic_kept records {a_slope:+.3f}, TReC {r.slope_t:+.3f} (p {r.pval_t:.2e}), joint_ok '
              f'{r.joint_ok}, final {r.final_stat} p {r.pval_nominal:.2e}; signs agree {ok}', flush=True)
    print(f'known answer: ASE_b sign equals the sign of beta and of the A slope in {agree} of {len(C)} genes')
    same_lead = same_sign = 0
    for g, d in df.groupby('phenotype_id', sort=False):
        lead = d.assign(p=d.pval_nominal.fillna(np.inf)).sort_values(['p', 'chisq'], ascending=[True, False],
                                                                       kind='stable').iloc[0]
        sp = split[split.phenotype_id == g].sort_values('pval_nominal', kind='stable').iloc[0]
        same_lead += lead.variant_id == sp.variant_id
        same_sign += np.sign(lead.slope) == np.sign(sp.slope)
        print(f'  {g}: TReCASE lead {lead.variant_id} ({lead.final_stat}) slope {lead.slope:+.3f} p '
              f'{lead.pval_nominal:.2e}; split lead {sp.variant_id} slope {sp.slope:+.3f} p {sp.pval_nominal:.2e}')
    n = df.phenotype_id.nunique()
    print(f'lead against the split arm: same variant in {same_lead} of {n} genes, same slope sign in {same_sign} of '
          f'{n} (reported, not a stop)', flush=True)


def projection(sec_per_variant, overhead, meta):
    """Full-run hours at JOBS processes from seconds per variant = c0 + c1 x admitted allele-specific reads.

    `sec_per_variant`: asSeq's seconds per tested variant for each smoke gene; `overhead`: seconds
    per Rscript process outside trecase (R start, library(asSeq), reading inputs). The reads are each
    gene's sum of Y1 + Y2 over the records asSeq admits (the ASE loops run over them, trecase.c:97-116,
    ase.c:39-58); tested pairs per gene come from the split arm's full-run output. Fitted on the
    smoke genes only, so a rough guide.
    """
    split = pd.read_parquet(RA.RESULTS / 'beta0.8' / 'split' / 'nominal_rep000.parquet', columns=['phenotype_id'])
    nv = split.phenotype_id.value_counts().reindex(meta['genes']).values
    runs = [(b, r) for b in meta['betas'] for r in range(meta['n_datasets'][str(b)])]
    reads = {}
    for b, r in runs:
        ds = np.load(DATASETS / f'beta{b}' / f'rep{r:03d}.npz')
        Y1, Y2, _ = allelic_counts(ds, slice(None))
        h = Y1 + Y2
        reads[(b, r)] = np.where(h >= MIN_AS_READS, h, 0).sum(1)
    gi = [meta['genes'].index(g) for g in sec_per_variant]
    x = reads[(0.8, 0)][gi]
    y = np.array(list(sec_per_variant.values()))
    c1, c0 = np.polyfit(x, y, 1)
    per_gene = np.concatenate([nv * np.maximum(c0 + c1 * reads[k], 0) for k in runs]) + overhead
    total = per_gene.sum()
    print(f'timing model: seconds per tested variant = {c0:.3g} + {c1:.3g} x admitted allele-specific reads '
          f'(smoke genes: reads {np.round(x).astype(int).tolist()}, s per variant {np.round(y, 4).tolist()}), '
          f'plus {overhead:.1f} s per Rscript process; '
          f'full run {len(runs)} datasets x {len(meta["genes"])} genes: {total / 3600:.2f} CPU-hours, '
          f'{total / JOBS / 3600:.2f} h at {JOBS} processes; slowest single gene {per_gene.max() / 60:.1f} min',
          flush=True)
    return total / JOBS / 3600


def main():
    tag = ('smoke' if SMOKE_VARIANTS else 'smoke_whole_genes') if SMOKE else None
    out = OUT / tag if SMOKE else OUT
    work = WORK / tag if SMOKE else WORK
    S = load(work)
    meta = json.loads((DATASETS / 'meta.json').read_text())
    if S['genes'] != meta['genes'] or S['order'] != meta['donors']:
        raise SystemExit(f'{DATASETS / "meta.json"}: genes or donors differ from the loader inputs')
    genes = list(SMOKE_GENES) if SMOKE else S['genes']
    runs = [('beta0.8', 0)] if SMOKE else [(f'beta{b}', r) for b in meta['betas']
                                           for r in range(meta['n_datasets'][str(b)])]
    gdir = work / 'genotypes'
    keep = smoke_subset(S, dict(np.load(DATASETS / 'beta0.8' / 'rep000.npz')), genes) if SMOKE and SMOKE_VARIANTS else None
    ids = write_genotypes(S, genes, gdir, keep)
    chrom = {g: chrom_int(S['gp'].loc[g, 'chr']) for g in genes}
    expected = {g: S['scanned'][g] & set(ids[chrom[g]]) for g in genes}
    if keep is None and any(expected[g] != S['scanned'][g] for g in genes):
        raise SystemExit('a tested variant with varying dosage is missing from its chromosome marker table')
    check_windows(S, genes, ids, expected)
    n_t = pd.Series({g: len(expected[g]) for g in genes})
    print(f'{R_RUNNER}; {len(runs)} datasets x {len(genes)} genes, {JOBS} Rscript processes; tested variants with '
          f'varying dosage per gene {n_t.min()}-{n_t.max()}, {int(n_t.sum()):,} in all', flush=True)
    t_all = time.perf_counter()
    jobs, summary, causal = {}, {}, []
    ex = cf.ThreadPoolExecutor(JOBS)
    try:
        for sc, r in runs:
            ds = dict(np.load(DATASETS / sc / f'rep{r:03d}.npz'))
            ddir = work / sc / f'rep{r:03d}'
            if ddir.exists():
                shutil.rmtree(ddir)
            n_cov = write_dataset(S, ds, ddir, genes)
            (ddir / 'out').mkdir()
            print(f'{sc} rep {r:03d}: inputs written to {ddir} ({n_cov} covariates, no constant column)', flush=True)
            jobs[(sc, r)] = (ds, ddir, [ex.submit(run_gene, ddir, gdir, g, ddir / 'out' / g) for g in genes])
        for (sc, r), (ds, ddir, futs) in jobs.items():
            parts, status, secs, border, logs = [], [], [], [], []
            for g, f in zip(genes, futs):
                secs.append(f.result())
                d, st, b, log = convert(g, ddir / 'out' / g, ids[chrom[g]], expected[g])
                parts.append(d)
                status.append(st)
                border.append(b)
                logs.append(log)
            df = pd.concat(parts, ignore_index=True)
            (out / sc / ARM).mkdir(parents=True, exist_ok=True)
            RA.write_parquet(df, out / sc / ARM / f'nominal_rep{r:03d}.parquet', RA.fingerprint(ds, ARM), 'log2')
            c, C = dataset_counts(df, ds, S, genes, expected, status, secs, border, logs)
            summary[f'{sc} rep {r:03d}'] = c
            causal.append(C)
            short = {k: v for k, v in c.items() if not k.endswith('_seconds')}
            print(f'{sc} rep {r:03d}: {json.dumps(short)}; seconds per gene inside trecase median '
                  f'{np.median(c["trecase_seconds"]):.1f} max {max(c["trecase_seconds"]):.1f}, per Rscript process '
                  f'median {np.median(c["wall_seconds"]):.1f} max {max(c["wall_seconds"]):.1f}; elapsed '
                  f'{(time.perf_counter() - t_all) / 60:.1f} min', flush=True)
    finally:
        ex.shutdown(cancel_futures=True)   # a failed gene cancels the queue rather than waiting it out
    tot = lambda k: sum(c[k] for c in summary.values())
    pooled = dict(
        tests=tot('rows'), joint_na=tot('joint_na'), joint_na_share=tot('joint_na') / tot('rows'),
        trec_p_below_strong=tot('trec_p_below_strong'), joint_na_trec_p_below_strong=tot('joint_na_trec_p_below_strong'),
        final_trec_when_joint_na=tot('final_trec_joint_na'), final_na=tot('final_na'),
        final_joint=tot('final_joint'), final_trec_trans_rejects=tot('final_trec_trans_rejects'),
        joint_fail_theta=sum(c['joint_na_by_trace']['joint_theta'] for c in summary.values()),
        causal_not_run=[x for c in summary.values() for x in c['causal_not_run']], causal_nonnull=tot('causal_nonnull'),
        causal_joint_na=tot('causal_joint_na'), causal_joint_na_share=tot('causal_joint_na') / tot('causal_nonnull'),
        causal_final_joint=tot('causal_final_joint'), causal_final_trec=tot('causal_final_trec'),
        causal_final_na=tot('causal_final_na'))
    print(f'pooled over {len(summary)} datasets: {json.dumps(pooled)}', flush=True)
    MD.write_atomic(out / 'summary.json', lambda fh: fh.write(MD.dumps(dict(
        per_dataset=summary, pooled=pooled, jobs=JOBS, smoke=SMOKE, smoke_variants=SMOKE_VARIANTS, genes=genes,
        strong_p=STRONG_P, wall_minutes=(time.perf_counter() - t_all) / 60))), 'w')
    if SMOKE:
        ds, ddir, _ = jobs[('beta0.8', 0)]
        smoke_checks(pd.read_parquet(out / 'beta0.8' / ARM / 'nominal_rep000.parquet'), causal[0], ds, S)
        secs = dict(zip(genes, summary['beta0.8 rep 000']['trecase_seconds']))
        wall = summary['beta0.8 rep 000']['wall_seconds']
        print('asSeq seconds per smoke gene (inside trecase / Rscript process): '
              + ', '.join(f'{g} {s:.1f} / {w:.1f} s for {len(expected[g])} variants'
                          for (g, s), w in zip(secs.items(), wall)))
        hours = projection({g: s / len(expected[g]) for g, s in secs.items()},
                           float(np.median([w - s for s, w in zip(secs.values(), wall)])), meta)
        print(f'full run projected {hours:.2f} h at {JOBS} processes against the {MAX_HOURS} h rule '
              f'({"whole-gene" if SMOKE_VARIANTS is None else "variant-subset"} timings): '
              f'{"RUN" if hours < MAX_HOURS else "DO NOT RUN"}')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
