"""Paths, parameters and helpers shared by the simulated-effects benchmark scripts (run order: run_all.sh).

Design: docs/simulation_benchmark_spec.md (superseded header) and README.md here. Every
parameter is written once, in this file or at the top of the script that owns it.
"""
import contextlib
import gzip
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))   # the repository's tensorqtl, ahead of an installed upstream copy

from tensorqtl import mixqtl_replication as MX               # noqa: E402
from tensorqtl.hapmixqtl import map_nominal                  # noqa: E402

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
if any(k.startswith('PLASMODE_') for k in os.environ):   # renamed 2026-10-01; an ignored PLASMODE_ROOT would send outputs into the delivered root
    raise SystemExit('PLASMODE_* variables are set; they are now SIMULATED_EFFECTS_* (renamed 2026-10-01): '
                     + ', '.join(sorted(k for k in os.environ if k.startswith('PLASMODE_'))))
GENE_SET = os.environ.get('SIMULATED_EFFECTS_GENE_SET', 'corrected_null_store_20260925')   # the gene set this run uses: a key of GENE_SETS
ACCEPTANCE = os.environ.get('SIMULATED_EFFECTS_ACCEPTANCE') == '1'   # set by 99_acceptance.py for itself and the steps it runs: ROOT is then the set's acceptance_root
GENE_SETS = {   # per gene set: its directory under D (genes.txt, regions.bed, gene_design.tsv; for the default set also its stored
                # 200-permutation gibbs null run), this pipeline's output directory, 99_acceptance.py's output directory, the committed run of the previous code (scripts/plasmode/ before f0c0b07) that
                # 99_acceptance.py compares with and whose RASQUAL and TReCASE results stage_joint_results copies, and the stored runs of that
                # gene set the scripts read (None: the set has none, and each reader prints a skip); a new gene set adds an entry
    'corrected_null_store_20260925': dict(
        gene_dir='corrected_null_store_20260925',
        root='plasmode_meier_20260927',                                    # every output: the one directory this pipeline may write outside 99_acceptance.py (task of 2026-09-27, Meier's correction)
        acceptance_root='plasmode2_acceptance_20260927',                   # ROOT under 99_acceptance.py, so the acceptance never writes into a delivered run
        committed='plasmode_20260926',                                     # the previous code at commit 3aac315
        hybrid_null='hybrid_weights_null_20260926',                        # the stored split / unit / plus_one null runs, with the gibbs run in gene_dir (06 ANCHOR, 01 REPRO_DRAWS)
        before_df_fix='plasmode_20260926/summary_before_df_fix.json',      # the arms scored before commit 8a06803 (08 BEFORE; a record, not regenerable)
        df_fix='allelic_df_fix_20260927/summary.json',                     # the stored null re-run under 8a06803 (08 DF_FIX; scripts/allelic_df_null_check.py)
        trecase_smoke='plasmode_20260926/results_trecase_asseq/smoke/summary.json',    # the 2026-09-26 TReCASE smoke run (08: its largest theta gradient)
        ladder='ladder'),                                                  # 07's output directory under root (08 section 3.8)
    'stratum30_100': dict(   # 100 genes at 30-100 median haplotype-informative reads over admitted allelic donors (select_stratum_genes.py)
        gene_dir='plasmode_stratum30_100_20260927/gene_set',
        root='plasmode_lowcov_meier_20260927',
        acceptance_root='plasmode2_stratum_acceptance_20260927',
        committed='plasmode_stratum30_100_20260927',                       # the previous code, commits 15aac90 to d3247e0
        hybrid_null=None, before_df_fix=None, df_fix=None, trecase_smoke=None, ladder=None)}
GS = GENE_SETS[GENE_SET]
GENE_DIR = D / GS['gene_dir']
GENES, REGIONS, GENE_DESIGN = GENE_DIR / 'genes.txt', GENE_DIR / 'regions.bed', GENE_DIR / 'gene_design.tsv'
ROOT, HYBRID_NULL, BEFORE_DF_FIX, DF_FIX, TRECASE_SMOKE, COMMITTED = (
    D / GS[k] if GS[k] else None for k in ('acceptance_root' if ACCEPTANCE else 'root', 'hybrid_null', 'before_df_fix', 'df_fix',
                                           'trecase_smoke', 'committed'))
if os.environ.get('SIMULATED_EFFECTS_ROOT') and not ACCEPTANCE:   # a fresh output root instead of the gene set's delivered one
    ROOT = Path(os.environ['SIMULATED_EFFECTS_ROOT'])
DATASETS, RESULTS = ROOT / 'datasets', ROOT / 'results'
JOINT = {'rasqual': ROOT / 'results_rasqual', 'trecase': ROOT / 'results_trecase'}
NATIVE_COUNTS = D / 'native_counts_wasp_20260928'   # featureCounts totals and WASP-filtered strand-split exonic phASER haplotype counts from the STAR BAMs (scripts/native_counts.py)
NATIVE = ROOT / 'native'                       # 05b_native_arms.py: edger/, datasets/, results/, results_trecase/, trecase_work/, facts.json
NATIVE_DATASETS = NATIVE / 'datasets'
NATIVE_RESULTS = {'split_native': NATIVE / 'results', 'trecase_native': NATIVE / 'results_trecase'}   # the native-input arms (task 2026-09-28)
NATIVE_ARMS = tuple(NATIVE_RESULTS)
COMMITTED_JOINT = {'rasqual': COMMITTED / 'results_rasqual', 'trecase': COMMITTED / 'results_trecase_asseq'}
EIGENMT = ROOT / 'eigenmt_m_eff.tsv'   # 03: eigenMT's effective number of tests per gene over its tested variants (06 reads it)
CHECKS, SUMMARY, REPORT = ROOT / 'checks', ROOT / 'summary.json', ROOT / 'report'
LADDER = ROOT / GS['ladder'] if GS['ladder'] else None
SEED = 42                      # one master seed; every stream is SeedSequence(SEED, spawn_key=...)
KAPPA = 0.5                    # summaries_from_point_estimates' pseudocount
EXPRESSIBLE_MIN = 0.5          # reads; the zero-haplotype rule (docs/pipeline_rules.md)
EPS = 1e-12                    # allelic admission Va > EPS: hapmixqtl._zero_degenerate_ase_weights
ROUNDING_TOL = 1e-6            # Salmon sums: pT - pL - pR measured down to -1.8e-12 (point estimates), -1.0e-11 (draws), 2026-09-26
PACKAGES = ('numpy', 'scipy', 'pandas', 'pyarrow', 'torch', 'matplotlib', 'threadpoolctl')   # requirements.txt
HAPMIX_ARMS = ('gibbs', 'split', 'unit', 'plus_one')
CONFIG = {'split': 'hybrid', 'unit': 'unit', 'plus_one': 'plus_one'}   # config_variances names
MIXQTL_ARMS = {'mixqtl': MX.PUBLISHED_CUTOFFS, 'mixqtl_permissive': MX.PACKAGE_DEFAULT_CUTOFFS}   # mixqtl_replication.py:167-170
TENSORQTL = 'tensorqtl'        # tensorqtl.cis on the total phenotype T alone, unweighted (user request 2026-09-27)
ARMS = HAPMIX_ARMS + tuple(MIXQTL_ARMS) + (TENSORQTL,)
UNITS = {**{a: 'log2' for a in HAPMIX_ARMS + (TENSORQTL,) + NATIVE_ARMS}, **{a: 'natural log' for a in MIXQTL_ARMS}}   # mixQTL's response is natural log
DOF_COLS = ['dof_nominal', 'dof_a', 'dof_t', 'allelic_admitted']   # map_nominal's t references (commit 8a06803)
META_KEY, UNIT_KEY = b'plasmode_input_sha256', b'plasmode_slope_unit'


# ---------------------------------------------------------------------------------------------------------------------------
#  Inputs and helpers this benchmark carries itself, so that it imports nothing from scripts/ and only public names from
#  tensorqtl. Copied on 2026-10-01 from compare_mixqtl_replication.py (load_point_estimate_inputs with the part of
#  load_inputs it keeps, gene_variant_index, mixqtl_gene), run_hapmixqtl_from_salmon.py (read_phased_vcf, read_edger_dir),
#  corrected_null_store.py (OLD, rates_by_gene), hybrid_weights_null.py (config_variances), null_permutation_instrument.py
#  and se_fixes.py (fit_channels, resid_out), compare_pipelines.py (RASQUAL_FIELDS) and tensorqtl/mixqtl_replication.py
#  (_stack_covariates); the refactoring check of that day reproduced every output file.
# ---------------------------------------------------------------------------------------------------------------------------

CACHE = D / 'cache' / 'gibbs_56b63c3b37ed5df8'     # the Gibbs cache: genes.txt, samples.txt, YL/YR/YT.npy
PE = CACHE / 'point_estimates'                     # pL/pR/pT.npy and edger/ (effective library sizes, eQTL gene filter)
COV = D / 'cov' / 'half_read_point_calibration_20260930'   # covariates.tsv and genotype_covariates.txt, half-read unit
VCF = D / 'prepped' / 'analysis.snps.maf01.vcf.gz'
GENE_TABLE = D / 'annot' / 'genes.tsv'
WIN, MAF = 1_000_000, 0.05                         # cis window (bp) and the tested set's MAF floor
OLD = D / 'protein_coding_null_store_20260925'     # the stored 2,000-permutation null (01 check d)
ALPHAS = (0.05, 0.01, 0.001)
CHANNELS = {'combined': 'pval_nominal', 'allelic': 'pval_a', 'total': 'pval_t'}
COLS = ['phenotype_id', 'variant_id', 'pval_nominal', 'slope', 'slope_se',
        'pval_a', 'slope_a', 'slope_a_se', 'pval_t', 'slope_t', 'slope_t_se']
RASQUAL_FIELDS = [
    'feature_id', 'rs_id', 'chrom', 'snp_pos', 'ref', 'alt',
    'allele_frequency', 'hwe_chisq', 'imputation_quality_ia',
    'log10_bh_qvalue', 'chisq', 'effect_size_pi', 'error_rate_delta',
    'ref_mapping_bias_phi', 'overdispersion_theta', 'snp_id_in_region',
    'n_feature_snps', 'n_tested_snps', 'n_iter_null', 'n_iter_alt',
    'tie_lead_snp', 'loglik_null', 'convergence', 'r2_prior_posterior_fsnps',
    'r2_prior_posterior_rsnp',
]


def read_phased_vcf(path, want_samples, regions=None, bcftools='bcftools'):
    """variant_df, dosage[V,N], xL[V,N], xR[V,N] and the sample order for the phased biallelic SNPs of the VCF, fetched
    by bcftools for the BED `regions` when given."""
    if regions is not None:
        proc = subprocess.Popen([bcftools, 'view', '-R', str(regions), str(path)],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        return _parse_phased_vcf(proc.stdout, want_samples)
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt') as fh:
        return _parse_phased_vcf(fh, want_samples)


def _parse_phased_vcf(fh, want_samples):
    ids, chroms, poss, refs, alts = [], [], [], [], []
    XL, XR = [], []
    order = None
    for line in fh:
        if line.startswith('##'):
            continue
        f = line.rstrip('\n').split('\t')
        if line.startswith('#CHROM'):
            vcf_samples = f[9:]
            keep = [i for i, s in enumerate(vcf_samples) if s in want_samples]
            if not keep:
                raise SystemExit(
                    'no VCF samples matched the manifest.\n'
                    f'  VCF: {vcf_samples[:4]}\n  manifest: {list(want_samples)[:4]}')
            order = [vcf_samples[i] for i in keep]
            continue
        if len(f) < 10 or len(f[3]) != 1 or len(f[4]) != 1 or ',' in f[4]:
            continue                                    # biallelic SNPs only
        gt_i = f[8].split(':').index('GT') if 'GT' in f[8] else 0
        xl = np.zeros(len(keep), np.int8); xr = np.zeros(len(keep), np.int8)
        ok = True
        for k, i in enumerate(keep):
            gt = f[9 + i].split(':')[gt_i]
            if '|' not in gt:
                ok = False; break                       # unphased -> skip
            a, b = gt.split('|')[:2]
            if a in '.' or b in '.':
                ok = False; break
            xl[k] = 1 if a != '0' else 0
            xr[k] = 1 if b != '0' else 0
        if not ok:
            continue
        # a joined (split multi-allelic, "a;b") or missing ID names no single record
        ids.append(f[2] if (f[2] != '.' and ';' not in f[2])
                   else f'{f[0]}_{f[1]}_{f[3]}_{f[4]}')
        chroms.append(f[0]); poss.append(int(f[1])); refs.append(f[3]); alts.append(f[4])
        XL.append(xl); XR.append(xr)
    if not ids:
        raise SystemExit('no phased biallelic SNPs read from the VCF')
    XL = np.array(XL, np.int8); XR = np.array(XR, np.int8)
    vdf = pd.DataFrame({'chrom': [str(c) for c in chroms], 'pos': poss,
                        'ref': refs, 'alt': alts}, index=ids)
    return vdf, XL + XR, XL, XR, order


def read_edger_dir(edger_dir, samples):
    """(effective library sizes lib.size x TMM aligned to `samples`, the edgeR-kept gene list)."""
    d = Path(edger_dir)
    es = pd.read_csv(d / 'edger_samples.tsv', sep='\t', dtype={'sample': str}).set_index('sample')
    missing = [s for s in samples if s not in es.index]
    if missing:
        raise SystemExit(f'{len(missing)} samples have no edgeR library size in {d}, e.g. {missing[:3]}')
    return es.loc[list(samples), 'eff_lib_size'].to_numpy(float), (d / 'calibration_genes.txt').read_text().split()


def load_point_estimate_inputs(gene_list, regions, cov=COV):
    """The gene set's inputs under the 2026-09-25 pipeline rules: Gibbs draws YL/YR/YT and Salmon point estimates
    pL/pR/pT [genes, cache samples], phased genotypes of the cis windows (`regions`) with the tested rows `idx` (in a
    window, outside the gene body, MAF >= MAF), edgeR effective library sizes (`eff_lib` in cache order, `lib_size` in
    VCF order, `keep` mapping VCF to cache order), and the RNA-tied (`cov_df`) and genotype-tied (`geno_cov_df`)
    covariates of `cov`. Refuses genes outside the eQTL gene filter the expression PCs were built on."""
    genes_all = (CACHE / 'genes.txt').read_text().split()
    samples = (CACHE / 'samples.txt').read_text().split()
    genes = [line.strip() for line in open(gene_list) if line.strip()]
    cal = set((PE / 'edger' / 'calibration_genes.txt').read_text().split())
    off = [g for g in genes if g not in cal]
    if off:
        raise SystemExit(f'{len(off)} genes fail the eQTL gene filter the expression PCs '
                         f'were built on, e.g. {off[:5]}')
    gi = {g: i for i, g in enumerate(genes_all)}
    rows = [gi[g] for g in genes]
    I = {k: np.asarray(np.load(CACHE / f'{k}.npy', mmap_mode='r')[rows]) for k in ('YL', 'YR', 'YT')}
    gp = pd.read_csv(GENE_TABLE, sep='\t', header=None, dtype={1: str})
    gp.columns = ['gene', 'chr', 'start', 'end', 'pos']
    gp = gp.set_index('gene')
    gp['chr'] = gp['chr'].astype(str).str.strip()
    vdf, dos, xL, xR, order = read_phased_vcf(VCF, set(samples), regions=regions)
    keep = [samples.index(s) for s in order]
    order = list(order)
    vdf['chrom'] = vdf['chrom'].astype(str)
    pos, ch = vdf['pos'].values, vdf['chrom'].values
    in_body = np.zeros(len(vdf), bool)
    in_win = np.zeros(len(vdf), bool)
    for g in genes:
        r = gp.loc[g]
        same = ch == str(r['chr'])
        in_body |= same & (pos >= int(r['start'])) & (pos <= int(r['end']))
        in_win |= same & (np.abs(pos - int(r['pos'])) <= WIN)
    af = dos.mean(1) / 2.0
    idx = np.where(in_win & ~in_body & (np.minimum(af, 1 - af) >= MAF))[0]
    for k in ('pL', 'pR', 'pT'):
        I[k] = np.asarray(np.load(PE / f'{k}.npy', mmap_mode='r')[rows])
    eff_lib, _ = read_edger_dir(PE / 'edger', samples)
    cov_all = pd.read_csv(Path(cov) / 'covariates.tsv', sep='\t', index_col=0)
    cov_all.index = cov_all.index.astype(str)
    cov_all = cov_all.loc[order]
    gcols = (Path(cov) / 'genotype_covariates.txt').read_text().split()
    I.update(genes=genes, order=order, keep=keep, vdf=vdf, dos=dos, xL=xL, xR=xR, idx=idx, gp=gp,
             eff_lib=eff_lib, lib_size=eff_lib[keep], geno_cov_df=cov_all[gcols], cov_df=cov_all.drop(columns=gcols))
    return I


def gene_variant_index(I, g):
    """Variant rows within I['idx'] that fall in gene g's cis window."""
    r = I['gp'].loc[g]
    v = I['vdf'].iloc[I['idx']]
    same = v['chrom'].values == str(r['chr'])
    return np.where(same & (np.abs(v['pos'].values - int(r['pos'])) <= WIN))[0]


def mixqtl_gene(I, g, j, y1, y2, yt, perm=None):
    """mixQTL mode (tensorqtl.mixqtl_replication.mixqtl_scan) on one gene: its lead's statistic, slope and counts."""
    vsel = gene_variant_index(I, g)
    if vsel.size == 0:
        return None
    keep = I['keep']
    h1 = I['xL'][I['idx']][:, keep][vsel].T.astype(float)   # [N, P]
    h2 = I['xR'][I['idx']][:, keep][vsel].T.astype(float)
    cov = I['cov_df'].values
    G = I['geno_cov_df'].values if I.get('geno_cov_df') is not None else None
    lib = I['lib_size']
    a1, a2, at = y1[j], y2[j], yt[j]
    if perm is not None:
        # the RNA record, its covariates and its library size move; the genotype PCs stay with the genotypes
        a1, a2, at = a1[perm], a2[perm], at[perm]
        cov = cov[perm]
        lib = lib[perm]
    out = MX.mixqtl_scan(a1, a2, at, lib, h1, h2, covariates=cov, genotype_covariates=G)
    stat = np.abs(out['meta']['stat'])
    if not np.isfinite(stat).any():
        return None
    k = int(np.nanargmax(stat))
    v = I['vdf'].iloc[I['idx']].iloc[vsel]
    return dict(gene=g, stat=float(stat[k] ** 2),
                variant_id=str(v.index[k]),
                beta=float(out['meta']['beta'][k]),
                se=float(out['meta']['se'][k]),
                method=str(out['meta']['method'][k]),
                n_trc=int(out['trc']['sample_size']),
                n_asc=int(out['asc']['sample_size']),
                n_cov_selected=int(out['cov_selected'].sum())
                if out['cov_selected'] is not None else 0,
                num_var=int(np.isfinite(stat).sum()))


def stack_covariates(covariates, genotype_covariates):
    """RNA-tied columns first, then the genotype-tied ones; None if neither."""
    parts = [np.asarray(c, float) for c in (covariates, genotype_covariates) if c is not None]
    if not parts:
        return None
    if len({p.shape[0] for p in parts}) != 1:
        raise ValueError(f'covariate blocks have different sample counts: {[p.shape for p in parts]}')
    return np.column_stack(parts)


def config_variances(config, Va, Vt):
    """(allelic, total) working variances; Va is 0 where a pair is excluded."""
    if config == 'hybrid':
        return Va, np.ones_like(Vt)
    if config == 'plus_one':
        return np.where(Va > EPS, Va + 1.0, 0.0), Vt + 1.0
    if config == 'unit':
        return np.where(Va > EPS, 1.0, 0.0), np.ones_like(Vt)
    raise SystemExit(f'unknown config {config}')


def rates_by_gene(files, genes, col, gene_filter=None):
    """Per gene, the count of finite `col` p-values below each of ALPHAS and the count of finite ones, over `files`."""
    gix = {g: i for i, g in enumerate(genes)}
    K = {al: np.zeros(len(genes)) for al in ALPHAS}
    n = np.zeros(len(genes))
    for fp in files:
        d = pd.read_parquet(fp, columns=['phenotype_id', col])
        if gene_filter is not None:
            d = d[d.phenotype_id.isin(gene_filter)]
        d = d[np.isfinite(d[col])]
        gi = d.phenotype_id.map(gix).values
        n += np.bincount(gi, minlength=len(genes))
        for al in ALPHAS:
            K[al] += np.bincount(gi, weights=(d[col].values < al), minlength=len(genes))
    return K, n


def resid_out(X, Z, w):
    """Residualize columns of X on Z in the sqrt(w)-weighted space."""
    sw = np.sqrt(w)[:, None]
    Xw, Zw = X * sw, Z * sw
    if Zw.shape[1] == 0:
        return Xw
    q, _ = np.linalg.qr(Zw)
    return Xw - q @ (q.T @ Xw)


def fit_channels(a, s, va, t, g, vt, C):
    """Per channel, the through-origin weighted allelic fit and the weighted total fit on [1, C], each with its fitted
    residual scale, and their inverse-variance combination; None where either channel cannot be fitted."""
    ka = np.isfinite(a) & np.isfinite(va) & (va > EPS) & np.isfinite(s)
    kt = np.isfinite(t) & np.isfinite(vt) & (vt > EPS) & np.isfinite(g)
    if ka.sum() < 5 or kt.sum() < 10:
        return None
    wa = 1.0 / va[ka]
    ya, xa = a[ka] * np.sqrt(wa), s[ka] * np.sqrt(wa)
    xxa = float(xa @ xa)
    if xxa <= 0:
        return None
    ba = float(xa @ ya) / xxa
    ea = ya - ba * xa
    dofa = max(int(ka.sum()) - 1, 1)
    sea2 = float(ea @ ea) / dofa / xxa
    wt = 1.0 / vt[kt]
    Z = np.column_stack([np.ones(kt.sum()), C[kt]])
    yt = resid_out(t[kt][:, None], Z, wt).ravel()
    xt = resid_out(g[kt][:, None], Z, wt).ravel()
    xxt = float(xt @ xt)
    if xxt <= 0:
        return None
    bt = float(xt @ yt) / xxt
    et = yt - bt * xt
    doft = max(int(kt.sum()) - 1 - Z.shape[1], 1)
    set2 = float(et @ et) / doft / xxt
    if not (np.isfinite(sea2) and np.isfinite(set2)) or sea2 <= 0 or set2 <= 0:
        return None
    prec = 1 / sea2 + 1 / set2
    b = (ba / sea2 + bt / set2) / prec
    se2 = 1 / prec
    dof = min(dofa, doft)
    return dict(ba=ba, sea=np.sqrt(sea2), dofa=dofa, n_a=int(ka.sum()),
                bt=bt, set=np.sqrt(set2), doft=doft, n_t=int(kt.sum()),
                b=b, se=np.sqrt(se2), dof=dof,
                t2_a=ba ** 2 / sea2, t2_t=bt ** 2 / set2, t2_b=b ** 2 / se2)


def module(name):
    """Import a numbered script of this directory (e.g. '06_score')."""
    sys.path.insert(0, str(HERE))
    return importlib.import_module(name)


def write_atomic(path, write, mode='wb'):
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, mode) as fh:
        write(fh)
    os.replace(tmp, path)


def dumps(obj):
    """JSON with NaN written as null; any other non-finite value stops the write."""
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        return None if isinstance(x, float) and np.isnan(x) else x
    return json.dumps(clean(obj), indent=1, allow_nan=False)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, lambda fh: fh.write(dumps(obj)), 'w')


def quiet(fn, *args, **kwargs):
    """Call fn with stdout captured; forward only its WARNING lines."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = fn(*args, **kwargs)
    for line in buf.getvalue().splitlines():
        if 'WARNING' in line:
            print(f'{fn.__name__}: {line.strip()}', flush=True)
    return res


def versions():
    """Python, the pinned packages, the repository HEAD and working-tree state and the sha256 of every script here,
    written to ROOT/versions.log (run_all.sh and 99_acceptance.py); the lines."""
    git = lambda *a: subprocess.run(['git', '-C', str(HERE), *a], capture_output=True, text=True, check=True).stdout.strip()   # noqa: E731
    scripts = sorted(p for p in HERE.iterdir() if p.suffix in ('.py', '.R', '.sh'))
    lines = [time.strftime('%Y-%m-%d %H:%M:%S'), f'repo {git("rev-parse", "HEAD")}; git status --porcelain:', git('status', '--porcelain'),
             *(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p}' for p in scripts), f'python {sys.version.split()[0]}',
             *(f'{p}=={importlib.metadata.version(p)}' for p in PACKAGES)]
    ROOT.mkdir(parents=True, exist_ok=True)
    write_atomic(ROOT / 'versions.log', lambda fh: fh.write('\n'.join(lines) + '\n'), 'w')
    return lines


def load():
    """The cache inputs of GENE_SET, validated once: I (the loader's dict), R (arrays over the kept donors),
    tested (each gene's tested variant rows, the loader's idx set within its cis window). Counts are non-negative and
    the total is at least the paired haplotypes (to ROUNDING_TOL) in the point estimates and in every Gibbs draw;
    02's thinning and truth rely on both without re-checking (moved records are column permutations of R)."""
    I = load_point_estimate_inputs(gene_list=str(GENES), regions=str(REGIONS))
    keep = I['keep']
    R = {k: I[k][:, keep] for k in ('pL', 'pR', 'pT', 'YL', 'YR', 'YT')}
    R['eff_lib'] = I['eff_lib'][keep]
    tested = [I['idx'][gene_variant_index(I, g)] for g in I['genes']]
    nt = np.array([len(t) for t in tested])
    G, N = R['pL'].shape
    if (nt == 0).any():
        raise SystemExit(f'{int((nt == 0).sum())} genes have no tested variant in {REGIONS}')
    for k in ('pL', 'pR', 'YL', 'YR'):
        if R[k].min() < 0:
            raise SystemExit(f'negative Salmon count in {k}: {R[k].min()}')
    rem = {what: float((R[t] - R[l] - R[r]).min()) for what, (l, r, t) in
           (('point estimates', ('pL', 'pR', 'pT')), ('Gibbs draws', ('YL', 'YR', 'YT')))}
    for what, u in rem.items():
        if u < -ROUNDING_TOL:
            raise SystemExit(f'{what}: total minus paired haplotypes is {u:.3g} < -{ROUNDING_TOL}')
    print(f'read {G} genes x {N} donors x {R["YL"].shape[2]} Gibbs draws from {GENES}; {len(I["vdf"]):,} variants, '
          f'{len(I["idx"]):,} tested; tested variants per gene {nt.min()} / {int(np.median(nt))} / {nt.max()}; '
          f'min total minus paired haplotypes {rem["point estimates"]:.1e} (point estimates), {rem["Gibbs draws"]:.1e} (draws)',
          flush=True)
    return I, R, tested


def setup(I):
    """Genotype frames, covariates and tested variants in the loader's gene and donor order; the tested set's phased
    alleles are 0 or 1 and its ALT dosage is xL + xR (checked here once; 02 and 07 rely on it)."""
    genes, order, idx = list(I['genes']), list(I['order']), I['idx']
    vdf = I['vdf'].iloc[idx]
    if not vdf.index.is_unique:
        raise SystemExit('variant ids in the tested set are not unique')
    xL, xR = I['xL'][idx], I['xR'][idx]
    if not (np.isin(xL, (0, 1)).all() and np.isin(xR, (0, 1)).all() and np.array_equal(I['dos'][idx], xL + xR)):
        raise SystemExit('tested set: phased alleles outside {0, 1} or ALT dosage differing from xL + xR')
    frame = lambda M: pd.DataFrame(M[idx], index=vdf.index, columns=order)   # noqa: E731
    tested_rows = {g: idx[gene_variant_index(I, g)] for g in genes}
    tested = {g: set(I['vdf'].index[r].astype(str)) for g, r in tested_rows.items()}
    dos = I['dos'][idx]
    constant = set(vdf.index[(dos == dos[:, [0]]).all(1)].astype(str))
    return dict(I=I, genes=genes, order=order, vdf=vdf, gdf=frame(I['dos']), xLdf=frame(I['xL']), xRdf=frame(I['xR']),
                gp=I['gp'].loc[genes][['chr', 'pos']], tested=tested, tested_rows=tested_rows,
                n_tested=pd.Series({g: len(v) for g, v in tested.items()}),
                scanned={g: v - constant for g, v in tested.items()},
                rows={str(v): i for i, v in enumerate(I['vdf'].index)})


def runs(meta):
    """[(scenario, dataset index)] of a datasets directory's meta.json."""
    return [(f'beta{b}', r) for b in meta['betas'] for r in range(meta['n_datasets'][str(b)])]


def load_dataset(datasets, sc, r):
    return dict(np.load(datasets / sc / f'rep{r:03d}.npz'))


def allelic_kept(pL, pR, Va):
    """Pairs the allelic channel fits: Va > EPS and not exactly one haplotype below EXPRESSIBLE_MIN."""
    return (Va > EPS) & ~((pL < EXPRESSIBLE_MIN) ^ (pR < EXPRESSIBLE_MIN))


def arm_variances(ds, arm):
    """(allelic, total) working variances of an arm after the allelic admission, and the pairs it zeroed. split_native
    (native counts) admits every pair with a + b > 0: the zero-haplotype rule answers Salmon's point estimate putting one
    copy at exactly zero (docs/pipeline_rules.md), while on alignment counts a one-sided zero is a counting outcome."""
    if arm == 'split_native':
        return (*config_variances(CONFIG['split'], ds['Va'], ds['Vt']), 0)
    kept = allelic_kept(ds['pL'], ds['pR'], ds['Va'])
    Va = np.where(kept, ds['Va'], 0.0)
    n_zeroed = int(((ds['Va'] > EPS) & ~kept).sum())
    if arm == 'gibbs':
        return Va, ds['Vt'], n_zeroed
    return (*config_variances(CONFIG[arm], Va, ds['Vt']), n_zeroed)


def fingerprint(ds, arm):
    """sha256 of the dataset arrays and the arm name, stored in every results file and checked by score."""
    h = hashlib.sha256(arm.encode())
    for k in ('A', 'T', 'Va', 'Vt', 'pL', 'pR', 'pT', 'eff_lib', 'perm', 'swap', 'causal_variant'):
        h.update(np.ascontiguousarray(ds[k]).tobytes())
    return h.hexdigest()


def write_parquet(df, path, sha, unit):
    tab = pa.Table.from_pandas(df, preserve_index=False)
    tab = tab.replace_schema_metadata({**tab.schema.metadata, META_KEY: sha.encode(), UNIT_KEY: unit.encode()})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    pq.write_table(tab, tmp, compression='zstd')
    tmp.rename(path)


def stored_fingerprint(path):
    return pq.read_schema(path).metadata[META_KEY].decode()


def read_results(path, columns):
    """A results file with every slope and se in log2 units (mixQTL's natural-log columns divided by ln 2)."""
    d = pd.read_parquet(path, columns=columns)
    unit = pq.read_schema(path).metadata[UNIT_KEY].decode()
    if unit == 'natural log':
        for c in columns:
            if c.startswith('slope'):
                d[c] = d[c].astype(float) / np.log(2.0)
    elif unit != 'log2':
        raise SystemExit(f'{path}: unknown slope unit {unit!r}')
    d['variant_id'] = d['variant_id'].astype(str)
    return d


def phenotypes(S, ds, arm):
    """map_nominal's frames for an arm: A, T, working Va and Vt, and the RNA-tied covariates in record order."""
    Va, Vt, n_zeroed = arm_variances(ds, arm)
    ph = lambda M: pd.DataFrame(M, index=S['genes'], columns=S['order'])   # noqa: E731
    cov = pd.DataFrame(S['I']['cov_df'].values[ds['perm']], index=S['order'], columns=S['I']['cov_df'].columns)
    return ph(ds['A']), ph(ds['T']), ph(Va), ph(Vt), cov, n_zeroed


def run_nominal(S, ds, arm, scratch, keep_a=None, keep_t=None):
    """map_nominal in default mode on the arm's inputs (allelic channel through the origin, genotype PCs in place);
    the tested pairs with COLS + DOF_COLS, and the donor-gene pairs the allelic admission zeroed."""
    A, T, Va, Vt, cov, n_zeroed = phenotypes(S, ds, arm)
    scratch.mkdir(parents=True, exist_ok=True)
    for q in scratch.glob('*'):
        q.unlink()
    masks = {} if keep_a is None else dict(keep_a_df=pd.DataFrame(keep_a, index=A.index, columns=A.columns),
                                           keep_t_df=pd.DataFrame(keep_t, index=A.index, columns=A.columns))
    quiet(map_nominal, S['gdf'], S['vdf'][['chrom', 'pos']], A, T, Va, Vt, S['gp'], xL_df=S['xLdf'], xR_df=S['xRdf'],
          prefix='n', covariates_df=cov, genotype_covariates_df=S['I']['geno_cov_df'], window=WIN,
          output_dir=str(scratch), verbose=False, ase_covariates_df=None, **masks)
    df = pd.concat([pd.read_parquet(q, columns=COLS + DOF_COLS) for q in sorted(scratch.glob('n*.parquet'))],
                   ignore_index=True)
    df['variant_id'] = df['variant_id'].astype(str)
    per = df.groupby('phenotype_id').size().reindex(S['genes'])
    if not per.equals(S['n_tested'].reindex(S['genes'])):
        raise SystemExit(f'map_nominal returned {len(df):,} rows against {int(S["n_tested"].sum()):,} tested pairs')
    return df, n_zeroed


def stage_joint_results():
    """The committed run's RASQUAL and TReCASE results (COMMITTED_JOINT) into JOINT, copied once, with a summary.json each
    (RASQUAL's derived from the committed run's log, which alone holds its counts): 99_acceptance.py, and run_all.sh with
    the argument staged in place of 04 and 05 (hours each)."""
    for arm, src in COMMITTED_JOINT.items():
        for f in sorted(src.glob(f'beta*/{arm}/nominal_rep*.parquet')):
            dst = JOINT[arm] / f.parent.parent.name / arm / f.name
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                write_atomic(dst, lambda fh, f=f: fh.write(f.read_bytes()))
    src = COMMITTED_JOINT['trecase'] / 'summary.json'
    write_atomic(JOINT['trecase'] / 'summary.json', lambda fh: fh.write(src.read_bytes()))
    log = (COMMITTED_JOINT['rasqual'] / 'run_rasqual.log').read_text()
    per = {}
    for k, *m in re.findall(r'(?m)^(beta\S+ rep \d+): ([\d,]+) rows written; excluded non-converged (\d+), pseudo-fSNP '
                            r'rows (\d+); tested variants with no RASQUAL row (\d+); chisq <= 0 \(slope_se NaN\) (\d+); '
                            r'genes where RASQUAL did not admit the pseudo fSNP (\d+); non-null causal variants without a '
                            r'row: non-converged (\d+), absent (\d+) \(of (\d+)\)', log):
        v = [int(x.replace(',', '')) for x in m]
        per[k] = dict(zip(('rows', 'nonconv', 'pseudo', 'absent', 'chisq_le0', 'no_fsnp', 'causal_nonconv', 'causal_absent',
                           'causal_nonnull'), v))
        per[k]['tests'] = per[k]['rows'] + per[k]['nonconv'] + per[k]['absent']
    for k, h, inf, z in re.findall(r'(?m)^(beta\S+ rep \d+): \d+ covariates; pseudo fSNP (\d+) het of (\d+) informative '
                                   r'pairs.*AS 0,0 (\d+)$', log):
        per[k].update(het=int(h), informative=int(inf), as00=int(z))
    if len(per) != 10 or not all('het' in v for v in per.values()):
        raise SystemExit(f'{COMMITTED_JOINT["rasqual"]}/run_rasqual.log: {len(per)} dataset lines parsed')
    pooled = {k: sum(v[k] for v in per.values()) for k in ('rows', 'tests', 'nonconv', 'absent', 'chisq_le0', 'no_fsnp',
                                                             'causal_nonconv', 'causal_absent', 'causal_nonnull')}
    write_json(JOINT['rasqual'] / 'summary.json', dict(per_dataset=per, pooled=pooled,
                                                       source=str(COMMITTED_JOINT['rasqual'] / 'run_rasqual.log')))
    print(f'staged the committed joint results of {COMMITTED} into {ROOT}; RASQUAL pooled {pooled}', flush=True)
