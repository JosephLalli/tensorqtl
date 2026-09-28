"""Paths, parameters and helpers shared by the plasmode benchmark scripts (run order: run_all.sh).

Design: docs/simulation_benchmark_spec.md (superseded header) and README.md here. Every
parameter is written once, in this file or at the top of the script that owns it.
"""
import contextlib
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
sys.path.insert(0, str(HERE.parents[0]))   # scripts/: the loader and the stored-null helpers
sys.path.insert(0, str(HERE.parents[1]))   # the repository: tensorqtl

import compare_mixqtl_replication as CM                      # noqa: E402
import corrected_null_store as CNS                           # noqa: E402
from hybrid_weights_null import config_variances             # noqa: E402
from tensorqtl import mixqtl_replication as MX               # noqa: E402
from tensorqtl.hapmixqtl import map_nominal                  # noqa: E402

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
GENE_SET = os.environ.get('PLASMODE_GENE_SET', 'corrected_null_store_20260925')   # the gene set this run uses: a key of GENE_SETS
ACCEPTANCE = os.environ.get('PLASMODE_ACCEPTANCE') == '1'   # set by 99_acceptance.py for itself and the steps it runs: ROOT is then the set's acceptance_root
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
DATASETS, RESULTS = ROOT / 'datasets', ROOT / 'results'
JOINT = {'rasqual': ROOT / 'results_rasqual', 'trecase': ROOT / 'results_trecase'}
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
CONFIG = {'split': 'hybrid', 'unit': 'unit', 'plus_one': 'plus_one'}   # hybrid_weights_null.config_variances names
MIXQTL_ARMS = {'mixqtl': MX.PUBLISHED_CUTOFFS, 'mixqtl_permissive': MX.PACKAGE_DEFAULT_CUTOFFS}   # mixqtl_replication.py:167-170
TENSORQTL = 'tensorqtl'        # tensorqtl.cis on the total phenotype T alone, unweighted (user request 2026-09-27)
ARMS = HAPMIX_ARMS + tuple(MIXQTL_ARMS) + (TENSORQTL,)
UNITS = {**{a: 'log2' for a in HAPMIX_ARMS + (TENSORQTL,)}, **{a: 'natural log' for a in MIXQTL_ARMS}}   # mixQTL's response is natural log
COLS, CHANNELS, ALPHAS = CNS.COLS, CNS.CHANNELS, CNS.ALPHAS
DOF_COLS = ['dof_nominal', 'dof_a', 'dof_t', 'allelic_admitted']   # map_nominal's t references (commit 8a06803)
META_KEY, UNIT_KEY = b'plasmode_input_sha256', b'plasmode_slope_unit'


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
    I = CM.load_point_estimate_inputs(gene_list=str(GENES), regions=str(REGIONS))
    keep = I['keep']
    R = {k: I[k][:, keep] for k in ('pL', 'pR', 'pT', 'YL', 'YR', 'YT')}
    R['eff_lib'] = I['eff_lib'][keep]
    tested = [I['idx'][CM.gene_variant_index(I, g)] for g in I['genes']]
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
    tested_rows = {g: idx[CM.gene_variant_index(I, g)] for g in genes}
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
    """(allelic, total) working variances of an arm after the allelic admission, and the pairs it zeroed."""
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
          prefix='n', covariates_df=cov, genotype_covariates_df=S['I']['geno_cov_df'], window=CM.WIN,
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
