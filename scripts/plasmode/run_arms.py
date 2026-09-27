"""Map every plasmode dataset under six arms: four hapmixQTL weightings
(map_nominal, and map_cis for gene-level permutation p on every dataset) and
mixQTL mode at two cutoff settings (mixqtl_scan, and mixqtl_permutation_scan
under the timing rule below).

Datasets come from make_datasets.py (records_signflip permutation of the real
cohort's Salmon records, effects injected by binomial thinning).

hapmixQTL ARMS, per dataset:
  1. Allelic admission from the dataset's own thinned point estimates
     (make_datasets.allelic_kept): Va := 0 where Va' <= EPS or exactly one of
     pL', pR' is below 0.5 reads (the state of a pair with no allelic
     information; corrected_null_store.py's `drop` rule). The total channel
     is untouched. The number of donor-gene pairs zeroed is printed per run.
  2. The arm's working variances:
       gibbs     (Va', Vt), Gibbs variance in both channels (the shipped default)
       split     hybrid_weights_null.config_variances('hybrid'): Gibbs variance
                 in the allelic channel, unit variance in the total channel
       unit      config_variances('unit'): 1 for every included pair
       plus_one  config_variances('plus_one'): v + 1 in both channels
  3. map_nominal as hybrid_weights_null.py calls it: the RNA-tied covariate rows
     in the dataset's permuted order (row i is real record perm[i]), the
     genotype PCs in place, allelic channel through the origin
     (ase_covariates_df=None), window CM.WIN, default mode. A is already
     swapped in the dataset (swapped records had pL and pR exchanged), so no
     sign is applied here. Stored as map_nominal writes it (p-values float64,
     slopes and se float32, map_nominal's own dtypes), with DOF_COLS: each
     p's t reference and whether the gene's allelic channel entered the
     combination (hapmixqtl.MIN_ALLELIC_DONORS = 15 informative allelic
     donors; below it the combined slope, se and p are the total channel's).
     The log line names the genes below that floor.
  GATE, per run: at each gene's causal variant, slope_a and slope_t equal
  null_permutation_instrument.fit_channels on the same arm inputs within
  GATE_TOL of the slope's se (corrected_null_store.py's gate; map_nominal
  computes in float32). A non-finite ratio stops the script, naming the gene
  and field.

mixQTL ARMS (mixQTL mode, the no-draws comparator), per dataset and gene:
  mixqtl             MX.PUBLISHED_CUTOFFS, the module defaults
                     (tensorqtl/mixqtl_replication.py:167-168, bound as
                     TRC_CUTOFF 100 / ASC_CUTOFF 50 / ASC_CAP 1000 /
                     WEIGHT_CAP 10 at :179-182; the GTEx v8 driver's values)
  mixqtl_permissive  MX.PACKAGE_DEFAULT_CUTOFFS (mixqtl_replication.py:169-170:
                     trc_cutoff 20, asc_cutoff 5, weight_cap 100, asc_cap 5000,
                     the R signature), named 'permissive' in
                     scripts/comparator_null_2000.py:115
  Inputs follow compare_mixqtl_replication.mixqtl_gene with the permutation
  already applied by the dataset: MX.inputs_from_point_estimates on the
  dataset's thinned pL', pR', pT' (never the draws); lib_size = the dataset's
  eff_lib (record order); covariates = cov_df rows in the order perm;
  genotype PCs in place; h1 / h2 = xL / xR of the gene's tested variants, in
  place. mixqtl_gene returns only the lead, so MX.mixqtl_scan is called here
  and mixqtl_gene is the gate. Stored per tested variant in
  corrected_null_store COLS: meta -> pval_nominal, slope, slope_se; asc ->
  pval_a, slope_a, slope_a_se; trc -> pval_t, slope_t, slope_t_se; plus
  `method` (meta, trc or asc: which estimate the meta columns hold when one
  channel has fewer than MX.META_N_CUTOFF samples). Each p is the module's
  own reference (normal above META_N_CUTOFF samples, t below). Slopes and se
  are in NATURAL LOG, as the module returns them (mixQTL's response is
  natural log by design); the parquet metadata records the unit and score.py
  divides them by ln 2.
  CAVEAT: the published asc_cap = 1000 makes allelic admission depend on the
  injected effect, because thinning pulls heterozygous records carrying the
  lower-expressed allele down into [50, 1000].
  GATE (published arm, every dataset): CM.mixqtl_gene on a copy of the loader
  inputs with cov_df rows in the order perm, lib_size = eff_lib and the
  variant rows restricted to the gene's tested set returns, per gene, the
  lead (largest |meta stat|); its variant, beta and se must equal this
  scan's exactly. mixqtl_gene takes no cutoffs, so the permissive arm shares
  the gated construction but is not gated itself.

TESTED VARIANTS. map_nominal is given genotype frames restricted to the
pipeline's tested set CM idx: in a cis window of one of the 100 genes,
outside every gene body, MAF >= 0.05 over the 92 donors. Within a gene's
window (inclusive, |pos - TSS| <= CM.WIN in both CM.gene_variant_index and
genotypeio.get_cis_ranges) this is exactly the tested set, so map_nominal
returns the tested pairs and nothing else (row count checked per run); the
mixQTL arms are given the same rows. Per-variant regressions do not depend on
other variants, so the tested rows are those of a full-frame run as
hybrid_weights_null.py makes it: checked on a smoke run (beta 0.8, rep 0,
gibbs), allelic columns identical, combined and total within 3.6e-6 se
(float32 batching), no rejection at 0.05 / 0.01 / 0.001 changed among 487,454
tests. 507 tested variants (0.10% of tested pairs; TPPP 473 of 7,333, NUDT4B
29, MBOAT7 4, CCT6A 1) have every donor heterozygous, ALT dosage 1 in all 92:
the allelic channel is informative there and the total channel is not
(map_nominal and mixQTL's trc channel report no total estimate).

map_cis, EVERY DATASET, the four hapmixQTL arms (user decision 2026-09-26):
gene-level pval_perm and pval_beta (the Beta approximation) from NPERM
records_signflip permutations, the arm's own weights, the covariates and
genotype PCs as map_nominal gets them. torch picks the device at
tensorqtl/hapmixqtl.py:2127 (CUDA when available); the log prints
torch.cuda.is_available() and that device. Seed: one integer per dataset
index r, SeedSequence(SEED, spawn_key=(MAPCIS_KEY, r)).generate_state(1)[0],
shared by the four arms and by the scenarios, so the permutation null is
paired across weightings and effect sizes as the datasets are.
map_cis(seed=...) calls np.random.seed: the one place this script touches
global numpy RNG state, and nothing here draws from it otherwise. Its gate:
num_var equals the gene's count of tested variants with varying dosage
(map_cis drops the 507 constant ones as monomorphic), the lead is one of
them, its slope equals map_nominal's at that variant within GATE_TOL of the
se. Its pval_nominal against map_nominal's smallest is logged, not a stop:
at a lead the p is so small that float32 rounding of the statistic moves it
by up to 1.8e-3 relative (2026-09-26 full run, beta 0.2 rep 001) while the
slopes agree to 7.4e-6 se; and since commit 8a06803 each pair's combined p
has its own Welch-Satterthwaite dof, so the lead (largest |t|) need not hold
the gene's smallest pval_nominal at all.

mixQTL GENE-LEVEL p, under the TIMING RULE (user decision 2026-09-26).
mixqtl_permutation_scan is CPU NumPy. On the first dataset the published arm
runs it gene by gene against MIXQTL_PERM_BUDGET seconds. If the dataset
finishes within the budget, both mixQTL arms get pval_perm on every dataset;
if not, it stops at the budget, the partial result is discarded, the mixQTL
arms get no gene-level permutation p, and one printed line says so. The
decision and the measurement are written to RESULTS/mixqtl_permutation.json
either way (score.py reads it). The null is mixQTL's published one, not
changed for mixQTL mode: the phenotype bundle (y1, y2, ytotal, library size)
and the RNA-tied covariates move by one of MIXQTL_NPERM record permutations,
the haplotypes and genotype PCs stay, the offset is refitted per permutation,
and no haplotype labels are swapped. Indices from SeedSequence(SEED,
spawn_key=(MIXQTL_PERM_KEY, r)), shared by genes, both arms and the
scenarios. Observed statistic = max |meta stat| over the gene's tested
variants from mixqtl_scan; pval_perm = (1 + #{finite permuted maxima >=
observed}) / (1 + #finite), as scripts/mixqtl_gene_level_typeI.py computes
it; the port has no Beta approximation. Gate, per gene: the identity
permutation reproduces the observed maximum within IDENTITY_RTOL relative
(the tolerance of tests/test_mixqtl_replication.py). Smoke 2026-09-26 (host
load ~100): 307 s for 24 of 100 genes (114,778 of 487,454 tested variants),
about 1,305 s per dataset in proportion to tested variants, so the mixQTL
arms got no gene-level p; map_cis took 17.9-21.0 s per dataset per arm on
one NVIDIA L4.

Output: RESULTS/<scenario>/<arm>/nominal_repNNN.parquet (tested variants,
COLS, plus DOF_COLS for hapmixQTL arms and `method` for mixQTL arms); cis_repNNN.parquet (hapmixQTL: CIS_COLS
from map_cis; mixQTL, when the timing rule admits it: MIXQTL_CIS_COLS);
RESULTS/mixqtl_permutation.json. Parquet metadata: the sha256 of the dataset
arrays and the arm (checked by score.py), and the slope unit. Every output is
recomputed on every run. Seconds per dataset per arm and step (map_nominal,
map_cis, mixqtl_scan, the mixqtl_gene gate, mixqtl_permutation_scan) are
printed at the end.
Usage: run_arms.py [datasets_dir [results_dir]]
"""
import contextlib
import hashlib
import io
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch                                                    # noqa: E402

import make_datasets as MD                                      # noqa: E402
import compare_mixqtl_replication as CM                         # noqa: E402
import corrected_null_store as CNS                              # noqa: E402
from hybrid_weights_null import config_variances                # noqa: E402
from null_permutation_instrument import fit_channels            # noqa: E402
from tensorqtl import mixqtl_replication as MX                  # noqa: E402
from tensorqtl.hapmixqtl import map_cis, map_nominal            # noqa: E402

DATASETS = MD.ROOT / 'datasets'
RESULTS = MD.ROOT / 'results'
HAPMIX_ARMS = ('gibbs', 'split', 'unit', 'plus_one')
CONFIG = {'split': 'hybrid', 'unit': 'unit', 'plus_one': 'plus_one'}   # hybrid_weights_null names
MIXQTL_ARMS = {'mixqtl': MX.PUBLISHED_CUTOFFS, 'mixqtl_permissive': MX.PACKAGE_DEFAULT_CUTOFFS}
ARMS = HAPMIX_ARMS + tuple(MIXQTL_ARMS)
UNITS = {**{a: 'log2' for a in HAPMIX_ARMS}, **{a: 'natural log' for a in MIXQTL_ARMS}}
SEED = 42
MAPCIS_KEY, MIXQTL_PERM_KEY = 4, 5   # spawn keys after make_datasets' 1 / 2 / 3
NPERM = 1000                  # map_cis on every dataset (user decision 2026-09-26): measured ~19 s per dataset per arm, 100 genes, Beta approximation, GPU
PERM_SCHEME = 'records_signflip'
MIXQTL_NPERM = 1000           # user decision 2026-09-26: as map_cis
MIXQTL_PERM_BUDGET = 300.0    # s per dataset, user decision 2026-09-26: above it the mixQTL arms get no gene-level p
MIXQTL_PERM = False if MD.GENE_SET == 'corrected_null_store' else None   # False: timed 2026-09-26 on that set's smoke (results_smoke/mixqtl_permutation.json): 307 s for 24 of 100 genes, ~1,305 s per dataset, over the budget; None (every other gene set) re-times on the first dataset
GATE_TOL = 1e-3               # corrected_null_store.py gate, max |diff| / se
IDENTITY_RTOL = 1e-9          # tests/test_mixqtl_replication.py: identity permutation vs observed maximum
META_KEY, UNIT_KEY = b'plasmode_input_sha256', b'plasmode_slope_unit'
CIS_COLS = ['phenotype_id', 'variant_id', 'num_var', 'pval_nominal', 'slope', 'slope_se',
            'slope_a', 'slope_a_se', 'slope_t', 'slope_t_se', 'pval_perm', 'pval_beta',
            'beta_shape1', 'beta_shape2', 'true_df']
MIXQTL_CIS_COLS = ['phenotype_id', 'variant_id', 'stat_obs', 'pval_perm', 'n_perm_finite']
MIXQTL_PERM_JSON = 'mixqtl_permutation.json'
MIXQTL_PERM_TIMED = MD.ROOT / 'results_smoke' / MIXQTL_PERM_JSON   # the timing behind MIXQTL_PERM = False
DOF_COLS = ['dof_nominal', 'dof_a', 'dof_t', 'allelic_admitted']   # map_nominal's t references, since commit 8a06803


def quiet(fn, *args, **kwargs):
    """Call fn with its stdout captured; forward every captured line that contains WARNING."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = fn(*args, **kwargs)
    for line in buf.getvalue().splitlines():
        if 'WARNING' in line:
            print(f'{fn.__name__}: {line.strip()}', flush=True)
    return res


def setup(I):
    """Genotype frames, covariates and tested variants in the loader's gene and donor order."""
    genes, order, idx = list(I['genes']), list(I['order']), I['idx']
    vdf = I['vdf'].iloc[idx]
    if not vdf.index.is_unique:
        raise SystemExit('variant ids in the tested set are not unique')
    frame = lambda M: pd.DataFrame(M[idx], index=vdf.index, columns=order)
    tested_rows = {g: idx[CM.gene_variant_index(I, g)] for g in genes}
    tested = {g: set(I['vdf'].index[r].astype(str)) for g, r in tested_rows.items()}
    n_tested = pd.Series({g: len(v) for g, v in tested.items()})
    dos = I['dos'][idx]
    constant = set(vdf.index[(dos == dos[:, [0]]).all(1)].astype(str))
    scanned = {g: v - constant for g, v in tested.items()}
    rows = {str(v): i for i, v in enumerate(I['vdf'].index)}
    print(f'{len(genes)} genes x {len(order)} donors; {len(vdf):,} variants in the tested set; tested '
          f'gene-variant pairs {int(n_tested.sum()):,} (per gene {n_tested.min()}-{n_tested.max()}), of which '
          f'{int(n_tested.sum()) - sum(map(len, scanned.values()))} at constant ALT dosage in '
          f'{sum(len(tested[g]) > len(scanned[g]) for g in genes)} genes; covariates '
          f'{I["cov_df"].shape[1]} RNA-tied + {I["geno_cov_df"].shape[1]} genotype PCs', flush=True)
    return dict(I=I, genes=genes, order=order, vdf=vdf, gdf=frame(I['dos']), xLdf=frame(I['xL']),
                xRdf=frame(I['xR']), gp=I['gp'].loc[genes][['chr', 'pos']], tested=tested,
                tested_rows=tested_rows, n_tested=n_tested, scanned=scanned, rows=rows)


def arm_variances(ds, arm):
    """(allelic, total) working variances after the allelic admission, and the pairs it zeroed."""
    kept = MD.allelic_kept(ds['pL'], ds['pR'], ds['Va'])
    Va = np.where(kept, ds['Va'], 0.0)
    n_zeroed = int(((ds['Va'] > MD.EPS) & ~kept).sum())
    if arm == 'gibbs':
        return Va, ds['Vt'], n_zeroed
    return (*config_variances(CONFIG[arm], Va, ds['Vt']), n_zeroed)


def fingerprint(ds, arm):
    h = hashlib.sha256(arm.encode())
    for k in ('A', 'T', 'Va', 'Vt', 'pL', 'pR', 'pT', 'eff_lib', 'perm', 'swap', 'causal_variant'):
        h.update(np.ascontiguousarray(ds[k]).tobytes())
    return h.hexdigest()


def stored_fingerprint(path):
    return pq.read_schema(path).metadata[META_KEY].decode()


def stored_unit(path):
    return pq.read_schema(path).metadata[UNIT_KEY].decode()


def write_parquet(df, path, sha, unit):
    tab = pa.Table.from_pandas(df, preserve_index=False)
    tab = tab.replace_schema_metadata({**tab.schema.metadata, META_KEY: sha.encode(), UNIT_KEY: unit.encode()})
    tmp = path.with_name(path.name + '.tmp')
    pq.write_table(tab, tmp, compression='zstd')
    tmp.rename(path)


def inputs(S, ds, arm):
    genes, order = S['genes'], S['order']
    Va, Vt, n_zeroed = arm_variances(ds, arm)
    ph = lambda M: pd.DataFrame(M, index=genes, columns=order)
    cov = pd.DataFrame(S['I']['cov_df'].values[ds['perm']], index=order, columns=S['I']['cov_df'].columns)
    return ph(ds['A']), ph(ds['T']), ph(Va), ph(Vt), cov, n_zeroed


def run_nominal(S, ds, arm, scratch):
    A, T, Va, Vt, cov, n_zeroed = inputs(S, ds, arm)
    scratch.mkdir(parents=True, exist_ok=True)
    for q in scratch.glob('*'):
        q.unlink()
    quiet(map_nominal, S['gdf'], S['vdf'][['chrom', 'pos']], A, T, Va, Vt, S['gp'],
          xL_df=S['xLdf'], xR_df=S['xRdf'], prefix='n', covariates_df=cov,
          genotype_covariates_df=S['I']['geno_cov_df'], window=CM.WIN,
          output_dir=str(scratch), verbose=False, ase_covariates_df=None)
    df = pd.concat([pd.read_parquet(q, columns=CNS.COLS + DOF_COLS) for q in sorted(scratch.glob('n*.parquet'))],
                   ignore_index=True)
    df['variant_id'] = df['variant_id'].astype(str)
    n_exp = int(S['n_tested'].sum())
    per = df.groupby('phenotype_id').size().reindex(S['genes'])
    if len(df) != n_exp or not per.equals(S['n_tested'].reindex(S['genes'])):
        raise SystemExit(f'map_nominal returned {len(df):,} rows against {n_exp:,} tested pairs')
    return df, gate_nominal(S, ds, df, Va.values, Vt.values), n_zeroed


def gate_nominal(S, ds, df, Va, Vt):
    """map_nominal's channel slopes at each causal variant against fit_channels."""
    I = S['I']
    Cg = np.column_stack([I['cov_df'].values[ds['perm']], I['geno_cov_df'].values])
    d = df.set_index(['phenotype_id', 'variant_id'])
    rec = []
    for k, g in enumerate(S['genes']):
        v = str(ds['causal_variant'][k])
        if v not in S['tested'][g]:
            raise SystemExit(f'causal variant {v} of {g} is not a tested variant')
        j = S['rows'][v]
        fc = fit_channels(ds['A'][k], (I['xL'][j] - I['xR'][j]).astype(float), Va[k], ds['T'][k],
                          I['dos'][j].astype(float) / 2.0, Vt[k], Cg)
        if fc is None:
            continue
        r = d.loc[(g, v)]
        rec.append((g, 'slope_a', abs(float(r.slope_a) - fc['ba']) / float(r.slope_a_se)))
        rec.append((g, 'slope_t', abs(float(r.slope_t) - fc['bt']) / float(r.slope_t_se)))
    ratio = np.array([x[2] for x in rec])
    bad = [f'{g} {f}' for g, f, x in rec if not np.isfinite(x)]
    if bad:
        raise SystemExit(f'GATE: non-finite |map_nominal - fit_channels| / se at the causal variant: {bad[:5]}')
    worst = float(ratio.max()) if len(ratio) else float('nan')
    if not worst < GATE_TOL:
        raise SystemExit(f'GATE FAILED: map_nominal vs fit_channels max |diff| / se {worst:.2e} over '
                         f'{len(rec) // 2} genes')
    return worst, len(rec) // 2


def mixqtl_values(ds):
    """mixQTL's inputs: the dataset's thinned point estimates, never the Gibbs draws."""
    return MX.inputs_from_point_estimates(ds['pL'], ds['pR'], ds['pT'])


def run_mixqtl(S, ds, cutoffs):
    """mixqtl_scan per gene on the tested variants; COLS in natural log, plus the meta method; each lead and max |meta stat|."""
    I = S['I']
    y1, y2, yt = mixqtl_values(ds)
    cov, G = I['cov_df'].values[ds['perm']], I['geno_cov_df'].values
    parts, leads, smax, n_asc, n_trc = [], {}, {}, [], []
    for k, g in enumerate(S['genes']):
        rows = S['tested_rows'][g]
        h1, h2 = I['xL'][rows].T.astype(float), I['xR'][rows].T.astype(float)
        o = MX.mixqtl_scan(y1[k], y2[k], yt[k], ds['eff_lib'], h1, h2, covariates=cov,
                           genotype_covariates=G, **cutoffs)
        m, a, t = o['meta'], o['asc'], o['trc']
        vid = I['vdf'].index[rows].astype(str)
        parts.append(pd.DataFrame(dict(
            phenotype_id=g, variant_id=vid, pval_nominal=m['pval'], slope=m['beta'], slope_se=m['se'],
            pval_a=a['pval'], slope_a=a['beta'], slope_a_se=a['se'],
            pval_t=t['pval'], slope_t=t['beta'], slope_t_se=t['se'], method=m['method'].astype(str))))
        stat = np.abs(m['stat'])
        if np.isfinite(stat).any():
            i = int(np.nanargmax(stat))
            leads[g] = (str(vid[i]), float(m['beta'][i]), float(m['se'][i]))
            smax[g] = float(stat[i])
        else:
            leads[g], smax[g] = None, float('nan')
        n_asc.append(a['sample_size'])
        n_trc.append(t['sample_size'])
    df = pd.concat(parts, ignore_index=True)
    if len(df) != int(S['n_tested'].sum()):
        raise SystemExit(f'mixqtl_scan returned {len(df):,} rows against {int(S["n_tested"].sum()):,} tested pairs')
    return df, leads, smax, np.array(n_asc), np.array(n_trc)


def gate_mixqtl(S, ds, leads):
    """compare_mixqtl_replication.mixqtl_gene on the same inputs must return the same lead, beta and se.

    The loader inputs are restricted to the gene's tested rows before the call (mixqtl_gene
    otherwise copies all 473,144 tested rows per gene, 3.3 s a gene here), so the gate checks
    the haplotype construction, the record order of covariates and library size, and the
    value pairing, not the tested set, which setup takes from the same CM.gene_variant_index.
    """
    I = S['I']
    base = dict(I, cov_df=I['cov_df'].iloc[ds['perm']], lib_size=ds['eff_lib'])
    y1, y2, yt = mixqtl_values(ds)
    bad = []
    for k, g in enumerate(S['genes']):
        rows = S['tested_rows'][g]
        Ig = dict(base, vdf=I['vdf'].iloc[rows], xL=I['xL'][rows], xR=I['xR'][rows], idx=np.arange(len(rows)))
        r = CM.mixqtl_gene(Ig, g, k, y1, y2, yt)
        want = None if r is None else (r['variant_id'], r['beta'], r['se'])
        if want != leads[g]:
            bad.append(f'{g}: mixqtl_gene {want}, scan {leads[g]}')
    if bad:
        raise SystemExit(f'GATE FAILED: mixqtl lead differs from compare_mixqtl_replication.mixqtl_gene in '
                         f'{len(bad)} genes, e.g. {bad[:3]}')
    return len(S['genes'])


def mixqtl_gene_level(S, ds, cutoffs, leads, smax, perm_idx, budget):
    """pval_perm per gene from mixqtl_permutation_scan (module docstring).

    Returns (frame, seconds, genes done); the frame is None when `budget` (seconds, or None for
    no budget) ran out before the last gene.
    """
    I = S['I']
    y1, y2, yt = mixqtl_values(ds)
    kw = dict(covariates=I['cov_df'].values[ds['perm']], genotype_covariates=I['geno_cov_df'].values, **cutoffs)
    ident = np.arange(len(S['order']))[None, :]
    rec, bad = [], []
    t0 = time.perf_counter()
    for k, g in enumerate(S['genes']):
        rows = S['tested_rows'][g]
        args = (y1[k], y2[k], yt[k], ds['eff_lib'], I['xL'][rows].T.astype(float), I['xR'][rows].T.astype(float))
        s_id = MX.mixqtl_permutation_scan(*args, ident, **kw)[0]
        s_obs = smax[g]
        if not ((np.isnan(s_id) and np.isnan(s_obs)) or abs(s_id - s_obs) <= IDENTITY_RTOL * abs(s_obs)):
            bad.append(f'{g}: identity {s_id!r}, observed {s_obs!r}')
        s = MX.mixqtl_permutation_scan(*args, perm_idx, **kw)
        ok = np.isfinite(s)
        p = (np.sum(s[ok] >= s_obs) + 1) / (ok.sum() + 1) if np.isfinite(s_obs) else float('nan')
        rec.append(dict(phenotype_id=g, variant_id=None if leads[g] is None else leads[g][0], stat_obs=s_obs,
                        pval_perm=p, n_perm_finite=int(ok.sum())))
        if budget is not None and time.perf_counter() - t0 > budget:
            break
    if bad:
        raise SystemExit(f'GATE FAILED: mixqtl_permutation_scan at the identity permutation differs from the '
                         f'observed maximum |meta stat| in {len(bad)} genes, e.g. {bad[:3]}')
    dt = time.perf_counter() - t0
    if budget is not None and dt > budget:
        return None, dt, len(rec)
    return pd.DataFrame(rec)[MIXQTL_CIS_COLS], dt, len(rec)


def cis_seed(r):
    return int(np.random.SeedSequence(SEED, spawn_key=(MAPCIS_KEY, r)).generate_state(1)[0])


def mixqtl_perm_idx(n, r):
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(MIXQTL_PERM_KEY, r)))
    return np.array([rng.permutation(n) for _ in range(MIXQTL_NPERM)])


def run_cis(S, ds, arm, nominal, seed):
    """map_cis on the arm's inputs, gated against map_nominal (module docstring)."""
    A, T, Va, Vt, cov, _ = inputs(S, ds, arm)
    # tau_refit=True is inert in default mode (map_cis refits only when tau_mode == 'estimate')
    res = quiet(map_cis, S['gdf'], S['vdf'][['chrom', 'pos']], A, T, Va, Vt, S['gp'],
                xL_df=S['xLdf'], xR_df=S['xRdf'], covariates_df=cov,
                genotype_covariates_df=S['I']['geno_cov_df'], window=CM.WIN, nperm=NPERM,
                seed=seed, perm_scheme=PERM_SCHEME, tau_refit=True, verbose=False, ase_covariates_df=None)
    res = res.reset_index()[CIS_COLS]
    res['variant_id'] = res['variant_id'].astype(str)
    if list(res.phenotype_id) != S['genes']:
        raise SystemExit(f'map_cis returned {len(res)} genes, not the {len(S["genes"])} in order')
    nom = nominal.set_index(['phenotype_id', 'variant_id'])
    sc = nominal[[v in S['scanned'][g] for g, v in zip(nominal.phenotype_id, nominal.variant_id)]]
    best = sc.loc[sc.groupby('phenotype_id').pval_nominal.idxmin()].set_index('phenotype_id')
    rb, rp = [], []
    for r in res.itertuples():
        want = S['scanned'][r.phenotype_id]
        if r.num_var != len(want) or r.variant_id not in want:
            raise SystemExit(f'map_cis scanned {r.num_var} variants of {r.phenotype_id} (tested with '
                             f'varying dosage {len(want)}), lead {r.variant_id}')
        m, b = nom.loc[(r.phenotype_id, r.variant_id)], best.loc[r.phenotype_id]
        rb.append(abs(r.slope - float(m.slope)) / float(m.slope_se))
        rp.append(abs(r.pval_nominal - b.pval_nominal) / b.pval_nominal)
    rb, rp = np.array(rb), np.array(rp)
    if not (np.isfinite(rb).all() and np.isfinite(rp).all()):
        raise SystemExit(f'map_cis gate: non-finite ratio in genes '
                         f'{[g for g, x, y in zip(res.phenotype_id, rb, rp) if not (np.isfinite(x) and np.isfinite(y))][:5]}')
    if not rb.max() < GATE_TOL:
        raise SystemExit(f'map_cis lead disagrees with map_nominal: slope {rb.max():.2e} se')
    return res, float(rb.max()), float(rp.max())


def main():
    datasets = Path(sys.argv[1]) if len(sys.argv) > 1 else DATASETS
    results = Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # tensorqtl/hapmixqtl.py:2127
    print(f'torch {torch.__version__}; torch.cuda.is_available() {torch.cuda.is_available()}; map_cis device '
          f'{device}' + (f' ({torch.cuda.get_device_name(device)})' if device.type == 'cuda' else ''), flush=True)
    S = setup(CM.load_point_estimate_inputs(gene_list=str(MD.GENES), regions=str(MD.REGIONS)))
    meta = json.loads((datasets / 'meta.json').read_text())
    if S['genes'] != meta['genes'] or S['order'] != meta['donors']:
        raise SystemExit(f'{datasets / "meta.json"}: genes or donors differ from the cache inputs')
    scenarios = [(f'beta{b}', meta['n_datasets'][str(b)]) for b in meta['betas']]
    files = {sc: sorted((datasets / sc).glob('rep*.npz')) for sc, _ in scenarios}
    for sc, n in scenarios:
        if len(files[sc]) != n:
            raise SystemExit(f'{datasets / sc}: {len(files[sc])} datasets, meta.json says {n}')
    print(f'{datasets}: datasets per scenario {dict(scenarios)}; arms {ARMS}; map_cis ({NPERM:,} permutations) '
          f'on every dataset for {HAPMIX_ARMS}; mixQTL gene-level p by the timing rule ({MIXQTL_NPERM:,} '
          f'permutations, budget {MIXQTL_PERM_BUDGET:.0f} s per dataset)', flush=True)
    results.mkdir(parents=True, exist_ok=True)
    scratch = results / 'scratch'
    secs = {}
    mixqtl_perm = MIXQTL_PERM
    if mixqtl_perm is False:
        rec = dict(json.loads(MIXQTL_PERM_TIMED.read_text()), source=str(MIXQTL_PERM_TIMED))
        MD.write_atomic(results / MIXQTL_PERM_JSON, lambda fh: fh.write(MD.dumps(rec)), 'w')
    n_var = int(S['n_tested'].sum())
    for sc, _ in scenarios:
        for f in files[sc]:
            r = int(f.stem[3:])
            ds = dict(np.load(f))
            seed, midx = cis_seed(r), mixqtl_perm_idx(len(S['order']), r)
            for arm in ARMS:
                out = results / sc / arm
                out.mkdir(parents=True, exist_ok=True)
                sha = fingerprint(ds, arm)
                tag = f'{sc} rep {r:03d} {arm:17s}'
                if arm in HAPMIX_ARMS:
                    t0 = time.perf_counter()
                    nominal, (worst, n_gate), n_zeroed = run_nominal(S, ds, arm, scratch)
                    secs.setdefault((arm, 'map_nominal'), []).append(time.perf_counter() - t0)
                    write_parquet(nominal, out / f'nominal_rep{r:03d}.parquet', sha, UNITS[arm])
                    below = sorted(nominal.phenotype_id[~nominal.allelic_admitted].unique())
                    print(f'{tag} map_nominal {len(nominal):,} rows; allelic admission zeroed {n_zeroed} donor-gene '
                          f'pairs; genes with the allelic channel out of the combination {len(below)} {below}; '
                          f'gate at causal variants {worst:.1e} se over {n_gate} genes; '
                          f'{secs[(arm, "map_nominal")][-1]:.1f} s', flush=True)
                    t0 = time.perf_counter()
                    cis, wb, wp = run_cis(S, ds, arm, nominal, seed)
                    secs.setdefault((arm, 'map_cis'), []).append(time.perf_counter() - t0)
                    write_parquet(cis, out / f'cis_rep{r:03d}.parquet', sha, UNITS[arm])
                    print(f'{tag} map_cis seed {seed}, {len(cis)} genes, NaN pval_beta '
                          f'{int(cis.pval_beta.isna().sum())}; lead vs map_nominal: slope {wb:.1e} se, p {wp:.1e} '
                          f'relative; {secs[(arm, "map_cis")][-1]:.1f} s', flush=True)
                    continue
                t0 = time.perf_counter()
                nominal, leads, smax, n_asc, n_trc = run_mixqtl(S, ds, MIXQTL_ARMS[arm])
                secs.setdefault((arm, 'mixqtl_scan'), []).append(time.perf_counter() - t0)
                gated = 'not gated'
                if arm == 'mixqtl':
                    t0 = time.perf_counter()
                    gated = f'lead equals mixqtl_gene in {gate_mixqtl(S, ds, leads)} genes'
                    secs.setdefault((arm, 'mixqtl_gene gate'), []).append(time.perf_counter() - t0)
                write_parquet(nominal, out / f'nominal_rep{r:03d}.parquet', sha, UNITS[arm])
                print(f'{tag} mixqtl_scan {len(nominal):,} rows; genes with allelic samples >= '
                      f'{MX.META_N_CUTOFF}: {int((n_asc >= MX.META_N_CUTOFF).sum())}, total samples >= '
                      f'{MX.META_N_CUTOFF}: {int((n_trc >= MX.META_N_CUTOFF).sum())}; allelic samples per '
                      f'gene median {int(np.median(n_asc))} [{n_asc.min()}, {n_asc.max()}]; genes with no '
                      f'finite meta p {sum(v is None for v in leads.values())}; meta estimate from both '
                      f'channels in {(nominal.method == "meta").mean():.3f} of tests; {gated}; '
                      f'{secs[(arm, "mixqtl_scan")][-1]:.1f} s', flush=True)
                if mixqtl_perm is False:
                    continue
                budget = MIXQTL_PERM_BUDGET if mixqtl_perm is None else None
                gl, dt, done = mixqtl_gene_level(S, ds, MIXQTL_ARMS[arm], leads, smax, midx, budget)
                if mixqtl_perm is None:
                    mixqtl_perm = gl is not None
                    v_done = int(S['n_tested'].iloc[:done].sum())
                    rec = dict(included=mixqtl_perm, rule='user decision 2026-09-26: mixqtl_permutation_scan on '
                               'every dataset only if one dataset takes at most budget_s',
                               budget_s=MIXQTL_PERM_BUDGET, nperm=MIXQTL_NPERM, timed_on=f'{sc} rep {r:03d} {arm}',
                               seconds=dt, genes_done=done, genes=len(S['genes']), tested_variants_done=v_done,
                               tested_variants=n_var, seconds_per_dataset_extrapolated=dt * n_var / v_done)
                    MD.write_atomic(results / MIXQTL_PERM_JSON, lambda fh: fh.write(MD.dumps(rec)), 'w')
                    if not mixqtl_perm:
                        print(f'mixQTL arms: NO gene-level permutation p. mixqtl_permutation_scan '
                              f'({MIXQTL_NPERM:,} permutations, CPU NumPy) passed the {MIXQTL_PERM_BUDGET:.0f} s '
                              f'budget per dataset after {done} of {len(S["genes"])} genes of {sc} rep {r:03d} '
                              f'({dt:.0f} s, {v_done:,} of {n_var:,} tested variants); in proportion to tested '
                              f'variants one dataset takes ~{dt * n_var / v_done:.0f} s', flush=True)
                        continue
                secs.setdefault((arm, 'mixqtl_permutation_scan'), []).append(dt)
                write_parquet(gl, out / f'cis_rep{r:03d}.parquet', sha, UNITS[arm])
                print(f'{tag} mixqtl_permutation_scan {len(gl)} genes, NaN pval_perm '
                      f'{int(gl.pval_perm.isna().sum())}, genes with fewer than {MIXQTL_NPERM} finite permuted '
                      f'maxima {int((gl.n_perm_finite < MIXQTL_NPERM).sum())}; identity gate passed; {dt:.1f} s',
                      flush=True)
    if scratch.exists():
        shutil.rmtree(scratch)
    for (arm, step), v in secs.items():
        print(f'{arm:17s} {step:24s} {len(v)} datasets, seconds per dataset median {np.median(v):.1f} '
              f'[{min(v):.1f}, {max(v):.1f}]')
    print(f'wrote {results}')


if __name__ == '__main__':
    main()
