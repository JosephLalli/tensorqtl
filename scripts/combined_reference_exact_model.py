"""Which t reference calibrates the combined statistic under the exact model,
as a function of the allelic donor count?

QUESTION (2026-09-27). Since 8a06803 the combined statistic t = slope / slope_se
is referred to the Welch-Satterthwaite df nu_WS = (w_a + w_t)^2 / (w_a^2/nu_a +
w_t^2/nu_t), w_k = 1/se_k^2: the df of a linear combination of independent
variance estimates, matched on its first two moments, with the channel weights
treated as fixed. On the stored null it raised admitted genes' combined rate
at 0.001 by +0.00013 to +0.00018 over the old 73 df. The weights are estimated
from the same residuals, so the combination is a Graybill-Deal mean (a weighted
mean of independent estimates with weights the reciprocals of their ESTIMATED
variances), whose reported variance 1/sum(w) is too small.

EXACT MODEL. The 100 genes of corrected_null_store_20260925, their real
working variances after the zero-haplotype admission under two weightings
(hybrid_weights_null.config_variances: split = allelic Gibbs variance, total
unit; unit = 1 in both channels), their phased genotypes, the 14 RNA-tied
covariates and 3 genotype PCs, all unpermuted. Phenotypes are replaced by pure
noise at the model's variance Var(eps) = sigma^2 v, with no genetic effect:
    A = sigma_a sqrt(v_a) z_a,  T = sigma_t sqrt(v_t) z_t,  z ~ N(0, 1)
(A = 0 where v_a = 0). DEVIATION from the request's A = sqrt(Va) N(0, 1):
sigma_a and sigma_t are each gene's real null-model fitted scales in that
configuration (shipped residualizer, unpermuted data), not 1. The combined
reference's error depends on the allelic share f_a = w_a / (w_a + w_t), which
the ratio sigma_a / sigma_t sets. The request's literal sigma = 1 is run as a
secondary arm (sigma_one_probe.json: N_PROBE replicates paired with the main
run's first ones, every reference and both channels, median f_a) to show
what the combination becomes without the real scales. A channel's own t is
invariant to its sigma, so only the combination feels this choice. REALISM:
summary.json 'realism' compares f_a and nu_WS gene by gene with the stored
null (allelic_df_fix_20260927 draw 0). Where the simulated allelic share is
below the stored null's, both the Graybill-Deal variance error (which grows
with f_a f_t while f_a < 0.5) and nu_WS are smaller in the simulation, so the
simulated WS excess understates the stored-null design's in direction.

DONOR COUNT. For each gene and replicate, a random order of the gene's
informative allelic donors (v_a > 0) is drawn; level n keeps the first n
(nested across levels, so levels are paired) and sets v_a = 0 for the rest:
n in 15, 20, 30, 40 and all. A level above the gene's own count is skipped.
The 5 genes below MIN_ALLELIC_DONORS = 15 are left out of every level (at
"all" they would be tested on the total channel alone).

FITS through the shipped code: _prepare_channels (tau_mode='zero', fitted
scale, allelic channel through the origin) and calculate_hapmixqtl_nominal
(fitted=True, return_info=True), per (gene, replicate, level, configuration),
on a fixed random subset of N_VAR = 200 of each gene's tested variants (2,295
to 12,942 per gene). Why 200: the quantity under test is the channel scales'
estimation noise, and every variant of one (gene, replicate) shares it, so
resolution comes from gene-replicates and more variants per gene add little;
storage binds (at 200 each gene file is 16 MB, 1.5 GB in all). Runtime does
not: a call takes 3.6 ms (run.log: 356,000 calls in 21.5 min) and costs about
the same at more variants. 200 variants sample the heterozygote
configurations that set f_a. N_REP = 400 replicates per gene: 95 x 400 =
38,000 independent gene-replicates at n = 15, twice the 20,000 of the
unstored measurement in _satterthwaite_dof's docstring. Expected rejections
at 0.001 per (configuration, level): n_genes x 400 x 200 x 0.001 = 7,600
tests at n = 15 (6,240 at n = 40); if the 200 variants of a gene-replicate
were one test, 38 (31). The truth lies between; the gene-clustered interval
measures it, and summary.json 'resolution' records its realized half-width
per alpha.

REFERENCES for the combined t, all from the same fits:
  WS     p = 2 t_sf(|t|, nu_WS), nu_WS the shipped dof_nominal.
  Meier  t_M = t / sqrt(1 + 4 f_a f_t (1/nu_a + 1/nu_t)), p = 2 t_sf(|t_M|, nu_WS).
         Meier, P. (1953) Variance of a weighted mean. Biometrics 9(1):59-73,
         doi:10.2307/3001633 (Crossref; the request cited 9:295, which does
         not match). Derivation: write w_hat_k = w_k (1 + d_k), f_k =
         w_k / W, W = sum_k w_k, with d_k independent of the channel slopes
         under normality. w_hat_k is w_k times nu_k / chi2_{nu_k}, so to first
         order E[d_k] = 2/nu_k (E[nu/chi2_nu] = nu/(nu - 2)) and Var(d_k) =
         2/nu_k. With f_hat_k = f_k (1 + u_k), sum_k f_k u_k = 0, the true
         variance of the weighted mean is (1/W)[1 + E sum_k f_k u_k^2] =
         (1/W)(1 + 2S), S = sum_k f_k(1 - f_k)/nu_k (the means of d_k cancel
         in the shares to first order): Meier's result. The reported variance
         1/W_hat = (1/W) / (1 + sum_k f_k d_k) has expectation, to second
         order, (1/W)[1 - sum_k f_k E d_k + Var(sum_k f_k d_k)] = (1/W)(1 -
         2S), which needs E[d_k] = 2/nu_k (with E[d_k] = 0 it would be
         (1/W)[1 + 2 sum_k f_k^2/nu_k]). The ratio (1 + 2S)/(1 - 2S) is
         1 + 4S to first order in 1/nu, and with two channels f_a(1 - f_a) =
         f_t(1 - f_t) = f_a f_t. Evaluated at the estimated shares.
  min    p = 2 t_sf(|t|, min(nu_a, nu_t)) over the channels that carry weight:
         nu_t where the allelic channel has no informative heterozygote among
         the kept donors (w_a = 0, the combination IS the total channel).
  old    p = 2 t_sf(|t|, 73), the shared reference before 8a06803.
  Checks of the simulation: allelic alone, 2 t_sf(|t_a|, dof_a); total alone,
  2 t_sf(|t_t|, dof_t). Both are exact under the model.

PRE-REGISTERED RULE (RULE below): a reference is calibrated if, in every n_a
level (15, 20, 30, 40, all) and both configurations (split, unit), its pooled
rejection rate at 0.05, 0.01 and 0.001 has a gene-clustered 95% interval
(genes resampled with replacement 2,000 times, pooled rate recomputed, 2.5 and
97.5 percentiles) that contains the nominal alpha: 30 checks per reference.
The allelic-alone and total-alone p, exact by construction, are put through
the same 30 checks; their failure counts are the rule's own false-failure
floor (1.5 of 30 expected at 95% coverage if the checks were independent;
they are not: the total channel's t does not depend on the configuration or
the allelic donor count, and the allelic channel's shares z_a across both),
against which the four references' counts are read. PROVENANCE: RULE is the
string summary.json has carried since the 2026-09-27 run; no artifact of that
run shows it was written before the fits (the script was uncommitted and was
edited after the run). From this revision on, analysis_fingerprint() hashes
RULE with the code that applies it (pvals, rate_check, verdicts); run()
writes it to gate.json before the simulation loop and summarize() stops if
it has changed. For the 2026-09-27 run it is recorded after the fact.
SECONDARY, not judged: rates on pairs where both channels carry weight
(w_a > 0, fixed by the design, not by the data); median reference df per
level; the effective df of the pooled combined t by moments, nu_eff =
2 V / (V - 1) with V = mean(t^2) (a t_nu has variance nu/(nu - 2)), with a
gene-clustered interval, and by tail, the df at which t_isf(alpha/2, nu)
equals the pooled empirical 1 - alpha quantile of |t|. A reference df below
the effective df costs power only in proportion to the gap. FIXED-DF
DIAGNOSTIC: the combined t referred to every integer df in DF_GRID, with the
same intervals; which fixed dfs would pass all three alphas per (configuration,
level). The df is chosen from the same fits, so it is a description of where
the needed df sits, never a reference judged by the rule.

GATES: (1) before the main loop, replicate 0 at n = 15 for every gene through
map_nominal equals the direct call (channel and combined slopes and SEs within
1e-4 se, dof_nominal within 1e-6 relative, pval_nominal within 1e-6 of the direct
call's t on nu_WS); (2) every call admits the allelic channel, dof_a = n - 1, dof_t = 73;
(3) nu_WS recomputed from the stored f_a equals the shipped dof_nominal within
1e-4 relative.

OUTPUTS (OUT): genes/<gene>.npz, per-gene fits (resumable; each file carries a
fingerprint of the parameters, the simulating source and hapmixqtl.py, and a
mismatch stops the run; summarize() also stops if the current source no
longer matches it); gate.json; sigma_one_probe.json; summary.json.
Usage: CUDA_VISIBLE_DEVICES=0 combined_reference_exact_model.py
           [--summarize-only | --sigma-one-probe]

SHIPPED 2026-09-27 (user decision, after this run): hapmixqtl.py applies the
Meier arm's factor to the combined SE in default mode. fit() returns the
shipped, corrected statistic, which the map_nominal gate compares like for
like; uncorrected() recovers t = t_M sqrt(M) and SE = SE_M / sqrt(M) from the
factor M that calculate_hapmixqtl_nominal returns in its info dict, and every
reference here is defined on that uncorrected t. The stored 2026-09-27 run
was made under the uncorrected code, its fingerprint covers that
hapmixqtl.py, and it cannot be re-summarised under the shipped source
(tests/test_hapmixqtl_meier.py reads its gene files directly as the known
answer and checks that uncorrected() reproduces their t).
"""
import contextlib
import hashlib
import inspect
import io
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import optimize, stats

import compare_mixqtl_replication as CM
import corrected_null_store as CNS
import hybrid_weights_null as HW
import tensorqtl.hapmixqtl as HM

OUT = CNS.D / 'combined_reference_exact_model_20260927'
SEED = 42                                   # project master seed; child streams by spawn key
LEVELS = (15, 20, 30, 40, 'all')            # informative allelic donors kept; 15 = MIN_ALLELIC_DONORS
CONFIGS = {'split': 'hybrid', 'unit': 'unit'}   # name here -> hybrid_weights_null.config_variances name
N_VAR = 200                                 # tested variants per gene (docstring: why 200)
N_REP = 400                                 # replicates per gene (docstring: resolution at 0.001)
ALPHAS = (0.05, 0.01, 0.001)
N_RESAMPLE = 2000                           # gene-clustered resamples, as every stored-null summary
OLD_DOF = 73                                # N - 2 - 17 covariates, the reference before 8a06803
EPS = 1e-12                                 # informative-donor threshold, as corrected_null_store
GATE_TOL = 1e-4                             # in se; the fits are the same float32 code
REFS = ('WS', 'Meier', 'min', 'old')
ALONE = ('allelic', 'total')
STREAM_VARIANTS, STREAM_NOISE, STREAM_RESAMPLE, STREAM_PROBE = 0, 1, 2, 3
N_PROBE = 20                                # sigma = 1 arm: replicates 0-19; 380,000 tests, 380 expected at 0.001, at n = 15
DF_GRID = np.arange(20, 151)                # fixed-df diagnostic grid (docstring: FIXED-DF DIAGNOSTIC)
STORED_NULL = CNS.D / 'allelic_df_fix_20260927' / 'draws'   # stored null after 8a06803, draw 0 per config
RULE = ('calibrated if, in every n_a level (15, 20, 30, 40, all) and both configurations (split, '
        'unit), the pooled rejection rate at 0.05, 0.01 and 0.001 has a gene-clustered 95% interval '
        '(2,000 gene resamples) containing the nominal alpha; the allelic-alone and total-alone p, '
        'exact by construction, go through the same 30 checks as the rule\'s false-failure floor')


def load():
    """Inputs, working variances and real null-model scales, per configuration."""
    if not Path(HM.__file__).resolve().is_relative_to(Path(CM.REPO).resolve()):
        raise SystemExit(f'hapmixqtl from {HM.__file__}: not this checkout')
    with contextlib.redirect_stdout(io.StringIO()):
        I = CM.load_point_estimate_inputs(gene_list=str(CNS.OUT / 'genes.txt'),
                                          regions=str(CNS.OUT / 'regions.bed'),
                                          cov=f'{CM.D}/cov/log2cpm1_point_calibration_20260925')   # the build the stored 2026-09-27 run used
    keep, order = I['keep'], I['order']
    A, T, Va, Vt, _ = HM.summaries_from_point_estimates(I['pL'], I['pR'], I['pT'], I['eff_lib'],
                                                        I['YL'], I['YR'], I['YT'])
    A, T, Va, Vt = A[:, keep], T[:, keep], Va[:, keep], Vt[:, keep]
    Va = np.where((I['pL'][:, keep] < 0.5) ^ (I['pR'][:, keep] < 0.5), 0.0, Va)
    genes = list(I['genes'])
    n_a = (Va > EPS).sum(1)
    design = pd.read_csv(CNS.OUT / 'gene_design.tsv', sep='\t').set_index('gene')
    if not np.array_equal(n_a, design.loc[genes, 'n_allelic_drop'].values):
        raise SystemExit('allelic donor counts differ from corrected_null_store gene_design.tsv')
    C, _ = HM._combine_covariates(I['cov_df'], I['geno_cov_df'], order)
    dev = torch.device('cuda')
    C_t = torch.tensor(C.values, dtype=torch.float32, device=dev)
    V = {c: HW.config_variances(name, Va, Vt) for c, name in CONFIGS.items()}
    if not all(np.array_equal(v[0] > EPS, Va > EPS) for v in V.values()):
        raise SystemExit('configurations differ in which allelic donors are informative')
    f = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)
    scale = {}
    for c, (va, vt) in V.items():
        s = np.zeros((len(genes), 2))
        for k in range(len(genes)):
            if n_a[k] < LEVELS[0]:
                continue
            wa, wt, ra, rt = HM._prepare_channels(f(A[k]), f(T[k]), f(va[k]), f(vt[k]), C_t, 'zero', dev,
                                                  ase_covariates_t=None, fitted_scale=True)
            for j, (y, w, r) in enumerate(((A[k], wa, ra), (T[k], wt, rt))):
                e = r.transform((f(y) * w).unsqueeze(0))
                s[k, j] = float((e * e).sum()) / (int((w != 0).sum()) - r.Q_t.shape[1])
        scale[c] = s
    variants = {}
    for k, g in enumerate(genes):
        tidx = I['idx'][CM.gene_variant_index(I, g)]
        rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(STREAM_VARIANTS, k)))
        variants[g] = np.sort(rng.choice(tidx, N_VAR, replace=False))
    adm = [g for g, n in zip(genes, n_a) if n >= LEVELS[0]]
    print(f'{len(genes)} genes, {len(order)} donors, {C.shape[1]} covariates; {len(adm)} with >= {LEVELS[0]} '
          f'informative allelic donors; left out: ' + ', '.join(f'{g} ({n})' for g, n in zip(genes, n_a)
                                                                 if n < LEVELS[0]), flush=True)
    for c, s in scale.items():
        ok = s[:, 0] > 0
        print(f'{c}: real null-model scales, median sigma_a^2 {np.median(s[ok, 0]):.4f}, '
              f'sigma_t^2 {np.median(s[ok, 1]):.4f}', flush=True)
    return dict(I=I, genes=genes, n_a=n_a, V=V, scale=scale, C_t=C_t, dev=dev, variants=variants, C=C)


def levels_for(n):
    return [(j, n if L == 'all' else L) for j, L in enumerate(LEVELS) if L == 'all' or L <= n]


def draw(k, r, inf, N):
    """Noise and nested donor order of (gene k, replicate r), shared by every level and configuration."""
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(STREAM_NOISE, k, r)))
    return rng.standard_normal(N), rng.standard_normal(N), rng.permutation(inf)


def inputs(S, k, c, n, z_a, z_t, perm):
    va_c, vt_c = S['V'][c]
    va = np.zeros_like(va_c[k])
    va[perm[:n]] = va_c[k][perm[:n]]
    s2a, s2t = S['scale'][c][k]
    return np.sqrt(s2a * va) * z_a, np.sqrt(s2t * vt_c[k]) * z_t, va, vt_c[k]


def fit(S, G_t, X_t, a, t, va, vt):
    """One shipped fit; returns (tstat, slope, slope_se, slope_a, se_a, slope_t, se_t) arrays and the info dict."""
    f = lambda x: torch.tensor(x, dtype=torch.float32, device=S['dev'])
    a_t, t_t = f(a), f(t)
    wa, wt, ra, rt = HM._prepare_channels(a_t, t_t, f(va), f(vt), S['C_t'], 'zero', S['dev'],
                                          ase_covariates_t=None, fitted_scale=True)
    out = HM.calculate_hapmixqtl_nominal(G_t, X_t, a_t, t_t, wa, wt, ra, rt, fitted=True, return_info=True)
    return torch.stack(out[:7]).cpu().numpy().astype(np.float64), out[7]


def uncorrected(o, info):
    """fit()'s output with the combined t and SE without Meier's factor: t sqrt(M) and SE / sqrt(M), M from
    the info dict (NaN where no channel carries weight, where the SE was not scaled)."""
    m = np.sqrt(np.nan_to_num(info['meier_factor'].cpu().numpy(), nan=1.0))
    o = o.copy()
    o[0] *= m
    o[2] /= m
    return o, info


def simulate_gene(S, k):
    I, g = S['I'], S['genes'][k]
    sel = S['variants'][g]
    G_t = torch.tensor(I['dos'][sel], dtype=torch.float32, device=S['dev'])
    X_t = torch.tensor(I['xL'][sel].astype(np.float32) - I['xR'][sel], device=S['dev'])
    inf = np.where(S['V']['split'][0][k] > EPS)[0]
    lev = levels_for(len(inf))
    shp = (len(CONFIGS), len(LEVELS), N_REP, N_VAR)
    R = {x: np.full(shp, np.nan, np.float32) for x in ('t', 't_a', 't_t', 'f_a', 'nu_ws')}
    R.update({x: np.full(shp[:3], np.nan) for x in ('dof_a', 'dof_t')})
    admitted = np.zeros(shp[:3], bool)
    for r in range(N_REP):
        z_a, z_t, perm = draw(k, r, inf, len(I['order']))
        for ci, c in enumerate(CONFIGS):
            for j, n in lev:
                o, info = uncorrected(*fit(S, G_t, X_t, *inputs(S, k, c, n, z_a, z_t, perm)))
                with np.errstate(divide='ignore', invalid='ignore'):
                    ok_a = np.isfinite(o[4]) & (o[4] > 0)
                    w_a = np.where(ok_a, 1.0 / o[4] ** 2, 0.0)
                    w_t = 1.0 / o[6] ** 2
                    R['t'][ci, j, r] = o[0]
                    R['t_a'][ci, j, r] = np.where(ok_a, o[3] / o[4], np.nan)
                    R['t_t'][ci, j, r] = o[5] / o[6]
                R['f_a'][ci, j, r] = w_a / (w_a + w_t)
                R['nu_ws'][ci, j, r] = info['dof_nominal'].cpu().numpy()
                R['dof_a'][ci, j, r], R['dof_t'][ci, j, r] = info['dof_a'], info['dof_t']
                admitted[ci, j, r] = info['allelic_admitted']
    for ci in range(len(CONFIGS)):
        for j, n in lev:
            if not (admitted[ci, j].all() and (R['dof_a'][ci, j] == n - 1).all() and (R['dof_t'][ci, j] == OLD_DOF).all()):
                raise SystemExit(f'GATE FAILED {g}: admission or channel dof at n = {n}')
    return R, np.array([j for j, _ in lev])


def fingerprint():
    h = hashlib.sha256()
    h.update(json.dumps([SEED, LEVELS, CONFIGS, N_VAR, N_REP, EPS]).encode())
    for fn in (load, levels_for, draw, inputs, fit, simulate_gene):
        h.update(inspect.getsource(fn).encode())
    h.update(Path(HM.__file__).read_bytes())
    return h.hexdigest()


def analysis_fingerprint():
    """The rule and the code that applies it (docstring: PROVENANCE)."""
    h = hashlib.sha256(RULE.encode())
    h.update(json.dumps([ALPHAS, N_RESAMPLE, OLD_DOF, STREAM_RESAMPLE]).encode())
    for fn in (pvals, rate_check, verdicts):
        h.update(inspect.getsource(fn).encode())
    return h.hexdigest()


def write_json(path, obj):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.rename(path)


def gate_map_nominal(S):
    """Replicate 0 at n = 15, every admitted gene, through map_nominal against the direct fit."""
    I, genes = S['I'], [g for g, n in zip(S['genes'], S['n_a']) if n >= LEVELS[0]]
    order, vdf = I['order'], I['vdf']
    rows = np.unique(np.concatenate([S['variants'][g] for g in genes]))
    gdf = pd.DataFrame(I['dos'][rows], index=vdf.index[rows], columns=order)
    xLdf = pd.DataFrame(I['xL'][rows], index=vdf.index[rows], columns=order)
    xRdf = pd.DataFrame(I['xR'][rows], index=vdf.index[rows], columns=order)
    gp = I['gp'].loc[genes][['chr', 'pos']]
    scratch = OUT / 'scratch'
    res = {}
    for c in CONFIGS:
        frames, direct = {x: [] for x in 'ATab'}, []
        for g in genes:
            k = S['genes'].index(g)
            inf = np.where(S['V']['split'][0][k] > EPS)[0]
            a, t, va, vt = inputs(S, k, c, LEVELS[0], *draw(k, 0, inf, len(order)))
            for x, v in zip('ATab', (a, t, va, vt)):
                frames[x].append(v)
            sel = S['variants'][g]
            G_t = torch.tensor(I['dos'][sel], dtype=torch.float32, device=S['dev'])
            X_t = torch.tensor(I['xL'][sel].astype(np.float32) - I['xR'][sel], device=S['dev'])
            o, info = fit(S, G_t, X_t, a, t, va, vt)
            nu = info['dof_nominal'].cpu().numpy()
            direct.append(pd.DataFrame(dict(
                phenotype_id=g, variant_id=vdf.index[sel].astype(str), slope=o[1], slope_se=o[2],
                slope_a=o[3], slope_a_se=o[4], slope_t=o[5], slope_t_se=o[6], dof_nominal=nu,
                p_ws=2 * stats.t.sf(np.abs(o[0].astype(np.float32)), nu))))
        mk = lambda x: pd.DataFrame(np.array(frames[x]), index=genes, columns=order)
        scratch.mkdir(exist_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):
            HM.map_nominal(gdf, vdf.iloc[rows][['chrom', 'pos']], mk('A'), mk('T'), mk('a'), mk('b'), gp,
                           xL_df=xLdf, xR_df=xRdf, prefix='g', covariates_df=I['cov_df'],
                           genotype_covariates_df=I['geno_cov_df'], window=CM.WIN,
                           output_dir=str(scratch), verbose=False, ase_covariates_df=None)
        m = pd.concat([pd.read_parquet(q) for q in sorted(scratch.glob('g*.parquet'))], ignore_index=True)
        shutil.rmtree(scratch)
        m['variant_id'] = m['variant_id'].astype(str)
        d = pd.concat(direct, ignore_index=True)
        j = d.merge(m, on=['phenotype_id', 'variant_id'], suffixes=('', '_m'), how='left')
        dev = lambda x, y, s: float(np.nanmax(np.abs(j[x].values - j[y].values) / j[s].values))
        res[c] = dict(
            pairs=len(d), missing_in_map_nominal=int(j['slope_m'].isna().sum()),
            slopes_over_se=max(dev(x, x + '_m', s) for x, s in (('slope', 'slope_se'), ('slope_a', 'slope_a_se'),
                                                                ('slope_t', 'slope_t_se'))),
            ses_over_se=max(dev(s, s + '_m', s) for s in ('slope_se', 'slope_a_se', 'slope_t_se')),
            dof_nominal_rel=dev('dof_nominal', 'dof_nominal_m', 'dof_nominal'),
            pval_nominal_vs_ws=float(np.nanmax(np.abs(j['p_ws'] - j['pval_nominal']))))
        print(f'gate map_nominal {c}: ' + '  '.join(f'{x} {v:.2e}' if isinstance(v, float) else f'{x} {v}'
                                                    for x, v in res[c].items()), flush=True)
        r = res[c]
        if not (r['missing_in_map_nominal'] == 0 and r['slopes_over_se'] < GATE_TOL and r['ses_over_se'] < GATE_TOL
                and r['dof_nominal_rel'] < 1e-6 and r['pval_nominal_vs_ws'] < 1e-6):
            raise SystemExit(f'GATE FAILED: map_nominal {c}')
    return res


def run():
    OUT.mkdir(exist_ok=True)
    (OUT / 'genes').mkdir(exist_ok=True)
    fp = fingerprint()
    t0 = time.time()
    S = load()
    print(f'loaded in {(time.time() - t0) / 60:.1f} min; fingerprint {fp[:12]}', flush=True)
    gate = gate_map_nominal(S)
    write_json(OUT / 'gate.json', dict(map_nominal=gate, fingerprint=fp, analysis_fingerprint=analysis_fingerprint()))
    todo = [k for k, n in enumerate(S['n_a']) if n >= LEVELS[0]]
    t0 = time.time()
    for i, k in enumerate(todo):
        fo = OUT / 'genes' / f'{S["genes"][k]}.npz'
        if fo.exists():
            if str(np.load(fo)['fingerprint']) != fp:
                raise SystemExit(f'{fo} was written by a different version; delete {OUT / "genes"} and rerun')
            print(f'  skip {fo.name}: exists with this fingerprint', flush=True)
            continue
        R, lev = simulate_gene(S, k)
        s2 = np.array([S['scale'][c][k] for c in CONFIGS])
        with open(fo.with_suffix('.tmp'), 'wb') as fh:
            np.savez(fh, **R, levels=lev, n_a=S['n_a'][k], scale=s2, fingerprint=fp)
        fo.with_suffix('.tmp').rename(fo)
        if i % 10 == 9 or i == len(todo) - 1:
            print(f'  gene {i + 1}/{len(todo)}  ({(time.time() - t0) / 60:.1f} min)', flush=True)
    return S


def pvals(z):
    """Per test: p for every reference and each channel alone, plus the reference dfs."""
    t, f_a, nu_ws = z['t'].astype(np.float64), z['f_a'].astype(np.float64), z['nu_ws'].astype(np.float64)
    nu_a, nu_t = z['nu_a'], z['nu_t']
    meier = 1 + 4 * f_a * (1 - f_a) * (1 / nu_a + 1 / nu_t)
    nu_min = np.where(f_a > 0, np.minimum(nu_a, nu_t), nu_t)
    p = {'WS': 2 * stats.t.sf(np.abs(t), nu_ws), 'Meier': 2 * stats.t.sf(np.abs(t) / np.sqrt(meier), nu_ws),
         'min': 2 * stats.t.sf(np.abs(t), nu_min), 'old': 2 * stats.t.sf(np.abs(t), OLD_DOF),
         'allelic': 2 * stats.t.sf(np.abs(z['t_a'].astype(np.float64)), nu_a),
         'total': 2 * stats.t.sf(np.abs(z['t_t'].astype(np.float64)), nu_t)}
    return p, dict(WS=nu_ws, meier_factor=meier, min=nu_min)


def rate_check(K, n, B, al):
    """Pooled rate over genes, its gene-clustered 95% interval (each row of B resamples genes), and RULE's test."""
    b = K[B].sum(1) / n[B].sum(1)
    lo, hi = float(np.quantile(b, .025)), float(np.quantile(b, .975))
    rate = K.sum() / n.sum()
    return dict(rate=float(rate), lo=lo, hi=hi, ratio=float(rate / al), contains_nominal=bool(lo <= al <= hi))


def verdicts(rates):
    """RULE applied: 30 checks per reference and per channel alone."""
    chk = [(c, L, al) for c in CONFIGS for L in map(str, LEVELS) for al in map(str, ALPHAS)]
    out = {}
    for x in REFS + ALONE:
        fails = [f'{c} {L} {al}' for c, L, al in chk if not rates[c][L]['refs'][x][al]['contains_nominal']]
        out[x] = dict(calibrated=not fails, n_checks=len(chk), n_failed=len(fails), failed=fails)
    return out


def gene_counts(z):
    """One gene's rejection counts per reference and alpha: all tests, and where both channels carry weight."""
    p, d = pvals(z)
    both = z['f_a'].astype(np.float64) > 0
    ok = {x: np.isfinite(v) for x, v in p.items()}
    K = {x: {al: (p[x][ok[x]] < al).sum() for al in ALPHAS} for x in p}
    Kb = {x: {al: (p[x][ok[x] & both] < al).sum() for al in ALPHAS} for x in REFS}
    return K, Kb, {x: ok[x].sum() for x in p}, both.sum(), d


def df_tail(q, alpha):
    """df at which t_isf(alpha/2, nu) equals the empirical quantile q; inf at or below the normal quantile."""
    if q <= stats.norm.isf(alpha / 2):
        return float('inf')
    return float(optimize.brentq(lambda nu: stats.t.isf(alpha / 2, nu) - q, 1.0, 1e7))


def sigma_one_probe(S):
    """The request's literal A = sqrt(Va) N(0, 1), T = sqrt(Vt) N(0, 1) (sigma_a = sigma_t = 1) on replicates
    0..N_PROBE-1 of the main run's noise and donor orders, every level and configuration."""
    S1 = dict(S, scale={c: np.ones_like(s) for c, s in S['scale'].items()})
    I, t0 = S['I'], time.time()
    per = {}                                    # (config index, level index) -> gene -> per-test arrays
    for k in [k for k, n in enumerate(S['n_a']) if n >= LEVELS[0]]:
        g, sel = S['genes'][k], S['variants'][S['genes'][k]]
        G_t = torch.tensor(I['dos'][sel], dtype=torch.float32, device=S['dev'])
        X_t = torch.tensor(I['xL'][sel].astype(np.float32) - I['xR'][sel], device=S['dev'])
        inf = np.where(S['V']['split'][0][k] > EPS)[0]
        acc = {}
        for r in range(N_PROBE):
            z_a, z_t, perm = draw(k, r, inf, len(I['order']))
            for ci, c in enumerate(CONFIGS):
                for j, n in levels_for(len(inf)):
                    o, info = uncorrected(*fit(S1, G_t, X_t, *inputs(S1, k, c, n, z_a, z_t, perm)))
                    if not (info['allelic_admitted'] and info['dof_a'] == n - 1 and info['dof_t'] == OLD_DOF):
                        raise SystemExit(f'GATE FAILED probe {g}: admission or channel dof at n = {n}')
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ok_a = np.isfinite(o[4]) & (o[4] > 0)
                        w_a = np.where(ok_a, 1.0 / o[4] ** 2, 0.0)
                        w_t = 1.0 / o[6] ** 2
                        z = dict(t=o[0], t_a=np.where(ok_a, o[3] / o[4], np.nan), t_t=o[5] / o[6],
                                 f_a=w_a / (w_a + w_t), nu_ws=info['dof_nominal'].cpu().numpy().astype(np.float64),
                                 nu_a=np.full(N_VAR, float(info['dof_a'])), nu_t=np.full(N_VAR, float(info['dof_t'])))
                    for x, v in z.items():
                        acc.setdefault((ci, j), {}).setdefault(x, []).append(v)
        for key, d in acc.items():
            per.setdefault(key, {})[g] = {x: np.concatenate(v) for x, v in d.items()}
    print(f'sigma = 1 probe: {N_PROBE} replicates in {(time.time() - t0) / 60:.1f} min', flush=True)
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(STREAM_PROBE,)))
    out = dict(n_probe=N_PROBE, n_var=N_VAR, fingerprint=fingerprint(), rates={})
    for j, L in enumerate(LEVELS):
        gl = list(per[(0, j)])
        B = rng.integers(0, len(gl), size=(N_RESAMPLE, len(gl)))
        for ci, c in enumerate(CONFIGS):
            C = [gene_counts(per[(ci, j)][g]) for g in gl]
            K = {x: {al: np.array([cc[0][x][al] for cc in C], float) for al in ALPHAS} for x in REFS + ALONE}
            n = {x: np.array([cc[2][x] for cc in C], float) for x in REFS + ALONE}
            fa = np.concatenate([per[(ci, j)][g]['f_a'] for g in gl])
            out['rates'].setdefault(c, {})[str(L)] = dict(
                n_genes=len(gl), n_tests=int(n['WS'].sum()), median_f_a_where_both=float(np.median(fa[fa > 0])),
                refs={x: {str(al): rate_check(K[x][al], n[x], B, al) for al in ALPHAS} for x in REFS + ALONE})
    write_json(OUT / 'sigma_one_probe.json', out)


def breakdown(res):
    """Where each reference's checks fail: per alpha over the 10 (configuration, level) pairs, and per level."""
    out = {}
    for x in REFS + ALONE:
        v = [res['rates'][c][L]['refs'][x] for c in CONFIGS for L in map(str, LEVELS)]
        out[x] = dict(by_alpha={}, failed_per_level={})
        for al in map(str, ALPHAS):
            r = [w[al]['ratio'] for w in v]
            f = [w[al]['ratio'] for w in v if not w[al]['contains_nominal']]
            out[x]['by_alpha'][al] = dict(ratio_min=min(r), ratio_max=max(r), n_failed=len(f),
                                          failed_ratio_min=min(f) if f else None, failed_ratio_max=max(f) if f else None,
                                          n_above_nominal=sum(w[al]['rate'] > float(al) for w in v))
        for L in map(str, LEVELS):
            out[x]['failed_per_level'][L] = sum(not res['rates'][c][L]['refs'][x][al]['contains_nominal']
                                                for c in CONFIGS for al in map(str, ALPHAS))
    return out


def resolution(res):
    """The rule's resolution: interval half-width over nominal, range over the 10 (configuration, level) pairs."""
    return {al: {x: dict(min=min(h), max=max(h)) for x in REFS + ALONE
                 for h in [[(w['hi'] - w['lo']) / 2 / float(al) for c in CONFIGS for L in map(str, LEVELS)
                            for w in [res['rates'][c][L]['refs'][x][al]]]]} for al in map(str, ALPHAS)}


def realism(Z):
    """f_a and nu_WS where both channels carry weight, per gene: simulated at all donors against stored null draw 0."""
    j, out = LEVELS.index('all'), {}
    for ci, c in enumerate(CONFIGS):
        d = pd.read_parquet(STORED_NULL / f'{c}_000.parquet',
                            columns=['phenotype_id', 'allelic_admitted', 'slope_a_se', 'slope_t_se', 'dof_nominal'])
        d = d[d.allelic_admitted]
        se_a, se_t = d.slope_a_se.values.astype(float), d.slope_t_se.values.astype(float)
        w_a = np.where(np.isfinite(se_a) & (se_a > 0), 1 / se_a ** 2, 0.0)
        d = d.assign(f_a=w_a / (w_a + 1 / se_t ** 2))
        st = d[d.f_a > 0].groupby('phenotype_id').agg(f_a=('f_a', 'median'), nu=('dof_nominal', 'median'))
        sim = {}
        for g, z in Z.items():
            fa, nu = z['f_a'][ci, j].ravel().astype(np.float64), z['nu_ws'][ci, j].ravel().astype(np.float64)
            sim[g] = dict(f_a=np.median(fa[fa > 0]), nu=np.median(nu[fa > 0]))
        sim = pd.DataFrame(sim).T
        g = sorted(set(st.index) & set(sim.index))
        ratio = sim.loc[g, 'f_a'] / st.loc[g, 'f_a']
        out[c] = dict(
            test_weighted_stored=dict(median_f_a_where_both=float(d.f_a[d.f_a > 0].median()),
                                      share_both=float((d.f_a > 0).mean()), n_tests=len(d)),
            n_genes_common=len(g), n_genes_stored=len(st), n_genes_simulated=len(sim),
            per_gene_median_f_a=dict(stored=float(st.loc[g, 'f_a'].median()), simulated=float(sim.loc[g, 'f_a'].median())),
            per_gene_ratio_simulated_over_stored=dict(median=float(ratio.median()), q25=float(ratio.quantile(.25)),
                                                      q75=float(ratio.quantile(.75)), n_below_1=int((ratio < 1).sum())),
            per_gene_median_nu_ws=dict(stored=float(st.loc[g, 'nu'].median()), simulated=float(sim.loc[g, 'nu'].median())))
    return out


def summarize():
    gate = json.loads((OUT / 'gate.json').read_text())
    fp, afp = gate['fingerprint'], analysis_fingerprint()
    if fingerprint() != fp:
        raise SystemExit('simulating source or hapmixqtl.py changed since the run: delete genes/ and rerun')
    if 'analysis_fingerprint' in gate and gate['analysis_fingerprint'] != afp:
        raise SystemExit('RULE or the code applying it changed since the run')
    probe = json.loads((OUT / 'sigma_one_probe.json').read_text())
    if probe['fingerprint'] != fp:
        raise SystemExit('sigma_one_probe.json was written by a different version')
    Z = {f.stem: dict(np.load(f)) for f in sorted((OUT / 'genes').glob('*.npz'))}
    if any(str(z['fingerprint']) != fp for z in Z.values()):
        raise SystemExit('gene files from different versions')
    genes = list(Z)
    print(f'\n{len(genes)} genes, {N_REP} replicates, {N_VAR} variants each; fingerprint {fp[:12]}')
    rrng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(STREAM_RESAMPLE,)))
    res = dict(rule=RULE, n_rep=N_REP, n_var=N_VAR, n_resample=N_RESAMPLE, seed=SEED, levels=[str(L) for L in LEVELS],
               gate=gate, analysis_fingerprint=dict(
                   now=afp, at_run=gate['analysis_fingerprint'] if 'analysis_fingerprint' in gate else
                   'not recorded: the run predates analysis_fingerprint(); RULE is the string summary.json carried'),
               rates={}, dof={}, fixed_df={}, verdict={})
    worst_ws, grid_vs_old = 0.0, 0.0
    tails = {}
    crit = np.stack([stats.t.isf(al / 2, DF_GRID) for al in ALPHAS], 1)      # (df, alpha)
    for j, L in enumerate(LEVELS):
        gl = [g for g in genes if j in Z[g]['levels']]
        B = rrng.integers(0, len(gl), size=(N_RESAMPLE, len(gl)))
        M = np.stack([np.bincount(b, minlength=len(gl)) for b in B]).astype(np.float64)   # the same resamples
        for ci, c in enumerate(CONFIGS):
            K = {x: {al: np.zeros(len(gl)) for al in ALPHAS} for x in REFS + ALONE}
            n = {x: np.zeros(len(gl)) for x in REFS + ALONE}
            Kb = {x: {al: np.zeros(len(gl)) for al in ALPHAS} for x in REFS}
            nb, ss = np.zeros(len(gl)), np.zeros(len(gl))
            Kg = np.zeros((len(gl), len(DF_GRID), len(ALPHAS)))
            dfs, fas, mf_both, abs_t = {x: [] for x in ('WS', 'meier_factor', 'min')}, [], [], []
            for i, g in enumerate(gl):
                z0 = Z[g]
                z = {x: z0[x][ci, j].ravel() for x in ('t', 't_a', 't_t', 'f_a', 'nu_ws')}
                z['nu_a'] = np.repeat(z0['dof_a'][ci, j], N_VAR)
                z['nu_t'] = np.repeat(z0['dof_t'][ci, j], N_VAR)
                fa = z['f_a'].astype(np.float64)
                with np.errstate(divide='ignore'):
                    ws = np.where(fa > 0, 1 / (fa ** 2 / z['nu_a'] + (1 - fa) ** 2 / z['nu_t']), z['nu_t'])
                worst_ws = max(worst_ws, float(np.max(np.abs(ws / z['nu_ws'] - 1))))
                k_, kb_, n_, nb[i], d = gene_counts(z)
                for x in REFS + ALONE:
                    n[x][i] = n_[x]
                    for al in ALPHAS:
                        K[x][al][i] = k_[x][al]
                        if x in REFS:
                            Kb[x][al][i] = kb_[x][al]
                tt = z['t'].astype(np.float64)
                if n_['WS'] != len(tt):
                    raise SystemExit(f'{g} {c} {L}: non-finite combined statistic')
                ss[i] = (tt ** 2).sum()
                at = np.sort(np.abs(tt))
                abs_t.append(at)
                Kg[i] = len(at) - np.searchsorted(at, crit, side='right')
                for x in dfs:
                    dfs[x].append(d[x])
                fas.append(fa[fa > 0])
                mf_both.append(d['meier_factor'][fa > 0])
            if not all((n[x] == n['WS']).all() for x in REFS) or not (n['total'] == n['WS']).all():
                raise SystemExit(f'{c} {L}: test counts differ between references')
            row = dict(n_genes=len(gl), n_tests=int(n['WS'].sum()), n_tests_allelic=int(n['allelic'].sum()),
                       share_both_channels=float(nb.sum() / n['WS'].sum()),
                       median_f_a_where_both=float(np.median(np.concatenate(fas))), refs={}, both_channels={})
            for x in REFS + ALONE:
                row['refs'][x] = {str(al): rate_check(K[x][al], n[x], B, al) for al in ALPHAS}
            for x in REFS:
                row['both_channels'][x] = {str(al): float(Kb[x][al].sum() / nb.sum()) for al in ALPHAS}
            V = ss.sum() / n['WS'].sum()
            Vb = ss[B].sum(1) / n['WS'][B].sum(1)
            nu_m = lambda v: 2 * v / (v - 1) if v > 1 else float('inf')
            at = np.concatenate(abs_t)
            q = {al: float(np.quantile(at, 1 - al)) for al in ALPHAS}
            tails[(c, L)] = {str(al): df_tail(q[al], al) for al in ALPHAS}
            med = {x: float(np.median(np.concatenate(v))) for x, v in dfs.items()}
            critf = {'WS': lambda al: stats.t.isf(al / 2, med['WS']),
                     'Meier': lambda al: stats.t.isf(al / 2, med['WS']) * np.sqrt(med['meier_factor']),
                     'min': lambda al: stats.t.isf(al / 2, med['min']),
                     'old': lambda al: stats.t.isf(al / 2, OLD_DOF)}
            res['dof'].setdefault(c, {})[str(L)] = dict(
                empirical_critical_abs_t={str(al): q[al] for al in ALPHAS},
                critical_abs_t_at_median_df={x: {str(al): float(fn(al)) for al in ALPHAS} for x, fn in critf.items()},
                median_WS=med['WS'],
                median_meier_factor=med['meier_factor'],
                median_meier_factor_where_both=float(np.median(np.concatenate(mf_both))),
                median_min=med['min'], old=OLD_DOF,
                effective_moment=nu_m(V), effective_moment_lo=nu_m(float(np.quantile(Vb, .975))),
                effective_moment_hi=nu_m(float(np.quantile(Vb, .025))), mean_t2=float(V),
                effective_tail=tails[(c, L)])
            res['rates'].setdefault(c, {})[str(L)] = row
            # fixed-df diagnostic: counts at every df in DF_GRID, resampled with the same B (integer sums, exact)
            nt = n['WS']
            Rg = (M @ Kg.reshape(len(gl), -1)) / (M @ nt)[:, None]
            lo_g = np.quantile(Rg, .025, axis=0).reshape(Kg.shape[1:])
            hi_g = np.quantile(Rg, .975, axis=0).reshape(Kg.shape[1:])
            rate_g = Kg.sum(0) / nt.sum()
            inside = (lo_g <= np.array(ALPHAS)) & (np.array(ALPHAS) <= hi_g)
            k73 = int(np.where(DF_GRID == OLD_DOF)[0][0])
            grid_vs_old = max(grid_vs_old, *(abs(v - row['refs']['old'][str(al)][w]) for a, al in enumerate(ALPHAS)
                                             for w, v in (('rate', rate_g[k73, a]), ('lo', lo_g[k73, a]),
                                                          ('hi', hi_g[k73, a]))))
            dt = int(round(tails[(c, L)]['0.01']))
            kt = int(np.where(DF_GRID == dt)[0][0])
            res['fixed_df'].setdefault(c, {})[str(L)] = dict(
                pass_all_three=DF_GRID[inside.all(1)].tolist(),
                pass_by_alpha={str(al): DF_GRID[inside[:, a]].tolist() for a, al in enumerate(ALPHAS)},
                df_tail_0_01=dt,
                at_df_tail_0_01={str(al): dict(rate=float(rate_g[kt, a]), lo=float(lo_g[kt, a]), hi=float(hi_g[kt, a]),
                                               ratio=float(rate_g[kt, a] / al)) for a, al in enumerate(ALPHAS)})
    if not worst_ws < 1e-4:
        raise SystemExit(f'GATE FAILED: nu_WS recomputed from f_a differs from dof_nominal by {worst_ws:.2e}')
    if not grid_vs_old < 1e-9:
        raise SystemExit(f'fixed-df grid at {OLD_DOF} does not reproduce the old reference ({grid_vs_old:.2e})')
    res['nu_ws_recompute_max_rel'] = worst_ws
    res['fixed_df_grid_at_73_vs_old_max_abs'] = grid_vs_old
    res['verdict'] = verdicts(res['rates'])
    res['breakdown'] = breakdown(res)
    res['resolution'] = resolution(res)
    res['total_t_identical_across_configs_and_levels'] = bool(all(
        np.array_equal(z['t_t'][ci, j], z['t_t'][0, z['levels'][0]], equal_nan=True)
        for z in Z.values() for ci in range(len(CONFIGS)) for j in z['levels']))
    res['realism'] = realism(Z)
    res['sigma_one_probe'] = probe
    write_json(OUT / 'summary.json', res)
    report(res)


def spans(v):
    """[38, 39, 40, 44] -> '38-40, 44'."""
    out, s = [], None
    for a, b in zip(v, list(v[1:]) + [None]):
        s = a if s is None else s
        if b != a + 1:
            out.append(f'{s}' if s == a else f'{s}-{a}')
            s = None
    return ', '.join(out) or 'none'


def report(res):
    fr = lambda v: f'{v["rate"]:.5f} [{v["lo"]:.5f}, {v["hi"]:.5f}] {v["ratio"]:.3f}x{"" if v["contains_nominal"] else "*"}'
    print('\nrate [gene-clustered 95% interval] ratio-to-nominal; * = interval excludes nominal')
    print(f'{"config":6s} {"n_a":>4s} {"ref":8s} ' + ''.join(f'{"alpha " + str(al):39s}' for al in ALPHAS))
    for c in CONFIGS:
        for L in map(str, LEVELS):
            r = res['rates'][c][L]
            print(f'{c:6s} {L:>4s} {r["n_genes"]} genes, {r["n_tests"]:,} tests, {r["share_both_channels"]:.3f} with '
                  f'both channels (median f_a there {r["median_f_a_where_both"]:.3f})')
            for x in REFS + ALONE:
                print(f'{"":6s} {"":4s} {x:8s} ' + ''.join(f'{fr(r["refs"][x][str(al)]):39s}' for al in ALPHAS))
    print('\nverdict under the pre-registered rule (30 checks each); failed checks per level, of 6 (2 configs x 3 alphas)')
    for x, v in res['verdict'].items():
        print(f'{x:8s} {"CALIBRATED" if v["calibrated"] else "not calibrated"}: {v["n_failed"]}/{v["n_checks"]} fail;  '
              + '  '.join(f'{L}: {k}' for L, k in res['breakdown'][x]['failed_per_level'].items()))
    print('\nper alpha over the 10 (config, level) pairs: ratio range; failures (ratio range of the failures)')
    for x in REFS + ALONE:
        print(f'{x:8s} ' + '   '.join(
            f'{al}: {b["ratio_min"]:.3f}-{b["ratio_max"]:.3f}x, {b["n_failed"]} fail'
            + (f' ({b["failed_ratio_min"]:.3f}-{b["failed_ratio_max"]:.3f}x)' if b['n_failed'] else '')
            for al, b in res['breakdown'][x]['by_alpha'].items()))
    print('\nrule resolution: interval half-width / nominal, range over the 10 pairs')
    for al, v in res['resolution'].items():
        print(f'{al:6s} ' + '  '.join(f'{x} {w["min"]:.3f}-{w["max"]:.3f}' for x, w in v.items()))
    print(f'allelic alone above nominal at 0.05 in {res["breakdown"]["allelic"]["by_alpha"]["0.05"]["n_above_nominal"]}'
          f'/10 pairs; total-channel t identical across configurations and levels: '
          f'{res["total_t_identical_across_configs_and_levels"]}')
    print('\nsecondary: rates on pairs where both channels carry weight, 0.05 / 0.01 / 0.001')
    for c in CONFIGS:
        for L in map(str, LEVELS):
            b = res['rates'][c][L]['both_channels']
            print(f'{c:6s} {L:>4s} ' + '   '.join(f'{x} ' + ' / '.join(f'{b[x][str(al)]:.5f}' for al in ALPHAS)
                                                  for x in REFS))
    print('\nreference df (medians over tests) against the effective df of the pooled combined t')
    for c in CONFIGS:
        for L in map(str, LEVELS):
            d = res['dof'][c][L]
            print(f'{c:6s} {L:>4s} WS {d["median_WS"]:6.1f}  Meier WS x factor {d["median_meier_factor"]:.4f}  '
                  f'min {d["median_min"]:5.1f}  old {d["old"]}  |  effective: moment {d["effective_moment"]:.0f} '
                  f'[{d["effective_moment_lo"]:.0f}, {d["effective_moment_hi"]:.0f}]  tail '
                  + ' / '.join(f'{d["effective_tail"][str(al)]:.0f}' for al in ALPHAS)
                  + f'  |  WS over moment {d["median_WS"] / d["effective_moment"]:.2f}, over tail at 0.001 '
                  f'{d["median_WS"] / d["effective_tail"]["0.001"]:.2f}')
    print('\nfixed-df diagnostic (combined t without the Meier factor; df chosen from these fits, not judged):')
    print('df matched to the tail at 0.01, its ratios at 0.05 / 0.01 / 0.001 (* = excludes nominal), and every integer '
          f'df in {DF_GRID[0]}-{DF_GRID[-1]} passing all three alphas')
    for c in CONFIGS:
        for L in map(str, LEVELS):
            f = res['fixed_df'][c][L]
            print(f'{c:6s} {L:>4s} df {f["df_tail_0_01"]:3d}: ' + ' / '.join(
                f'{w["ratio"]:.3f}x{"" if w["lo"] <= float(al) <= w["hi"] else "*"}' for al, w in f['at_df_tail_0_01'].items())
                + f'   pass all three: {spans(f["pass_all_three"])}')
    inter = set.intersection(*[set(res['fixed_df'][c][L]['pass_all_three']) for c in CONFIGS for L in map(str, LEVELS)])
    print(f'one fixed df passing all three alphas in every (config, level): {spans(sorted(inter))}')
    print('\ncritical |t| at 0.001: the pooled empirical 0.999 quantile (calibrated) against each reference at its median df')
    for c in CONFIGS:
        for L in map(str, LEVELS):
            d = res['dof'][c][L]
            print(f'{c:6s} {L:>4s} empirical {d["empirical_critical_abs_t"]["0.001"]:.3f}  ' + '  '.join(
                f'{x} {d["critical_abs_t_at_median_df"][x]["0.001"]:.3f}' for x in REFS))
    print('\nrealism, where both channels carry weight: simulated (all donors) against stored null draw 0')
    for c, v in res['realism'].items():
        s, r = res['rates'][c]['all'], v['per_gene_ratio_simulated_over_stored']
        print(f'{c}: test-weighted median f_a {s["median_f_a_where_both"]:.3f} vs '
              f'{v["test_weighted_stored"]["median_f_a_where_both"]:.3f}; {v["n_genes_common"]} genes, median of per-gene '
              f'median f_a {v["per_gene_median_f_a"]["simulated"]:.3f} vs {v["per_gene_median_f_a"]["stored"]:.3f}, per-gene '
              f'ratio {r["median"]:.2f} [IQR {r["q25"]:.2f}-{r["q75"]:.2f}], {r["n_below_1"]} below 1; per-gene median '
              f'nu_WS {v["per_gene_median_nu_ws"]["simulated"]:.1f} vs {v["per_gene_median_nu_ws"]["stored"]:.1f}')
    p = res['sigma_one_probe']
    print(f'\nsigma = 1 probe ({p["n_probe"]} replicates): median f_a where both; ratio at 0.05/0.01/0.001 (* excludes nominal)')
    for c in CONFIGS:
        for L in map(str, LEVELS):
            r = p['rates'][c][L]
            print(f'{c:6s} {L:>4s} f_a {r["median_f_a_where_both"]:.3f} {r["n_tests"]:,} tests  ' + '  '.join(
                f'{x} ' + '/'.join(f'{w["ratio"]:.2f}{"" if w["contains_nominal"] else "*"}' for w in r['refs'][x].values())
                for x in REFS + ALONE))
    a = res['analysis_fingerprint']
    print(f'\nnu_WS recomputed from f_a vs shipped dof_nominal, max relative difference {res["nu_ws_recompute_max_rel"]:.1e}; '
          f'fixed-df grid at {OLD_DOF} vs old reference {res["fixed_df_grid_at_73_vs_old_max_abs"]:.1e}')
    print(f'fingerprint {res["gate"]["fingerprint"][:12]} (matches the current source); analysis fingerprint '
          f'{a["now"][:12]}; at run: {a["at_run"]}')


def main():
    if '--sigma-one-probe' in sys.argv:
        sigma_one_probe(load())
    elif '--summarize-only' not in sys.argv:
        sigma_one_probe(run())
    summarize()
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
