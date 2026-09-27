"""Plasmode cis-eQTL datasets built from the real cohort's Salmon output.

User decision 2026-09-26: no Salmon simulation beyond the simplest
assumptions, and the rule for the Gibbs variance is read from Salmon's own
code. A dataset is the BrainVar cohort's own Salmon quantification (point
estimates and 200 Gibbs draws, 100 genes of corrected_null_store_20260925)
with the genotype association broken and a known cis effect injected. This is
the plasmode approach of Gerard 2020, "Data-based RNA-seq simulations by
binomial thinning", BMC Bioinformatics: real counts are thinned binomially so
a chosen signal is added while the data keep their own noise, depth and
dependence. Run order and outputs: run_all.sh; design:
docs/simulation_benchmark_spec.md.

1. BREAK THE ASSOCIATION. Donor records move against fixed genotypes by a
   permutation perm, and each moved record's L/R labels are swapped with
   probability one half (the records_signflip construction). Column i of
   every stored [G, N] array is real record perm[i]; where swap[i] = -1 its
   L and R were exchanged (pL<->pR, YL<->YR). pT, the Gibbs draws and the
   record's edgeR effective library size (stored as eff_lib) move with it;
   genotypes, the observed phase xL/xR and the genotype PCs stay in place.
   Dataset r's RNA-tied covariates are cov_df rows taken in the order perm.
   Streams: perm and swap from SeedSequence(42, spawn_key=(1, r)); the null
   set, causal variants and effect signs from (2, r); the binomial draws from
   (3, r, round(1000 |beta|)). Dataset r therefore has the same permutation,
   causal variants and signs in every scenario, and the same null genes in
   every scenario with beta > 0, so scenarios are paired.

2. INJECT BY BINOMIAL THINNING. Scenarios |beta| in BETAS, N_DATASETS[beta]
   datasets each. For beta > 0 a fraction NULL_FRACTION of genes is null.
   beta = 0 is the null anchor: every gene null and no thinning (every factor
   1, which the generator reproduces bit for bit, check (a) of
   check_generator.py). Every gene draws one causal variant uniformly from
   its tested variants (cis window, outside the gene bodies, MAF >= 0.05: the
   pipeline's idx set) and a sign for the scenario's |beta|, a log2 allelic
   fold change, ALT over REF. Haplotype h of the record at genotype position
   i carries allele x_ih (the record's L is paired with xL[i], its R with
   xR[i], the pairing of the pipeline's allelic regression A ~ s = xL - xR).
   Its factor is f_ih = 2^-|beta| if it carries the lower-expressed allele
   (REF when beta > 0, ALT when beta < 0), else 1. Then
       pL' = thin(pL, f_iL),  pR' = thin(pR, f_iR)
       U   = pT - pL - pR     (reads on transcripts whose two copies were
                               identical and collapsed, carrying both
                               haplotypes)
       U'  = thin(U, (f_iL + f_iR) / 2),  pT' = pL' + pR' + U'
   thin(y, f) = Binomial(floor(y), f) + f (y - floor(y)), the identity at
   f = 1. pT' is computed as pT minus the reads removed, which is the same
   sum and is exactly pT at f = 1. U is clipped at 0 after asserting it is
   above -ROUNDING_TOL, since Salmon's sums round.
   Depth matching: null genes are thinned too, by the mean of the factors a
   non-null configuration would have given them (a causal variant and sign
   are drawn for every gene; for null genes every f_ih is replaced by the
   gene's mean over donors and haplotypes), so null and non-null genes sit
   at matched depth and differ only in the genotype dependence.
   A side the point estimate already put at zero stays zero; per
   donor-gene pair, `expressible` records whether min(pL, pR) >= 0.5 before
   thinning.

3. VARIANCE RULES.
   TOTAL. Every YT Gibbs draw is thinned as the point estimates are (its
   haplotype parts YL, YR by f_iL, f_iR, its remainder by the mean), and T
   and Vt are summaries_from_point_estimates(pL', pR', pT', L, YL, YR, YT').
   Binomial thinning of a Poisson count of mean y by f gives a Poisson count
   of mean f y, so this is exact for the Gamma shot noise of Salmon's draws;
   check (b) of check_generator.py measures it (median Fano factor of YT
   draws 0.990-0.994 thinned at f = 0.5 against 0.985-0.993 real, by band of
   haplotype-informative reads).
   ALLELIC. The thinned YL, YR draws are NOT used for the allelic variance:
       Va' = dv * q(pL', pR') / q(pL, pR) + q_a(pL', pR')
   dv = across-draw variance (ddof=0) of log2((YL + 0.5)/(YR + 0.5)) from
   the REAL draws of that record; q(x, y) = 1/(x + 0.5) + 1/(y + 0.5);
   q_a = q / ln(2)^2, the counting term summaries_from_point_estimates adds;
   pL', pR' the realized thinned point estimates. Where pL' + pR' = 0,
   Va' = 0 (the no_cov guard: no allelic information). At f = 1 this is
   summaries_from_point_estimates' Va bit for bit, which check (a) pins; the
   function's own Va is discarded.
   WHY. Salmon 1.10.3's Gibbs round (src/CollapsedGibbsSampler.cpp; saved
   copies brainvar_hapmix_deploy/mixqtl_algorithm_review_20260914/
   salmon_variance_theory_20260915/CollapsedGibbsSampler.cpp and
   /mnt/ssd/lalli/tmp_salmonsrc/, byte-identical) draws each transcript's
   rate from Gamma(count + prior, 1/(0.1 + effLen)) (line 149), reassigns
   the reads of each multi-transcript equivalence class by a multinomial on
   those rates (lines 257-265), and writes mu x effLen x scale (line 507).
   For an L/R pair, reads in classes holding both haplotypes carry no
   information about the haplotype fraction, so the delta method gives the
   draw variance beyond Gamma shot noise as s^2 (1/u_L + 1/u_R) / ln(2)^2,
   with u_h the reads in classes holding only haplotype h of the gene and s
   the ambiguous share of the gene's paired reads. check_salmon_premise.py
   verifies this on donor 100_D1's dumped equivalence classes: median
   observed/predicted 0.989 (IQR 0.795-1.169) over the 3,589 of 3,595 genes
   with u_L, u_R >= 20 that have positive excess and s > 0; Spearman rank
   correlation with the observed excess 0.877 against 0.491 for the counting
   term; median haplotype-informative share 0.117. Under thinning u shrinks
   by f and s is unchanged, so both parts of the draw variance scale by 1/f:
   exponent 1 in q. The q-ratio equals the u-based form when informative
   reads split between the haplotypes like total reads.
   ERROR BOUND. The Gibbs prior is 1 pseudo-read per transcript copy:
   SalmonDefaults.hpp:76 sets useVBOpt{true}, :85 perTranscriptPrior{true}
   and :87 vbPrior{1e-2}, so CollapsedGibbsSampler.cpp:369 takes
   prior = max(1, vbPrior) = 1 (the per-nucleotide branch, line 316, needs
   --perNucleotidePrior, which the production command did not pass: the
   cmd_info.json of the quant directories listed in cohort/salmon.tsv, e.g.
   100_R1, carries no prior flag). That prior does not scale with depth.
   Draw mean over point estimate is 1.014 at 100-999 and 1.09 at 10-99 total
   reads (tmp_scratch/draw_vs_point.py), so the rule overstates Va by at
   most (1 - f) x ~1.4% / ~9% there.
   WHAT THE RULE CANNOT DO. The point estimate is Salmon's VB optimum, not a
   read count. VB zeroes one haplotype more often at lower depth, and
   binomial thinning cannot create those zeros. No known-answer test against
   Salmon at reduced depth exists in this repository.

4. SUMMARIES. A = log2((pL' + 0.5)/(pR' + 0.5)), T, Va', Vt. Va' is stored
   before the zero-haplotype drop (one side below 0.5 reads), which is
   applied downstream by allelic_kept from the stored pL', pR'. L is the real
   edgeR effective library size: it is a property of a whole library over
   41,552 genes and thinning 100 genes changes it negligibly. Measured in the
   2026-09-26 smoke run (2 datasets per scenario), reads removed per donor
   over the median effective library size: median 9.2e-4, max 1.5e-3 /
   2.9e-3 / 5.0e-3 at |beta| = 0.2 / 0.4 / 0.8. At the maximum, a library
   size updated for the removed reads would move that donor's T by at most
   log2(1 / (1 - 0.005)) = 0.007 log2 units. Each run records the same ratio
   in meta.json (library_size_change).

5. TRUTH per dataset: which genes are null, the causal variant id and MAF,
   signed beta, per-donor true log2 aFC d_i = beta (x_iL - x_iR) (0 for
   homozygous donors and null genes), the expressible flags, the thinning
   factors, the permutation and the swap signs. Two scales, both 0 for null
   genes:
   COUNT SCALE. Allelic truth = beta, the slope of the log2 haplotype-count
   ratio on s = xL - xR. Total truth, per gene and dataset = the
   least-squares slope, with intercept, of the exact log2 total fold
   log2((n_low f + (2 - n_low)) / 2) on g/2 over the dataset's 92 donors
   (n_low = the donor's haplotypes carrying the lower-expressed allele, g =
   ALT dosage, f = 2^-|beta|); the fold holds for the expected pT' because U
   is thinned by (f_iL + f_iR)/2, which is that fold. The linear total
   channel reads beta only when genotype counts are symmetric, hence the
   per-dataset slope. NaN where the causal variant has one dosage in every
   donor (no total-channel estimand).
   PIPELINE SCALE: the noise-free shift of the pipeline's own phenotypes, from
   the REAL point estimates in the dataset's record order and swap, with the
   dataset's factors:
       A*_i = log2((f_iL pL + 0.5)/(f_iR pR + 0.5)) - log2((pL + 0.5)/(pR + 0.5))
       T*_i = log2(k E[pT'] + 1) - log2(k pT + 1),  k = 1e6 / L,
              E[pT'] = pT - (1 - f_iL) pL - (1 - f_iR) pR - (1 - fbar_i) U
   (E[pT'] = f_iL pL + f_iR pR + fbar_i U up to Salmon's rounding of U, and
   exactly pT at f = 1). Allelic pipeline truth = unweighted least-squares
   slope through the origin of A* on s over the records allelic_kept admits
   in the thinned data (the records every hapmixQTL arm fits); total
   pipeline truth = unweighted least-squares slope with intercept of T* on
   g/2 over all donors. The difference from the count scale is the
   attenuation of the transforms themselves (the +0.5 pseudocount of A, the
   +1 of log2(CPM + 1)), which is largest at low depth. It is unweighted by
   definition, so a weighted arm's bias against it still contains the
   weights' re-targeting of a depth-heterogeneous shift.
   truth.tsv holds one row per dataset x non-null gene (none for beta = 0).

WHAT THIS CANNOT ANSWER: CANNOT_ANSWER below, copied into meta.json.

Output: OUT/beta<beta>/rep<NNN>.npz, OUT/meta.json, OUT/truth.tsv.
Usage: make_datasets.py [n_datasets [out_dir]]   (defaults N_DATASETS, OUT; an
n_datasets argument replaces every scenario's count, e.g. 1 for a smoke run)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import compare_mixqtl_replication as CM                             # noqa: E402
from tensorqtl.hapmixqtl import LN2, summaries_from_point_estimates  # noqa: E402

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
ROOT = D / 'plasmode_20260926'          # every output of this benchmark
GENES = D / 'corrected_null_store_20260925' / 'genes.txt'
REGIONS = D / 'corrected_null_store_20260925' / 'regions.bed'
OUT = ROOT / 'datasets'
SEED = 42
BETAS = (0.0, 0.2, 0.4, 0.8)   # user decision 2026-09-26; 0.0 is the null anchor
N_DATASETS = {0.0: 1, 0.2: 3, 0.4: 3, 0.8: 3}   # user decision 2026-09-26: one anchor dataset, 3 per effect size
NULL_FRACTION = 0.5            # user decision 2026-09-26, for beta > 0
KAPPA = 0.5                    # summaries_from_point_estimates' default pseudocount
EXPRESSIBLE_MIN = 0.5          # reads; the pipeline's zero-haplotype rule (docs/pipeline_rules.md)
EPS = 1e-12                    # allelic admission Va > EPS: hapmixqtl.py:181,199 (_zero_degenerate_ase_weights)
ROUNDING_TOL = 1e-6            # Salmon sums: pT - pL - pR measured down to -1.8e-12, draws -1.0e-11
PERM_KEY, DESIGN_KEY, THIN_KEY = 1, 2, 3
CANNOT_ANSWER = [
    'No oracle variance exists in real data, so only relative efficiency between weightings is '
    'measured, never efficiency against the true error variance.',
    'The null is the record permutation, so these data cannot tell which permutation rule for '
    'genotype PCs (tied to the genotypes or moving with the record) is right.',
    'Effects are thinning only (expression can only go down) and are injected at one variant per gene.',
    'Thinning adds model-like binomial noise, which dilutes the real data\'s weight-residual coupling '
    'on thinned records by about the factor f, so null rates of the 1/v arms drift toward nominal as '
    'beta rises and must be read against a beta = 0 anchor, not as evidence for the weights.',
    'The pipeline\'s transforms attenuate a count-scale fold at low depth (the +0.5 pseudocount of the '
    'allelic ratio, the +1 of log2(CPM + 1)), so bias against the count-scale truth mixes that '
    'attenuation with estimator bias; the pipeline-scale truth separates them, but it is an unweighted '
    'slope, so a weighted arm\'s bias against it still includes how its weights re-target a shift that '
    'varies with depth.',
]


def load():
    """Real inputs in VCF donor order, and each gene's tested variant rows."""
    I = CM.load_point_estimate_inputs(gene_list=str(GENES), regions=str(REGIONS))
    keep = I['keep']
    R = {k: I[k][:, keep] for k in ('pL', 'pR', 'pT', 'YL', 'YR', 'YT')}
    R['eff_lib'] = I['eff_lib'][keep]
    tested = [I['idx'][CM.gene_variant_index(I, g)] for g in I['genes']]
    nt = np.array([len(t) for t in tested])
    G, N = R['pL'].shape
    print(f'read {G} genes x {N} donors x {R["YL"].shape[2]} Gibbs draws from {GENES}; '
          f'{len(I["vdf"]):,} variants read, {len(I["idx"]):,} pass the tested filter; '
          f'tested variants per gene min {nt.min()} / median {int(np.median(nt))} / max {nt.max()}',
          flush=True)
    if (nt == 0).any():
        raise SystemExit(f'{int((nt == 0).sum())} genes have no tested variant in {REGIONS}')
    for k in ('pL', 'pR', 'YL', 'YR'):
        if R[k].min() < 0:
            raise SystemExit(f'negative Salmon count in {k}: {R[k].min()}')
    hap = R['pL'] + R['pR']
    print(f'donor-gene pairs {G * N:,}; with haplotype-informative reads (pL + pR > 0) '
          f'{int((hap > 0).sum()):,}; with min(pL, pR) >= {EXPRESSIBLE_MIN} '
          f'{int((np.minimum(R["pL"], R["pR"]) >= EXPRESSIBLE_MIN).sum()):,}', flush=True)
    return I, R, tested


def allelic_kept(pL, pR, Va):
    """Pairs the allelic channel fits: Va > EPS and not exactly one haplotype below EXPRESSIBLE_MIN."""
    return (Va > EPS) & ~((pL < EXPRESSIBLE_MIN) ^ (pR < EXPRESSIBLE_MIN))


def thin(y, f, rng):
    """Binomial thinning of a fractional Salmon count; the identity at f = 1."""
    n = np.floor(y)
    return rng.binomial(n.astype(np.int64), f) + f * (y - n)


def move_records(R, perm, swap):
    """Column i takes real record perm[i]; L and R exchanged where swap[i] = -1."""
    s2, s3 = (swap < 0)[None, :], (swap < 0)[None, :, None]
    pL, pR, YL, YR = (R[k][:, perm] for k in ('pL', 'pR', 'YL', 'YR'))
    return dict(pL=np.where(s2, pR, pL), pR=np.where(s2, pL, pR), pT=R['pT'][:, perm],
                YL=np.where(s3, YR, YL), YR=np.where(s3, YL, YR), YT=R['YT'][:, perm],
                eff_lib=R['eff_lib'][perm])


def remainder(L, R, T, what):
    """U = T - L - R clipped at 0, after checking it is above -ROUNDING_TOL."""
    U = T - L - R
    if U.min() < -ROUNDING_TOL:
        raise SystemExit(f'{what}: total minus paired haplotypes is {U.min():.3g} < -{ROUNDING_TOL}')
    return np.maximum(U, 0.0)


def thin_haplotypes(L, R, T, fl, fr, rng, what):
    Uc = remainder(L, R, T, what)
    L2, R2, U2 = thin(L, fl, rng), thin(R, fr, rng), thin(Uc, (fl + fr) / 2, rng)
    return L2, R2, T - ((L - L2) + (R - R2) + (Uc - U2))


def allelic_variance(pL, pR, pL2, pR2, YL, YR):
    """Va' = dv q(pL', pR') / q(pL, pR) + q_a(pL', pR'), dv from the real draws; 0 where pL' + pR' = 0.

    Written in the operation order of summaries_from_point_estimates so that at
    pL' = pL, pR' = pR it returns that function's Va bit for bit.
    """
    dv = np.log2((YL + KAPPA) / (YR + KAPPA)).var(axis=2, ddof=0)
    q = 1.0 / (pL + KAPPA) + 1.0 / (pR + KAPPA)
    q2 = 1.0 / (pL2 + KAPPA) + 1.0 / (pR2 + KAPPA)
    Va = dv * (q2 / q) + q2 / LN2 ** 2
    return np.where((pL2 + pR2) <= 0, 0.0, Va)


def generate(R, perm, swap, fL, fR, rng):
    """Move records, thin haplotype L by fL and R by fR ([G, N]), summarize."""
    M = move_records(R, perm, swap)
    out = dict(expressible=np.minimum(M['pL'], M['pR']) >= EXPRESSIBLE_MIN, eff_lib=M['eff_lib'])
    out['pL'], out['pR'], out['pT'] = thin_haplotypes(M['pL'], M['pR'], M['pT'], fL, fR, rng,
                                                      'point estimates')
    _, _, out['YT'] = thin_haplotypes(M['YL'], M['YR'], M['YT'], fL[..., None], fR[..., None], rng,
                                      'Gibbs draws')
    out['A'], out['T'], _, out['Vt'], _ = summaries_from_point_estimates(
        out['pL'], out['pR'], out['pT'], M['eff_lib'], M['YL'], M['YR'], out['YT'])
    out['Va'] = allelic_variance(M['pL'], M['pR'], out['pL'], out['pR'], M['YL'], M['YR'])
    out['removed'] = (M['pT'] - out['pT']).sum(0)
    out['moved'] = M
    return out


def record_permutation(N, r):
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(PERM_KEY, r)))
    return rng.permutation(N), (rng.integers(0, 2, N) * 2 - 1).astype(np.int8)


def slope_with_intercept(x, y):
    """Least-squares slope of y on x with intercept, per row; NaN where x is constant."""
    xc = x - x.mean(1, keepdims=True)
    den = (xc ** 2).sum(1)
    out = np.full(len(x), np.nan)
    ok = den > 0
    out[ok] = (xc[ok] * y[ok]).sum(1) / den[ok]
    return out


def total_truth(g, n_low, f):
    """Count scale: slope, with intercept, of log2((n_low f + 2 - n_low) / 2) on g / 2, per gene row."""
    return slope_with_intercept(g / 2, np.log2((n_low * f[:, None] + (2 - n_low)) / 2))


def pipeline_truth(M, fL, fR, s, g, kept):
    """Pipeline scale: noise-free shifts A*, T* of the real records, regressed as the pipeline does."""
    pL, pR, pT = M['pL'], M['pR'], M['pT']
    a_star = np.log2((fL * pL + KAPPA) / (fR * pR + KAPPA)) - np.log2((pL + KAPPA) / (pR + KAPPA))
    U = remainder(pL, pR, pT, 'point estimates')
    k = 1e6 / M['eff_lib'][None, :]
    pT_exp = pT - (1 - fL) * pL - (1 - fR) * pR - (1 - (fL + fR) / 2) * U
    t_star = np.log2(k * pT_exp + 1.0) - np.log2(k * pT + 1.0)
    sk = s * kept
    ss = (sk ** 2).sum(1)
    ba = np.full(len(s), np.nan)
    ok = ss > 0
    ba[ok] = (sk[ok] * a_star[ok]).sum(1) / ss[ok]
    return ba, slope_with_intercept(g / 2, t_star)


def build_dataset(I, R, tested, beta, null_fraction, r):
    """Dataset r of scenario |beta|: permuted, swapped, thinned records and their truth."""
    G, N = R['pL'].shape
    perm, swap = record_permutation(N, r)
    drng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(DESIGN_KEY, r)))
    is_null = np.zeros(G, bool)
    is_null[drng.permutation(G)[:int(round(null_fraction * G))]] = True
    j = np.array([t[drng.integers(len(t))] for t in tested])
    b = drng.choice([-1.0, 1.0], size=G) * beta
    xL, xR = I['xL'][j].astype(float), I['xR'][j].astype(float)
    if not (np.isin(xL, (0, 1)).all() and np.isin(xR, (0, 1)).all()):
        raise SystemExit('phased alleles outside {0, 1} at a causal variant')
    g = I['dos'][j].astype(float)
    if not np.array_equal(g, xL + xR):
        raise SystemExit('ALT dosage differs from xL + xR at a causal variant')
    lower = np.where(b > 0, 0.0, 1.0)[:, None]
    fL = np.where(xL == lower, 2.0 ** -beta, 1.0)
    fR = np.where(xR == lower, 2.0 ** -beta, 1.0)
    fbar = (fL.sum(1) + fR.sum(1)) / (2 * N)
    fL[is_null], fR[is_null] = fbar[is_null, None], fbar[is_null, None]
    trng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(THIN_KEY, r, int(round(1000 * beta)))))
    out = generate(R, perm, swap, fL, fR, trng)
    af = g.mean(1) / 2.0
    n_low = (xL == lower).astype(float) + (xR == lower)
    tt = total_truth(g, n_low, np.full(G, 2.0 ** -beta))
    kept = allelic_kept(out['pL'], out['pR'], out['Va'])
    ba_p, bt_p = pipeline_truth(out.pop('moved'), fL, fR, xL - xR, g, kept)
    out.update(perm=perm, swap=swap, is_null=is_null, beta=b, fL=fL, fR=fR, fbar=fbar,
               causal_variant=np.asarray(I['vdf'].index[j].astype(str), dtype=str), causal_row=j,
               causal_maf=np.minimum(af, 1 - af), kept=kept,
               d=np.where(is_null[:, None], 0.0, b[:, None] * (xL - xR)), het=xL != xR,
               allelic_truth=np.where(is_null, 0.0, b), total_truth=np.where(is_null, 0.0, tt),
               allelic_truth_pipeline=np.where(is_null, 0.0, ba_p),
               total_truth_pipeline=np.where(is_null, 0.0, bt_p))
    return out


STORED = ('A', 'T', 'Va', 'Vt', 'pL', 'pR', 'pT', 'eff_lib', 'perm', 'swap', 'is_null', 'beta',
          'causal_variant', 'causal_maf', 'd', 'expressible', 'fL', 'fR', 'allelic_truth', 'total_truth',
          'allelic_truth_pipeline', 'total_truth_pipeline')


def write_atomic(path, write, mode='wb'):
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, mode) as fh:
        write(fh)
    os.replace(tmp, path)


def dumps(obj):
    """JSON with every NaN written as null; a stray non-finite value that is not NaN stops the write."""
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, float) and np.isnan(x):
            return None
        return x
    return json.dumps(clean(obj), indent=1, allow_nan=False)


def main():
    counts = {b: int(sys.argv[1]) for b in BETAS} if len(sys.argv) > 1 else dict(N_DATASETS)
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    out.mkdir(parents=True, exist_ok=True)
    stale = sorted(str(p) for p in out.glob('beta*/rep*.npz')
                   if float(p.parent.name[4:]) not in counts or int(p.stem[3:]) >= counts[float(p.parent.name[4:])])
    if stale:
        raise SystemExit(f'{len(stale)} datasets outside this run (datasets per beta {counts}) exist, '
                         f'e.g. {stale[0]}; remove them')
    I, R, tested = load()
    G, N = R['pL'].shape
    hap_real = np.median(R['pL'] + R['pR'], axis=1)
    med_lib = float(np.median(R['eff_lib']))
    rows, removed, seconds = [], [], []
    for beta in BETAS:
        nf = 1.0 if beta == 0 else NULL_FRACTION
        (out / f'beta{beta}').mkdir(exist_ok=True)
        for r in range(counts[beta]):
            t0 = time.perf_counter()
            ds = build_dataset(I, R, tested, beta, nf, r)
            write_atomic(out / f'beta{beta}' / f'rep{r:03d}.npz',
                         lambda fh: np.savez(fh, **{k: ds[k] for k in STORED}))
            seconds.append(time.perf_counter() - t0)
            nn = ~ds['is_null']
            het = ds['het'] & nn[:, None]
            informative = het & (ds['Va'] > EPS)
            if beta > 0:
                removed.append(ds['removed'] / med_lib)
            for k in np.where(nn)[0]:
                rows.append(dict(beta_abs=beta, rep=r, gene=I['genes'][k],
                                 causal_variant=ds['causal_variant'][k], causal_maf=ds['causal_maf'][k],
                                 beta=ds['beta'][k], allelic_truth=ds['allelic_truth'][k],
                                 total_truth=ds['total_truth'][k],
                                 allelic_truth_pipeline=ds['allelic_truth_pipeline'][k],
                                 total_truth_pipeline=ds['total_truth_pipeline'][k],
                                 n_het=int(ds['het'][k].sum()),
                                 n_het_expressible=int((ds['het'][k] & ds['expressible'][k]).sum()),
                                 n_het_informative=int(informative[k].sum()),
                                 n_het_kept=int((ds['het'][k] & ds['kept'][k]).sum()),
                                 mean_factor=ds['fbar'][k], median_hap_reads_real=hap_real[k]))
            if nn.any():
                tr = ds['total_truth'][nn] / ds['beta'][nn]
                ap = ds['allelic_truth_pipeline'][nn] / ds['beta'][nn]
                tp = ds['total_truth_pipeline'][nn] / ds['total_truth'][nn]
                eff = (f'het donor-gene pairs in non-null genes {int(het.sum())}, expressible '
                       f'{(het & ds["expressible"]).sum() / het.sum():.3f}, informative after thinning '
                       f'{informative.sum() / het.sum():.3f}, kept by the allelic admission '
                       f'{(het & ds["kept"]).sum() / het.sum():.3f}; total truth / beta median '
                       f'{np.nanmedian(tr):.3f} [{np.nanmin(tr):.3f}, {np.nanmax(tr):.3f}]; pipeline-scale / '
                       f'count-scale truth median allelic {np.nanmedian(ap):.3f} [{np.nanmin(ap):.3f}, '
                       f'{np.nanmax(ap):.3f}], total {np.nanmedian(tp):.3f} [{np.nanmin(tp):.3f}, '
                       f'{np.nanmax(tp):.3f}]; non-finite truths among non-null genes: count-scale total '
                       f'{int((~np.isfinite(tr)).sum())}, pipeline allelic {int((~np.isfinite(ap)).sum())}, '
                       f'pipeline total {int((~np.isfinite(tp)).sum())}')
            else:
                eff = 'no thinning'
            print(f'beta {beta} rep {r:03d}: {int(nn.sum())} non-null / {int((~nn).sum())} null genes; {eff}; '
                  f'reads removed per donor / median library max {ds["removed"].max() / med_lib:.2e}; '
                  f'{seconds[-1]:.2f} s', flush=True)
    truth = pd.DataFrame(rows)
    write_atomic(out / 'truth.tsv', lambda fh: truth.to_csv(fh, sep='\t', index=False), 'w')
    removed = np.concatenate(removed) if removed else None
    meta = dict(
        genes=list(I['genes']), donors=list(I['order']), gene_list=str(GENES), regions=str(REGIONS),
        n_genes=G, n_donors=N, n_datasets={str(b): counts[b] for b in BETAS}, betas=list(BETAS),
        null_fraction={str(b): (1.0 if b == 0 else NULL_FRACTION) for b in BETAS},
        seed=SEED, streams={'perm and swap': f'SeedSequence({SEED}, spawn_key=({PERM_KEY}, r))',
                            'null set, causal variant, sign': f'SeedSequence({SEED}, spawn_key=({DESIGN_KEY}, r))',
                            'binomial thinning': f'SeedSequence({SEED}, spawn_key=({THIN_KEY}, r, round(1000*|beta|)))'},
        conventions={
            'arrays': '[G, N] in genes x donors order above; donors in VCF (genotype) order',
            'perm': 'column i holds real record perm[i] (values, Gibbs draws, library size)',
            'swap': '-1 where that record\'s L and R were exchanged before thinning',
            'eff_lib': 'edgeR effective library size of the record in each column (real, unthinned)',
            'covariates': 'RNA-tied covariates of dataset r are compare_mixqtl_replication.'
                          'load_point_estimate_inputs cov_df rows in the order perm; genotype PCs '
                          '(geno_cov_df) stay unpermuted',
            'Va': 'dv * q(pL\', pR\') / q(pL, pR) + q_a(pL\', pR\'), dv the across-draw variance of '
                  'log2((YL+0.5)/(YR+0.5)) over the real (unthinned) Gibbs draws of the record, '
                  'q(x, y) = 1/(x+0.5) + 1/(y+0.5), q_a = q / ln(2)^2; 0 where pL\' + pR\' = 0; '
                  'zero-haplotype drop not applied (allelic_kept applies it)',
            'Vt': 'summaries_from_point_estimates on the thinned YT Gibbs draws',
            'beta': 'signed log2 allelic fold change ALT over REF, drawn for every gene; '
                    'null genes (is_null) carry no effect and were thinned uniformly by their mean factor',
            'allelic_truth': 'count scale: beta for non-null genes (slope of log2 L/R on xL - xR), 0 for null genes',
            'total_truth': 'count scale: least-squares slope with intercept of log2((n_low f + 2 - n_low)/2) '
                           'on ALT dosage / 2 over the 92 donors; 0 for null genes',
            'allelic_truth_pipeline': 'unweighted through-origin slope of A* on xL - xR over the records '
                                      'allelic_kept admits; 0 for null genes',
            'total_truth_pipeline': 'unweighted slope with intercept of T* on ALT dosage / 2 over all '
                                    'donors; 0 for null genes',
            'd': 'true log2 aFC on A = log2(L/R) per donor, beta (xL - xR); 0 for null genes',
            'expressible': f'min(pL, pR) >= {EXPRESSIBLE_MIN} before thinning',
            'units': 'A log2((L+0.5)/(R+0.5)); T log2(CPM+1) on the real edgeR effective library size'},
        library_size_change=None if removed is None else dict(
            statistic='reads removed per donor over the median effective library size, datasets with beta > 0',
            median=float(np.median(removed)), max=float(removed.max())),
        cannot_answer=CANNOT_ANSWER)
    write_atomic(out / 'meta.json', lambda fh: fh.write(dumps(meta)), 'w')
    lib = ('none (no beta > 0)' if removed is None
           else f'median {np.median(removed):.2e}, max {removed.max():.2e}')
    per = truth.groupby(['beta_abs', 'rep']).size() if len(truth) else pd.Series([0])
    exp = truth.n_het_expressible.sum() / truth.n_het.sum() if len(truth) else float('nan')
    print(f'wrote {sum(counts.values())} datasets (per beta {counts}) to {out}: {G} genes x {N} donors; '
          f'truth.tsv {len(truth)} rows (non-null genes per dataset with beta > 0 {per.min()}-{per.max()}); '
          f'expressible share of het donor-gene pairs in non-null genes {exp:.3f}; '
          f'reads removed per donor / median library (beta > 0): {lib}; seconds per dataset median '
          f'{np.median(seconds):.2f}, max {max(seconds):.2f}', flush=True)


if __name__ == '__main__':
    main()
