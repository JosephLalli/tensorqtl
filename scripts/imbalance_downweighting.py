"""Why some genes are CONSERVATIVE under the records null, and whether the
pooled allelic excess is the net of two opposing weight-residual couplings.

THE QUESTION. Under the shared null instrument
(scripts/null_permutation_instrument.py; 46 genes x 92 donors x 2,000
records permutations at RASQUAL's observed lead), the allelic nominal p is
anticonservative when pooled (0.069 at 0.05). Per gene the first-order closed
form of sdratio_a^2 is

    R_g = n_a * sum(w z^2) / (sum(w) * sum(z^2)) = 1 + Cov(w, z^2)/(mean w * mean z^2)

with w = 1/va (the record's weight) and z^2 = a^2/va (its whitened squared
residual under the null). R_g = 1 in expectation under the shipped model
Var(a) = sigma^2 va. Genes with R_g > 1 carry large z^2 at HIGH weight and are
anticonservative; genes with R_g < 1 carry large z^2 at LOW weight and are
conservative. This script asks what the low-weight, large-z^2 records ARE,
why the Gibbs variance is large for them, and what the pooled rate would be
with each coupling alone.

DEFINITIONS, used throughout (natural-log units):

  mL, mR   posterior-mean haplotype counts over the 200 Gibbs draws
  q_k      1/(mL + 1/2) + 1/(mR + 1/2), the Poisson (counting) variance of
           the log ratio with the Haldane-Anscombe half-count; it is exactly
           the count_noise term already added into va
  AF       ambiguity factor, va / q_k. Because va = Gibbs across-draw variance
           + q_k, AF - 1 is the pure Gibbs part in units of counting variance.
  keff     effective binomial sample size implied by the draws,
           pbar(1-pbar)/Var_draws(p) with p = yL/(yL+yR) per draw: the number
           of reads a binomial split would need to be as uncertain as the
           Gibbs posterior is
  corrLR   across-draw correlation of yL and yR; strongly negative when the
           Gibbs variance is reads being traded between the two haplotype
           copies (assignment ambiguity), near zero when it is independent
           shot noise
  kinf, ap, qinf
           phASER (read-back phasing of RNA alignments at heterozygous SNPs,
           Castel et al. 2016) haplotype counts aCount + bCount per donor-gene,
           ap = log((aCount+1/2)/(bCount+1/2)), qinf = 1/(aCount+1/2) +
           1/(bCount+1/2). An INDEPENDENT measurement of the same donor's
           allelic ratio from aligned reads, without Salmon's EM. Used only
           where gw_phased == 1 (haplotype A anchored to the genome-wide phase)
  dz       discordance z, (a - ap)/sqrt(va + qinf): how far Salmon's
           posterior-mean log ratio sits from phASER's, in units of the two
           stated uncertainties. a and ap share reads, so their errors are
           positively correlated and this sd is an over-statement; |dz| >= 3
           is therefore a conservative flag of a Salmon point estimate that
           the aligned reads contradict.

DESIGN. Held fixed: the instrument's 46 genes, fixed lead variant, the 92
donors in genotype order, the allelic regressor s = xL - xR, the allelic fit
(weighted least squares through the origin, sigma_a^2 = RSS/(n_a - 1), t^2
referred to F(1, n_a - 1)), the total channel exactly as the instrument
reported it, and the permutation stream RandomState(42) with
perms = [rng.permutation(92) for _ in range(2000)]. What varies, per arm, is
ONLY the allelic record set (a, va) that is permuted:

  real                   the instrument's records (GATED, see below)
  drop_lowW_top{1,3}     remove the 1 or 3 largest-z^2 records among the
                         records below the gene's median weight
  drop_lowW_rand{1,3}    control: remove 1 or 3 RANDOM records below the median
                         weight (20 seeded draws per gene)
  drop_highW_top{1,3}    mirror: largest-z^2 records above the median weight
  poisson_medAF          va' = median_g(AF) * q_k: each record's own ambiguity
                         inflation replaced by the gene's median, keeping the
                         gene's overall scale and the counting shape
  drop_discordant        remove records with |dz| >= 3 (a Salmon-error filter,
                         not a weight filter)
  phaser                 phASER records (ap, qinf) in place of Salmon's
  phaser_a_salmon_v      phASER point estimate ap at Salmon's va
  salmon_a_phaser_v      Salmon's a at phASER's qinf
  neutralize_lowT        z of the bottom weight tercile replaced by
                         sigma * N(0,1), sigma^2 = mean z^2 of the other two
                         terciles (20 seeded record sets): only the coupling
                         carried by mid/high-weight records remains
  neutralize_highT       the same for the top weight tercile
  neutralize_both        low and high terciles both neutralized (reference)
  shuffle_a              a shuffled against va within gene (20 seeded sets):
                         what the pooled rate would be if the Gibbs variance
                         did not track each record's own imbalance
  model, model_drop_*    SELECTION CONTROL: every z drawn from sigma*N(0,1) at
                         the real weights (the shipped model, 20 seeded sets),
                         then the same drop_*_top rule applied. Removing the
                         largest z^2 of a weight half moves R by construction;
                         the real-data change is reported net of this one

The combined (inverse-variance meta-analysis) statistic is reported for every
arm by recombining the arm's allelic slope and standard error with the
instrument's total-channel slope, se and dof for the same (gene, permutation).

GATES (the script aborts on failure):

  gate_summaries  compute_summaries_from_gibbs on the Gibbs cache, mapped by
                  sample id (never by position), reproduces the instrument's
                  a and va exactly (max |diff| 0)
  gate_real       the vectorized allelic t^2 of arm 'real' reproduces
                  null_long.tsv.gz t2_a for every (gene, perm) to relative
                  1e-9, and the pooled allelic and combined rejection counts
                  at 0.05 / 0.01 / 0.001 equal the instrument's exactly
  gate_R          R_g computed here equals the closed form on the npz arrays

Intervals: gene-clustered percentile bootstrap (2,000 resamples of the 46
genes, SeedSequence(42) child stream); for differences between arms, a PAIRED
gene-clustered bootstrap of the difference (same resampled genes for both
arms), which is the noise floor for that comparison. Seeded arms also report
the Monte Carlo sd of the pooled rate across their record sets.

Master seed 42: the permutation stream is RandomState(42) (the instrument's);
everything else draws from np.random.SeedSequence(42) child streams.

Outputs in /mnt/ssd/lalli/brainvar_hapmix_deploy/imbalance_downweighting_20260925/:
  records.tsv        one row per admitted (gene, donor) record
  top_records.tsv    per gene, the 6 largest-z^2 records with shares
  per_gene.tsv       R, covariance decomposition, AF statistics, per-arm R/rates
  arms_pooled.tsv    pooled rates per arm, channel and alpha, with intervals
  summary.json       everything headline, with the quantities behind it
  fig_af_vs_imbalance.png, fig_R_by_arm.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
INST = D / 'nominal_p_null_instrument_20260925'
CACHE = D / 'cache' / 'gibbs_56b63c3b37ed5df8'
OUT = D / 'imbalance_downweighting_20260925'
SEED, EPS, KAPPA = 42, 1e-12, 0.5
N_PERM, N_BOOT, N_MC = 2000, 2000, 20
ALPHAS = (0.05, 0.01, 0.001)
DZ_FLAG = 3.0

SS = np.random.SeedSequence(SEED)
(SS_BOOT, SS_RAND, SS_NEUT, SS_SHUF, SS_NULLSP, SS_MODEL) = SS.spawn(6)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load():
    import run_hapmixqtl_from_salmon as H
    d = np.load(INST / 'inputs_at_lead.npz', allow_pickle=True)
    genes, donors = list(d['genes']), list(d['donors'])
    cg = open(CACHE / 'genes.txt').read().split()
    cs = open(CACHE / 'samples.txt').read().split()
    rows = [cg.index(g) for g in genes]
    keep = [cs.index(s) for s in donors]          # by sample id, never position
    mm = {k: np.load(CACHE / f'{k}.npy', mmap_mode='r') for k in ('YL', 'YR', 'YT')}
    Y = {k: np.asarray(mm[k][rows])[:, keep] for k in mm}
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        A, _, Va, _, _ = H.compute_summaries_from_gibbs(Y['YL'], Y['YR'], yT=Y['YT'])
    da = float(np.nanmax(np.abs(A - d['a'])))
    dv = float(np.nanmax(np.abs(Va - d['va'])))
    print(f'gate_summaries: max|da| {da:.3g}, max|dva| {dv:.3g}')
    if da > 0 or dv > 0:
        raise SystemExit('gate_summaries FAILED: cache does not reproduce the npz')
    return d, genes, donors, Y, dict(max_abs_da=da, max_abs_dva=dv)


def load_phaser(genes):
    man = pd.read_csv(D / 'phaser_manifest.tsv', sep='\t', header=None,
                      names=['donor', 'prefix'])
    gs = set(genes)
    parts = []
    for donor, pfx in man.itertuples(index=False):
        g = pd.read_csv(f'{pfx}.gene_ae.txt', sep='\t',
                        usecols=['name', 'aCount', 'bCount', 'n_variants', 'gw_phased'])
        g = g[g.name.isin(gs)].copy()
        g['donor'] = donor
        parts.append(g)
    P = pd.concat(parts).rename(columns={'name': 'gene'})
    if P.duplicated(['gene', 'donor']).any():
        raise SystemExit('phASER: duplicated (gene, donor) rows')
    return P


# --------------------------------------------------------------------------
# per-record table
# --------------------------------------------------------------------------
def record_table(d, genes, donors, Y, P):
    YL, YR = Y['YL'], Y['YR']
    mL, mR = YL.mean(2), YR.mean(2)
    qk = 1.0 / (mL + KAPPA) + 1.0 / (mR + KAPPA)
    cL = YL - mL[..., None]
    cR = YR - mR[..., None]
    with np.errstate(all='ignore'):
        corr = (cL * cR).mean(2) / np.sqrt((cL ** 2).mean(2) * (cR ** 2).mean(2))
        p = YL / (YL + YR)
        vp = np.nanvar(p, axis=2)
        pb = mL / (mL + mR)
        keff = pb * (1 - pb) / vp
    rec = []
    for i, g in enumerate(genes):
        a, va, s = d['a'][i], d['va'][i], d['s'][i]
        ok = np.isfinite(a) & np.isfinite(va) & (va > EPS)
        for j in np.where(ok)[0]:
            rec.append(dict(gene=g, stratum=str(d['strata'][i]), gi=i, j=j,
                            donor=donors[j], a=a[j], va=va[j], mL=mL[i, j],
                            mR=mR[i, j], n=mL[i, j] + mR[i, j], qk=qk[i, j],
                            AF=va[j] / qk[i, j], keff=keff[i, j],
                            corrLR=corr[i, j], s=s[j], w=1 / va[j],
                            z2=a[j] ** 2 / va[j]))
    R = pd.DataFrame(rec)
    R = R.merge(P, on=['gene', 'donor'], how='left')
    R['kinf'] = R.aCount + R.bCount
    R['ap'] = np.log((R.aCount + KAPPA) / (R.bCount + KAPPA))
    R['qinf'] = 1 / (R.aCount + KAPPA) + 1 / (R.bCount + KAPPA)
    R['comparable'] = (R.gw_phased == 1) & (R.kinf >= 1)
    R['dz'] = np.where(R.comparable, (R.a - R.ap) / np.sqrt(R.va + R.qinf), np.nan)
    R['discordant'] = R.comparable & (R.dz.abs() >= DZ_FLAG)
    R['absa'] = R.a.abs()
    # within-gene shares and weight terciles
    R['w_share'] = R.w / R.groupby('gene').w.transform('sum')
    R['z2_share'] = R.z2 / R.groupby('gene').z2.transform('sum')
    R['w_rank'] = R.groupby('gene').w.rank(pct=True)
    R['w_tercile'] = pd.cut(R.w_rank, [0, 1 / 3, 2 / 3, 1.0001],
                            labels=['low', 'mid', 'high']).astype(str)
    R['medAF_gene'] = R.groupby('gene').AF.transform('median')
    return R


def variance_budget(R):
    """Within-gene variance budget of Salmon's a against phASER's ap.

    With both centred on their gene mean, cov(a, ap) estimates the variance of
    the allelic ratio the two measurements share. What var(a) holds beyond
    that shared part and beyond Salmon's stated mean va is variance Salmon's
    posterior does not declare; likewise for ap against qinf. The naive slope
    of a on ap is attenuated by ap's own counting noise, so it is reported
    with the reliability correction 1 - mean(qinf)/var(ap)."""
    out = {}
    for kmin in (50, 200):
        c = R[R.comparable & (R.kinf >= kmin)]
        a = c.a - c.groupby('gene').a.transform('mean')
        p = c.ap - c.groupby('gene').ap.transform('mean')
        va_, vp_, cv = float(a.var()), float(p.var()), float(np.cov(a, p)[0, 1])
        rel = 1 - c.qinf.mean() / vp_
        out[f'kinf>={kmin}'] = dict(
            n=len(c), var_a=va_, var_ap=vp_, cov_a_ap=cv,
            mean_va=float(c.va.mean()), mean_qinf=float(c.qinf.mean()),
            undeclared_var_a=va_ - cv - float(c.va.mean()),
            undeclared_var_ap=vp_ - cv - float(c.qinf.mean()),
            naive_slope_a_on_ap=cv / vp_, reliability_ap=float(rel),
            corrected_slope_a_on_ap=cv / vp_ / rel,
            median_abs_a_over_abs_ap=float(c.a.abs().median() / c.ap.abs().median()))
    return out


def R_of(w, z2):
    w, z2 = np.asarray(w, float), np.asarray(z2, float)
    return len(w) * np.sum(w * z2) / (np.sum(w) * np.sum(z2))


# --------------------------------------------------------------------------
# the vectorized allelic channel on the instrument's permutation stream
# --------------------------------------------------------------------------
def perm_stream():
    rng = np.random.RandomState(SEED)
    return np.array([rng.permutation(92) for _ in range(N_PERM)])


def allelic_perm(a_full, va_full, s, perms):
    """a_full, va_full: (92,) record arrays, NaN / <=EPS = not admitted.
    Returns dict of (P,) arrays: ba, sea2, t2, dofa, valid."""
    ok = np.isfinite(a_full) & np.isfinite(va_full) & (va_full > EPS)
    w = np.where(ok, 1.0 / np.where(ok, va_full, 1.0), 0.0)
    wa = np.where(ok, w * np.where(ok, a_full, 0.0), 0.0)
    S = float(np.sum(wa * np.where(ok, a_full, 0.0)))
    n = int(ok.sum())
    num = wa[perms] @ s
    den = w[perms] @ (s * s)
    valid = den > 0
    with np.errstate(all='ignore'):
        ba = num / den
        rss = S - num ** 2 / den
        sea2 = rss / (n - 1) / den
        t2 = (n - 1) * num ** 2 / (den * S - num ** 2)
    valid &= np.isfinite(t2) & (sea2 > 0)
    return dict(ba=ba, sea2=sea2, t2=t2, dofa=np.full(len(perms), max(n - 1, 1)),
                valid=valid, n=n)


def combine(al, tot):
    """Inverse-variance meta-analysis with the instrument's total channel."""
    set2 = tot['set'].values ** 2
    prec = 1 / al['sea2'] + 1 / set2
    b = (al['ba'] / al['sea2'] + tot['bt'].values / set2) / prec
    t2b = b ** 2 * prec
    dof = np.minimum(al['dofa'], tot['doft'].values)
    return sps.f.sf(t2b, 1, dof)


# --------------------------------------------------------------------------
# rates and intervals
# --------------------------------------------------------------------------
def counts(pdict, alpha):
    k = np.array([np.sum(p < alpha) for p in pdict.values()], float)
    n = np.array([len(p) for p in pdict.values()], float)
    return k, n


BOOT_IDX = np.random.default_rng(SS_BOOT).integers(0, 46, size=(N_BOOT, 46))


def rate_ci(pdict, alpha):
    k, n = counts(pdict, alpha)
    b = k[BOOT_IDX].sum(1) / n[BOOT_IDX].sum(1)
    return float(k.sum() / n.sum()), float(np.quantile(b, .025)), float(np.quantile(b, .975))


def diff_ci(pX, p0, alpha):
    kx, nx = counts(pX, alpha)
    k0, n0 = counts(p0, alpha)
    b = kx[BOOT_IDX].sum(1) / nx[BOOT_IDX].sum(1) - k0[BOOT_IDX].sum(1) / n0[BOOT_IDX].sum(1)
    return (float(kx.sum() / nx.sum() - k0.sum() / n0.sum()),
            float(np.quantile(b, .025)), float(np.quantile(b, .975)))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    OUT.mkdir(exist_ok=True)
    d, genes, donors, Y, g0 = load()
    P = load_phaser(genes)
    R = record_table(d, genes, donors, Y, P)
    s_all = d['s'].astype(float)
    NL = pd.read_csv(INST / 'null_long.tsv.gz', sep='\t')
    NL = NL.set_index(['gene', 'perm']).sort_index()
    perms = perm_stream()
    summary = dict(gate_summaries=g0, n_records=len(R), n_genes=len(genes))

    # ---------------- per-gene R, gate_R ---------------------------------
    per = []
    for g, x in R.groupby('gene', sort=False):
        Rg = R_of(x.w, x.z2)
        n = len(x)
        wb, zb = x.w.mean(), x.z2.mean()
        contrib = (x.w - wb) * (x.z2 - zb) / (n * wb * zb)
        big = x.z2 > zb
        hiw = x.w > wb
        row = dict(gene=g, stratum=x.stratum.iloc[0], n_a=n, R=Rg,
                   R_minus_1_check=float(contrib.sum()),
                   C_low=float(contrib[x.w_tercile == 'low'].sum()),
                   C_mid=float(contrib[x.w_tercile == 'mid'].sum()),
                   C_high=float(contrib[x.w_tercile == 'high'].sum()),
                   Q_bigz2_highw=float(contrib[big & hiw].sum()),
                   Q_bigz2_loww=float(contrib[big & ~hiw].sum()),
                   Q_smallz2_highw=float(contrib[~big & hiw].sum()),
                   Q_smallz2_loww=float(contrib[~big & ~hiw].sum()),
                   medAF=float(x.AF.median()), med_n=float(x.n.median()),
                   sp_AF_absa=float(sps.spearmanr(x.AF, x.absa)[0]),
                   sp_AF_logn=float(sps.spearmanr(x.AF, x.n)[0]),
                   sp_AF_z2=float(sps.spearmanr(x.AF, x.z2)[0]),
                   sp_w_z2=float(sps.spearmanr(x.w, x.z2)[0]),
                   n_discordant=int(x.discordant.sum()),
                   n_comparable=int(x.comparable.sum()))
        c = x[x.comparable & (x.kinf >= 10)]
        row['sp_a_ap'] = float(sps.spearmanr(c.a, c.ap)[0]) if len(c) > 5 else np.nan
        row['median_keff_over_kinf'] = float((c.keff / c.kinf).median()) if len(c) else np.nan
        per.append(row)
    per = pd.DataFrame(per).set_index('gene')
    gR = float(np.max(np.abs(per.R - 1 - per.R_minus_1_check)))
    print(f'gate_R: max |(R-1) - sum of covariance contributions| {gR:.2e}')
    if gR > 1e-9:
        raise SystemExit('gate_R FAILED')
    summary['gate_R_max_abs'] = gR

    # ---------------- arms: record sets --------------------------------
    gi = {g: i for i, g in enumerate(genes)}
    base = {g: (d['a'][gi[g]].astype(float).copy(), d['va'][gi[g]].astype(float).copy())
            for g in genes}
    Rg_ = {g: x for g, x in R.groupby('gene', sort=False)}
    rng_rand = np.random.default_rng(SS_RAND)
    rng_neut = np.random.default_rng(SS_NEUT)
    rng_shuf = np.random.default_rng(SS_SHUF)
    rng_model = np.random.default_rng(SS_MODEL)

    def drop(g, js):
        a, va = base[g][0].copy(), base[g][1].copy()
        va[list(js)] = 0.0
        return a, va

    def arm_sets(g):
        """Yields (arm, [(a, va), ...]) -- seeded arms give N_MC record sets."""
        x = Rg_[g]
        a0, v0 = base[g]
        out = {'real': [(a0, v0)]}
        lo = x[x.w < x.w.median()]
        hi = x[x.w >= x.w.median()]
        for K in (1, 3):
            out[f'drop_lowW_top{K}'] = [drop(g, lo.nlargest(K, 'z2').j)]
            out[f'drop_highW_top{K}'] = [drop(g, hi.nlargest(K, 'z2').j)]
            out[f'drop_lowW_rand{K}'] = [drop(g, rng_rand.choice(lo.j.values, K, replace=False))
                                         for _ in range(N_MC)]
        v = v0.copy()
        v[x.j.values] = x.medAF_gene.values * x.qk.values
        out['poisson_medAF'] = [(a0, v)]
        out['drop_discordant'] = [drop(g, x[x.discordant].j)]
        c = x[x.comparable]
        aP = np.full(92, np.nan); vP = np.zeros(92)
        aP[c.j.values] = c.ap.values; vP[c.j.values] = c.qinf.values
        out['phaser'] = [(aP, vP)]
        vS = np.zeros(92); vS[c.j.values] = c.va.values
        out['phaser_a_salmon_v'] = [(aP, vS)]
        aS = np.full(92, np.nan); aS[c.j.values] = c.a.values
        out['salmon_a_phaser_v'] = [(aS, vP)]
        for arm, terc in (('neutralize_lowT', ['low']), ('neutralize_highT', ['high']),
                          ('neutralize_both', ['low', 'high'])):
            sel = x.w_tercile.isin(terc).values
            sig2 = x.z2.values[~sel].mean()
            sets = []
            for _ in range(N_MC):
                z = np.sqrt(x.z2.values) * np.sign(x.a.values)
                z = np.where(sel, rng_neut.standard_normal(len(x)) * np.sqrt(sig2), z)
                a = a0.copy(); a[x.j.values] = z * np.sqrt(x.va.values)
                sets.append((a, v0))
            out[arm] = sets
        sets = []
        for _ in range(N_MC):
            a = a0.copy(); a[x.j.values] = rng_shuf.permutation(x.a.values)
            sets.append((a, v0))
        out['shuffle_a'] = sets
        # MODEL CONTROL for the drop_*_top rules: every z drawn from
        # sigma * N(0,1) at the real weights (the shipped model exactly), then
        # the same selection rule applied. Removing the largest z^2 among any
        # subset raises or lowers R by construction; the change these arms
        # show is the selection artifact the real-data change must clear.
        sig2 = x.z2.values.mean()
        med = x.w.median()
        m_sets = {k: [] for k in ('model', 'model_drop_lowW_top1', 'model_drop_lowW_top3',
                                  'model_drop_highW_top1', 'model_drop_highW_top3')}
        for _ in range(N_MC):
            z = rng_model.standard_normal(len(x)) * np.sqrt(sig2)
            a = a0.copy(); a[x.j.values] = z * np.sqrt(x.va.values)
            m_sets['model'].append((a, v0))
            z2m = pd.Series(z ** 2, index=x.index)
            for K in (1, 3):
                jl = x.j[z2m[x.w < med].nlargest(K).index]
                jh = x.j[z2m[x.w >= med].nlargest(K).index]
                v = v0.copy(); v[jl.values] = 0.0
                m_sets[f'model_drop_lowW_top{K}'].append((a, v))
                v = v0.copy(); v[jh.values] = 0.0
                m_sets[f'model_drop_highW_top{K}'].append((a, v))
        out.update(m_sets)
        return out

    ARMS = None
    pa = {}; pb = {}; Rarm = {}; ndrop = {}; nvalid_lost = {}
    gate_rel = 0.0
    for g in genes:
        s = s_all[gi[g]]
        tot = NL.loc[g]
        sets = arm_sets(g)
        if ARMS is None:
            ARMS = list(sets)
            for arm in ARMS:
                pa[arm], pb[arm], Rarm[arm], ndrop[arm], nvalid_lost[arm] = {}, {}, {}, {}, {}
        for arm, lst in sets.items():
            pas, pbs, rs = [], [], []
            for a, va in lst:
                al = allelic_perm(a, va, s, perms)
                keep = al['valid']
                if arm == 'real':
                    ref = tot['t2_a'].values
                    rel = np.max(np.abs(al['t2'] - ref) / np.abs(ref))
                    gate_rel = max(gate_rel, float(rel))
                    if not keep.all() or al['n'] != int(tot.n_a.iloc[0]):
                        raise SystemExit(f'gate_real FAILED at {g}: validity / n_a')
                p_a = sps.f.sf(al['t2'], 1, al['dofa'])[keep]
                p_b = combine(al, tot)[keep]
                pas.append(p_a); pbs.append(p_b)
                ok = np.isfinite(a) & (va > EPS)
                ww = 1 / va[ok]; zz = a[ok] ** 2 / va[ok]
                rs.append(R_of(ww, zz))
                nvalid_lost[arm][g] = int((~keep).sum())
            pa[arm][g] = np.concatenate(pas)
            pb[arm][g] = np.concatenate(pbs)
            Rarm[arm][g] = float(np.mean(rs))
            ndrop[arm][g] = int(Rg_[g].shape[0] - np.sum(np.isfinite(lst[0][0]) & (lst[0][1] > EPS)))
    print(f'gate_real: max relative |t2_a - instrument| {gate_rel:.2e}')
    if gate_rel > 1e-9:
        raise SystemExit('gate_real FAILED: t2_a does not reproduce the instrument')
    for alpha in ALPHAS:
        ka = sum(int(np.sum(NL.loc[g].p_a.values < alpha)) for g in genes)
        kb = sum(int(np.sum(NL.loc[g].p_b.values < alpha)) for g in genes)
        ma = sum(int(np.sum(pa['real'][g] < alpha)) for g in genes)
        mb = sum(int(np.sum(pb['real'][g] < alpha)) for g in genes)
        print(f'gate_real at {alpha}: allelic {ma} vs {ka}, combined {mb} vs {kb}')
        if ka != ma or kb != mb:
            raise SystemExit('gate_real FAILED: pooled rejection counts differ')
    summary['gate_real_max_rel_t2a'] = gate_rel

    # ---------------- pooled rates per arm ------------------------------
    rows = []
    for arm in ARMS:
        for ch, pdict in (('allelic', pa[arm]), ('combined', pb[arm])):
            for alpha in ALPHAS:
                est, lo, hi = rate_ci(pdict, alpha)
                dd, dlo, dhi = diff_ci(pdict, (pa if ch == 'allelic' else pb)['real'], alpha)
                r = dict(arm=arm, channel=ch, alpha=alpha, rate=est, lo=lo, hi=hi,
                         diff_vs_real=dd, diff_lo=dlo, diff_hi=dhi,
                         n_pvals=int(sum(len(v) for v in pdict.values())))
                # Monte Carlo sd across seeded record sets
                nset = len(next(iter(pdict.values()))) // N_PERM
                if nset > 1:
                    by = []
                    for m in range(nset):
                        k = sum(np.sum(p[m * N_PERM:(m + 1) * N_PERM] < alpha)
                                for p in pdict.values() if len(p) == nset * N_PERM)
                        nn = sum(N_PERM for p in pdict.values() if len(p) == nset * N_PERM)
                        by.append(k / nn)
                    r['mc_sd_across_sets'] = float(np.std(by, ddof=1))
                rows.append(r)
        Rs = pd.Series(Rarm[arm])
        per[f'R_{arm}'] = Rs
        per[f'rej05_a_{arm}'] = pd.Series({g: float(np.mean(pa[arm][g] < .05)) for g in genes})
        per[f'rej01_a_{arm}'] = pd.Series({g: float(np.mean(pa[arm][g] < .01)) for g in genes})
        per[f'rej001_a_{arm}'] = pd.Series({g: float(np.mean(pa[arm][g] < .001)) for g in genes})
        per[f'ndrop_{arm}'] = pd.Series(ndrop[arm])
    AP = pd.DataFrame(rows)
    AP.to_csv(OUT / 'arms_pooled.tsv', sep='\t', index=False)
    sign0 = np.sign(per.R - 1)
    summary['arms'] = {}
    for arm in ARMS:
        sub = AP[AP.arm == arm]
        sA = np.sign(per[f'R_{arm}'] - 1)
        summary['arms'][arm] = dict(
            rates={f'{r.channel}@{r.alpha}': dict(rate=r.rate, lo=r.lo, hi=r.hi,
                                                   diff_vs_real=r.diff_vs_real,
                                                   diff_lo=r.diff_lo, diff_hi=r.diff_hi,
                                                   mc_sd=r.get('mc_sd_across_sets', None)
                                                   if not pd.isna(r.get('mc_sd_across_sets', np.nan)) else None)
                   for r in sub.to_dict('records') for r in [pd.Series(r)]},
            n_genes_R_below_1=int((per[f'R_{arm}'] < 1).sum()),
            n_genes_R_sign_change=int((sA != sign0).sum()),
            genes_flipped_to_above=sorted(per.index[(sign0 < 0) & (sA > 0)]),
            genes_flipped_to_below=sorted(per.index[(sign0 > 0) & (sA < 0)]),
            median_R=float(per[f'R_{arm}'].median()),
            sd_logR=float(np.log(per[f'R_{arm}']).std()),
            records_removed_total=int(per[f'ndrop_{arm}'].sum()),
            perms_invalid_total=int(sum(nvalid_lost[arm].values())))
        print(f'{arm:22s} ' + '  '.join(
            f"{r.channel[:3]}@{r.alpha}: {r.rate:.4f} [{r.lo:.4f},{r.hi:.4f}] d={r.diff_vs_real:+.4f} [{r.diff_lo:+.4f},{r.diff_hi:+.4f}]"
            for r in sub.itertuples() if r.channel == 'allelic')
            + f'  R<1: {summary["arms"][arm]["n_genes_R_below_1"]}, sign changes {summary["arms"][arm]["n_genes_R_sign_change"]}')

    # ---------------- (a) ambiguity factor vs imbalance and depth -------
    rngn = np.random.default_rng(SS_NULLSP)
    null_sp = []
    for g, x in R.groupby('gene', sort=False):
        sig = np.sqrt(x.z2.mean())
        sims = [sps.spearmanr(x.AF, np.abs(rngn.standard_normal(len(x)) * sig * np.sqrt(x.va)))[0]
                for _ in range(200)]
        null_sp.append(np.mean(sims))
    per['sp_AF_absa_model'] = null_sp
    c = R[R.comparable & (R.kinf >= 10)]
    conc = R[R.comparable & ~R.discordant]
    disc = R[R.discordant]
    # partial Spearman AF ~ |a| given depth, within gene: rank-residualize on log n
    def partial(x):
        rk = lambda v: sps.rankdata(v) / len(v)
        X = np.column_stack([np.ones(len(x)), rk(np.log(x.n))])
        ra = rk(x.AF) - X @ np.linalg.lstsq(X, rk(x.AF), rcond=None)[0]
        rb = rk(x.absa) - X @ np.linalg.lstsq(X, rk(x.absa), rcond=None)[0]
        return float(np.corrcoef(ra, rb)[0, 1])
    per['partial_sp_AF_absa_given_n'] = R.groupby('gene', sort=False).apply(partial, include_groups=False)
    terc = R.groupby('gene', group_keys=False).apply(
        lambda x: pd.qcut(x.AF.rank(method='first'), 3, labels=['lowAF', 'midAF', 'highAF']),
        include_groups=False)
    R['AF_tercile'] = terc.astype(str)
    byAF = R[R.comparable].groupby('AF_tercile').agg(
        n=('dz', 'size'), mean_dz2=('dz', lambda v: float(np.mean(v ** 2))),
        frac_discordant=('discordant', 'mean'), median_absa=('absa', 'median'),
        median_abs_ap=('ap', lambda v: float(np.median(np.abs(v)))),
        mean_z2=('z2', 'mean'))
    a_sec = dict(
        AF_quantiles=dict(zip(['10', '25', '50', '75', '90'],
                              np.quantile(R.AF, [.1, .25, .5, .75, .9]).round(3).tolist())),
        corrLR_quantiles=dict(zip(['10', '25', '50', '75', '90'],
                                  np.quantile(R.corrLR, [.1, .25, .5, .75, .9]).round(3).tolist())),
        pooled_sp_AF_absa=float(sps.spearmanr(R.AF, R.absa)[0]),
        pooled_sp_AF_n=float(sps.spearmanr(R.AF, R.n)[0]),
        pooled_sp_AF_inv_keff_frac=float(sps.spearmanr(R.AF, R.n / R.keff)[0]),
        within_gene_median_sp_AF_absa=float(per.sp_AF_absa.median()),
        within_gene_median_sp_AF_absa_model=float(per.sp_AF_absa_model.median()),
        within_gene_median_partial_sp_AF_absa_given_n=float(per.partial_sp_AF_absa_given_n.median()),
        within_gene_median_sp_AF_n=float(per.sp_AF_logn.median()),
        within_gene_median_sp_AF_z2=float(per.sp_AF_z2.median()),
        n_genes_sp_AF_absa_above_model=int((per.sp_AF_absa > per.sp_AF_absa_model).sum()),
        comparable_records_kinf10=len(c),
        sp_a_ap_pooled=float(sps.spearmanr(c.a, c.ap)[0]),
        n_genes_sp_a_ap_negative=int((per.sp_a_ap < 0).sum()),
        min_gene_sp_a_ap=float(per.sp_a_ap.min()),
        keff_over_kinf_quantiles=dict(zip(['10', '25', '50', '75', '90'],
                                          np.quantile(c.keff / c.kinf, [.1, .25, .5, .75, .9]).round(3).tolist())),
        va_over_qinf_quantiles=dict(zip(['10', '25', '50', '75', '90'],
                                        np.quantile(c.va / c.qinf, [.1, .25, .5, .75, .9]).round(3).tolist())),
        sp_keff_kinf=float(sps.spearmanr(c.keff, c.kinf)[0]),
        sp_AF_n_over_kinf=float(sps.spearmanr(c.AF, c.n / c.kinf)[0]),
        n_comparable=int(R.comparable.sum()),
        n_discordant=int(R.discordant.sum()),
        frac_discordant=float(R.discordant.sum() / R.comparable.sum()),
        sp_AF_absa_concordant=float(sps.spearmanr(conc.AF, conc.absa)[0]),
        sp_AF_absa_discordant=float(sps.spearmanr(disc.AF, disc.absa)[0]),
        median_AF_concordant=float(conc.AF.median()),
        median_AF_discordant=float(disc.AF.median()),
        frac_discordant_salmon_more_extreme=float((disc.absa > disc.ap.abs()).mean()),
        median_absa_discordant=float(disc.absa.median()),
        median_abs_ap_discordant=float(disc.ap.abs().median()),
        variance_budget=variance_budget(R),
        by_AF_tercile=byAF.round(4).to_dict('index'))
    # EXTERNAL CHECK OF THE RECORD-SPECIFIC INFLATION. If Salmon's va is right
    # in SHAPE, the Salmon-phASER discrepancy standardized by (v + qinf) has
    # the same mean in every within-gene AF tercile. Compare with the two
    # alternatives that drop the record-specific ambiguity: the gene's median
    # AF times the counting variance, and the counting variance alone.
    cmpd = R[R.comparable & (R.kinf >= 50)].copy()
    gl = sorted(cmpd.gene.unique())
    ext = {}
    rng_ext = np.random.default_rng(SS_BOOT.spawn(1)[0])
    bidx = rng_ext.integers(0, len(gl), size=(N_BOOT, len(gl)))
    for nm, v in (('gibbs_va', cmpd.va), ('poisson_x_gene_medAF', cmpd.medAF_gene * cmpd.qk),
                  ('poisson_only', cmpd.qk)):
        dz2 = (cmpd.a - cmpd.ap) ** 2 / (v + cmpd.qinf)
        tab = pd.DataFrame(dict(gene=cmpd.gene, t=cmpd.AF_tercile, dz2=dz2))
        S = tab.pivot_table(index='gene', columns='t', values='dz2', aggfunc='sum').reindex(gl).fillna(0)
        N_ = tab.pivot_table(index='gene', columns='t', values='dz2', aggfunc='size').reindex(gl).fillna(0)
        m = {t: float(S[t].sum() / N_[t].sum()) for t in ('lowAF', 'midAF', 'highAF')}
        bs = S.values[bidx].sum(1) / N_.values[bidx].sum(1)
        cols = list(S.columns)
        ratio = bs[:, cols.index('highAF')] / bs[:, cols.index('lowAF')]
        ext[nm] = dict(mean_dz2_by_AF_tercile=m, high_over_low=m['highAF'] / m['lowAF'],
                       high_over_low_lo=float(np.quantile(ratio, .025)),
                       high_over_low_hi=float(np.quantile(ratio, .975)),
                       n_records=len(cmpd))
        print(f'external shape check [{nm}]: mean dz^2 by AF tercile {m}, '
              f'high/low {m["highAF"] / m["lowAF"]:.3f} '
              f'[{ext[nm]["high_over_low_lo"]:.3f},{ext[nm]["high_over_low_hi"]:.3f}]')
    a_sec['external_shape_check_kinf50'] = ext
    summary['a_ambiguity'] = a_sec

    # ---------------- (b) conservative genes ----------------------------
    top = (R.sort_values('z2', ascending=False).groupby('gene', sort=False).head(6)
           [['gene', 'donor', 'a', 'ap', 'n', 'kinf', 'keff', 'AF', 'medAF_gene', 'w',
             'w_share', 'w_tercile', 'z2', 'z2_share', 'dz', 'discordant', 's', 'corrLR',
             'n_variants']])
    top.to_csv(OUT / 'top_records.tsv', sep='\t', index=False)
    cons = per.index[per.R < 1]
    b_sec = {}
    for g in cons:
        x = R[R.gene == g]
        loBig = x[(x.w_tercile == 'low') & (x.z2 > x.z2.mean())]
        b_sec[g] = dict(
            R=float(per.loc[g, 'R']),
            C_low=float(per.loc[g, 'C_low']), C_high=float(per.loc[g, 'C_high']),
            lowT_bigz2_n=len(loBig),
            lowT_bigz2_z2_share=float(loBig.z2_share.sum()),
            lowT_bigz2_w_share=float(loBig.w_share.sum()),
            lowT_bigz2_n_discordant=int(loBig.discordant.sum()),
            lowT_bigz2_n_comparable=int(loBig.comparable.sum()),
            lowT_bigz2_median_AF_over_gene=float((loBig.AF / loBig.medAF_gene).median()) if len(loBig) else None,
            top3=x.nlargest(3, 'z2')[['donor', 'a', 'ap', 'n', 'kinf', 'AF', 'w_share',
                                      'z2_share', 'dz', 's']].round(4).to_dict('records'))
    summary['b_conservative_genes'] = b_sec
    # pooled over all genes: where do discordant records sit?
    summary['b_discordant_by_weight_tercile'] = (
        R[R.comparable].groupby('w_tercile').agg(n=('discordant', 'size'),
                                                  n_discordant=('discordant', 'sum'),
                                                  z2_share_of_discordant=('z2_share', lambda v: float(v[R.loc[v.index, 'discordant']].sum())))
        .to_dict('index'))
    # R on concordant (plus non-comparable) records only = drop_discordant arm's R
    summary['b_R_below_1_genes'] = sorted(cons)

    # ---------------- (d) covariance decomposition ----------------------
    d_sec = dict(
        sum_C_low=float(per.C_low.sum()), sum_C_mid=float(per.C_mid.sum()),
        sum_C_high=float(per.C_high.sum()),
        sum_Q_bigz2_highw=float(per.Q_bigz2_highw.sum()),
        sum_Q_bigz2_loww=float(per.Q_bigz2_loww.sum()),
        sum_Q_smallz2_highw=float(per.Q_smallz2_highw.sum()),
        sum_Q_smallz2_loww=float(per.Q_smallz2_loww.sum()),
        mean_R_minus_1=float((per.R - 1).mean()),
        n_genes_C_low_negative=int((per.C_low < 0).sum()),
        n_genes_C_high_positive=int((per.C_high > 0).sum()))
    ex = {arm: {str(al): summary['arms'][arm]['rates'][f'allelic@{al}']['rate'] - al
                for al in ALPHAS} for arm in ('real', 'neutralize_lowT', 'neutralize_highT',
                                              'neutralize_both')}
    d_sec['excess'] = ex
    d_sec['additivity_residual'] = {str(al): ex['real'][str(al)] - (ex['neutralize_lowT'][str(al)]
                                    + ex['neutralize_highT'][str(al)] - ex['neutralize_both'][str(al)])
                                    for al in ALPHAS}
    # selection-corrected effect of each removal rule: real change minus the
    # change the same rule makes on model-generated records at the same weights
    sel = {}
    for K in (1, 3):
        for side in ('lowW', 'highW'):
            arm, marm = f'drop_{side}_top{K}', f'model_drop_{side}_top{K}'
            sel[arm] = {}
            for ch in ('allelic', 'combined'):
                pr = pa if ch == 'allelic' else pb
                for al in ALPHAS:
                    k = lambda p: counts(p, al)
                    kr, nr = k(pr['real']); kx, nx = k(pr[arm])
                    km, nm = k(pr['model']); kmx, nmx = k(pr[marm])
                    f = lambda kk, nn, ii: kk[ii].sum(-1) / nn[ii].sum(-1)
                    est = (kx.sum() / nx.sum() - kr.sum() / nr.sum()) - \
                          (kmx.sum() / nmx.sum() - km.sum() / nm.sum())
                    bt = (f(kx, nx, BOOT_IDX) - f(kr, nr, BOOT_IDX)) - \
                         (f(kmx, nmx, BOOT_IDX) - f(km, nm, BOOT_IDX))
                    sel[arm][f'{ch}@{al}'] = dict(
                        real_change=float(kx.sum() / nx.sum() - kr.sum() / nr.sum()),
                        model_change=float(kmx.sum() / nmx.sum() - km.sum() / nm.sum()),
                        corrected=float(est), lo=float(np.quantile(bt, .025)),
                        hi=float(np.quantile(bt, .975)),
                        rate_if_removed_corrected=float(kr.sum() / nr.sum() + est))
    d_sec['selection_corrected_removal'] = sel
    # why are the negative-coupling records low weight? log w = log n + log(p q)
    # - log AF + (small kappa terms): decompose each record's log-weight deficit
    # against its gene's median into depth, imbalance (binomial p q) and
    # ambiguity (AF) parts
    R['pq'] = (R.mL / R.n) * (R.mR / R.n)
    R['lw_depth'] = np.log(R.n) - R.groupby('gene').n.transform(lambda v: np.median(np.log(v)))
    R['lw_pq'] = np.log(R.pq) - R.groupby('gene').pq.transform(lambda v: np.median(np.log(v)))
    R['lw_AF'] = -(np.log(R.AF) - R.groupby('gene').AF.transform(lambda v: np.median(np.log(v))))
    R['lw_total'] = np.log(R.w) - R.groupby('gene').w.transform(lambda v: np.median(np.log(v)))
    neg = R[(R.w_tercile == 'low') & (R.z2 > R.groupby('gene').z2.transform('mean'))]
    pos = R[(R.w_tercile == 'high') & (R.z2 > R.groupby('gene').z2.transform('mean'))]
    def dec(x):
        return dict(n=len(x), n_discordant=int(x.discordant.sum()),
                    n_comparable=int(x.comparable.sum()),
                    median_lw_total=float(x.lw_total.median()),
                    median_lw_depth=float(x.lw_depth.median()),
                    median_lw_pq=float(x.lw_pq.median()),
                    median_lw_AF=float(x.lw_AF.median()),
                    median_absa=float(x.absa.median()),
                    median_abs_ap=float(x.ap.abs().median()),
                    median_AF_over_gene=float((x.AF / x.medAF_gene).median()),
                    frac_phaser_confirms_imbalance=float(
                        ((x.ap.abs() > 0.5 * x.absa) & (np.sign(x.ap) == np.sign(x.a)))[x.comparable].mean()))
    d_sec['negative_part_records'] = dec(neg)
    d_sec['negative_part_records_in_R_below_1_genes'] = dec(neg[neg.gene.isin(cons)])
    d_sec['positive_part_records'] = dec(pos)
    d_sec['positive_part_records_in_R_above_1_genes'] = dec(pos[~pos.gene.isin(cons)])
    summary['d_decomposition'] = d_sec

    per.to_csv(OUT / 'per_gene.tsv', sep='\t')
    R.to_csv(OUT / 'records.tsv', sep='\t', index=False)

    # ---------------- figures -------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    cc = R[R.comparable & (R.kinf >= 10)]
    for flag, col, lab in ((False, '#8a8a8a', f'|dz| < {DZ_FLAG:g}'), (True, '#c0392b', f'|dz| >= {DZ_FLAG:g}')):
        x = cc[cc.discordant == flag]
        ax[0].scatter(x.absa, x.AF / x.medAF_gene, s=5, c=col, alpha=.5, label=lab)
        ax[1].scatter(x.ap.abs(), x.absa, s=5, c=col, alpha=.5, label=lab)
    ax[0].set_xscale('symlog', linthresh=0.05); ax[0].set_yscale('log')
    ax[0].set_xlabel('|a| (Salmon, natural log)'); ax[0].set_ylabel('AF / gene median AF')
    ax[0].legend(frameon=False); ax[0].set_title('Ambiguity factor against imbalance')
    ax[1].plot([0, 4], [0, 4], 'k:', lw=1)
    ax[1].set_xlabel('|ap| (phASER, natural log)'); ax[1].set_ylabel('|a| (Salmon)')
    ax[1].set_title('Salmon against phASER allelic ratio')
    fig.tight_layout(); fig.savefig(OUT / 'fig_af_vs_imbalance.png', dpi=130); plt.close(fig)
    arms_show = ['real', 'drop_lowW_top1', 'poisson_medAF', 'drop_discordant', 'phaser']
    fig, ax = plt.subplots(figsize=(11, 4.5))
    order = per.R.sort_values().index
    for k, arm in enumerate(arms_show):
        ax.plot(range(len(order)), per.loc[order, f'R_{arm}'], 'o', ms=4, label=arm)
    ax.axhline(1, c='k', lw=.8); ax.set_yscale('log')
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_ylabel('R_g'); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / 'fig_R_by_arm.png', dpi=130); plt.close(fig)

    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: summary[k] for k in ('a_ambiguity', 'd_decomposition')}, indent=1, default=float))
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
