"""What explains the TOTAL channel's excess nominal rejection under the donor-
record permutation null (0.060 / 0.0136 / 0.0014 at 0.05 / 0.01 / 0.001)?

QUESTION. hapmixQTL's total channel is weighted least squares of log total
expression t on dosage/2, weights w = 1/v_t (v_t the Gibbs variance of t plus the
count_noise Poisson term), intercept and 17 covariates partialled out in the
sqrt(w)-weighted space, residual scale fitted, t^2 referred to F(1, n_t - 19).
On the shared 2,000-permutation instrument
(scripts/null_permutation_instrument.py) it rejects above nominal. This script
decomposes that excess into mechanisms, each measured.

THE ALGEBRA, AND HOW COVARIATE PARTIALLING INTERACTS WITH PERMUTED WEIGHTS.
Under perm_scheme='records' a donor's t, v_t (hence w) AND covariate row move
together; only the genotype stays with the position. In the record frame let
Zw = sqrt(w) [1, C], P = QQ' from one QR of Zw, h_i = P_ii (weighted hat values),
r = (I - P) sqrt(w) t (whitened covariate residuals), q = sqrt(w) r = w e with e
the weighted-least-squares residual on the original scale,
M = diag(sqrt w) (I - P) diag(sqrt w), S = sum r^2. Every one of these is a
property of the SET of records and is identical in every permutation: the
covariate fit, its residuals and the hat values do not move. What the
permutation changes is only which genotype lands on which record. With u the
genotype vector placed in record order (u[prm] = g),

    num = u' q,   den = u' M u,   t2_t = dof num^2 / (den S - num^2),  dof = n - 19.

Because the intercept puts sqrt(w) 1 in range(P), sum q = 0 and M 1 = 0, so over
permutations E[num^2] = (m2 - c) sum q^2 and E[den] = (m2 - c) sum w (1 - h),
with m2 = mean(g^2), c the mean off-diagonal product of g. The first-order
closed form of the variance ratio of the statistic is therefore

    R_t = (n - 18) sum w r^2 / ( sum w (1 - h) * sum r^2 ),

which has expectation 1 under the shipped model (E r_i^2 = sigma^2 (1 - h_i)):
a donor record whose squared whitened residual z_i^2 = r_i^2/(1 - h_i) is large
where its weight is large inflates the permutation variance of the slope beyond
what the fitted residual scale reports. Leverage enters only through the
record-fixed (1 - h_i); the genotype-side quantity that varies by permutation is
den, i.e. how much of the placed genotype the weighted covariates absorb.

DESIGN. Held fixed throughout: the 46 genes, each gene's RASQUAL observed lead
variant, the 92 donors, the covariate matrix, the estimator (the instrument's
fit, re-expressed as above) and the permutation stream
rng = RandomState(42); perms = [rng.permutation(92) for _ in range(2000)].
What varies is the RECORD SET the estimator is permuted over, one mechanism at a
time (data-side arms; the estimator is never changed except in the OLS control):

  REAL         the real records (gated against the instrument)
  OLS_CONTROL  the real records, unit weights (R_t = 1 identically under a
               records permutation, whatever the variances; a control on the
               algebra, NOT evidence about the variance shape)
  MODEL        e ~ N(0, v_t) at each record's own v_t: the shipped model's world
               (sigma^2 = 1, immaterial because the statistic fits its own scale)
  DECOUPLE     the real leverage-corrected whitened residuals z shuffled among
               records against (w, C): real marginal shape, coupling removed
  ISOLATE_OWN  Gaussian e with each record's own realized variance e^2/(1-h):
               real weight-variance coupling, Gaussian shape, random sign
  SIGNFLIP     each record's own |z| with a random sign: real coupling and real
               magnitudes, sign asymmetry between weight and residual removed
  POWER        e ~ N(0, v_t^gamma_g), gamma_g the per-gene ML exponent of
               Var(e) against v_t fitted from the real residuals (a circular,
               isolating arm: it asks whether a smooth power-law shape error
               carries the excess, not whether gamma is known)
  NB_GEN       generative: counts ~ negative binomial (Poisson-gamma) at each
               record's fitted mean and a per-gene biological dispersion phi_g
               estimated from the real residual variance; t and v_t RECOMPUTED
               from the simulated count (Gibbs-to-Poisson ratio kept per record),
               so the weight depends on the simulated outcome exactly as it
               does in the real data (w is ~ proportional to the count)
  NB_EXPW      as NB_GEN, but v_t evaluated at the EXPECTED count: same
               variance mis-shape (additive biological dispersion), weights no
               longer a function of the realized outcome
  POIS_GEN     as NB_GEN with phi = 0: variance shape correct (shot noise only),
               weights still a function of the realized outcome

Simulated arms use N_SIM record sets per gene drawn from child streams of
SeedSequence(42); every set is permuted with the SAME 2,000 permutations, so
arm differences are not permutation noise. The Monte Carlo sd of an arm's
pooled rate ACROSS record sets is the spread a single real record set would
show if the arm's mechanism were the truth, and is the floor against which the
REAL rate is compared.

Also measured: per-gene R_t and its MODEL band; the scale-mixture prediction
(each gene's statistic taken as sqrt(R_t) x t_dof) and the scale/shape
partition (t2 divided by each gene's own permutation variance of the t
statistic relative to t_dof); z^2 against realized depth (log w), against
outcome-independent depth (log library size, covariate-predicted log w); the
hat-value distribution and whether rejection tracks the fraction of the placed
genotype the covariates absorb; kurtosis of z by weight tercile; record
concentration of q; the drop in R_t from removing each gene's single most
R-influential record, against the same selection applied to MODEL record sets
(N_LOO_MODEL of them, the floor for "concentrated in one record"); which donors
recur as the dominant record, against the MODEL recurrence floor; and, for
every simulated arm, whether the scale-mixture prediction from that arm's own
R reproduces that arm's rate (if it does for every mechanism, the rate is a
function of R alone and the question reduces to what makes R exceed 1).

Figures: fig_R_t_band_and_prediction.png (per-gene R_t against the MODEL 95%
band; per-gene REAL rate against the scale-mixture prediction) and
fig_arm_ladder.png (every arm's pooled rate at the three levels).

GATES (abort on failure):
  1. REAL reproduces the instrument's per-(gene, perm) t2_t to relative 1e-9
     wherever t2_t >= 1e-3 and to absolute 1e-12 where t2_t < 1e-3 (below 1e-3 the
     numerator is a near-total cancellation, so relative agreement is limited
     by float64 rounding of a quantity of order 1e-8: the worst case is t2 =
     1.1e-7 differing by 1.2e-16; those p-values exceed 0.97), per-(gene, perm)
     p_t to absolute 1e-10, and the pooled p_t counts at 0.05 / 0.01 / 0.001
     exactly.
  2. Hat values computed from a permuted design equal the record-order hat
     values permuted (the record-invariance claim), to 1e-10.
  3. t and v_t recomputed from the Gibbs-cache draws (mapped by sample id,
     never by position) reproduce the instrument's arrays to 1e-9, so the
     posterior-mean count mT used by the generative arms belongs to the same
     donor-gene pair as the record it is attached to.

Outputs: /mnt/ssd/lalli/brainvar_hapmix_deploy/total_channel_null_calibration_20260925/
Natural-log units throughout.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
INST = D / 'nominal_p_null_instrument_20260925'
CACHE = D / 'cache' / 'gibbs_56b63c3b37ed5df8'
OUT = D / 'total_channel_null_calibration_20260925'
SEED = 42
N_PERM = 2000
N_SIM = 200
N_BOOT = 2000
N_LOO_MODEL = 50        # MODEL record sets used as the floor for record concentration
ALPHAS = (0.05, 0.01, 0.001)
KAPPA = 0.5
SIM_ARMS = ('MODEL', 'DECOUPLE', 'ISOLATE_OWN', 'SIGNFLIP', 'POWER',
            'NB_GEN', 'NB_EXPW', 'POIS_GEN')


# --------------------------------------------------------------------------
# core algebra
# --------------------------------------------------------------------------
def setup(w, Z, U):
    """Record-invariant pieces for weights w, design Z, placed genotypes U."""
    sw = np.sqrt(w)
    Qq, _ = np.linalg.qr(Z * sw[:, None])
    h = (Qq ** 2).sum(1)
    M = (np.eye(len(w)) - Qq @ Qq.T) * sw[:, None] * sw[None, :]
    den = ((U @ M) * U).sum(1)
    return dict(sw=sw, Qq=Qq, h=h, M=M, den=den, trM=float((w * (1 - h)).sum()))


def t2_stats(Y, st, U, dof):
    """t2 for each permutation (rows) and each record set (columns of Y)."""
    r = st['sw'][:, None] * Y
    r = r - st['Qq'] @ (st['Qq'].T @ r)
    q = st['sw'][:, None] * r
    num = U @ q
    S = (r ** 2).sum(0)
    t2 = dof * num ** 2 / (st['den'][:, None] * S[None, :] - num ** 2)
    return t2, r, num


def R_closed(r, st, n, p):
    """First-order variance ratio (n - p) sum w r^2 / (sum w(1-h) sum r^2)."""
    w = st['sw'] ** 2
    return (n - p) * (w[:, None] * r ** 2).sum(0) / (st['trM'] * (r ** 2).sum(0))


def gamma_ml(z2, vt):
    """Per-gene ML exponent in Var(e) = s^2 v_t^gamma, from z^2 ~ s^2 v^(gamma-1).
    Returns gamma_hat, LR statistic vs gamma = 1 and vs gamma = 0."""
    lv = np.log(vt)
    grid = np.arange(-2.0, 3.0001, 0.002)

    def nll(gm):
        return 0.5 * len(z2) * np.log(np.mean(z2 * np.exp((1 - gm) * lv))) \
            + 0.5 * (gm - 1) * lv.sum()
    vals = np.array([nll(gm) for gm in grid])
    gh = grid[np.argmin(vals)]
    return float(gh), float(2 * (nll(1.0) - vals.min())), float(2 * (nll(0.0) - vals.min()))


def R_of_records(t, w, Zd, n, p):
    """Closed-form R_t of a record set, no permutation needed."""
    sw = np.sqrt(w)
    Qq, _ = np.linalg.qr(Zd * sw[:, None])
    r = sw * t
    r = r - Qq @ (Qq.T @ r)
    h = (Qq ** 2).sum(1)
    return (n - p) * (w * r ** 2).sum() / ((w * (1 - h)).sum() * (r ** 2).sum())


def boot_rate(k, n, idx):
    """Pooled rate and gene-clustered percentile bootstrap interval."""
    est = k.sum() / n.sum()
    b = k[idx].sum(1) / n[idx].sum(1)
    return float(est), float(np.quantile(b, .025)), float(np.quantile(b, .975)), b


def figures(P, summ, Rsim, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    col = {'HIGH': '#2a6fdb', 'MID': '#e08a1e', 'LOW': '#2e9e5b'}
    Q = P.sort_values('R_t').reset_index(drop=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(Q))
    ax[0].vlines(x, Q.R_model_q025, Q.R_model_q975, color='#bbbbbb', lw=4,
                 label='MODEL 95% band')
    ax[0].scatter(x, Q.R_t, c=[col[s_] for s_ in Q.stratum], s=22, zorder=3)
    ax[0].axhline(1, color='k', lw=.6)
    ax[0].set_xticks(x); ax[0].set_xticklabels(Q.gene, rotation=90, fontsize=6)
    ax[0].set_ylabel('R_t (closed-form variance ratio)')
    ax[0].set_title('Per-gene R_t against the shipped model\'s band')
    for s_, c_ in col.items():
        ax[0].scatter([], [], c=c_, label=s_)
    ax[0].legend(fontsize=7)
    a = '0.05'
    ax[1].scatter(P[f'pred_SCALE_MIX_R_t_{0.05}'], P['rej_REAL_0.05'],
                  c=[col[s_] for s_ in P.stratum], s=22)
    lim = [0.03, 0.095]
    ax[1].plot(lim, lim, 'k', lw=.6)
    ax[1].errorbar([0.034], [0.09], yerr=[1.96 * np.sqrt(.05 * .95 / N_PERM)], color='k',
                   capsize=3)
    ax[1].text(0.036, 0.09, 'binomial 95%\nat one gene', fontsize=7, va='center')
    for _, r_ in P.iterrows():
        if r_['rej_REAL_0.05'] > 0.075:
            ax[1].annotate(r_.gene, (r_[f'pred_SCALE_MIX_R_t_{0.05}'], r_['rej_REAL_0.05']),
                           fontsize=7)
    ax[1].set_xlabel('scale-mixture prediction from R_t, at 0.05')
    ax[1].set_ylabel('permutation rejection rate at 0.05 (REAL)')
    ax[1].set_title('R_t alone predicts each gene\'s rate')
    fig.tight_layout(); fig.savefig(out / 'fig_R_t_band_and_prediction.png', dpi=140)
    plt.close(fig)
    arms = ['REAL', 'REAL_SCALE_REMOVED', 'SCALE_MIX_R_t', 'MODEL', 'DECOUPLE', 'POWER',
            'ISOLATE_OWN', 'SIGNFLIP', 'POIS_GEN', 'NB_EXPW', 'NB_GEN', 'OLS_CONTROL']
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
    for i, a in enumerate(('0.05', '0.01', '0.001')):
        for j, arm in enumerate(arms):
            d = summ['arms'][arm][a]
            if 'lo' in d:
                lo, hi = d['lo'], d['hi']
            else:
                lo, hi = d['rate'] - 1.96 * d['mc_sd'], d['rate'] + 1.96 * d['mc_sd']
            ax[i].errorbar(d['rate'], j, xerr=[[d['rate'] - lo], [hi - d['rate']]], fmt='o',
                           color='#c0392b' if arm == 'REAL' else '#34495e', capsize=2)
        ax[i].axvline(float(a), color='k', lw=.6)
        ax[i].axvline(summ['arms']['REAL'][a]['rate'], color='#c0392b', lw=.6, ls='--')
        ax[i].set_yticks(range(len(arms)))
        ax[i].set_yticklabels(arms if i == 0 else [], fontsize=8)
        ax[i].set_title(f'total-channel rejection at {a}')
        ax[i].xaxis.set_major_locator(plt.MaxNLocator(5))
        ax[i].invert_yaxis()
    fig.text(0.5, 0.005, 'REAL / REAL_SCALE_REMOVED / SCALE_MIX / OLS: gene-clustered 95% '
             'interval; simulated arms: +-1.96 MC sd across record sets', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out / 'fig_arm_ladder.png', dpi=140)
    plt.close(fig)


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    Z_ = np.load(INST / 'inputs_at_lead.npz', allow_pickle=True)
    genes = [str(x) for x in Z_['genes']]
    strata = [str(x) for x in Z_['strata']]
    donors = [str(x) for x in Z_['donors']]
    T, VT, G, C = Z_['t'], Z_['vt'], Z_['g'], Z_['C']
    lib = Z_['lib_size'].astype(float)
    NG, N = T.shape
    assert np.all(VT > 1e-12) and np.all(np.isfinite(T)), 'all donors must be admitted'
    n = N
    p_cols = 1 + C.shape[1]
    dof = n - 1 - p_cols
    Zd = np.column_stack([np.ones(N), C])
    crit = {a: sps.f.isf(a, 1, dof) for a in ALPHAS}
    tcrit = {a: sps.t.isf(a / 2, dof) for a in ALPHAS}
    L = pd.read_csv(INST / 'null_long.tsv.gz', sep='\t')
    L = L.set_index(['gene', 'perm']).sort_index()

    # the instrument's permutation stream
    rng = np.random.RandomState(SEED)
    PERMS = np.stack([rng.permutation(N) for _ in range(N_PERM)])
    rows = np.arange(N_PERM)[:, None]

    # child streams for everything else
    ss = np.random.SeedSequence(SEED).spawn(len(SIM_ARMS) + 2)
    arm_rng = {a: np.random.default_rng(s) for a, s in zip(SIM_ARMS, ss[:len(SIM_ARMS)])}
    boot_rng = np.random.default_rng(ss[len(SIM_ARMS)])

    # ---- Gibbs cache: posterior-mean total count, mapped by sample id ----
    cg = open(CACHE / 'genes.txt').read().split()
    cs = open(CACHE / 'samples.txt').read().split()
    keep = [cs.index(s) for s in donors]
    grow = [cg.index(g) for g in genes]
    YT = np.load(CACHE / 'YT.npy', mmap_mode='r')
    yt = np.stack([np.asarray(YT[i])[keep] for i in grow])       # genes x donors x draws
    mT = yt.mean(-1)
    td = np.log(yt / 2 + KAPPA)
    t_re = td.mean(-1)
    vt_re = td.var(-1) + 1.0 / (mT + 2 * KAPPA)
    jgap = float(max(np.max(np.abs(t_re - T)), np.max(np.abs(vt_re - VT) / VT)))
    print(f'gate 3: t and v_t recomputed from the mapped draws, max diff {jgap:.3e}',
          flush=True)
    if jgap > 1e-9:
        raise SystemExit('gate 3 FAILED: Gibbs cache does not map onto the instrument')
    del yt, td
    # per-record Gibbs-to-Poisson ratio of the Gibbs part of v_t
    gib = VT - 1.0 / (mT + 2 * KAPPA)
    cgib = gib * (mT + 1) ** 2 / np.maximum(mT, 1e-9)

    real_t2 = np.zeros((NG, N_PERM))
    real_p = np.zeros((NG, N_PERM))
    real_ts = np.zeros((NG, N_PERM))
    ols_t2 = np.zeros((NG, N_PERM))
    rej = {a: np.zeros((NG, N_SIM, len(ALPHAS)), int) for a in SIM_ARMS}
    Rsim = {a: np.zeros((NG, N_SIM)) for a in SIM_ARMS}
    shape_sim = np.zeros((NG, N_SIM, len(ALPHAS)), int)   # MODEL, scale-corrected
    loo_model = np.zeros((NG, N_LOO_MODEL))
    topq_model = np.zeros((NG, N_LOO_MODEL), int)
    topq_model_all = np.zeros((NG, N_SIM), int)
    donor_z2_model = np.zeros((NG, N, N_SIM))
    pg, rec_rows = [], []
    gate2_max = 0.0
    gate1 = {'max_rel_t2_ge_1e-3': 0.0, 'max_abs_t2_lt_1e-3': 0.0, 'max_abs_p': 0.0}
    for k, gname in enumerate(genes):
        t, vt, g = T[k], VT[k], G[k]
        w = 1.0 / vt
        U = np.zeros((N_PERM, N))
        U[rows, PERMS] = g[None, :]
        st = setup(w, Zd, U)

        # ---- REAL -------------------------------------------------------
        t2, r, num = t2_stats(t[:, None], st, U, dof)
        t2 = t2[:, 0]; r = r[:, 0]; num = num[:, 0]
        ref = L.loc[gname].t2_t.values
        big = ref >= 1e-3
        rel = float(np.max(np.abs(t2[big] - ref[big]) / ref[big]))
        adf = float(np.max(np.abs(t2[~big] - ref[~big]), initial=0.0))
        pdf = float(np.max(np.abs(sps.f.sf(t2, 1, dof) - L.loc[gname].p_t.values)))
        gate1['max_rel_t2_ge_1e-3'] = max(gate1['max_rel_t2_ge_1e-3'], rel)
        gate1['max_abs_t2_lt_1e-3'] = max(gate1['max_abs_t2_lt_1e-3'], adf)
        gate1['max_abs_p'] = max(gate1['max_abs_p'], pdf)
        if rel > 1e-9 or adf > 1e-12 or pdf > 1e-10:
            raise SystemExit(f'gate 1 FAILED at {gname}: rel {rel:.2e} abs {adf:.2e} p {pdf:.2e}')
        real_t2[k] = t2
        real_p[k] = sps.f.sf(t2, 1, dof)
        real_ts[k] = np.sign(num) * np.sqrt(t2)

        # ---- gate 2: hat values are record properties ------------------
        for prm in PERMS[:3]:
            Qp, _ = np.linalg.qr(Zd[prm] * np.sqrt(w[prm])[:, None])
            gate2_max = max(gate2_max, float(np.max(np.abs((Qp ** 2).sum(1) - st['h'][prm]))))

        # ---- OLS control ------------------------------------------------
        st1 = setup(np.ones(N), Zd, U)
        ols_t2[k] = t2_stats(t[:, None], st1, U, dof)[0][:, 0]

        # ---- record anatomy ----------------------------------------------
        h = st['h']
        z = r / np.sqrt(1 - h)
        z2 = z ** 2
        e = r / st['sw']
        q = st['sw'] * r
        R_real = float(R_closed(r[:, None], st, n, p_cols)[0])
        gh, lr1, lr0 = gamma_ml(z2, vt)
        # outcome-independent depth: covariate-predicted log w, and log library
        lw = np.log(w)
        lw_pred = Zd @ np.linalg.lstsq(Zd, lw, rcond=None)[0]
        zs = z2 / z2.mean()
        sl_real = np.polyfit(lw, zs, 1)[0]
        sl_pred = np.polyfit(lw_pred, zs, 1)[0]
        sl_lib = np.polyfit(np.log(lib), zs, 1)[0]
        sp_real = sps.spearmanr(lw, z2)[0]
        sp_pred = sps.spearmanr(lw_pred, z2)[0]
        sp_lib = sps.spearmanr(np.log(lib), z2)[0]
        # fitted mean for the generative arms
        beta_f = np.linalg.lstsq(Zd * st['sw'][:, None], t * st['sw'], rcond=None)[0]
        f = Zd @ beta_f
        mu = np.maximum(2 * np.exp(f) - 1, 0.5)
        e_ols = t - Zd @ np.linalg.lstsq(Zd, t, rcond=None)[0]
        s2_ols = float(e_ols @ e_ols / (n - p_cols))
        shot = float(np.mean(1.0 / (mT[k] + 1)))
        phi = max(s2_ols - shot, 1e-4)
        # g-side: fraction of the placed genotype's weighted variance absorbed
        # by the 17 covariates (intercept-only denominator as the reference)
        wu = (U * w[None, :]).sum(1) / w.sum()
        den_int = ((U - wu[:, None]) ** 2 * w[None, :]).sum(1)
        absorbed = 1 - st['den'] / den_int
        tstat = np.sign(num) * np.sqrt(t2)
        cvar = float(np.var(tstat, ddof=1) / (dof / (dof - 2)))
        R_loo = np.array([R_of_records(np.delete(t, j), np.delete(w, j),
                                       np.delete(Zd, j, 0), n - 1, p_cols)
                          for j in range(N)])
        j_inf = int(np.argmax(R_real - R_loo))
        j_q = int(np.argmax(q ** 2))
        pg.append(dict(gene=gname, stratum=strata[k], n=n,
                       R_loo_top=float(R_loo[j_inf]), top_influence_donor=donors[j_inf],
                       top_influence_z_std=float(z[j_inf] / np.sqrt(z2.mean())),
                       top_influence_wrank=float(sps.rankdata(w)[j_inf] / N),
                       top_q_donor=donors[j_q],
                       median_count=float(np.median(mT[k])),
                       kish=float(w.sum() ** 2 / (w ** 2).sum()),
                       n_over_kish=float(n * (w ** 2).sum() / w.sum() ** 2),
                       w_fold=float(w.max() / w.min()),
                       corr_t_logw=float(np.corrcoef(t, lw)[0, 1]),
                       R_t=R_real, t_var_ratio=cvar,
                       gamma_hat=gh, LR_gamma1=lr1, LR_gamma0=lr0,
                       phi=phi, s2_ols=s2_ols, shot_var=shot,
                       sigma2_fit=float(r @ r / (n - p_cols)),
                       slope_z2_logw=sl_real, slope_z2_logw_pred=sl_pred,
                       slope_z2_loglib=sl_lib, spear_z2_logw=sp_real,
                       spear_z2_logw_pred=sp_pred, spear_z2_loglib=sp_lib,
                       h_max=float(h.max()), h_mean=float(h.mean()),
                       n_h_gt_0p5=int((h > 0.5).sum()),
                       q_conc=float((q ** 4).sum() / ((q ** 2).sum()) ** 2),
                       q_maxshare=float((q ** 2).max() / (q ** 2).sum()),
                       z_kurt=float(sps.kurtosis(z)),
                       maf=float(min(g.mean(), 1 - g.mean())),
                       absorbed_med=float(np.median(absorbed)),
                       absorbed_q95=float(np.quantile(absorbed, .95))))
        # within-gene weight tercile, standardized z, and the absorbed fraction
        terc = np.searchsorted(np.quantile(w, [1 / 3, 2 / 3]), w)
        for j in range(N):
            rec_rows.append(dict(gene=gname, record=j, donor=donors[j], w=w[j],
                                 vt=vt[j], mT=mT[k, j], e=e[j], z=z[j] / np.sqrt(z2.mean()),
                                 h=h[j], q2_share=q[j] ** 2 / (q ** 2).sum(),
                                 w_tercile=int(terc[j])))
        # absorbed-fraction quintile vs rejection at 0.05 (REAL)
        aq = np.searchsorted(np.quantile(absorbed, [.2, .4, .6, .8]), absorbed)
        pg[-1].update({f'rej05_absorbed_q{i}': float((real_p[k][aq == i] < .05).mean())
                       for i in range(5)})
        pg[-1]['absorbed_quintile'] = aq  # removed before writing

        # ---- simulated arms ---------------------------------------------
        sq = np.sqrt(vt)[:, None]
        Ysets = {}
        Ysets['MODEL'] = sq * arm_rng['MODEL'].standard_normal((N, N_SIM))
        Ysets['DECOUPLE'] = sq * np.stack([z[arm_rng['DECOUPLE'].permutation(N)]
                                           for _ in range(N_SIM)], 1)
        Ysets['ISOLATE_OWN'] = sq * np.abs(z)[:, None] * \
            arm_rng['ISOLATE_OWN'].standard_normal((N, N_SIM))
        Ysets['SIGNFLIP'] = sq * z[:, None] * \
            arm_rng['SIGNFLIP'].choice([-1.0, 1.0], size=(N, N_SIM))
        Ysets['POWER'] = (vt ** (gh / 2))[:, None] * \
            arm_rng['POWER'].standard_normal((N, N_SIM))
        # NB_EXPW: v_t at the expected count, fixed across sets
        vt_exp = cgib[k] * mu / (mu + 1) ** 2 + 1.0 / (mu + 1)
        lam = mu[:, None] * arm_rng['NB_EXPW'].gamma(1 / phi, phi, size=(N, N_SIM))
        cnt = arm_rng['NB_EXPW'].poisson(lam)
        Ysets['NB_EXPW'] = np.log(cnt / 2 + KAPPA)
        # floor for record concentration: the same leave-one-record-out drop and
        # the same top-q donor, on MODEL record sets
        for s_ in range(N_LOO_MODEL):
            y_ = Ysets['MODEL'][:, s_]
            Rf = R_of_records(y_, w, Zd, n, p_cols)
            Rl = np.array([R_of_records(np.delete(y_, j), np.delete(w, j),
                                        np.delete(Zd, j, 0), n - 1, p_cols) for j in range(N)])
            loo_model[k, s_] = float(np.max(Rf - Rl))
            rr_ = st['sw'] * y_
            rr_ = rr_ - st['Qq'] @ (st['Qq'].T @ rr_)
            topq_model[k, s_] = int(np.argmax((st['sw'] * rr_) ** 2))
        rM = st['sw'][:, None] * Ysets['MODEL']
        rM = rM - st['Qq'] @ (st['Qq'].T @ rM)
        zM2 = rM ** 2 / (1 - h)[:, None]
        donor_z2_model[k] = zM2 / zM2.mean(0, keepdims=True)
        topq_model_all[k] = np.argmax((st['sw'][:, None] * rM) ** 2, axis=0)
        for arm, Y in Ysets.items():
            stx = st if arm != 'NB_EXPW' else setup(1.0 / vt_exp, Zd, U)
            t2s, rs, nums = t2_stats(Y, stx, U, dof)
            for ai, a in enumerate(ALPHAS):
                rej[arm][k, :, ai] = (t2s > crit[a]).sum(0)
            Rsim[arm][k] = R_closed(rs, stx, n, p_cols)
            if arm == 'MODEL':
                ts = np.sign(nums) * np.sqrt(t2s)
                cv = ts.var(0, ddof=1) / (dof / (dof - 2))
                for ai, a in enumerate(ALPHAS):
                    shape_sim[k, :, ai] = (t2s / cv[None, :] > crit[a]).sum(0)
        # NB_GEN / POIS_GEN: weights recomputed from the simulated count
        for arm, ph in (('NB_GEN', phi), ('POIS_GEN', 0.0)):
            rg = arm_rng[arm]
            for s_ in range(N_SIM):
                lam = mu * (rg.gamma(1 / ph, ph, size=N) if ph > 0 else 1.0)
                c_ = rg.poisson(lam).astype(float)
                ts_ = np.log(c_ / 2 + KAPPA)
                vs_ = cgib[k] * c_ / (c_ + 1) ** 2 + 1.0 / (c_ + 1)
                stx = setup(1.0 / vs_, Zd, U)
                t2s, rs, _ = t2_stats(ts_[:, None], stx, U, dof)
                for ai, a in enumerate(ALPHAS):
                    rej[arm][k, s_, ai] = int((t2s[:, 0] > crit[a]).sum())
                Rsim[arm][k, s_] = R_closed(rs, stx, n, p_cols)[0]
        print(f'  {k + 1:2d}/{NG} {gname:9s} R_t {R_real:.3f} gamma {gh:+.2f} '
              f'rej05 {np.mean(real_p[k] < .05):.4f}  [{time.time() - t0:.0f}s]', flush=True)

    if gate2_max > 1e-10:
        raise SystemExit(f'gate 2 FAILED: hat values not record-invariant ({gate2_max:.2e})')
    print(f'gate 2: hat values record-invariant, max diff {gate2_max:.2e}')

    # ---- gate 1, pooled rates exactly -------------------------------------
    pinst = np.stack([L.loc[gn].p_t.values for gn in genes])
    for a in ALPHAS:
        mine, theirs = (real_p < a).sum(), (pinst < a).sum()
        if mine != theirs:
            raise SystemExit(f'gate 1 FAILED: pooled count at {a}: {mine} vs {theirs}')
    print('gate 1: per-(gene, perm) t2_t to 1e-9 and pooled p_t counts exact')

    # ---- pooled rates ------------------------------------------------------
    idx = boot_rng.integers(0, NG, size=(N_BOOT, NG))
    nn = np.full(NG, N_PERM)
    summ = dict(n_genes=NG, n_perm=N_PERM, n_sim=N_SIM, dof=dof,
                gates=dict(gate1=dict(status='PASSED', pooled_counts='exact', **gate1),
                           gate2_hat_max_diff=gate2_max, gate3_jensen_max=jgap),
                arms={})
    boots = {}

    def real_like(name, t2mat):
        out = {}
        for a in ALPHAS:
            kk = (t2mat > crit[a]).sum(1)
            est, lo, hi, b = boot_rate(kk, nn, idx)
            boots[(name, a)] = b
            out[str(a)] = dict(rate=est, lo=lo, hi=hi, ratio=est / a)
        return out
    summ['arms']['REAL'] = real_like('REAL', real_t2)
    summ['arms']['OLS_CONTROL'] = real_like('OLS_CONTROL', ols_t2)
    # scale/shape partition: divide each gene's t2 by its own permutation
    # variance of t relative to t_dof
    cvar = np.array([r_['t_var_ratio'] for r_ in pg])
    summ['arms']['REAL_SCALE_REMOVED'] = real_like('REAL_SCALE_REMOVED', real_t2 / cvar[:, None])
    # scale-mixture predictions
    Rreal = np.array([r_['R_t'] for r_ in pg])
    Rmod_mean = Rsim['MODEL'].mean(1)
    for nm, sc in (('SCALE_MIX_R_t', Rreal / Rmod_mean), ('SCALE_MIX_tvar', cvar)):
        out = {}
        for a in ALPHAS:
            pr = 2 * sps.t.sf(tcrit[a] / np.sqrt(sc), dof)
            b = pr[idx].mean(1)
            boots[(nm, a)] = b
            out[str(a)] = dict(rate=float(pr.mean()), lo=float(np.quantile(b, .025)),
                               hi=float(np.quantile(b, .975)), ratio=float(pr.mean() / a))
            for r_, v in zip(pg, pr):
                r_[f'pred_{nm}_{a}'] = float(v)
        summ['arms'][nm] = out
    for arm in SIM_ARMS:
        out = {}
        for ai, a in enumerate(ALPHAS):
            per_set = rej[arm][:, :, ai].sum(0) / (NG * N_PERM)
            out[str(a)] = dict(rate=float(per_set.mean()), mc_sd=float(per_set.std(ddof=1)),
                               mc_se_of_mean=float(per_set.std(ddof=1) / np.sqrt(N_SIM)),
                               ratio=float(per_set.mean() / a),
                               real_minus_arm=float(summ['arms']['REAL'][str(a)]['rate'] - per_set.mean()),
                               real_z_vs_arm=float((summ['arms']['REAL'][str(a)]['rate'] - per_set.mean())
                                                   / per_set.std(ddof=1)),
                               q025=float(np.quantile(per_set, .025)),
                               q975=float(np.quantile(per_set, .975)))
            for kk, r_ in enumerate(pg):
                r_[f'rej_{arm}_{a}'] = float(rej[arm][kk, :, ai].mean() / N_PERM)
        summ['arms'][arm] = out
    out = {}
    for ai, a in enumerate(ALPHAS):
        per_set = shape_sim[:, :, ai].sum(0) / (NG * N_PERM)
        out[str(a)] = dict(rate=float(per_set.mean()), mc_sd=float(per_set.std(ddof=1)))
    summ['arms']['MODEL_SCALE_REMOVED'] = out
    # excess fractions (paired through the same gene bootstrap)
    frac = {}
    for a in ALPHAS:
        br, bs = boots[('REAL', a)], boots[('REAL_SCALE_REMOVED', a)]
        ex = br - a
        fr = (br - bs) / ex
        frac[str(a)] = dict(real_excess=float(summ['arms']['REAL'][str(a)]['rate'] - a),
                            scale_share=float((summ['arms']['REAL'][str(a)]['rate'] -
                                               summ['arms']['REAL_SCALE_REMOVED'][str(a)]['rate']) /
                                              (summ['arms']['REAL'][str(a)]['rate'] - a)),
                            scale_share_lo=float(np.quantile(fr, .025)),
                            scale_share_hi=float(np.quantile(fr, .975)),
                            Rt_mix_share=float((summ['arms']['SCALE_MIX_R_t'][str(a)]['rate'] - a) /
                                               (summ['arms']['REAL'][str(a)]['rate'] - a)),
                            tvar_mix_share=float((summ['arms']['SCALE_MIX_tvar'][str(a)]['rate'] - a) /
                                                 (summ['arms']['REAL'][str(a)]['rate'] - a)))
    summ['excess_partition'] = frac

    # ---- per gene -----------------------------------------------------------
    P = pd.DataFrame([{k_: v for k_, v in r_.items() if k_ != 'absorbed_quintile'} for r_ in pg])
    for ai, a in enumerate(ALPHAS):
        P[f'rej_REAL_{a}'] = (real_p < a).mean(1)
        P[f'rej_OLS_{a}'] = (ols_t2 > crit[a]).mean(1)
        P[f'rej_REAL_scale_removed_{a}'] = (real_t2 / cvar[:, None] > crit[a]).mean(1)
    P['R_model_mean'] = Rsim['MODEL'].mean(1)
    P['R_model_q025'] = np.quantile(Rsim['MODEL'], .025, axis=1)
    P['R_model_q975'] = np.quantile(Rsim['MODEL'], .975, axis=1)
    P['R_model_sd'] = Rsim['MODEL'].std(1, ddof=1)
    P['R_z'] = (P.R_t - P.R_model_mean) / P.R_model_sd
    for arm in ('NB_GEN', 'NB_EXPW', 'POIS_GEN', 'POWER', 'ISOLATE_OWN', 'DECOUPLE',
                'SIGNFLIP'):
        P[f'R_{arm}_mean'] = Rsim[arm].mean(1)
    P['kurt_tstat_perm'] = [float(sps.kurtosis(real_ts[k])) for k in range(NG)]
    P['excess05'] = P['rej_REAL_0.05'] - 0.05
    P['excess05_binom_se'] = np.sqrt(0.05 * 0.95 / N_PERM)
    P.to_csv(OUT / 'per_gene.tsv', sep='\t', index=False)
    pd.DataFrame(rec_rows).to_csv(OUT / 'records.tsv.gz', sep='\t', index=False)

    # ---- association summaries ---------------------------------------------
    assoc = {}
    for c in ('R_t', 'n_over_kish', 'gamma_hat', 'h_max', 'q_conc', 'q_maxshare',
              'z_kurt', 'median_count', 'w_fold', 'absorbed_q95', 'maf', 'phi'):
        rs = sps.spearmanr(P[c], P['t_var_ratio'])
        rr = sps.spearmanr(P[c], P['rej_REAL_0.01'])
        assoc[c] = dict(spearman_vs_tvar=float(rs[0]), p_vs_tvar=float(rs[1]),
                        spearman_vs_rej01=float(rr[0]), p_vs_rej01=float(rr[1]))
    summ['per_gene_associations'] = assoc
    summ['R_t'] = dict(median=float(P.R_t.median()), min=float(P.R_t.min()),
                       max=float(P.R_t.max()),
                       n_above_model_q975=int((P.R_t > P.R_model_q975).sum()),
                       n_below_model_q025=int((P.R_t < P.R_model_q025).sum()),
                       sd_log_real=float(np.log(P.R_t).std(ddof=1)),
                       sd_log_model_mean=float(np.mean([np.log(Rsim['MODEL'][k]).std(ddof=1)
                                                        for k in range(NG)])),
                       spearman_R_vs_tvar=float(sps.spearmanr(P.R_t, P.t_var_ratio)[0]),
                       NB_GEN_median=float(np.median(Rsim['NB_GEN'])),
                       NB_EXPW_median=float(np.median(Rsim['NB_EXPW'])),
                       POIS_GEN_median=float(np.median(Rsim['POIS_GEN'])),
                       spearman_R_vs_NBGEN_mean=float(sps.spearmanr(P.R_t, P.R_NB_GEN_mean)[0]),
                       spearman_R_vs_POWER_mean=float(sps.spearmanr(P.R_t, P.R_POWER_mean)[0]))
    summ['gamma'] = dict(median=float(P.gamma_hat.median()),
                         q25=float(P.gamma_hat.quantile(.25)), q75=float(P.gamma_hat.quantile(.75)),
                         n_LR1_gt_3p84=int((P.LR_gamma1 > 3.84).sum()),
                         n_LR0_gt_3p84=int((P.LR_gamma0 > 3.84).sum()),
                         n_gamma_lt_1=int((P.gamma_hat < 1).sum()))
    sgn = lambda x: dict(n_pos=int((x > 0).sum()), n=int(len(x)),
                         sign_p=float(sps.binomtest(int((x > 0).sum()), len(x)).pvalue),
                         median=float(np.median(x)))
    summ['depth'] = dict(spear_z2_logw=sgn(P.spear_z2_logw),
                         spear_z2_logw_pred=sgn(P.spear_z2_logw_pred),
                         spear_z2_loglib=sgn(P.spear_z2_loglib),
                         median_corr_t_logw=float(P.corr_t_logw.median()))
    Rr = pd.DataFrame(rec_rows)
    kt = Rr.groupby('w_tercile').z.agg(lambda x: float(sps.kurtosis(x)))
    summ['kurtosis_by_weight_tercile'] = {int(k_): v for k_, v in kt.items()}
    summ['hat'] = dict(median_h=float(Rr.h.median()), q99_h=float(Rr.h.quantile(.99)),
                       max_h=float(Rr.h.max()), n_h_gt_0p5=int((Rr.h > .5).sum()),
                       mean_h_theory=p_cols / n)
    # absorbed-fraction quintile vs rejection (REAL vs MODEL-free expectation 0.05)
    summ['rej05_by_absorbed_quintile'] = {
        i: float(P[f'rej05_absorbed_q{i}'].mean()) for i in range(5)}
    # does R alone predict every arm's rate? per record set, the scale-mixture
    # prediction from that set's own R (normalized by the gene's MODEL mean R)
    Rm = Rsim['MODEL'].mean(1)[:, None]
    summ['scale_mix_check_by_arm'] = {}
    for arm in SIM_ARMS:
        d_ = {}
        for ai, a in enumerate(ALPHAS):
            pr = 2 * sps.t.sf(tcrit[a] / np.sqrt(np.maximum(Rsim[arm], 1e-6) / Rm), dof)
            obs = rej[arm][:, :, ai].sum(0) / (NG * N_PERM)
            d_[str(a)] = dict(observed=float(obs.mean()), predicted=float(pr.mean()),
                              obs_mc_sd=float(obs.std(ddof=1)))
        summ['scale_mix_check_by_arm'][arm] = d_
    # the same prediction for REAL after dropping each gene's most R-influential
    # record (a prediction only: dropping a record changes n and the stream)
    Rloo = P.R_loo_top.values
    summ['scale_mix_drop_top_record'] = {
        str(a): float(np.mean(2 * sps.t.sf(tcrit[a] / np.sqrt(Rloo / Rmod_mean), dof - 1)))
        for a in ALPHAS}
    drop_real = P.R_t.values - Rloo
    summ['R_drop_top_record_floor'] = dict(
        mean_drop_real=float(drop_real.mean()), median_drop_real=float(np.median(drop_real)),
        mean_drop_model=float(loo_model.mean()),
        mean_drop_model_sd_of_46gene_mean=float(loo_model.mean(0).std(ddof=1)),
        n_genes_drop_above_model_q95=int((drop_real > np.quantile(loo_model, .95, axis=1)).sum()))
    maxrec = np.array([np.bincount(topq_model_all[:, s_], minlength=N).max()
                       for s_ in range(N_SIM)])
    summ['donor_recurrence_floor'] = dict(
        real_max_genes_as_top_q=int(P.top_q_donor.value_counts().max()),
        model_max_genes_as_top_q_mean=float(maxrec.mean()),
        model_max_genes_as_top_q_q95=float(np.quantile(maxrec, .95)),
        model_max_genes_as_top_q_max=int(maxrec.max()),
        model_frac_sets_max_ge_real=float(np.mean(maxrec >= P.top_q_donor.value_counts().max())),
        n_model_sets=N_SIM)
    summ['R_drop_top_record'] = dict(median_R=float(np.median(Rloo)),
                                     median_R_full=float(P.R_t.median()),
                                     raw_share_of_mean_excess=float(
                                         np.mean(P.R_t - Rloo) / np.mean(P.R_t - 1)),
                                     floor_corrected_share_of_mean_excess=float(
                                         (np.mean(P.R_t - Rloo) - loo_model.mean()) /
                                         np.mean(P.R_t - 1)))
    # donor-level view: the same donors recur as the dominant record
    Rr['z2'] = Rr.z ** 2
    don = Rr.groupby('donor').agg(mean_z2=('z2', 'mean'), mean_q2_share=('q2_share', 'mean'))
    don['lib_size'] = pd.Series(lib, index=donors).loc[don.index]
    don['n_genes_top_q'] = P.top_q_donor.value_counts().reindex(don.index).fillna(0).astype(int)
    don['n_genes_top_influence'] = P.top_influence_donor.value_counts().reindex(
        don.index).fillna(0).astype(int)
    Rr['wrank'] = Rr.groupby('gene').w.rank(pct=True)
    don['mean_w_rank'] = Rr.groupby('donor').wrank.mean().loc[don.index]
    cn = [str(x) for x in Z_['cov_names']]
    for c_ in ('rin', 'age_days', 'sex'):
        don[c_] = pd.Series(C[:, cn.index(c_)], index=donors).loc[don.index]
    don.sort_values('mean_z2', ascending=False).to_csv(OUT / 'donors.tsv', sep='\t')
    dz = donor_z2_model.mean(0)                      # donors x sets
    summ['donor_z2_floor'] = dict(
        real_max=float(don.mean_z2.max()), real_second=float(don.mean_z2.nlargest(2).iloc[1]),
        real_n_donors_above_model_max_q95=int((don.mean_z2 > np.quantile(dz.max(0), .95)).sum()),
        model_max_over_donors_mean=float(dz.max(0).mean()),
        model_max_over_donors_q95=float(np.quantile(dz.max(0), .95)),
        model_max_over_donors_max=float(dz.max()),
        model_donor_sd=float(dz.std(0, ddof=1).mean()),
        real_donor_sd=float(don.mean_z2.std(ddof=1)),
        spearman_mean_z2_vs_mean_w_rank=float(sps.spearmanr(don.mean_z2, don.mean_w_rank)[0]),
        p_mean_z2_vs_mean_w_rank=float(sps.spearmanr(don.mean_z2, don.mean_w_rank)[1]),
        spearman_mean_w_rank_vs_log_lib=float(sps.spearmanr(don.mean_w_rank,
                                                            np.log(don.lib_size))[0]))
    sp_ = sps.spearmanr(don.mean_z2, np.log(don.lib_size))
    summ['donor_level'] = dict(
        spearman_mean_z2_vs_log_lib=float(sp_[0]), p=float(sp_[1]),
        top_q_donor_counts=P.top_q_donor.value_counts().head(5).to_dict(),
        top_influence_donor_counts=P.top_influence_donor.value_counts().head(5).to_dict(),
        max_mean_z2=float(don.mean_z2.max()), max_mean_z2_donor=str(don.mean_z2.idxmax()))
    # per-gene agreement of the generative NB arm with REAL
    summ['nb_gen_per_gene'] = dict(
        spearman_rej05=float(sps.spearmanr(P['rej_REAL_0.05'], P['rej_NB_GEN_0.05'])[0]),
        by_stratum={st_: dict(real=float(g_['rej_REAL_0.05'].mean()),
                              nb_gen=float(g_['rej_NB_GEN_0.05'].mean()),
                              nb_expw=float(g_['rej_NB_EXPW_0.05'].mean()),
                              power=float(g_['rej_POWER_0.05'].mean()),
                              n=int(len(g_)), median_n_over_kish=float(g_.n_over_kish.median()),
                              median_phi_over_shot=float((g_.phi / g_.shot_var).median()))
                    for st_, g_ in P.groupby('stratum')})
    P.to_csv(OUT / 'per_gene.tsv', sep='\t', index=False)
    (OUT / 'summary.json').write_text(json.dumps(summ, indent=2, default=float))
    figures(P, summ, Rsim, OUT)

    # ---- print ------------------------------------------------------------
    print('\npooled rates (REAL/OLS/scale: gene-clustered 95% interval; '
          'simulated: mean, MC sd across record sets)')
    for arm, d in summ['arms'].items():
        line = f'  {arm:20s}'
        for a in ALPHAS:
            x = d[str(a)]
            if 'lo' in x:
                line += f'  {a}: {x["rate"]:.4f} [{x["lo"]:.4f},{x["hi"]:.4f}]'
            else:
                line += f'  {a}: {x["rate"]:.4f} (sd {x["mc_sd"]:.4f})'
        print(line)
    print('excess partition', json.dumps(frac, indent=1))
    print('R_t', json.dumps(summ['R_t'], indent=1))
    print('gamma', summ['gamma'])
    print('depth', json.dumps(summ['depth'], indent=1))
    print('kurtosis by weight tercile', summ['kurtosis_by_weight_tercile'])
    print('hat', summ['hat'])
    print('rej05 by absorbed quintile', summ['rej05_by_absorbed_quintile'])
    print('associations', json.dumps(assoc, indent=1))
    for k_ in ('scale_mix_check_by_arm', 'scale_mix_drop_top_record', 'R_drop_top_record',
               'R_drop_top_record_floor', 'donor_recurrence_floor', 'donor_z2_floor',
               'donor_level', 'nb_gen_per_gene'):
        print(k_, json.dumps(summ[k_], indent=1))
    cols = ['gene', 'stratum', 'median_count', 'n_over_kish', 'R_t', 'R_model_q975',
            't_var_ratio', 'gamma_hat', 'LR_gamma1', 'phi', 'h_max', 'q_maxshare',
            'rej_REAL_0.05', 'rej_REAL_0.01', 'rej_REAL_scale_removed_0.05',
            'rej_NB_GEN_0.05', 'rej_POWER_0.05', 'rej_ISOLATE_OWN_0.05']
    pd.set_option('display.width', 250)
    print(P[cols].sort_values('t_var_ratio').round(3).to_string(index=False))
    print(f'\nwrote {OUT} in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
