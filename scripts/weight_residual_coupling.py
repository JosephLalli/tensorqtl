"""Is within-gene coupling between the Gibbs weight and the whitened squared
residual the cause of the ALLELIC nominal-p excess, and was elimination 5
(the "variance shape is too small to matter" finding of 2026-09-24)
mis-measured?

THE NULL. The shared instrument (scripts/null_permutation_instrument.py,
nominal_p_null_instrument_20260925/): 46 genes x 92 donors at RASQUAL's observed
lead, 2,000 donor-record permutations drawn from RandomState(42), one
permutation per round applied to every gene. The allelic record of donor j --
log ratio a_j and Gibbs variance va_j -- moves as a unit; the phased genotype
difference s_i = xL_i - xR_i in {-1, 0, 1} stays with position i. The allelic
fit is weighted least squares through the origin with w = 1/va, fitted scale
sigma^2 = RSS/(n_a - 1), n_a = donors with va > 1e-12, reference F(1, n_a - 1).
Under a records permutation the statistic is exactly

    num = sum_j s_{pos(j)} w_j a_j,   den = sum_j s_{pos(j)}^2 w_j,
    S   = sum_j w_j a_j^2 = sum_j z_j^2   (z_j = sqrt(w_j) a_j, permutation-invariant)
    t^2 = (n_a - 1) num^2 / (den S - num^2)

so each arm is two matrix products against the inverse-permutation genotype
matrix. Only the allelic channel is analysed here.

THE COUPLING STATISTIC. R_g = n sum(w z^2) / (sum(w) sum(z^2))
= mean(w z^2) / (mean(w) mean(z^2)), equivalently 1 + Cov(w, z^2)/(mean w mean
z^2). Part (a) derives it as the first-order closed form of sdratio_a^2 under
records permutation, with the exact finite-population prefactors and the
mean-shift term, and checks the algebra against the instrument's per_gene.tsv.
That check is ALGEBRA, not evidence: R_g and sdratio_a are functions of the same
records, so their agreement holds on any data.

WHAT IS EVIDENCE.
  (b) the per-gene model-null distribution of R_g (z iid N(0,1) at the real w;
      10,000 draws per gene): per-gene 2.5/97.5% points, the number of genes
      outside them, the across-gene sd of log R observed against its
      model-null distribution (one draw per gene, replicated), a combined
      normal-score dispersion statistic, each with and without CALM2 and with
      each gene's single most influential record removed.
  (c) arms on the IDENTICAL permutation stream, each changing only the
      allelic record set (w, admission and s are held fixed):
        REAL            the data; gated against the instrument
        MODEL           z iid N(0,1) at the real w                 (40 sets)
        DECOUPLE        the gene's own z shuffled against its w     (40 sets)
                        -- both marginals kept, the pairing destroyed
        ISOLATE_OWN     z_j ~ N(0, z_obs_j^2): each record keeps its own
                        realized squared residual as a variance     (20 sets)
        ISOLATE_SMOOTH  z_j ~ N(0, mu_j), mu = c w^delta, the maximum-
                        likelihood power law per gene               (20 sets)
        ISOLATE_SMOOTH_LOO  as above, but mu_j fitted with record j left out
                        (20 sets)
      NON-TAUTOLOGY of the smooth arm: an OLS fit of z^2 on w with an
      intercept satisfies sum(w * fitted) = sum(w z^2) by its normal
      equations and so reproduces R_g exactly. The power law is fitted instead
      by Gaussian maximum likelihood for z_j ~ N(0, c w_j^delta), equivalently
      a Gamma generalized linear model (a regression for a positive response
      whose variance grows with the square of its mean) with log link on
      regressor log w; its score equations are sum(z^2/mu - 1) = 0 and
      sum((z^2/mu - 1) log w) = 0, which contain log w but not w, so
      sum(w mu) is NOT constrained to equal sum(w z^2). The script prints the
      implied R_smooth = mean(w mu)/(mean(w) mean(mu)) against R_g per gene to
      show the gap. The leave-one-out variant additionally keeps each record's
      own z^2 out of its own variance.
  (d) elimination 5: the pooled slope of the standardized z^2 on log v
      (-0.064, residual_diagnostics_20260924) against the within-gene slope,
      the exact attenuation identity, per-gene slopes and power-law
      exponents, and records-permuted Gaussian arms at Var(a) = v^gamma with
      gamma = 0.936 (the elimination-5 value), 1 + the pooled within-gene
      slope, and 1 + the mean per-gene slope (20 sets each). Because
      elimination 5 used a FRESH-data design (no permutation; each position
      keeps its own v and a new Gaussian a is drawn) on the COMBINED
      statistic, the same exponents and the per-gene ML power law are also
      run in that design on the allelic channel alone (2,000 draws per gene,
      separate child stream), separating design and channel from exponent.
      The functional
      that governs the permutation variance, R_g - 1 = Cov(w, z^2)/(mean w
      mean z^2), is compared with what the log-v regression can see.

INTERVALS. Rates at 0.05, 0.01 and 0.001. Every pooled rate has a gene-
clustered percentile bootstrap interval (genes resampled with replacement,
2,000 resamples, the rate recomputed); every simulated arm also has the Monte
Carlo sd of the pooled rate across its record sets. Differences from REAL use a
PAIRED gene bootstrap (the same resampled genes for both arms) plus the arm's
Monte Carlo standard error.

GATES (the script aborts on failure):
  G1  REAL t^2 equals null_long.tsv.gz t2_a for every (gene, perm) to relative
      1e-9, and the pooled rejection counts at 0.05/0.01/0.001 are identical.
  G2  REAL sdratio_a and meanshift_a equal per_gene.tsv to relative 1e-9.
  G3  the pooled slope of standardized z^2 on log v reproduces
      residual_diagnostics_20260924's -0.06393 to relative 1e-6.
  G4  the attenuation identity pooled = within x SSW/(SSW+SSB) holds to 1e-10.

Master seed 42. The permutation stream is the instrument's RandomState(42);
everything else draws from SeedSequence(42) child streams.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats as sps

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
INST = D / 'nominal_p_null_instrument_20260925'
RESID = D / 'residual_diagnostics_20260924'
OUT = D / 'weight_residual_coupling_20260925'
SEED, EPS = 42, 1e-12
N_PERM, N_BOOT, N_NULL = 2000, 2000, 10000
ALPHAS = (0.05, 0.01, 0.001)
K_MODEL, K_DEC, K_ISO, K_POW = 40, 40, 20, 20
CALM2 = 'CALM2'


# ------------------------------------------------------------------ helpers
def load():
    d = np.load(INST / 'inputs_at_lead.npz', allow_pickle=True)
    genes = [str(g) for g in d['genes']]
    a, va, s = d['a'].astype(float), d['va'].astype(float), d['s'].astype(float)
    keep = np.isfinite(a) & np.isfinite(va) & (va > EPS)
    w = np.where(keep, 1.0 / np.where(keep, va, 1.0), 0.0)
    a0 = np.where(keep, a, 0.0)
    return d, genes, a0, va, s, w, keep


def perm_stream(N):
    rng = np.random.RandomState(SEED)
    perms = np.array([rng.permutation(N) for _ in range(N_PERM)])
    inv = np.argsort(perms, axis=1)     # inv[p, j] = position record j lands at
    return perms, inv


def t2_sets(Z, w, keep, Sg):
    """t^2 for K whitened record sets Z (K, N; zero where not admitted).

    Sg (P, N): s at the position each record lands in, per permutation.
    Returns (K, P).
    """
    n = keep.sum()
    sw = np.sqrt(w)
    Pm = Z * sw[None, :]                       # p_j = w_j a_j = sqrt(w_j) z_j
    S = (Z ** 2).sum(1)                        # sum z^2, invariant
    num = Pm @ Sg.T                            # (K, P)
    den = (Sg ** 2) @ w                        # (P,)
    return (n - 1) * num ** 2 / (den[None, :] * S[:, None] - num ** 2)


def pvals(t2, n):
    return sps.f.sf(t2, 1, n - 1)


def rcoup(w, z2):
    """R_g = mean(w z^2)/(mean(w) mean(z^2)) over admitted records."""
    return (w * z2).mean(-1) / (w.mean() * z2.mean(-1))


def boot_idx(rng, G):
    return rng.integers(0, G, size=(N_BOOT, G))


def pooled_ci(per_gene_rate, idx):
    est = per_gene_rate.mean()
    b = per_gene_rate[idx].mean(1)
    return float(est), float(np.quantile(b, .025)), float(np.quantile(b, .975))


def fit_power(x, z2):
    """Gaussian ML for z ~ N(0, c exp(delta x)); returns (log c, delta).

    Profile: for fixed delta the ML log c is log mean(z^2 exp(-delta x)).
    """
    xc = x - x.mean()

    def prof(dl):
        return len(z2) * np.log(np.mean(z2 * np.exp(-dl * xc)))
    r = optimize.minimize_scalar(prof, bounds=(-4, 4), method='bounded',
                                 options=dict(xatol=1e-10))
    dl = r.x
    lc = np.log(np.mean(z2 * np.exp(-dl * xc))) - dl * x.mean()
    return lc, dl


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(exist_ok=True)
    d, genes, a0, va, s, w, keep = load()
    G, N = a0.shape
    perms, inv = perm_stream(N)
    ss = np.random.SeedSequence(SEED).spawn(8)
    rng_boot, rng_null, rng_model, rng_dec, rng_iso, rng_sm, rng_loo, rng_pow = \
        [np.random.default_rng(c) for c in ss]
    no_calm2 = np.array([g != CALM2 for g in genes])
    summary = dict(question='coupling-identification', n_genes=G, n_perm=N_PERM,
                   n_donors=N)

    Sg = [s[k][inv] for k in range(G)]                # (P, N) per gene
    Z = np.sqrt(w) * a0                               # observed whitened z
    nA = keep.sum(1)

    # ============================================================ G1: REAL
    L = pd.read_csv(INST / 'null_long.tsv.gz', sep='\t',
                    usecols=['gene', 'perm', 't2_a', 'p_a', 'ba', 'sea'])
    L = L.set_index(['gene', 'perm']).sort_index()
    t2_real = np.zeros((G, N_PERM))
    beta_real = np.zeros((G, N_PERM))
    se_real = np.zeros((G, N_PERM))
    maxrel = 0.0
    for k, g in enumerate(genes):
        kp = keep[k]
        t2 = t2_sets(Z[k][None, :], w[k], kp, Sg[k])[0]
        t2_real[k] = t2
        num = (Z[k] * np.sqrt(w[k])) @ Sg[k].T
        den = (Sg[k] ** 2) @ w[k]
        S = (Z[k] ** 2).sum()
        beta_real[k] = num / den
        se_real[k] = np.sqrt((S - num ** 2 / den) / (nA[k] - 1) / den)
        ref = L.loc[g].t2_a.reindex(range(N_PERM)).values
        if np.isnan(ref).any():
            raise SystemExit(f'G1 FAILED: instrument rows missing for {g}')
        maxrel = max(maxrel, float(np.max(np.abs(t2 - ref) / np.abs(ref))))
    p_real = np.array([pvals(t2_real[k], nA[k]) for k in range(G)])
    mine = [int((p_real < al).sum()) for al in ALPHAS]
    theirs = [int((L.p_a.values < al).sum()) for al in ALPHAS]
    print(f'G1 REAL vs instrument: max rel |dt2| {maxrel:.2e}; '
          f'counts mine {mine} instrument {theirs}')
    if maxrel > 1e-9 or mine != theirs:
        raise SystemExit('G1 FAILED: REAL arm does not reproduce the instrument')
    summary['gate_G1'] = dict(max_rel_t2=maxrel, counts=mine,
                              instrument_counts=theirs)

    # ======================================================= (a) identity
    PG = pd.read_csv(INST / 'per_gene.tsv', sep='\t', index_col='gene')
    rows = []
    g2max = 0.0
    for k, g in enumerate(genes):
        kp = keep[k]
        wk, zk2 = w[k][kp], Z[k][kp] ** 2
        n = int(kp.sum())
        p = (np.sqrt(w[k]) * Z[k])                    # w a, zero if not admitted
        sk = s[k]
        sbar, s2bar = sk.mean(), (sk ** 2).mean()
        var_s = sk.var()                              # population variance, 1/N
        S = zk2.sum()
        Rg = float(rcoup(wk, zk2))
        sum_p = p.sum()
        var_num = N / (N - 1) * var_s * ((p ** 2).sum() - sum_p ** 2 / N)
        E_den = s2bar * w[k].sum()
        R_fp = (n - 1) * var_num / (E_den * S)
        pref = (n - 1) / n * N / (N - 1) * var_s / s2bar
        mterm = n / N * sum_p ** 2 / (w[k].sum() * S)
        M2 = (n - 1) * (sbar * sum_p) ** 2 / (E_den * S)   # (E num)^2 / (E den S/(n-1))
        # empirical, from the REAL permutations
        b, se = beta_real[k], se_real[k]
        sdr = float(b.std(ddof=1) / np.sqrt((se ** 2).mean()))
        msh = float(b.mean() / np.sqrt((se ** 2).mean()))
        g2max = max(g2max, abs(sdr / PG.loc[g, 'sdratio_a'] - 1),
                    abs(msh - PG.loc[g, 'meanshift_a']) /
                    max(abs(PG.loc[g, 'meanshift_a']), 1e-12))
        num = p @ Sg[k].T
        rows.append(dict(
            gene=g, stratum=str(d['strata'][k]), n_a=n,
            n_het=int((sk != 0).sum()), sum_s=float(sk.sum()),
            R_g=Rg, prefactor=pref, mean_term=mterm, R_fp=R_fp,
            R_fp_check=pref * (Rg - mterm), M2=M2,
            var_num_formula=var_num, var_num_perm=float(num.var(ddof=0)),
            sdratio2=sdr ** 2, meanshift=msh,
            Et2=float(t2_real[k].mean()),
            log_sdr2_over_Rfp=float(np.log(sdr ** 2 / R_fp)),
            top_share=float((wk * zk2).max() / (wk * zk2).sum()),
            top_w_share=float(wk.max() / wk.sum()),
            kish_frac=float(wk.sum() ** 2 / (wk ** 2).sum() / n),
            sd_logw=float(np.log(wk).std()),
            rej05=float((p_real[k] < .05).mean()),
            rej01=float((p_real[k] < .01).mean()),
            rej001=float((p_real[k] < .001).mean())))
    A = pd.DataFrame(rows).set_index('gene')
    print(f'G2 REAL sdratio/meanshift vs per_gene.tsv: max rel {g2max:.2e}')
    if g2max > 1e-9:
        raise SystemExit('G2 FAILED')
    summary['gate_G2'] = g2max
    assert np.allclose(A.R_fp, A.R_fp_check, rtol=1e-12)
    vn = np.log(A.var_num_perm / A.var_num_formula)
    summary['identity'] = dict(
        statement=('sdratio_a^2 ~ R_fp = (n-1)/n * N/(N-1) * var_s/mean(s^2) * '
                   '[R_g - (n/N) (sum w a)^2/(sum w * sum z^2)]; '
                   'E[t^2] ~ R_fp + M2, M2 = (n-1)(mean(s) sum w a)^2 / '
                   '(mean(s^2) sum w * sum z^2)'),
        prefactor_range=[float(A.prefactor.min()), float(A.prefactor.max())],
        mean_term_max=float(A.mean_term.max()),
        M2_max=float(A.M2.max()), M2_median=float(A.M2.median()),
        var_num_perm_over_formula_median=float(np.exp(vn.median())),
        var_num_perm_over_formula_sd_log=float(vn.std()),
        var_num_mc_floor_sd_log=float(np.sqrt(2 / N_PERM)),
        log_sdr2_over_Rfp_median=float(A.log_sdr2_over_Rfp.median()),
        log_sdr2_over_Rfp_maxabs=float(A.log_sdr2_over_Rfp.abs().max()),
        log_sdr2_over_Rfp_maxabs_gene=str(A.log_sdr2_over_Rfp.abs().idxmax()),
        pearson_sdr2_Rfp=float(np.corrcoef(A.sdratio2, A.R_fp)[0, 1]),
        spearman_sdr_Rg=float(sps.spearmanr(A.sdratio2, A.R_g)[0]),
        pearson_Et2_Rfp_plus_M2=float(np.corrcoef(A.Et2, A.R_fp + A.M2)[0, 1]),
        spearman_rej05_Rg=float(sps.spearmanr(A.rej05, A.R_g)[0]),
        log_Et2_over_Rfp_M2_median=float(np.median(np.log(A.Et2 / (A.R_fp + A.M2)))),
        log_Et2_over_Rfp_M2_maxabs=float(np.abs(np.log(A.Et2 / (A.R_fp + A.M2))).max()),
        log_sdr2_over_Rfp_maxabs_noCALM2=float(
            A.log_sdr2_over_Rfp.drop(CALM2).abs().max()),
        pearson_sdr2_Rfp_noCALM2=float(np.corrcoef(
            A.sdratio2.drop(CALM2), A.R_fp.drop(CALM2))[0, 1]),
        note='ALGEBRA: R_fp and sdratio_a are functions of the same records; '
             'their agreement is an identity check, not evidence of coupling.')
    print(A[['R_g', 'R_fp', 'sdratio2', 'M2', 'top_share', 'rej05']]
          .sort_values('R_g').round(3).to_string())

    # ================================================ (b) model null of R_g
    nr = []
    Rnull = np.zeros((G, N_NULL))
    Rnull_drop = np.zeros((G, N_NULL))
    for k, g in enumerate(genes):
        kp = keep[k]
        wk, zk2 = w[k][kp], Z[k][kp] ** 2
        zz = rng_null.standard_normal((N_NULL, kp.sum())) ** 2
        Rnull[k] = rcoup(wk, zz)
        # drop the single most influential record (largest w z^2) per draw
        contrib = wk[None, :] * zz
        j = contrib.argmax(1)
        m = np.ones_like(zz, bool); m[np.arange(N_NULL), j] = False
        wm = np.where(m, wk[None, :], 0.0); zm = np.where(m, zz, 0.0)
        nm = kp.sum() - 1
        Rnull_drop[k] = (nm * (wm * zm).sum(1)) / (wm.sum(1) * zm.sum(1))
        jo = (wk * zk2).argmax()
        mo = np.ones(kp.sum(), bool); mo[jo] = False
        R_drop = float(rcoup(wk[mo], zk2[mo]))
        lo, hi = np.quantile(Rnull[k], [.025, .975])
        u = float((Rnull[k] < A.loc[g, 'R_g']).mean()
                  + 0.5 * (Rnull[k] == A.loc[g, 'R_g']).mean())
        ud = float((Rnull_drop[k] < R_drop).mean())
        nr.append(dict(gene=g, R_g=A.loc[g, 'R_g'], null_q025=lo,
                       null_median=float(np.median(Rnull[k])), null_q975=hi,
                       null_sd_logR=float(np.log(Rnull[k]).std()), u=u,
                       R_drop_top=R_drop,
                       null_drop_q025=float(np.quantile(Rnull_drop[k], .025)),
                       null_drop_q975=float(np.quantile(Rnull_drop[k], .975)),
                       u_drop=ud))
    NR = pd.DataFrame(nr).set_index('gene')
    A = A.join(NR.drop(columns='R_g'))

    def spread_test(mask, Robs, Rn):
        lo = np.log(Robs[mask]); sd_obs = float(lo.std(ddof=1))
        sd_null = np.log(Rn[mask]).std(axis=0, ddof=1)       # one draw per gene
        return dict(sd_obs=sd_obs, sd_null_median=float(np.median(sd_null)),
                    sd_null_q95=float(np.quantile(sd_null, .95)),
                    ratio_to_null_median=sd_obs / float(np.median(sd_null)),
                    p=float((sd_null >= sd_obs).mean()),
                    p_resolution=1.0 / N_NULL,
                    mean_logR_obs=float(lo.mean()),
                    mean_logR_null_median=float(np.median(np.log(Rn[mask]).mean(0))))

    def ns_test(u):
        uu = np.clip(u, 0.5 / N_NULL, 1 - 0.5 / N_NULL)
        q = sps.norm.ppf(uu)
        chi = float((q ** 2).sum())
        return dict(chi2=chi, df=len(q), p=float(sps.chi2.sf(chi, len(q))),
                    mean_normal_score=float(q.mean()))

    allm = np.ones(G, bool)
    Rg_arr = A.R_g.values; Rd_arr = A.R_drop_top.values
    bres = {}
    for lab, msk in (('all46', allm), ('noCALM2', no_calm2)):
        bres[lab] = dict(
            n_above_q975=int((A.R_g.values > A.null_q975.values)[msk].sum()),
            n_below_q025=int((A.R_g.values < A.null_q025.values)[msk].sum()),
            expected_each_side=0.025 * msk.sum(),
            binom_p_above=float(sps.binomtest(
                int((A.R_g.values > A.null_q975.values)[msk].sum()),
                int(msk.sum()), 0.025, alternative='greater').pvalue),
            binom_p_below=float(sps.binomtest(
                int((A.R_g.values < A.null_q025.values)[msk].sum()),
                int(msk.sum()), 0.025, alternative='greater').pvalue),
            spread=spread_test(msk, Rg_arr, Rnull),
            normal_score=ns_test(A.u.values[msk]),
            drop_top_record=dict(
                n_above_q975=int((Rd_arr > A.null_drop_q975.values)[msk].sum()),
                n_below_q025=int((Rd_arr < A.null_drop_q025.values)[msk].sum()),
                spread=spread_test(msk, Rd_arr, Rnull_drop),
                normal_score=ns_test(A.u_drop.values[msk])))
    summary['model_null_R'] = bres
    print(json.dumps(bres, indent=1))

    # ================================================= (d) slope arithmetic
    # standardized z^2 as residual_diagnostics: z^2/(S/n), mean exactly 1
    X, Y, gid = [], [], []
    slopes = []
    for k, g in enumerate(genes):
        kp = keep[k]
        zk2 = Z[k][kp] ** 2
        y = zk2 / zk2.mean()
        x = np.log(va[k][kp])
        X.append(x); Y.append(y); gid.append(np.full(len(x), k))
        sl = sps.linregress(x, y).slope
        lc, dl = fit_power(np.log(w[k][kp]), zk2)
        # R implied by the fitted log-v line (regressor log v, not w)
        f_lin = y.mean() + sl * (x - x.mean())
        R_lin = float(rcoup(w[k][kp], f_lin)) if (f_lin > 0).all() else \
            float((w[k][kp] * f_lin).mean() / (w[k][kp].mean() * f_lin.mean()))
        mu = np.exp(lc + dl * np.log(w[k][kp]))
        slopes.append(dict(gene=g, slope_logv=sl, delta_ml=dl,
                           gamma_ml=1 - dl, R_loglinear=R_lin,
                           R_smooth=float(rcoup(w[k][kp], mu)),
                           sum_wmu_over_sum_wz2=float((w[k][kp] * mu).sum() /
                                                      (w[k][kp] * zk2).sum()),
                           sd_logv=float(x.std())))
    X, Y, gid = map(np.concatenate, (X, Y, gid))
    pooled = sps.linregress(X, Y).slope
    ref_sl = json.loads((RESID / 'summary.json').read_text())['shape_allelic']['pooled_slope']
    print(f'G3 pooled slope {pooled:.6f} vs residual_diagnostics {ref_sl:.6f}')
    if abs(pooled / ref_sl - 1) > 1e-6:
        raise SystemExit('G3 FAILED')
    xm = np.array([X[gid == k].mean() for k in range(G)])
    ym = np.array([Y[gid == k].mean() for k in range(G)])
    nk = np.array([(gid == k).sum() for k in range(G)])
    xw = X - xm[gid]
    SSW, SSB = float((xw ** 2).sum()), float((nk * (xm - X.mean()) ** 2).sum())
    within = float((xw * (Y - ym[gid])).sum() / SSW)
    att = SSW / (SSW + SSB)
    print(f'G4 pooled {pooled:.6f} = within {within:.6f} x att {att:.4f} '
          f'= {within * att:.6f}; gene means of y max |1-ybar| '
          f'{np.abs(ym - 1).max():.1e}')
    if abs(pooled - within * att) > 1e-10:
        raise SystemExit('G4 FAILED')
    SL = pd.DataFrame(slopes).set_index('gene')
    A = A.join(SL)
    gam_within = 1 + within
    gam_mean = 1 + float(SL.slope_logv.mean())
    summary['elimination5'] = dict(
        pooled_slope=float(pooled), within_slope=within, SSW=SSW, SSB=SSB,
        attenuation=att, gamma_pooled=1 + float(pooled),
        gamma_within=gam_within, gamma_mean_pergene=gam_mean,
        pergene_slope_mean=float(SL.slope_logv.mean()),
        pergene_slope_median=float(SL.slope_logv.median()),
        pergene_slope_negative=int((SL.slope_logv < 0).sum()),
        delta_ml_median=float(SL.delta_ml.median()),
        delta_ml_mean=float(SL.delta_ml.mean()),
        spearman_Rloglinear_Rg=float(sps.spearmanr(SL.R_loglinear, A.R_g)[0]),
        median_Rg=float(A.R_g.median()),
        median_R_loglinear=float(SL.R_loglinear.median()),
        median_R_smooth=float(SL.R_smooth.median()),
        mean_abs_logR_obs=float(np.abs(np.log(A.R_g)).mean()),
        mean_abs_logR_loglinear=float(np.abs(np.log(SL.R_loglinear)).mean()),
        mean_abs_logR_smooth=float(np.abs(np.log(SL.R_smooth)).mean()),
        median_sum_wmu_over_sum_wz2=float(SL.sum_wmu_over_sum_wz2.median()))
    print(json.dumps(summary['elimination5'], indent=1))

    # ================================================================ (c) arms
    def run_arm(make_sets, K):
        """make_sets(k) -> (K, n_admitted) whitened record sets for gene k."""
        rates = np.zeros((G, K, len(ALPHAS)))
        Rs = np.zeros((G, K))
        for k in range(G):
            kp = keep[k]
            zs = make_sets(k)
            Zf = np.zeros((zs.shape[0], N)); Zf[:, kp] = zs
            t2 = t2_sets(Zf, w[k], kp, Sg[k])
            p = pvals(t2, nA[k])
            for ai, al in enumerate(ALPHAS):
                rates[k, :, ai] = (p < al).mean(1)
            Rs[k] = rcoup(w[k][kp], zs ** 2)
        return rates, Rs

    arms = {}
    arms['REAL'] = (np.stack([[(p_real[k] < al).mean() for al in ALPHAS]
                              for k in range(G)])[:, None, :],
                    A.R_g.values[:, None])
    zobs = [Z[k][keep[k]] for k in range(G)]
    wadm = [w[k][keep[k]] for k in range(G)]
    arms['MODEL'] = run_arm(lambda k: rng_model.standard_normal(
        (K_MODEL, len(zobs[k]))), K_MODEL)
    arms['DECOUPLE'] = run_arm(lambda k: np.stack(
        [rng_dec.permutation(zobs[k]) for _ in range(K_DEC)]), K_DEC)
    arms['ISOLATE_OWN'] = run_arm(lambda k: np.abs(zobs[k])[None, :] *
                                  rng_iso.standard_normal((K_ISO, len(zobs[k]))),
                                  K_ISO)
    mu_in, mu_loo = [], []
    for k in range(G):
        x = np.log(wadm[k]); z2 = zobs[k] ** 2
        lc, dl = fit_power(x, z2)
        mu_in.append(np.exp(lc + dl * x))
        ml = np.empty(len(x))
        for j in range(len(x)):
            m = np.ones(len(x), bool); m[j] = False
            lcj, dlj = fit_power(x[m], z2[m])
            ml[j] = np.exp(lcj + dlj * x[j])
        mu_loo.append(ml)
    A['R_smooth_loo'] = [float(rcoup(wadm[k], mu_loo[k])) for k in range(G)]
    arms['ISOLATE_SMOOTH'] = run_arm(lambda k: np.sqrt(mu_in[k])[None, :] *
                                     rng_sm.standard_normal((K_ISO, len(zobs[k]))),
                                     K_ISO)
    arms['ISOLATE_SMOOTH_LOO'] = run_arm(
        lambda k: np.sqrt(mu_loo[k])[None, :] *
        rng_loo.standard_normal((K_ISO, len(zobs[k]))), K_ISO)
    for lab, gam in (('POWER_g0.936_elim5', 0.936),
                     (f'POWER_g{gam_within:.3f}_within', gam_within),
                     (f'POWER_g{gam_mean:.3f}_pergene_mean', gam_mean)):
        # Var(a) = v^gamma  =>  Var(z) = w * v^gamma = w^(1-gamma)
        arms[lab] = run_arm(lambda k, gm=gam: np.sqrt(wadm[k] ** (1 - gm))[None, :] *
                            rng_pow.standard_normal((K_POW, len(zobs[k]))), K_POW)

    # FRESH-data analogue of elimination 5's design: no permutation, each
    # position keeps its own record and w, new Gaussian a ~ N(0, v^gamma) per
    # draw (the parametric_bootstrap_20260924 'shape' arm, allelic channel
    # alone). Not on the permutation stream; reported to separate the design
    # difference from the exponent difference.
    rng_fresh = np.random.default_rng(np.random.SeedSequence(SEED).spawn(9)[8])
    K_FRESH = 2000
    fresh = {}
    for lab, gm in (('FRESH_g0.936', 0.936), (f'FRESH_g{gam_within:.3f}', gam_within),
                    ('FRESH_ML_powerlaw', None), ('FRESH_model_g1', 1.0)):
        rr = np.zeros((G, len(ALPHAS)))
        for k in range(G):
            kp = keep[k]
            var_z = mu_in[k] if gm is None else wadm[k] ** (1 - gm)
            zs = np.sqrt(var_z)[None, :] * rng_fresh.standard_normal((K_FRESH, kp.sum()))
            Zf = np.zeros((K_FRESH, N)); Zf[:, kp] = zs
            p = pvals(t2_sets(Zf, w[k], kp, s[k][None, :])[:, 0], nA[k])
            rr[k] = [(p < al).mean() for al in ALPHAS]
        fresh[lab] = {str(al): float(rr[:, ai].mean()) for ai, al in enumerate(ALPHAS)}
        fresh[lab]['binomial_se_0.05'] = float(np.sqrt(.05 * .95 / (G * K_FRESH)))
    summary['fresh_design'] = fresh
    print('fresh-data design (allelic only):', json.dumps(fresh, indent=1))

    idx_all = boot_idx(rng_boot, G)
    idx_nc = boot_idx(rng_boot, G - 1)
    real_pg = arms['REAL'][0][:, 0, :]
    rows = []
    for name, (rates, Rs) in arms.items():
        K = rates.shape[1]
        for lab, msk, idx in (('all46', allm, idx_all), ('noCALM2', no_calm2, idx_nc)):
            pg = rates[msk].mean(1)                      # (G', alpha)
            rp = real_pg[msk]
            for ai, al in enumerate(ALPHAS):
                est, lo, hi = pooled_ci(pg[:, ai], idx)
                per_set = rates[msk][:, :, ai].mean(0)   # pooled rate per set
                mcsd = float(per_set.std(ddof=1)) if K > 1 else 0.0
                diff = rp[:, ai] - pg[:, ai]
                db = diff[idx].mean(1)
                rows.append(dict(
                    arm=name, genes=lab, alpha=al, K=K, rate=est, lo=lo, hi=hi,
                    mc_sd_across_sets=mcsd, mc_se_mean=mcsd / np.sqrt(K),
                    real_minus_arm=float(diff.mean()),
                    diff_lo=float(np.quantile(db, .025)),
                    diff_hi=float(np.quantile(db, .975)),
                    mean_logR=float(np.log(Rs[msk]).mean()),
                    sd_logR_across_genes=float(np.log(Rs[msk].mean(1)).std(ddof=1))
                    if K == 1 else float(np.median(np.log(Rs[msk]).std(0, ddof=1)))))
    AR = pd.DataFrame(rows)
    # share of the REAL-over-MODEL excess removed by decoupling, paired genes
    shares = {}
    rd, rm = arms['DECOUPLE'][0].mean(1), arms['MODEL'][0].mean(1)
    for ai, al in enumerate(ALPHAS):
        num_ = (real_pg[:, ai] - rd[:, ai])[idx_all].mean(1)
        den_ = (real_pg[:, ai] - rm[:, ai])[idx_all].mean(1)
        ok = den_ > 0
        est = float((real_pg[:, ai] - rd[:, ai]).mean() / (real_pg[:, ai] - rm[:, ai]).mean())
        shares[str(al)] = dict(share_removed_by_decoupling=est,
                               lo=float(np.quantile(num_[ok] / den_[ok], .025)),
                               hi=float(np.quantile(num_[ok] / den_[ok], .975)))
    summary['coupling_share_of_excess'] = shares
    summary['pergene_spearman_vs_REAL_rej0.05'] = {
        name: float(sps.spearmanr(rates[:, :, 0].mean(1), real_pg[:, 0])[0])
        for name, (rates, _) in arms.items() if name != 'REAL'}
    print(json.dumps(shares, indent=1))
    print(json.dumps(summary['pergene_spearman_vs_REAL_rej0.05'], indent=1))
    pd.set_option('display.width', 250)
    print(AR.round(5).to_string())
    AR.to_csv(OUT / 'arm_rates_pooled.tsv', sep='\t', index=False)
    pgt = []
    for name, (rates, Rs) in arms.items():
        for k, g in enumerate(genes):
            pgt.append(dict(arm=name, gene=g, R_mean=float(Rs[k].mean()),
                            **{f'rej{al}': float(rates[k, :, ai].mean())
                               for ai, al in enumerate(ALPHAS)}))
    pd.DataFrame(pgt).to_csv(OUT / 'arm_rates_per_gene.tsv', sep='\t', index=False)
    summary['arms'] = {f'{r.arm}|{r.genes}|{r.alpha}': dict(
        rate=r.rate, lo=r.lo, hi=r.hi, mc_sd=r.mc_sd_across_sets,
        mc_se_mean=r.mc_se_mean, real_minus_arm=r.real_minus_arm,
        diff_lo=r.diff_lo, diff_hi=r.diff_hi, K=int(r.K))
        for r in AR.itertuples()}

    # per-gene rejection split by coupling sign, REAL
    for lab, m in (('R_gt_1', A.R_g.values > 1), ('R_lt_1', A.R_g.values < 1)):
        summary[f'real_split_{lab}'] = dict(
            n_genes=int(m.sum()),
            **{str(al): float(real_pg[m, ai].mean()) for ai, al in enumerate(ALPHAS)})
    A.to_csv(OUT / 'per_gene_coupling.tsv', sep='\t')
    SL.to_csv(OUT / 'slopes_and_power_law.tsv', sep='\t')

    # ================================================================ figures
    Ao = A.sort_values('R_g')
    fig, ax = plt.subplots(figsize=(12, 4.8))
    xs = np.arange(G)
    ax.vlines(xs, Ao.null_q025, Ao.null_q975, color='0.7', lw=5,
              label='model-null 2.5-97.5% (z iid N(0,1) at real w)')
    ax.plot(xs, Ao.null_median, '_', color='0.35', ms=10, label='model-null median')
    out_ = (Ao.R_g > Ao.null_q975) | (Ao.R_g < Ao.null_q025)
    ax.plot(xs[~out_.values], Ao.R_g[~out_], 'o', color='tab:blue', label='observed R_g, inside band')
    ax.plot(xs[out_.values], Ao.R_g[out_], 'o', color='tab:red', label='observed R_g, outside band')
    ax.plot(xs, Ao.R_smooth, 'x', color='tab:green', label='R implied by ML power law (in-sample)')
    ax.set_yscale('log'); ax.axhline(1, color='k', lw=0.6)
    ax.set_xticks(xs); ax.set_xticklabels(Ao.index, rotation=90, fontsize=7)
    for lbl in ax.get_xticklabels():
        if lbl.get_text() == CALM2:
            lbl.set_color('tab:red'); lbl.set_fontweight('bold')
    ax.set_ylabel('R_g = mean(w z^2) / (mean(w) mean(z^2))')
    ax.set_title('Weight-residual coupling per gene, observed against its model-null band')
    ax.legend(fontsize=7, loc='upper left')
    fig.tight_layout(); fig.savefig(OUT / 'fig_Rg_vs_model_null.png', dpi=150)
    plt.close(fig)

    names = list(arms.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    for ai, al in enumerate(ALPHAS):
        ax = axes[ai]
        for off, lab, col in ((-0.15, 'all46', 'tab:blue'), (0.15, 'noCALM2', 'tab:orange')):
            sub = AR[(AR.alpha == al) & (AR.genes == lab)].set_index('arm').loc[names]
            yy = np.arange(len(names)) + off
            ax.errorbar(sub.rate / al, yy, xerr=[(sub.rate - sub.lo) / al,
                                                 (sub.hi - sub.rate) / al],
                        fmt='o', color=col, ms=4, capsize=2,
                        label='46 genes' if lab == 'all46' else 'without CALM2')
        ax.axvline(1, color='k', lw=0.7)
        ax.set_title(f'alpha = {al}')
        ax.set_xlabel('rejection rate / alpha (gene-clustered 95% interval)')
    axes[0].set_yticks(np.arange(len(names))); axes[0].set_yticklabels(names, fontsize=8)
    axes[0].invert_yaxis(); axes[0].legend(fontsize=8)
    fig.suptitle('Allelic nominal-p rejection under the records permutation, by arm '
                 '(identical permutation stream)')
    fig.tight_layout(); fig.savefig(OUT / 'fig_arm_ladder.png', dpi=150)
    plt.close(fig)

    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, default=float))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
