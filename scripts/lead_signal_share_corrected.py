"""How much of the nominal-p excess on the records null and on the sign-flip
null is the REAL allelic signal at RASQUAL's lead that the null draws carry,
measured with LEVERAGE-CORRECTED lead-removed residuals? And does that signal
explain why the allelic-only empirical permutation p called 7 genes under sign
flip against 11 under records on 2026-09-24?

QUESTION. Under a donor-records permutation (and under a haplotype-label sign
flip) the null draws are built from the observed records, which contain
whatever allelic effect the tested variant really has. That effect does not
vanish under permutation: it is carried into the null as extra, structured
residual magnitude at heterozygotes. The first round estimated its share of
the excess at ~18% at 0.05 with UNCORRECTED lead removal, which the gene-
property verifier showed is conservative by construction (0.0424 under a true
null), because the raw residual e = a - b_hat s of a through-origin weighted
least squares fit has Var(e_i) = sigma^2 va_i (1 - h_i), deflated at every
heterozygote by its weighted leverage h_i = w_i s_i^2 / sum_j w_j s_j^2.

DESIGN. 46 genes x 92 donors at each gene's RASQUAL observed lead, read from
nominal_p_null_instrument_20260925/inputs_at_lead.npz. Nothing is re-derived
from the Gibbs cache.

  Lead removal, allelic channel. b_obs = sum w s a / sum w s^2 over admitted
  donors (va > 1e-12, INCLUDING s = 0), e_i = a_i - b_obs s_i, and the
  corrected record e*_i = e_i / sqrt(1 - h_i). Under the Gaussian model
  Var(e*_i) = sigma^2 va_i exactly. e does not depend on the true slope, so
  lead removal of (true signal + noise) equals lead removal of the noise alone.
  Lead removal, total channel. In the sqrt(w_t) space, y = M sqrt(w_t) t and
  x0 = M sqrt(w_t) g with M the residual-maker of Z = [1, 17 covariates];
  e = y - b_t x0, h_i = (Z leverage)_i + x0_i^2 / sum x0^2 (leverage of the
  full design), record t* = (e_i / sqrt(1 - h_i)) / sqrt(w_t,i). The Z-fitted
  part is dropped because a records-permuted fit partials out the covariate
  row that travels with the record, so keeping it changes nothing.

  Nulls.
    records    the instrument's RandomState(42) stream, 2,000 permutations;
               each donor record (a, va, t, vt, covariate row) moves as a unit
               against the fixed genotype. Evaluated by relabelling: permuting
               records against a fixed genotype equals applying the inverse
               permutation to the genotype, so the covariate partialling of the
               record is computed once per gene.
    sign flip  haplotype-label swap, a_i -> r_i a_i with Rademacher r_i,
               records at their own positions. 2,000 draws on the first-round
               verifier's stream (SeedSequence(42).spawn(9)[5], default_rng,
               choice([-1, 1], (2000, 92))) so its numbers are reproduced.

  Arms (what varies is only the record set; weights, admission, genotypes
  and draws are fixed):
    REAL        the data; gated against the instrument per (gene, perm)
    LR          leverage-corrected lead-removed records (both channels)
    LR_UNCORR   uncorrected lead-removed records (e, not e*)
    LR_MIX      allelic corrected (e*), total UNcorrected (e). The total
                channel's covariate row travels with its record and is
                partialled again inside every permuted fit, so its raw
                residual already has the rank structure that fit expects;
                dividing by sqrt(1 - h) with the 19-column leverage then
                over-inflates it. Which removal is nominal in each channel is
                not assumed: the MODEL/DECOUPLE/SIGNRAND arms below measure it
                for all three variants, and the combined share is read from
                the variant that is nominal in both channels
    MODEL       z iid N(0,1) at the real weights (40 sets); with and without
                lead removal, the latter being the true-null check that the
                corrected lead removal is nominal
    DECOUPLE    the gene's own signed z = sqrt(w) a shuffled among admitted
                donors against w (40 sets): the real heavy-tailed marginal,
                coupling and lead association destroyed. With and without
                lead removal: the heavy-tailed selection control, i.e. how
                much lead removal changes the rate when there is no lead
                association to remove
    SIGNRAND    each record keeps its own |z| at its own weight, with a random
                sign (40 sets): real marginal AND real weight coupling kept,
                lead association destroyed. With and without lead removal:
                the second heavy-tailed selection control
  Controls follow the first-round correction that overturned two claims: the
  selection control keeps the real heavy-tailed z marginal, it is not
  Gaussian.

  TRUE-NULL CHECK (reported, not an abort gate, because a failure is itself
  a finding): MODEL_LR* against MODEL, per channel, null and tier, with the
  gene-clustered interval and the Monte Carlo sd of the paired difference.

  Lead-signal share at tier alpha = (REAL - LR) / (REAL - MODEL), the
  fraction of the excess over the Gaussian model that lead removal takes away.
  Net share subtracts the control's own lead-removal effect:
  ((REAL - LR) - (CTRL - CTRL_LR)) / (REAL - MODEL).

  Combined statistic: inverse-variance meta-analysis of the two channel
  slopes, t^2 against F(1, min(dof_a, dof_t)); under controls the allelic
  set k and the total set k are paired (independent shuffles per channel).

  (e) The 2026-09-24 7-vs-11 finding: map_cis's allelic-only pval_nominal,
  empirical p = (1 + #{p_null <= p_obs}) / (1 + n_draw), 30 records draws and
  30 sign-flip draws from one RandomState(42) (30 permutations, then 30
  choice([1, -1], 92)). map_cis refers the allelic-only t^2 to F(1, 73), the
  GLOBAL dof N - 2 - 17, not n_a - 1; because n_a is invariant under both nulls
  this changes no empirical p, and the reproduction below uses it only to
  match p values. Recomputed with lead-removed corrected null records, at 30
  draws (same streams) and at 2,000 draws. A null draw counts as "at least as
  extreme" when t2_null >= t2_obs * (1 - 1e-9): a sign flip that swaps every
  heterozygote together (or none) returns the observed t^2 exactly, map_cis
  counted those as ties in float32, and in float64 they differ by an ulp (LDLR,
  EMX2). The tolerance restores the tie and is what reproduces the 2026-09-24
  per-gene empirical p.

GATES (abort on failure):
  G1  REAL records arm vs null_long.tsv.gz: t2_a, t2_t, t2_b per (gene, perm)
      to relative 1e-9 (absolute 1e-9 where t2 < 1e-3), and the pooled counts
      of p_a, p_t, p_b below 0.05/0.01/0.001 exactly equal.
  G2  observed fit vs observed.tsv, same tolerance.
  G3  lead removal is signal-invariant: e* of (a + 3 b_obs s) equals e* of a
      to 1e-10 for every gene (both channels).
  G4  the 30-draw 2026-09-24 p values (allelic_null_schemes_20260924/pvals.tsv,
      both schemes) and observed p (perm_p_by_scheme_20260924/per_gene.tsv)
      reproduced to relative 1e-4 (map_cis is float32), and every per-gene
      empirical p and the 11 / 7 counts reproduced exactly.
  G5  the first-round sign-flip rates (raw 0.0982/0.0255/0.00325, lead-
      removed corrected 0.0759/0.0197/0.0026) reproduced to their printed
      rounding.

INTERVALS. Gene-clustered percentile bootstrap (genes resampled with
replacement, 2,000 resamples) for every pooled rate, difference and share;
Monte Carlo sd across simulated sets for every simulated arm. Master seed 42;
SeedSequence(42).spawn(16): child 5 is the first-round sign-flip stream and is
used only for that; children 10-15 are this script's.

OUTPUTS: brainvar_hapmix_deploy/lead_signal_share_corrected_20260925/
  summary.json, per_gene.tsv, emp_p_per_gene.tsv, fig_rates.png, fig_emp_p.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
INST = D / 'nominal_p_null_instrument_20260925'
OUT = D / 'lead_signal_share_corrected_20260925'
SEED, EPS = 42, 1e-12
N_PERM, N_BOOT, K = 2000, 2000, 40
ALPHAS = (0.05, 0.01, 0.001)
FIRST_ROUND_FLIP = dict(raw=(0.0982, 0.0255, 0.00325), lr=(0.0759, 0.0197, 0.0026))


# ------------------------------------------------------------ statistics
def allelic(Y, w, Sg, n):
    """Through-origin WLS for K record sets Y (K, N; 0 where not admitted)
    against P designs Sg (P, N). Returns ba, sea2 (K, P)."""
    num = (Y * w) @ Sg.T
    den = (Sg ** 2) @ w
    S = (w * Y ** 2).sum(1)
    rss = S[:, None] - num ** 2 / den[None, :]
    return num / den, rss / (n - 1) / den[None, :]


def total(Yw, Xres, dof):
    """Partialled records Yw (K, N) against partialled designs Xres (P, N)."""
    xy = Yw @ Xres.T
    xx = (Xres ** 2).sum(1)
    yy = (Yw ** 2).sum(1)
    return xy / xx, (yy[:, None] - xy ** 2 / xx) / dof / xx


def t2(b, se2):
    return b ** 2 / se2


def comb(ba, sea2, bt, set2):
    prec = 1 / sea2 + 1 / set2
    return (ba / sea2 + bt / set2) ** 2 / prec


def lr_allelic(a, w, s, keep, correct=True):
    b = (w * s * a).sum() / (w * s * s).sum()
    h = w * s * s / (w * s * s).sum()
    e = np.where(keep, a - b * s, 0.0)
    return e / np.sqrt(1 - h) if correct else e


def lr_total(y, x0, hz, correct=True):
    b = x0 @ y / (x0 @ x0)
    h = hz + x0 ** 2 / (x0 @ x0)
    e = y - b * x0
    return e / np.sqrt(1 - h) if correct else e


# ------------------------------------------------------------ summaries
def rates(t2m, dof):
    p = sps.f.sf(t2m, 1, dof)
    return np.stack([(p < al).mean(-1) for al in ALPHAS], -1)   # (..., 3)


def ci(per_gene, idx):
    """per_gene (G,) -> [estimate, lo, hi] pooled over genes."""
    b = per_gene[idx].mean(1)
    return [float(per_gene.mean()), float(np.quantile(b, .025)),
            float(np.quantile(b, .975))]


def ratio_ci(num, den, idx):
    b = num[idx].mean(1) / den[idx].mean(1)
    return [float(num.mean() / den.mean()), float(np.quantile(b, .025)),
            float(np.quantile(b, .975))]


def main():
    OUT.mkdir(exist_ok=True)
    z = np.load(INST / 'inputs_at_lead.npz', allow_pickle=True)
    genes = [str(g) for g in z['genes']]
    strata = np.array([str(x) for x in z['strata']])
    A, VA, S = z['a'].astype(float), z['va'].astype(float), z['s'].astype(float)
    T, VT, GD, C = z['t'].astype(float), z['vt'].astype(float), z['g'].astype(float), z['C']
    G, N = A.shape
    Z = np.column_stack([np.ones(N), C])

    rs = np.random.RandomState(SEED)
    perms = np.array([rs.permutation(N) for _ in range(N_PERM)])
    inv = np.argsort(perms, axis=1)          # record j lands at position inv[p, j]
    ss = np.random.SeedSequence(SEED).spawn(16)
    FLIP = np.random.default_rng(ss[5]).choice(np.array([-1.0, 1.0]), size=(N_PERM, N))
    r_model, r_dec, r_sgn, r_boot = [np.random.default_rng(c) for c in ss[10:14]]
    bidx = r_boot.integers(0, G, size=(N_BOOT, G))
    rs30 = np.random.RandomState(SEED)
    p30 = np.array([rs30.permutation(N) for _ in range(30)])
    f30 = np.array([rs30.choice([1, -1], size=N) for _ in range(30)]).astype(float)
    assert (p30 == perms[:30]).all()

    inst = pd.read_csv(INST / 'null_long.tsv.gz', sep='\t')
    inst_obs = pd.read_csv(INST / 'observed.tsv', sep='\t').set_index('gene')

    # per-arm per-gene rate arrays: key -> (G, K_or_1, 3)
    R = {}
    arms_rec = ['REAL', 'LR', 'LR_UNCORR', 'LR_MIX', 'MODEL', 'MODEL_LR', 'MODEL_LR_UNCORR',
                'MODEL_LR_MIX', 'DECOUPLE', 'DECOUPLE_LR', 'DECOUPLE_LR_MIX',
                'SIGNRAND', 'SIGNRAND_LR', 'SIGNRAND_LR_MIX']
    for ch in ('a', 't', 'b'):
        for arm in arms_rec:
            R[('rec', ch, arm)] = np.zeros((G, 1 if not arm.startswith(('MODEL', 'DECOUPLE', 'SIGNRAND')) else K, 3))
    for arm in arms_rec:
        R[('flip', 'a', arm)] = np.zeros_like(R[('rec', 'a', arm)])
    gate1 = dict(max_rel=0.0, n=0)
    my_counts = {c: np.zeros(3, int) for c in ('p_a', 'p_t', 'p_b')}
    per_gene = []
    emp = []          # (e) rows
    obs_t2a = np.zeros(G)
    ref30 = pd.read_csv(D / 'allelic_null_schemes_20260924' / 'pvals.tsv',
                        sep='\t').set_index(['scheme', 'gene', 'perm']).pval
    g4_max = 0.0
    lr_g3 = 0.0

    for k in range(G):
        a, va, s, t, vt, gd = A[k], VA[k], S[k], T[k], VT[k], GD[k]
        ka = np.isfinite(a) & np.isfinite(va) & (va > EPS)
        kt = np.isfinite(t) & np.isfinite(vt) & (vt > EPS)
        assert kt.all(), 'total channel admission is assumed complete'
        na = int(ka.sum()); dofa = na - 1; doft = N - Z.shape[1] - 1
        wa = np.where(ka, 1 / np.where(ka, va, 1), 0.0)
        a0 = np.where(ka, a, 0.0)
        wt = 1 / vt; swt = np.sqrt(wt)
        q, _ = np.linalg.qr(Z * swt[:, None])
        M = lambda v: v - (v @ q) @ q.T          # rows are vectors
        hz = (q ** 2).sum(1)
        Sg = s[inv]                              # (P, N) s at landing position
        Xres = M(swt[None, :] * gd[inv])         # (P, N)
        SgF = s[None, :] * FLIP
        y_obs = M(swt * t)
        x0 = M(swt * gd)

        # ---------------- REAL and lead-removed records ----------------
        e_a = lr_allelic(a0, wa, s, ka)
        e_a_u = lr_allelic(a0, wa, s, ka, correct=False)
        e_t = lr_total(y_obs, x0, hz)
        e_t_u = lr_total(y_obs, x0, hz, correct=False)
        # G3: signal invariance of lead removal
        b_obs = (wa * s * a0).sum() / (wa * s * s).sum()
        bt_obs = x0 @ y_obs / (x0 @ x0)
        lr_g3 = max(lr_g3,
                    np.abs(lr_allelic(np.where(ka, a0 + 3 * b_obs * s, 0), wa, s, ka) - e_a).max(),
                    np.abs(lr_total(y_obs + 3 * bt_obs * x0, x0, hz) - e_t).max())

        recs = dict(REAL=(a0, y_obs), LR=(e_a, M(e_t)), LR_UNCORR=(e_a_u, M(e_t_u)),
                    LR_MIX=(e_a, M(e_t_u)))
        for arm, (ya, yt) in recs.items():
            ba, sea2 = allelic(ya[None, :], wa, Sg, na)
            bt, set2 = total(yt[None, :], Xres, doft)
            ta, tt, tb = t2(ba, sea2), t2(bt, set2), comb(ba, sea2, bt, set2)
            R[('rec', 'a', arm)][k] = rates(ta, dofa)
            R[('rec', 't', arm)][k] = rates(tt, doft)
            R[('rec', 'b', arm)][k] = rates(tb, min(dofa, doft))
            baf, seaf = allelic(ya[None, :], wa, SgF, na)
            R[('flip', 'a', arm)][k] = rates(t2(baf, seaf), dofa)
            if arm == 'REAL':
                sub = inst[inst.gene == genes[k]].sort_values('perm')
                assert len(sub) == N_PERM and (sub.perm.values == np.arange(N_PERM)).all()
                for mine, col in ((ta[0], 't2_a'), (tt[0], 't2_t'), (tb[0], 't2_b')):
                    ref = sub[col].values
                    big = np.abs(ref) >= 1e-3
                    rel = np.abs(mine - ref) / np.where(big, np.abs(ref), 1.0)
                    gate1['max_rel'] = max(gate1['max_rel'], float(rel.max()))
                    gate1['n'] += len(ref)
                for c_, tm, d_ in (('p_a', ta[0], dofa), ('p_t', tt[0], doft),
                                   ('p_b', tb[0], min(dofa, doft))):
                    p = sps.f.sf(tm, 1, d_)
                    my_counts[c_] += [(p < al).sum() for al in ALPHAS]
                # G2 observed
                bo, so = allelic(a0[None, :], wa, s[None, :], na)
                bto, sto = total(y_obs[None, :], x0[None, :], doft)
                o = inst_obs.loc[genes[k]]
                for mine, col in ((t2(bo, so)[0, 0], 't2_a'), (t2(bto, sto)[0, 0], 't2_t'),
                                  (comb(bo, so, bto, sto)[0, 0], 't2_b')):
                    rel = abs(mine - o[col]) / (abs(o[col]) if abs(o[col]) >= 1e-3 else 1.0)
                    if rel > 1e-9:
                        raise SystemExit(f'G2 FAILED {genes[k]} {col} rel {rel:.2e}')
                obs_t2a[k] = t2(bo, so)[0, 0]
                obs_bo = float(bo[0, 0]); obs_ta = float(t2(bo, so)[0, 0])

        # ---------------- simulated record sets ------------------------
        idx_a = np.flatnonzero(ka)
        z_a = np.sqrt(wa[ka]) * a0[ka]
        z_t = y_obs                                 # standardized partialled residual
        def sets(kind, rng):
            Ya = np.zeros((K, N)); Yt = np.zeros((K, N))
            if kind == 'MODEL':
                Ya[:, ka] = rng.standard_normal((K, na)) / np.sqrt(wa[ka])
                Yt[:] = rng.standard_normal((K, N))
            elif kind == 'DECOUPLE':
                for j in range(K):
                    Ya[j, ka] = rng.permutation(z_a) / np.sqrt(wa[ka])
                    Yt[j] = rng.permutation(z_t)
            else:  # SIGNRAND
                Ya[:, ka] = rng.choice([-1.0, 1.0], (K, na)) * np.abs(z_a) / np.sqrt(wa[ka])
                Yt[:] = rng.choice([-1.0, 1.0], (K, N)) * np.abs(z_t)
            return Ya, M(Yt)
        for kind, rng in (('MODEL', r_model), ('DECOUPLE', r_dec), ('SIGNRAND', r_sgn)):
            Ya, Yt = sets(kind, rng)
            ya_lr = np.stack([lr_allelic(y, wa, s, ka) for y in Ya])
            variants = {kind: (Ya, Yt),
                        kind + '_LR': (ya_lr, M(np.stack([lr_total(y, x0, hz) for y in Yt]))),
                        kind + '_LR_MIX': (ya_lr, M(np.stack([lr_total(y, x0, hz, correct=False)
                                                              for y in Yt])))}
            if kind == 'MODEL':
                variants['MODEL_LR_UNCORR'] = (
                    np.stack([lr_allelic(y, wa, s, ka, correct=False) for y in Ya]),
                    M(np.stack([lr_total(y, x0, hz, correct=False) for y in Yt])))
            for arm, (ya, yt) in variants.items():
                ba, sea2 = allelic(ya, wa, Sg, na)
                bt, set2 = total(yt, Xres, doft)
                R[('rec', 'a', arm)][k] = rates(t2(ba, sea2), dofa)
                R[('rec', 't', arm)][k] = rates(t2(bt, set2), doft)
                R[('rec', 'b', arm)][k] = rates(comb(ba, sea2, bt, set2), min(dofa, doft))
                baf, seaf = allelic(ya, wa, SgF, na)
                R[('flip', 'a', arm)][k] = rates(t2(baf, seaf), dofa)

        # ---------------- (e) empirical p, 30 and 2,000 draws -----------
        inv30 = np.argsort(p30, axis=1)
        row = dict(gene=genes[k], stratum=strata[k], b_obs=obs_bo, t2_obs=obs_ta,
                   n_a=na, n_het=int(((s != 0) & ka).sum()))
        for lab, ya in (('raw', a0), ('lr', e_a)):
            for nd, Sr, Sf in ((30, s[inv30], s[None, :] * f30), (N_PERM, Sg, SgF)):
                ba, sea2 = allelic(ya[None, :], wa, Sr, na)
                tr = t2(ba, sea2)[0]
                baf, seaf = allelic(ya[None, :], wa, Sf, na)
                tf = t2(baf, seaf)[0]
                row[f'pemp_rec_{lab}_{nd}'] = (1 + int((tr >= obs_ta * (1 - 1e-9)).sum())) / (1 + nd)
                row[f'pemp_flip_{lab}_{nd}'] = (1 + int((tf >= obs_ta * (1 - 1e-9)).sum())) / (1 + nd)
                if lab == 'raw' and nd == 30:
                    for sch, tm in (('records', tr), ('sign_flip', tf)):
                        refp = ref30.loc[sch, genes[k]].sort_index().values
                        mp = sps.f.sf(tm, 1, N - 2 - C.shape[1])
                        g4_max = max(g4_max, float(np.max(np.abs(mp - refp) / refp)))
        emp.append(row)

        per_gene.append(dict(
            gene=genes[k], stratum=strata[k], n_a=na, n_het=row['n_het'],
            b_obs_a=obs_bo, t2_obs_a=obs_ta, max_h_a=float((wa * s * s / (wa * s * s).sum()).max()),
            **{f'{sch}_{arm}_{al}': float(R[(sch, 'a', arm)][k, :, i].mean())
               for sch in ('rec', 'flip') for arm in ('REAL', 'LR', 'MODEL', 'MODEL_LR', 'DECOUPLE', 'DECOUPLE_LR')
               for i, al in enumerate(ALPHAS)}))
        print(f'  {k + 1}/{G} {genes[k]}', flush=True)

    # ---------------- gates ----------------
    print(f"G1 REAL vs instrument: {gate1['n']} values, max rel {gate1['max_rel']:.2e}")
    if gate1['max_rel'] > 1e-9:
        raise SystemExit('G1 FAILED')
    for c_ in my_counts:
        ref = np.array([(inst[c_] < al).sum() for al in ALPHAS])
        print(f'G1 counts {c_}: mine {my_counts[c_].tolist()} instrument {ref.tolist()}')
        if (ref != my_counts[c_]).any():
            raise SystemExit('G1 FAILED (pooled counts)')
    print(f'G3 lead removal signal invariance: max |diff| {lr_g3:.2e}')
    if lr_g3 > 1e-10:
        raise SystemExit('G3 FAILED')
    E = pd.DataFrame(emp)
    old = pd.read_csv(D / 'perm_p_by_scheme_20260924' / 'per_gene.tsv', sep='\t').set_index('gene')
    Eg = E.set_index('gene')
    mp_obs = sps.f.sf(Eg.t2_obs, 1, N - 2 - C.shape[1])
    g4_obs = float(np.max(np.abs(mp_obs - old.loc[Eg.index, 'p_obs']) / old.loc[Eg.index, 'p_obs']))
    d_rec = np.abs(Eg.pemp_rec_raw_30 - old.loc[Eg.index, 'p_emp_records']).max()
    d_flp = np.abs(Eg.pemp_flip_raw_30 - old.loc[Eg.index, 'p_emp_sign_flip']).max()
    print(f'G4 30-draw p vs 2026-09-24: null max rel {g4_max:.2e}, observed max rel {g4_obs:.2e}, '
          f'p_emp max |diff| records {d_rec:.2e} sign flip {d_flp:.2e}')
    n_rec30 = int((Eg.pemp_rec_raw_30 <= 0.05).sum()); n_flp30 = int((Eg.pemp_flip_raw_30 <= 0.05).sum())
    if g4_max > 1e-4 or g4_obs > 1e-4 or d_rec > 1e-12 or d_flp > 1e-12 or (n_rec30, n_flp30) != (11, 7):
        raise SystemExit('G4 FAILED')
    fr = {arm: [float(R[('flip', 'a', arm)][:, 0, i].mean()) for i in range(3)] for arm in ('REAL', 'LR')}
    print(f"G5 first-round sign flip: raw {fr['REAL']} vs {FIRST_ROUND_FLIP['raw']}; "
          f"LR {fr['LR']} vs {FIRST_ROUND_FLIP['lr']}")
    for mine, ref in ((fr['REAL'], FIRST_ROUND_FLIP['raw']), (fr['LR'], FIRST_ROUND_FLIP['lr'])):
        for m_, r_ in zip(mine, ref):
            dec = len(str(r_).split('.')[1])
            if abs(round(m_, dec) - r_) > 1e-12:
                raise SystemExit('G5 FAILED')

    # ---------------- pooled rates, differences, shares ----------------
    summ = dict(investigation='lead-signal-leverage', n_genes=G, n_perm=N_PERM, K_sets=K,
                alphas=list(ALPHAS),
                gates=dict(G1_max_rel=gate1['max_rel'], G1_counts={c: my_counts[c].tolist() for c in my_counts},
                           G3_max_abs=float(lr_g3), G4_null_max_rel=g4_max, G4_obs_max_rel=g4_obs,
                           G4_counts=[n_rec30, n_flp30], G5=fr),
                rates={}, differences={}, shares={})

    def pooled(key):
        x = R[key]                       # (G, K, 3)
        out = {}
        for i, al in enumerate(ALPHAS):
            pg = x[:, :, i].mean(1)
            e = ci(pg, bidx)
            if x.shape[1] > 1:
                e.append(float(x[:, :, i].mean(0).std(ddof=1)))   # MC sd across sets
            out[str(al)] = e
        return out

    for (sch, ch, arm), x in R.items():
        summ['rates'].setdefault(f'{sch}_{ch}', {})[arm] = pooled((sch, ch, arm))

    def diff(k1, k2):
        """k1 - k2 (per-gene, set-averaged); paired MC sd when both simulated."""
        out = {}
        for i, al in enumerate(ALPHAS):
            d = R[k1][:, :, i].mean(1) - R[k2][:, :, i].mean(1)
            e = ci(d, bidx)
            if R[k1].shape[1] > 1 and R[k2].shape[1] > 1:
                e.append(float((R[k1][:, :, i] - R[k2][:, :, i]).mean(0).std(ddof=1)))
            out[str(al)] = e
        return out

    for sch, chs in (('rec', ('a', 't', 'b')), ('flip', ('a',))):
        for ch in chs:
            key = f'{sch}_{ch}'
            dd = {}
            for a1, a2 in (('REAL', 'LR'), ('REAL', 'LR_UNCORR'), ('REAL', 'MODEL'), ('LR', 'MODEL'),
                           ('REAL', 'LR_MIX'), ('MODEL', 'MODEL_LR_MIX'),
                           ('DECOUPLE', 'DECOUPLE_LR_MIX'), ('SIGNRAND', 'SIGNRAND_LR_MIX'),
                           ('MODEL', 'MODEL_LR'), ('MODEL', 'MODEL_LR_UNCORR'),
                           ('DECOUPLE', 'DECOUPLE_LR'), ('SIGNRAND', 'SIGNRAND_LR'),
                           ('REAL', 'SIGNRAND'), ('LR', 'DECOUPLE_LR'), ('LR', 'SIGNRAND_LR')):
                dd[f'{a1}-{a2}'] = diff((sch, ch, a1), (sch, ch, a2))
            summ['differences'][key] = dd
            sh = {}
            for i, al in enumerate(ALPHAS):
                real = R[(sch, ch, 'REAL')][:, 0, i]
                lr = R[(sch, ch, 'LR')][:, 0, i]
                lru = R[(sch, ch, 'LR_UNCORR')][:, 0, i]
                mod = R[(sch, ch, 'MODEL')][:, :, i].mean(1)
                lrm = R[(sch, ch, 'LR_MIX')][:, 0, i]
                ent = dict(share_corrected=ratio_ci(real - lr, real - mod, bidx),
                           share_uncorrected=ratio_ci(real - lru, real - mod, bidx),
                           share_mix=ratio_ci(real - lrm, real - mod, bidx))
                for c_ in ('DECOUPLE', 'SIGNRAND', 'MODEL'):
                    art = R[(sch, ch, c_)][:, :, i].mean(1) - R[(sch, ch, c_ + '_LR_MIX')][:, :, i].mean(1)
                    ent[f'net_share_mix_vs_{c_}'] = ratio_ci(real - lrm - art, real - mod, bidx)
                for c_ in ('DECOUPLE', 'SIGNRAND', 'MODEL'):
                    art = R[(sch, ch, c_)][:, :, i].mean(1) - R[(sch, ch, c_ + '_LR')][:, :, i].mean(1)
                    ent[f'net_share_vs_{c_}'] = ratio_ci(real - lr - art, real - mod, bidx)
                sh[str(al)] = ent
            summ['shares'][key] = sh

    # per stratum, allelic records at 0.05
    summ['by_stratum_rec_a_0.05'] = {
        st: {arm: float(R[('rec', 'a', arm)][strata == st, :, 0].mean()) for arm in ('REAL', 'LR', 'MODEL', 'DECOUPLE', 'DECOUPLE_LR')}
        for st in ('HIGH', 'MID', 'LOW')}

    # ---------------- (e) counts ----------------
    ee = {}
    for nd in (30, N_PERM):
        for lab in ('raw', 'lr'):
            r_, f_ = E[f'pemp_rec_{lab}_{nd}'], E[f'pemp_flip_{lab}_{nd}']
            d = f_ - r_
            up, dn = int((d > 0).sum()), int((d < 0).sum())
            ent = dict(n_called_05_records=int((r_ <= 0.05).sum()), n_called_05_sign_flip=int((f_ <= 0.05).sum()),
                       n_called_10_records=int((r_ <= 0.10).sum()), n_called_10_sign_flip=int((f_ <= 0.10).sum()),
                       n_called_01_records=int((r_ <= 0.01).sum()), n_called_01_sign_flip=int((f_ <= 0.01).sum()),
                       median_records=float(r_.median()), median_sign_flip=float(f_.median()),
                       median_diff=float(d.median()), mean_diff=float(d.mean()),
                       n_flip_larger=up, n_flip_smaller=dn,
                       sign_p=float(sps.binomtest(up, up + dn, 0.5).pvalue) if up + dn else 1.0)
            ee[f'{lab}_{nd}'] = ent
        # cross: raw vs lr within scheme
        for sch in ('rec', 'flip'):
            d = E[f'pemp_{sch}_lr_{nd}'] - E[f'pemp_{sch}_raw_{nd}']
            up, dn = int((d > 0).sum()), int((d < 0).sum())
            ee[f'{sch}_lr_minus_raw_{nd}'] = dict(median=float(d.median()), mean=float(d.mean()),
                                                  n_up=up, n_down=dn,
                                                  sign_p=float(sps.binomtest(up, up + dn, 0.5).pvalue) if up + dn else 1.0)
    # gene-clustered bootstrap for the paired call-count gap (2,000 draws)
    for lab in ('raw', 'lr'):
        gap = ((E[f'pemp_rec_{lab}_{N_PERM}'] <= 0.05).astype(int) - (E[f'pemp_flip_{lab}_{N_PERM}'] <= 0.05).astype(int)).values
        b = gap[bidx].sum(1)
        ee[f'gap_calls_05_{lab}_{N_PERM}'] = [int(gap.sum()), float(np.quantile(b, .025)), float(np.quantile(b, .975))]
        dmean = (E[f'pemp_flip_{lab}_{N_PERM}'] - E[f'pemp_rec_{lab}_{N_PERM}']).values
        ee[f'mean_diff_{lab}_{N_PERM}_ci'] = ci(dmean, bidx)
    dd = ((E[f'pemp_flip_raw_{N_PERM}'] - E[f'pemp_rec_raw_{N_PERM}']) -
          (E[f'pemp_flip_lr_{N_PERM}'] - E[f'pemp_rec_lr_{N_PERM}'])).values
    ee[f'gap_explained_by_lr_mean_{N_PERM}_ci'] = ci(dd, bidx)
    summ['empirical_p'] = ee
    E.to_csv(OUT / 'emp_p_per_gene.tsv', sep='\t', index=False)
    pd.DataFrame(per_gene).to_csv(OUT / 'per_gene.tsv', sep='\t', index=False)
    (OUT / 'summary.json').write_text(json.dumps(summ, indent=1))

    # ---------------- print ----------------
    def fmt(e):
        return f'{e[0]:.4f} [{e[1]:.4f},{e[2]:.4f}]' + (f' mc {e[3]:.4f}' if len(e) > 3 else '')
    for key, arms in summ['rates'].items():
        print(f'\n== rates {key}')
        for arm, v in arms.items():
            print(f'  {arm:16s} ' + '  '.join(fmt(v[str(al)]) for al in ALPHAS))
    for key, dd_ in summ['differences'].items():
        print(f'\n== differences {key}')
        for nm, v in dd_.items():
            print(f'  {nm:24s} ' + '  '.join(fmt(v[str(al)]) for al in ALPHAS))
    for key, sh in summ['shares'].items():
        print(f'\n== shares {key}')
        for al in ALPHAS:
            print(f'  {al}: ' + '  '.join(f'{nm} {fmt(v)}' for nm, v in sh[str(al)].items()))
    print('\n== by stratum', json.dumps(summ['by_stratum_rec_a_0.05']))
    print('\n== empirical p')
    for kk, v in ee.items():
        print(f'  {kk}: {v}')

    # ---------------- figures ----------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    show = ['REAL', 'LR', 'LR_UNCORR', 'LR_MIX', 'SIGNRAND', 'SIGNRAND_LR', 'DECOUPLE', 'DECOUPLE_LR', 'MODEL', 'MODEL_LR', 'MODEL_LR_UNCORR', 'MODEL_LR_MIX']
    for ax, key, title in zip(axes, ('rec_a', 'rec_t', 'rec_b', 'flip_a'),
                              ('allelic, records null', 'total, records null',
                               'combined, records null', 'allelic, sign-flip null')):
        v = summ['rates'][key]
        x = np.arange(len(show))
        for i, al in enumerate(ALPHAS):
            est = np.array([v[a_][str(al)][0] for a_ in show]) / al
            lo = np.array([v[a_][str(al)][1] for a_ in show]) / al
            hi = np.array([v[a_][str(al)][2] for a_ in show]) / al
            ax.errorbar(x + (i - 1) * 0.22, est, yerr=[est - lo, hi - est], fmt='o', ms=4,
                        label=f'alpha {al}')
        ax.axhline(1, color='k', lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(show, rotation=60, ha='right', fontsize=8)
        ax.set_title(title, fontsize=10); ax.set_ylabel('rejection rate / alpha')
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / 'fig_rates.png', dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, lab, title in zip(axes, ('raw', 'lr'), ('null from observed records',
                                                    'null from lead-removed corrected records')):
        xr = E[f'pemp_rec_{lab}_{N_PERM}']; xf = E[f'pemp_flip_{lab}_{N_PERM}']
        ax.loglog(xr, xf, 'o', ms=4)
        ax.plot([4e-4, 1], [4e-4, 1], 'k-', lw=0.8)
        ax.axvline(0.05, color='grey', ls=':'); ax.axhline(0.05, color='grey', ls=':')
        ax.set_xlabel('empirical p, records null (2,000 draws)')
        ax.set_ylabel('empirical p, sign-flip null (2,000 draws)')
        e_ = ee[f'{lab}_{N_PERM}']
        ax.set_title(f"{title}\ncalls at 0.05: records {e_['n_called_05_records']}, "
                     f"sign flip {e_['n_called_05_sign_flip']}", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / 'fig_emp_p.png', dpi=130); plt.close(fig)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
