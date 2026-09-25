"""How much of the ALLELIC nominal-p excess do single dominant records carry,
once the selection control keeps the real heavy-tailed residual marginal?

QUESTION. The first round (verification_checks/verify_dra/net_share.json)
dropped each gene's top record -- the record with the largest share of
sum_j w_j z_j^2, w = 1/va the Gibbs weight and z = sqrt(w) a the whitened
allelic log ratio -- and netted the resulting drop in rejection rate against
the same drop rule applied to GAUSSIAN record sets (z iid N(0,1) at the real
w). It reported net shares of the real-minus-model excess of 15% / 31% / 68%
at 0.05 / 0.01 / 0.001, and CALM2 657_D1 alone 57% at 0.001. A Gaussian
control has no heavy tail, so dropping its top record removes almost nothing;
any record set whose z has a heavy tail loses more when its top record goes,
whatever the pairing of z with w. The Gaussian net therefore credits to the
dominant record's PAIRING what may belong to the marginal's tail alone.

DESIGN. Everything runs on the instrument's permutation stream
(RandomState(42), 2,000 donor-record permutations, one per round applied to
every gene) at the fixed variant, allelic channel only. The allelic record
(a_j, va_j) moves as a unit; s = xL - xR stays with the position. With
num = sum s w a, den = sum s^2 w, S = sum z^2,
t^2 = (n - 1) num^2 / (den S - num^2), referred to F(1, n - 1), n = admitted
records (va > 1e-12, s = 0 included). Dropping a record zeroes its weight and
its z and lowers n by one, exactly as the first round did.

Arms, all on the same 2,000 permutations:
  REAL        the data; REAL-top1, REAL-top3 drop the gene's 1 or 3 records
              with the largest w z^2; REAL-657 drops CALM2's 657_D1 only.
  DECOUPLE    the gene's own signed z shuffled against its own w among the
              admitted records (K_DEC sets): both marginals kept, pairing
              destroyed. The SAME drop rule is applied to each shuffled set
              (its own argmax / top-3 of w z'^2). This is the heavy-tailed
              selection control.
  MODEL       z iid N(0,1) at the real w (K_MODEL sets), same drop rules:
              the first round's Gaussian control, re-run so that the
              statement being replaced is printed next to its replacement.

Held fixed across arms: s, admission, the permutation stream, the drop rule.
What varies: only the z attached to each weight (REAL / shuffled / Gaussian).

Net for drop rule k: (REAL - REAL_k) - (CTRL - CTRL_k). Shares are reported
against two denominators, both stated: the TOTAL allelic excess REAL - MODEL
(the first round's denominator) and the COUPLING excess REAL - DECOUPLE.
The heavy-tail part carried by top records is also decomposed:
(DEC - DEC_k) - (MODEL - MODEL_k), as a share of DEC - MODEL.

CALM2 657_D1 (w rank 2 of 84, z^2 = 215 of sum z^2 = 535, 91% of sum w z^2).
Is CALM2's DECOUPLE tail caused by 657_D1's z^2 being extreme wherever it
lands? CALM2-only arms, K_C sets each unless enumerated:
  DEC, DEC-top1 (rule), DEC-dropZ657 (drop whichever record carries z_657
  after the shuffle), DEC-dropW657 (drop record 657_D1's weight with whatever
  z it received), DEC-excl657 (657_D1's weight and z removed, the rest
  shuffled), DEC-excl657+2nd (also the record with the second-largest z^2),
  DEC-pin657 (657_D1's z kept at its own weight, the rest shuffled),
  SWAP657 (657_D1's z swapped with each of the other 83 records' z in turn,
  everything else in its real pairing; exhaustive, no Monte Carlo),
  MODEL+z657@random / MODEL+z657@own (Gaussian background, z_657 inserted at
  a random or at its own weight), and a LANDING decomposition: for each of
  the 84 weights, z_657 fixed there and the other 83 z shuffled (K_LAND sets
  per weight); the mean over weights equals DEC in expectation.

INTERVALS. Pooled rates at 0.05 / 0.01 / 0.001. Gene-clustered percentile
bootstrap (genes resampled with replacement, N_BOOT resamples, the SAME
resampled genes for every arm, so differences are paired). For share ratios,
resamples whose denominator is <= 0 are dropped and COUNTED. Every simulated
arm also reports the Monte Carlo sd of its pooled rate across record sets and
the Monte Carlo se of its mean; each net states whether it clears both floors.
Everything is repeated with CALM2 excluded (45 genes).

GATES (abort on failure):
  G1  REAL t^2 equals null_long.tsv.gz t2_a for every (gene, perm) to relative
      1e-9 (absolute 1e-11 where t2 < 1e-3); pooled rejection counts equal the
      instrument's p_a counts exactly.
  G2  REAL-top1 pooled rates equal the first round's `realx` exactly (same
      metric a^2/va^2, same argmax), so the drop rule is the one being
      corrected, not a different one.
  G3  CALM2's top record by w z^2 is 657_D1.
  G4  CALM2 LANDING mean over weights agrees with CALM2 DEC within 4 combined
      Monte Carlo se at every alpha.

Master seed 42: the permutation stream is the instrument's RandomState(42);
every simulated draw uses a SeedSequence(42) child stream.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
INST = D / 'nominal_p_null_instrument_20260925'
FIRST = D / 'nominal_p_hypotheses_20260925' / 'verification_checks' / 'verify_dra' / 'net_share.json'
OUT = D / 'dominant_record_share_corrected_20260925'
SCR = OUT / 'scratch'
SEED, EPS = 42, 1e-12
N_PERM, N_BOOT = 2000, 4000
ALPHAS = (0.05, 0.01, 0.001)
K_DEC, K_MODEL = 200, 200
K_C, K_LAND = 2000, 200
CALM2, REC657 = 'CALM2', '657_D1'


# ------------------------------------------------------------------ core
def t2_sets(Z, W, Sg):
    """t^2 for K record sets.

    Z (K, N) whitened residuals, W (K, N) weights; both zero at records not
    admitted or dropped. Sg (P, N): s at the position each record lands in.
    Returns t2 (K, P) and n (K,).
    """
    n = (W > 0).sum(1)
    num = (Z * np.sqrt(W)) @ Sg.T
    den = W @ (Sg ** 2).T
    S = (Z ** 2).sum(1)
    t2 = (n - 1)[:, None] * num ** 2 / (den * S[:, None] - num ** 2)
    return t2, n


def rej_counts(t2, n):
    """(K, 3) rejection counts over the P permutations, F(1, n - 1)."""
    p = sps.f.sf(t2, 1, (n - 1)[:, None])
    return np.stack([(p < al).sum(1) for al in ALPHAS], 1)


def drop_top(Z, W, k):
    """Zero the k records with the largest w z^2 in each set."""
    c = W * Z ** 2
    idx = np.argsort(-c, axis=1)[:, :k]
    Z2, W2 = Z.copy(), W.copy()
    rows = np.arange(Z.shape[0])[:, None]
    Z2[rows, idx] = 0.0
    W2[rows, idx] = 0.0
    return Z2, W2


def run_all_rules(Z, W, Sg):
    """Counts (K, 3) for base, top1, top3."""
    out = {'base': rej_counts(*t2_sets(Z, W, Sg))}
    for k in (1, 3):
        out[f'top{k}'] = rej_counts(*t2_sets(*drop_top(Z, W, k), Sg))
    return out


def q(x, lo=.025, hi=.975):
    return float(np.quantile(x, lo)), float(np.quantile(x, hi))


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    SCR.mkdir(exist_ok=True)
    d = np.load(INST / 'inputs_at_lead.npz', allow_pickle=True)
    genes = [str(g) for g in d['genes']]
    donors = [str(x) for x in d['donors']]
    a, va, s = d['a'].astype(float), d['va'].astype(float), d['s'].astype(float)
    G, N = a.shape
    keep = np.isfinite(a) & np.isfinite(va) & (va > EPS)
    w = np.where(keep, 1.0 / np.where(keep, va, 1.0), 0.0)
    z = np.where(keep, np.sqrt(w) * np.where(keep, a, 0.0), 0.0)
    rng = np.random.RandomState(SEED)
    perms = np.array([rng.permutation(N) for _ in range(N_PERM)])
    inv = np.argsort(perms, axis=1)          # inv[p, j]: position record j lands at
    Sg = [s[k][inv] for k in range(G)]
    ss = np.random.SeedSequence(SEED).spawn(12)
    rng_boot, rng_dec, rng_model, rng_cdec, rng_cmod, rng_land = \
        [np.random.default_rng(c) for c in ss[:6]]
    ci = genes.index(CALM2)
    j657 = donors.index(REC657)
    summary = dict(investigation='dominant-share-heavy-control', n_genes=G,
                   n_donors=N, n_perm=N_PERM, K_DEC=K_DEC, K_MODEL=K_MODEL,
                   K_C=K_C, K_LAND=K_LAND, n_boot=N_BOOT)

    # ============================================================ REAL + G1
    L = pd.read_csv(INST / 'null_long.tsv.gz', sep='\t',
                    usecols=['gene', 'perm', 't2_a', 'p_a'])
    L = L.set_index(['gene', 'perm']).sort_index()
    real = {r: np.zeros((G, 3), int) for r in ('base', 'top1', 'top3', 'drop657')}
    maxrel, maxabs = 0.0, 0.0
    top_rec = []
    for k, g in enumerate(genes):
        Zk, Wk = z[k][None, :], w[k][None, :]
        t2, n = t2_sets(Zk, Wk, Sg[k])
        ref = L.loc[g].t2_a.reindex(range(N_PERM)).values
        if np.isnan(ref).any():
            raise SystemExit(f'G1 FAILED: instrument rows missing for {g}')
        big = np.abs(ref) >= 1e-3
        maxrel = max(maxrel, float(np.max(np.abs(t2[0][big] - ref[big]) / np.abs(ref[big]))))
        if (~big).any():
            maxabs = max(maxabs, float(np.max(np.abs(t2[0][~big] - ref[~big]))))
        rr = run_all_rules(Zk, Wk, Sg[k])
        real['base'][k], real['top1'][k], real['top3'][k] = rr['base'][0], rr['top1'][0], rr['top3'][0]
        c = w[k] * z[k] ** 2
        top_rec.append(dict(gene=g, top1=donors[int(np.argmax(c))],
                            top1_share=float(c.max() / c.sum()),
                            top3_share=float(np.sort(c)[::-1][:3].sum() / c.sum()),
                            n_a=int(keep[k].sum())))
        if k == ci:
            Z2, W2 = Zk.copy(), Wk.copy()
            Z2[0, j657] = 0.0
            W2[0, j657] = 0.0
            real['drop657'][k] = rej_counts(*t2_sets(Z2, W2, Sg[k]))[0]
        else:
            real['drop657'][k] = real['base'][k]
    inst_counts = [int((L.p_a.values < al).sum()) for al in ALPHAS]
    mine = real['base'].sum(0).tolist()
    print(f'G1: max rel {maxrel:.2e}, max abs (t2<1e-3) {maxabs:.2e}, counts {mine} vs {inst_counts}', flush=True)
    if maxrel > 1e-9 or maxabs > 1e-11 or mine != inst_counts:
        raise SystemExit('G1 FAILED: REAL does not reproduce the instrument')
    summary['gate_G1'] = dict(max_rel_t2=maxrel, max_abs_t2_small=maxabs,
                              counts=mine, instrument_counts=inst_counts, passed=True)
    M = G * N_PERM
    first = json.load(open(FIRST))
    fx = [int(round(x * M)) for x in first['realx']]
    print(f'G2: REAL-top1 counts {real["top1"].sum(0).tolist()} vs first round {fx}', flush=True)
    if real['top1'].sum(0).tolist() != fx:
        raise SystemExit('G2 FAILED: top-1 drop rule differs from the first round')
    summary['gate_G2'] = dict(counts=real['top1'].sum(0).tolist(), first_round=fx, passed=True)
    TR = pd.DataFrame(top_rec).set_index('gene')
    if TR.loc[CALM2, 'top1'] != REC657:
        raise SystemExit('G3 FAILED: CALM2 top record is not 657_D1')
    summary['gate_G3'] = dict(calm2_top1=REC657, share=float(TR.loc[CALM2, 'top1_share']), passed=True)

    # ============================================ DECOUPLE and MODEL, 46 genes
    dec = {r: np.zeros((G, K_DEC, 3), int) for r in ('base', 'top1', 'top3')}
    mod = {r: np.zeros((G, K_MODEL, 3), int) for r in ('base', 'top1', 'top3')}
    for k in range(G):
        adm = np.where(keep[k])[0]
        Zd = np.zeros((K_DEC, N))
        for i in range(K_DEC):
            Zd[i, adm] = rng_dec.permutation(z[k][adm])
        Wd = np.broadcast_to(w[k], (K_DEC, N)).copy()
        rr = run_all_rules(Zd, Wd, Sg[k])
        for r in dec:
            dec[r][k] = rr[r]
        Zm = np.zeros((K_MODEL, N))
        Zm[:, adm] = rng_model.standard_normal((K_MODEL, len(adm)))
        Wm = np.broadcast_to(w[k], (K_MODEL, N)).copy()
        rr = run_all_rules(Zm, Wm, Sg[k])
        for r in mod:
            mod[r][k] = rr[r]
    # the CALM2-only tier's control: the rule (top1) applied in CALM2 only
    print(f'arms done {time.time() - t0:.0f}s', flush=True)

    # per-gene rates (G, 3); simulated arms averaged over sets
    R = {f'REAL_{r}': real[r] / N_PERM for r in real}
    for r in dec:
        R[f'DEC_{r}'] = dec[r].mean(1) / N_PERM
        R[f'MODEL_{r}'] = mod[r].mean(1) / N_PERM
    # CALM2-only tier: in non-CALM2 genes the rule changes nothing
    for arm, src in (('DEC', dec), ('MODEL', mod)):
        x = src['base'].mean(1) / N_PERM
        x = x.copy()
        x[ci] = src['top1'][ci].mean(0) / N_PERM
        R[f'{arm}_calm2top1'] = x
    R['REAL_calm2top1'] = R['REAL_drop657']

    boot_all = rng_boot.integers(0, G, size=(N_BOOT, G))
    idx45 = np.array([k for k in range(G) if k != ci])
    boot_45 = idx45[rng_boot.integers(0, len(idx45), size=(N_BOOT, len(idx45)))]

    def mc_sets(src, rule, gidx):
        """Pooled rate per record set over the gene subset: (K, 3)."""
        return src[rule][gidx].sum(0) / (len(gidx) * N_PERM)

    def mc_calm2_rule(src, gidx):
        """Per set pooled (base - rule-in-CALM2-only)."""
        if ci not in gidx:
            return np.zeros((src['base'].shape[1], 3))
        return (src['base'][ci] - src['top1'][ci]) / (len(gidx) * N_PERM)

    def tier_block(tier, gidx, bidx):
        rk = {'top1': 'top1', 'top3': 'top3', 'CALM2_657_only': 'calm2top1'}[tier]
        realx, decx, modx = R[f'REAL_{rk}'], R[f'DEC_{rk}'], R[f'MODEL_{rk}']
        rb, db, mb = R['REAL_base'], R['DEC_base'], R['MODEL_base']

        def stats(ix):
            m = lambda x: x[ix].mean(0) if ix.ndim == 1 else x[ix].mean(1)  # noqa: E731
            drop_real = m(rb) - m(realx)
            drop_dec = m(db) - m(decx)
            drop_mod = m(mb) - m(modx)
            e_tot = m(rb) - m(mb)
            e_coup = m(rb) - m(db)
            e_heavy = m(db) - m(mb)
            net_h = drop_real - drop_dec
            net_g = drop_real - drop_mod
            tail_part = drop_dec - drop_mod
            return dict(drop_real=drop_real, drop_dec=drop_dec, drop_model=drop_mod,
                        excess_total=e_tot, excess_coupling=e_coup, excess_heavy=e_heavy,
                        net_heavy=net_h, net_gauss=net_g, tail_part=tail_part)
        est = stats(gidx)
        bs = stats(bidx)
        # Monte Carlo: per set pooled drop in the control arms
        if tier == 'CALM2_657_only':
            dd = mc_calm2_rule(dec, gidx)
            dm = mc_calm2_rule(mod, gidx)
        else:
            dd = mc_sets(dec, 'base', gidx) - mc_sets(dec, rk, gidx)
            dm = mc_sets(mod, 'base', gidx) - mc_sets(mod, rk, gidx)
        dec_base_sets = mc_sets(dec, 'base', gidx)
        mod_base_sets = mc_sets(mod, 'base', gidx)
        out = {}
        for ai, al in enumerate(ALPHAS):
            e = {kk: float(v[ai]) for kk, v in est.items()}
            b = {kk: v[:, ai] for kk, v in bs.items()}
            mc_se_net_h = float(dd[:, ai].std(ddof=1) / np.sqrt(len(dd)))
            mc_se_net_g = float(dm[:, ai].std(ddof=1) / np.sqrt(len(dm)))
            mc_se_etot = float(mod_base_sets[:, ai].std(ddof=1) / np.sqrt(K_MODEL))
            mc_se_ecoup = float(dec_base_sets[:, ai].std(ddof=1) / np.sqrt(K_DEC))

            def share(num, den, bnum, bden):
                ok = bden > 0
                r = bnum[ok] / bden[ok]
                lo, hi = q(r) if ok.sum() > 10 else (np.nan, np.nan)
                return dict(share=num / den if den > 0 else np.nan, lo=lo, hi=hi,
                            n_boot_dropped_den_le0=int((~ok).sum()))
            nh_lo, nh_hi = q(b['net_heavy'])
            ng_lo, ng_hi = q(b['net_gauss'])
            tp_lo, tp_hi = q(b['tail_part'])
            out[str(al)] = dict(
                real_rate=float(rb[gidx].mean(0)[ai]),
                real_dropped_rate=float(realx[gidx].mean(0)[ai]),
                dec_rate=float(db[gidx].mean(0)[ai]),
                dec_dropped_rate=float(decx[gidx].mean(0)[ai]),
                model_rate=float(mb[gidx].mean(0)[ai]),
                model_dropped_rate=float(modx[gidx].mean(0)[ai]),
                drop_real=e['drop_real'], drop_real_ci=q(b['drop_real']),
                drop_dec=e['drop_dec'], drop_dec_mc_sd_sets=float(dd[:, ai].std(ddof=1)),
                drop_model=e['drop_model'], drop_model_mc_sd_sets=float(dm[:, ai].std(ddof=1)),
                excess_total=e['excess_total'], excess_total_ci=q(b['excess_total']),
                excess_total_mc_se=mc_se_etot,
                excess_coupling=e['excess_coupling'], excess_coupling_ci=q(b['excess_coupling']),
                excess_coupling_mc_se=mc_se_ecoup,
                excess_heavy=e['excess_heavy'], excess_heavy_ci=q(b['excess_heavy']),
                net_heavy=e['net_heavy'], net_heavy_ci=(nh_lo, nh_hi),
                net_heavy_mc_se=mc_se_net_h,
                net_heavy_clears_zero=bool(nh_lo > 0 and e['net_heavy'] > 2 * mc_se_net_h),
                net_gauss=e['net_gauss'], net_gauss_ci=(ng_lo, ng_hi),
                net_gauss_mc_se=mc_se_net_g,
                tail_part=e['tail_part'], tail_part_ci=(tp_lo, tp_hi),
                share_heavy_of_total=share(e['net_heavy'], e['excess_total'],
                                           b['net_heavy'], b['excess_total']),
                share_heavy_of_coupling=share(e['net_heavy'], e['excess_coupling'],
                                              b['net_heavy'], b['excess_coupling']),
                share_gauss_of_total=share(e['net_gauss'], e['excess_total'],
                                           b['net_gauss'], b['excess_total']),
                share_tailpart_of_heavy=share(e['tail_part'], e['excess_heavy'],
                                              b['tail_part'], b['excess_heavy']))
        return out

    shares = {}
    rows = []
    for gs_lab, gidx, bidx in (('all46', np.arange(G), boot_all), ('noCALM2_45', idx45, boot_45)):
        shares[gs_lab] = {}
        for tier in ('top1', 'top3', 'CALM2_657_only'):
            if gs_lab == 'noCALM2_45' and tier == 'CALM2_657_only':
                continue
            blk = tier_block(tier, gidx, bidx)
            shares[gs_lab][tier] = blk
            for al, v in blk.items():
                rows.append(dict(gene_set=gs_lab, tier=tier, alpha=al,
                                 real=v['real_rate'], real_dropped=v['real_dropped_rate'],
                                 dec=v['dec_rate'], dec_dropped=v['dec_dropped_rate'],
                                 model=v['model_rate'], model_dropped=v['model_dropped_rate'],
                                 net_heavy=v['net_heavy'], net_heavy_lo=v['net_heavy_ci'][0],
                                 net_heavy_hi=v['net_heavy_ci'][1], net_heavy_mc_se=v['net_heavy_mc_se'],
                                 share_heavy_total=v['share_heavy_of_total']['share'],
                                 share_heavy_total_lo=v['share_heavy_of_total']['lo'],
                                 share_heavy_total_hi=v['share_heavy_of_total']['hi'],
                                 share_heavy_coupling=v['share_heavy_of_coupling']['share'],
                                 share_heavy_coupling_lo=v['share_heavy_of_coupling']['lo'],
                                 share_heavy_coupling_hi=v['share_heavy_of_coupling']['hi'],
                                 share_gauss_total=v['share_gauss_of_total']['share'],
                                 share_gauss_total_lo=v['share_gauss_of_total']['lo'],
                                 share_gauss_total_hi=v['share_gauss_of_total']['hi'],
                                 share_tailpart_heavy=v['share_tailpart_of_heavy']['share']))
    summary['net_shares'] = shares
    NS = pd.DataFrame(rows)
    NS.to_csv(OUT / 'net_shares.tsv', sep='\t', index=False)
    pd.set_option('display.width', 250)
    print(NS.round(5).to_string(), flush=True)

    # arm pooled rates with bootstrap and MC sd
    arm_rows = []
    for gs_lab, gidx, bidx in (('all46', np.arange(G), boot_all), ('noCALM2_45', idx45, boot_45)):
        for arm in ('REAL', 'DEC', 'MODEL'):
            for rule in ('base', 'top1', 'top3'):
                x = R[f'{arm}_{rule}']
                src = dec if arm == 'DEC' else mod if arm == 'MODEL' else None
                for ai, al in enumerate(ALPHAS):
                    lo, hi = q(x[bidx].mean(1)[:, ai])
                    mcsd = float(mc_sets(src, rule, gidx)[:, ai].std(ddof=1)) if src else 0.0
                    arm_rows.append(dict(gene_set=gs_lab, arm=arm, rule=rule, alpha=al,
                                         rate=float(x[gidx].mean(0)[ai]), lo=lo, hi=hi,
                                         mc_sd_across_sets=mcsd))
    AR = pd.DataFrame(arm_rows)
    AR.to_csv(OUT / 'arm_rates.tsv', sep='\t', index=False)
    summary['arm_rates'] = {f'{r.gene_set}|{r.arm}|{r.rule}|{r.alpha}':
                            dict(rate=r.rate, lo=r.lo, hi=r.hi, mc_sd=r.mc_sd_across_sets)
                            for r in AR.itertuples()}

    # per gene: who carries the net
    pg = TR.copy()
    for ai, al in enumerate(ALPHAS):
        for key in ('REAL_base', 'REAL_top1', 'DEC_base', 'DEC_top1', 'MODEL_base', 'MODEL_top1'):
            pg[f'{key}_{al}'] = R[key][:, ai]
        pg[f'net_heavy_top1_{al}'] = ((R['REAL_base'] - R['REAL_top1']) - (R['DEC_base'] - R['DEC_top1']))[:, ai]
        pg[f'net_gauss_top1_{al}'] = ((R['REAL_base'] - R['REAL_top1']) - (R['MODEL_base'] - R['MODEL_top1']))[:, ai]
        # per-gene MC floor of the DEC drop (sd across sets / sqrt K)
        pg[f'dec_drop_mc_se_{al}'] = ((dec['base'][:, :, ai] - dec['top1'][:, :, ai]) / N_PERM).std(1, ddof=1) / np.sqrt(K_DEC)
    pg.to_csv(OUT / 'per_gene_drop.tsv', sep='\t')
    for al in ALPHAS:
        col = f'net_heavy_top1_{al}'
        tot = pg[col].sum()
        srt = pg[col].sort_values(ascending=False)
        summary.setdefault('per_gene_net_top1', {})[str(al)] = dict(
            sum=float(tot), top5=[(gg, float(v)) for gg, v in srt.head(5).items()],
            bottom3=[(gg, float(v)) for gg, v in srt.tail(3).items()],
            calm2_fraction_of_sum=float(pg.loc[CALM2, col] / tot) if tot != 0 else None)

    # ============ where does each gene's largest z^2 sit in its weight order?
    # Under DECOUPLE the weight rank of the record carrying the largest z^2 is
    # uniform on 1..n_a (the argmax is chosen by z alone), so this asks whether
    # CALM2's pairing of its extreme z with a top-2 weight is itself unusual
    # across the 46 genes.
    rng_rank = np.random.default_rng(ss[6])
    rk_rows = []
    for kk, g in enumerate(genes):
        ad = np.where(keep[kk])[0]
        wr = np.empty(len(ad), int)
        wr[np.argsort(-w[kk][ad])] = np.arange(1, len(ad) + 1)
        oz = np.argsort(-(z[kk][ad] ** 2))
        rk_rows.append(dict(gene=g, n_a=len(ad), maxz2_record=donors[ad[oz[0]]],
                            maxz2=float(z[kk][ad[oz[0]]] ** 2),
                            maxz2_over_sum=float(z[kk][ad[oz[0]]] ** 2 / (z[kk][ad] ** 2).sum()),
                            w_rank_of_maxz2=int(wr[oz[0]]),
                            u_maxz2=float((wr[oz[0]] - 0.5) / len(ad)),
                            u_top3z2_mean=float(((wr[oz[:3]] - 0.5) / len(ad)).mean())))
    RK = pd.DataFrame(rk_rows).set_index('gene')
    RK.to_csv(OUT / 'maxz2_weight_rank.tsv', sep='\t')
    nas = RK.n_a.values
    # exact Poisson-binomial for the count of genes with rank <= 2
    pr2 = 2.0 / nas
    dist = np.zeros(G + 1)
    dist[0] = 1.0
    for pp in pr2:
        dist[1:] = dist[1:] * (1 - pp) + dist[:-1] * pp
        dist[0] *= (1 - pp)
    obs2 = int((RK.w_rank_of_maxz2 <= 2).sum())
    # MC null for the mean normalized rank (lower = extreme z on high weight)
    NM = 100000
    sim_u = np.zeros(NM)
    sim_u3 = np.zeros(NM)
    for kk in range(G):
        n_ = nas[kk]
        r1 = rng_rank.integers(1, n_ + 1, NM)
        sim_u += (r1 - 0.5) / n_
        r3 = np.argsort(rng_rank.random((NM, n_)), 1)[:, :3] + 1
        sim_u3 += ((r3 - 0.5) / n_).mean(1)
    sim_u /= G
    sim_u3 /= G
    summary['maxz2_weight_rank'] = dict(
        calm2_rank=int(RK.loc[CALM2, 'w_rank_of_maxz2']), calm2_n_a=int(RK.loc[CALM2, 'n_a']),
        calm2_p_rank_le2_under_decouple=float(2.0 / RK.loc[CALM2, 'n_a']),
        n_genes_rank_le2=obs2, expected_rank_le2=float(pr2.sum()),
        p_ge_obs_rank_le2=float(dist[obs2:].sum()),
        genes_rank_le2=list(RK.index[RK.w_rank_of_maxz2 <= 2]),
        mean_u_maxz2=float(RK.u_maxz2.mean()), mean_u_null_q=q(sim_u),
        p_mean_u_le_obs=float((sim_u <= RK.u_maxz2.mean()).mean()),
        mean_u_top3z2=float(RK.u_top3z2_mean.mean()), mean_u_top3_null_q=q(sim_u3),
        p_mean_u_top3_le_obs=float((sim_u3 <= RK.u_top3z2_mean.mean()).mean()),
        mean_u_maxz2_noCALM2=float(RK.u_maxz2.drop(CALM2).mean()),
        note='u = (weight rank - 0.5)/n_a of the record carrying the largest z^2; '
             'uniform under DECOUPLE; small u = extreme residual on a high weight')
    print(json.dumps(summary['maxz2_weight_rank'], indent=1), flush=True)

    # ============================================================ CALM2 arms
    k = ci
    adm = np.where(keep[k])[0]
    na = len(adm)
    zc, wc, Sgc = z[k], w[k], Sg[k]
    pos657 = int(np.where(adm == j657)[0][0])
    z2 = zc[adm] ** 2
    order_z2 = np.argsort(-z2)
    j2nd = int(adm[order_z2[1]])
    wrank = {int(adm[i]): int(r) + 1 for r, i in enumerate(np.argsort(-wc[adm]))}
    calm2_info = dict(n_a=na, z657=float(zc[j657]), z2_657=float(zc[j657] ** 2),
                      sum_z2=float(z2.sum()), w657_rank_desc=wrank[j657],
                      share_wz2_657=float((wc * zc ** 2)[j657] / (wc * zc ** 2).sum()),
                      second_z2_record=donors[j2nd], second_z2=float(zc[j2nd] ** 2),
                      second_w_rank_desc=wrank[j2nd], s657_at_lead=float(s[k][j657]),
                      frac_positions_het=float((s[k] != 0).mean()),
                      w_top5_share_of_sum=float(np.sort(wc[adm])[::-1][:5].sum() / wc[adm].sum()),
                      w657_share_of_sum=float(wc[j657] / wc[adm].sum()))
    C = {}

    def add(lab, cnt, kind):
        cnt = np.atleast_2d(cnt)
        rate = cnt / N_PERM
        C[lab] = dict(kind=kind, K=int(cnt.shape[0]),
                      **{str(al): dict(rate=float(rate[:, ai].mean()),
                                       mc_sd_sets=float(rate[:, ai].std(ddof=1)) if cnt.shape[0] > 1 else None,
                                       mc_se_mean=float(rate[:, ai].std(ddof=1) / np.sqrt(cnt.shape[0]))
                                       if cnt.shape[0] > 1 else
                                       float(np.sqrt(rate[0, ai] * (1 - rate[0, ai]) / N_PERM)))
                         for ai, al in enumerate(ALPHAS)})

    add('REAL', real['base'][k], 'data (mc_se = binomial over the 2,000 permutations)')
    # REAL, split by whether 657_D1 lands at a heterozygous position
    t2r, nr = t2_sets(zc[None, :], wc[None, :], Sgc)
    pr = sps.f.sf(t2r[0], 1, nr[0] - 1)
    het = Sgc[:, j657] != 0
    calm2_real_split = dict(
        n_perms_657_at_het=int(het.sum()), frac=float(het.mean()),
        **{str(al): dict(rate_657_at_het=float((pr[het] < al).mean()),
                         rate_657_at_hom=float((pr[~het] < al).mean()),
                         frac_of_rejections_with_657_at_het=float(
                             (pr[het] < al).sum() / max((pr < al).sum(), 1)))
           for al in ALPHAS})
    add('REAL-drop657', real['drop657'][k], 'data')
    # DEC with record bookkeeping
    Zd = np.zeros((K_C, N))
    carrier = np.zeros(K_C, int)              # record that carries z_657 after the shuffle
    carrier2 = np.zeros(K_C, int)
    for i in range(K_C):
        pi = rng_cdec.permutation(na)
        Zd[i, adm] = zc[adm][pi]
        carrier[i] = adm[int(np.where(pi == pos657)[0][0])]
        carrier2[i] = adm[int(np.where(pi == order_z2[1])[0][0])]
    Wd = np.broadcast_to(wc, (K_C, N)).copy()
    rr = run_all_rules(Zd, Wd, Sgc)
    add('DEC', rr['base'], 'z shuffled against w')
    add('DEC-top1', rr['top1'], 'same drop rule on each shuffled set')
    rows_ = np.arange(K_C)
    Z2, W2 = Zd.copy(), Wd.copy()
    Z2[rows_, carrier] = 0.0
    W2[rows_, carrier] = 0.0
    add('DEC-dropZ657', rej_counts(*t2_sets(Z2, W2, Sgc)), 'record carrying z_657 dropped')
    Z2, W2 = Zd.copy(), Wd.copy()
    Z2[:, j657] = 0.0
    W2[:, j657] = 0.0
    add('DEC-dropW657', rej_counts(*t2_sets(Z2, W2, Sgc)), "657_D1's weight dropped with its shuffled z")
    top1_is_z657 = float((np.argmax(Wd * Zd ** 2, 1) == carrier).mean())
    # DEC excluding 657_D1 (weight and z), rest shuffled; and also the 2nd-largest z^2 record
    for lab, excl in (('DEC-excl657', [j657]), ('DEC-excl657+2nd', [j657, j2nd])):
        rest = np.array([j for j in adm if j not in excl])
        Ze = np.zeros((K_C, N))
        for i in range(K_C):
            Ze[i, rest] = rng_cdec.permutation(zc[rest])
        We = np.broadcast_to(np.where(np.isin(np.arange(N), rest), wc, 0.0), (K_C, N)).copy()
        add(lab, rej_counts(*t2_sets(Ze, We, Sgc)), f'records {excl} removed, rest shuffled')
    # DEC with 657 pinned at its own weight
    rest = np.array([j for j in adm if j != j657])
    Zp = np.zeros((K_C, N))
    Zp[:, j657] = zc[j657]
    for i in range(K_C):
        Zp[i, rest] = rng_cdec.permutation(zc[rest])
    add('DEC-pin657', rej_counts(*t2_sets(Zp, Wd, Sgc)), "z_657 at its own weight, others shuffled")
    # SWAP657 exhaustive
    Zs = np.broadcast_to(zc, (na - 1, N)).copy()
    partners = rest
    for i, j in enumerate(partners):
        Zs[i, j657], Zs[i, j] = zc[j], zc[j657]
    cs = rej_counts(*t2_sets(Zs, np.broadcast_to(wc, (na - 1, N)).copy(), Sgc))
    add('SWAP657', cs, "z_657 swapped with each other record's z (exhaustive, 83)")
    swap_by_partner = pd.DataFrame(dict(partner=[donors[j] for j in partners],
                                        w_rank_desc=[wrank[int(j)] for j in partners],
                                        partner_z2=zc[partners] ** 2,
                                        **{f'rate_{al}': cs[:, ai] / N_PERM for ai, al in enumerate(ALPHAS)}))
    swap_by_partner.sort_values('w_rank_desc').to_csv(OUT / 'calm2_swap657_by_partner.tsv', sep='\t', index=False)
    # MODEL background with z_657 inserted
    Zm = np.zeros((K_C, N))
    Zm[:, adm] = rng_cmod.standard_normal((K_C, na))
    add('MODEL', rej_counts(*t2_sets(Zm, Wd, Sgc)), 'z iid N(0,1)')
    Zo = Zm.copy()
    Zo[:, j657] = zc[j657]
    add('MODEL+z657@own', rej_counts(*t2_sets(Zo, Wd, Sgc)), 'Gaussian background, z_657 at its own weight')
    Zr = Zm.copy()
    land = adm[rng_cmod.integers(0, na, K_C)]
    Zr[rows_, land] = zc[j657]
    add('MODEL+z657@random', rej_counts(*t2_sets(Zr, Wd, Sgc)), 'Gaussian background, z_657 at a random weight')
    # LANDING decomposition: z_657 fixed at each weight, others shuffled
    land_rows = []
    land_cnt = np.zeros((na, K_LAND, 3))
    others_z = zc[rest]
    for li, j in enumerate(adm):
        tgt = np.array([x for x in adm if x != j])
        Zl = np.zeros((K_LAND, N))
        Zl[:, j] = zc[j657]
        for i in range(K_LAND):
            Zl[i, tgt] = rng_land.permutation(others_z)
        cnt = rej_counts(*t2_sets(Zl, np.broadcast_to(wc, (K_LAND, N)).copy(), Sgc))
        land_cnt[li] = cnt
        land_rows.append(dict(record=donors[j], w=float(wc[j]), w_rank_desc=wrank[int(j)],
                              w_share=float(wc[j] / wc[adm].sum()),
                              frac_perms_het=float((Sgc[:, j] != 0).mean()),
                              **{f'rate_{al}': float(cnt[:, ai].mean() / N_PERM) for ai, al in enumerate(ALPHAS)}))
    LD = pd.DataFrame(land_rows).sort_values('w_rank_desc')
    LD.to_csv(OUT / 'calm2_landing.tsv', sep='\t', index=False)
    land_mean = land_cnt.mean(1) / N_PERM          # (na, 3)
    g4 = {}
    for ai, al in enumerate(ALPHAS):
        lm = float(land_mean[:, ai].mean())
        # MC se of the landing mean: across-set sd within weight / sqrt(K_LAND), combined over weights
        se_l = float(np.sqrt(((land_cnt[:, :, ai] / N_PERM).var(1, ddof=1) / K_LAND).sum()) / na)
        se_d = C['DEC'][str(al)]['mc_se_mean']
        zz = (lm - C['DEC'][str(al)]['rate']) / np.sqrt(se_l ** 2 + se_d ** 2)
        g4[str(al)] = dict(landing_mean=lm, dec=C['DEC'][str(al)]['rate'], z=float(zz))
        ordered = LD[f'rate_{al}'].values           # sorted by weight rank
        g4[str(al)]['share_of_landing_tail_from_top5_weights'] = float(ordered[:5].sum() / ordered.sum())
        g4[str(al)]['share_of_landing_tail_from_top10_weights'] = float(ordered[:10].sum() / ordered.sum())
        g4[str(al)]['rate_excluding_top5_weight_landings'] = float(ordered[5:].mean())
        g4[str(al)]['rate_when_landing_on_top5_weights'] = float(ordered[:5].mean())
    print('G4 landing vs DEC', json.dumps(g4, indent=1), flush=True)
    if any(abs(v['z']) > 4 for v in g4.values()):
        raise SystemExit('G4 FAILED: landing decomposition does not average to DEC')
    summary['gate_G4'] = dict(**g4, passed=True)

    # survival fractions against CALM2 MODEL
    surv = {}
    for ai, al in enumerate(ALPHAS):
        al_ = str(al)
        m0 = C['MODEL'][al_]['rate']
        base_d = C['DEC'][al_]['rate'] - m0
        base_r = C['REAL'][al_]['rate'] - m0
        surv[al_] = dict(
            model=m0,
            dec_excess=base_d, real_excess=base_r,
            **{f'{lab}_excess_over_model': C[lab][al_]['rate'] - m0
               for lab in C if lab not in ('MODEL',)},
            swap657_survival_of_real_excess=(C['SWAP657'][al_]['rate'] - m0) / base_r if base_r > 0 else None,
            excl657_survival_of_dec_excess=(C['DEC-excl657'][al_]['rate'] - m0) / base_d if base_d > 0 else None,
            excl657_2nd_survival_of_dec_excess=(C['DEC-excl657+2nd'][al_]['rate'] - m0) / base_d if base_d > 0 else None,
            model_plus_z657_random_share_of_dec_excess=(C['MODEL+z657@random'][al_]['rate'] - m0) / base_d if base_d > 0 else None)
    summary['calm2'] = dict(info=calm2_info, arms=C, survival=surv, real_split_by_landing=calm2_real_split,
                            dec_top1_is_z657_carrier_frac=top1_is_z657)
    CT = pd.DataFrame([dict(arm=lab, kind=v['kind'], K=v['K'],
                            **{f'rate_{al}': v[str(al)]['rate'] for al in ALPHAS},
                            **{f'mcse_{al}': v[str(al)]['mc_se_mean'] for al in ALPHAS})
                       for lab, v in C.items()])
    CT.to_csv(OUT / 'calm2_arms.tsv', sep='\t', index=False)
    print(json.dumps(calm2_info, indent=1))
    print(CT.round(5).to_string(), flush=True)

    # ============================================================ figures
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ai, al in enumerate(ALPHAS):
        ax = axes[ai]
        labs, vals, los, his, cols = [], [], [], [], []
        for gs_lab, tiers in (('all46', ('top1', 'top3', 'CALM2_657_only')), ('noCALM2_45', ('top1', 'top3'))):
            for tier in tiers:
                v = shares[gs_lab][tier][str(al)]
                for kind, key, colr in (('Gaussian ctrl', 'share_gauss_of_total', '#9aa5b1'),
                                        ('heavy-tailed ctrl', 'share_heavy_of_total', '#2b6cb0')):
                    labs.append(f'{tier}\n{gs_lab}\n{kind}')
                    vals.append(v[key]['share'])
                    los.append(v[key]['lo'])
                    his.append(v[key]['hi'])
                    cols.append(colr)
        y = np.arange(len(labs))
        vals, los, his = np.array(vals), np.array(los), np.array(his)
        ax.barh(y, vals, color=cols)
        ax.errorbar(vals, y, xerr=[np.clip(vals - los, 0, None), np.clip(his - vals, 0, None)],
                    fmt='none', ecolor='k', lw=0.8)
        ax.axvline(0, color='k', lw=0.6)
        ax.set_title(f'nominal {al}')
        ax.set_xlabel('net share of total allelic excess (REAL - MODEL)')
        if ai == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(labs, fontsize=6)
        ax.set_xlim(-1.0, 1.2)
    fig.suptitle('Dominant-record net share: Gaussian vs heavy-tailed selection control '
                 '(gene-clustered 95% intervals)', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_net_share_controls.png', dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ai, al in ((1, 0.01), (2, 0.001)):
        ax = axes[ai - 1]
        ax.plot(LD.w_rank_desc, LD[f'rate_{al}'], 'o', ms=3, color='#2b6cb0',
                label='z_657 placed at this weight, others shuffled')
        ax.axhline(C['DEC'][str(al)]['rate'], color='#2b6cb0', ls='--', lw=0.8, label='DEC (all shuffled)')
        ax.axhline(C['DEC-excl657'][str(al)]['rate'], color='#c05621', ls='--', lw=0.8, label='DEC without 657_D1')
        ax.axhline(C['MODEL'][str(al)]['rate'], color='k', ls=':', lw=0.8, label='MODEL')
        ax.axhline(C['REAL'][str(al)]['rate'], color='#2f855a', lw=0.8, label='REAL')
        ax.axvline(calm2_info['w657_rank_desc'], color='grey', lw=0.5)
        ax.set_yscale('log')
        ax.set_ylim(5e-4, 0.3)
        ax.set_xlabel('weight rank of the landing record (1 = largest w)')
        ax.set_ylabel(f'CALM2 allelic rejection rate at {al}')
        ax.legend(fontsize=6)
    fig.suptitle('CALM2: where z_657 lands decides the tail', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_calm2_landing.png', dpi=130)
    plt.close(fig)

    summary['first_round_record'] = first
    summary['runtime_s'] = time.time() - t0
    json.dump(summary, open(OUT / 'summary.json', 'w'), indent=1, default=float)
    print(f'done {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
