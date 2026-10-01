"""Does the allelic channel's Gibbs variance grow as a donor's allele counts become lopsided, and does that give donors
that show a simulated effect more strongly less weight?

A (real records, real Gibbs draws; no simulation). Per admitted heterozygous donor-gene pair (both haplotypes >= 0.5
reads), with n = pL + pR and p = (pL + 0.5) / (n + 1): dv = across-draw variance of log2((YL + 0.5) / (YR + 0.5)), the
Gibbs variance without the counting term, and v = dv + (1/(pL + 0.5) + 1/(pR + 0.5)) / ln2^2, the shipped allelic
variance (prepare_default_inputs, count_noise=True). Within-gene least squares of log dv (and log v) on log(n + 1) and
log p(1 - p), gene-clustered bootstrap interval; the Poisson delta-method law Var = 1 / (n p (1 - p) ln2^2) predicts
slopes -1 and -1. Also the median of dv * (n + 1) * ln2^2 by bands of n and of |p - 0.5|: 4.0 at an even split under
the law, rising as 1 / (p (1 - p)).

B (simulated effects, beta 0.8 and the no-effect dataset; each thinned record's variance set by the benchmark's
formula). Per gene, Spearman correlation across its admitted heterozygotes between the weight 1 / v and the allelic
ratio in the effect's direction (sign(beta) (xL - xR) a; with no effect, (xL - xR) a); mean over genes with the
gene-clustered interval. C: the same with weights from each record before the effect was simulated.

  [PLASMODE_GENE_SET=stratum30_100] python3 scripts/allelic_weight_imbalance.py   # writes OUT/imbalance_<set>.json
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark' / 'plasmode'))
import common as C          # noqa: E402

G2 = C.module('02_make_datasets')
OUT = C.D / 'allelic_weight_imbalance_20260930'
LN2 = np.log(2.0)
N_BOOT, BOOT_KEY = 2000, 60
N_BANDS = [(1, 30), (30, 100), (100, 300), (300, np.inf)]
P_BANDS = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.25), (0.25, 0.5)]      # |p - 0.5|
MIN_HETS = 5                                                            # hets per gene for a correlation


def gene_moments(df, y):
    """Per gene, X'X and X'y of the within-gene-centred design (log(n + 1), log p(1 - p)) and response: the
    within-gene least-squares slope of any set of genes (with repeats) is solve(sum X'X, sum X'y)."""
    X = df[['log_n', 'log_pq']].to_numpy()
    X = X - df.groupby('gene')[['log_n', 'log_pq']].transform('mean').to_numpy()
    yy = df[y].to_numpy() - df.groupby('gene')[y].transform('mean').to_numpy()
    idx = pd.Index(df.gene.unique()).get_indexer(df.gene)
    G = idx.max() + 1
    XtX = np.zeros((G, 2, 2))
    Xty = np.zeros((G, 2))
    np.add.at(XtX, idx, X[:, :, None] * X[:, None, :])
    np.add.at(Xty, idx, X * yy[:, None])
    return XtX, Xty


def part_a(R):
    pL, pR = R['pL'], R['pR']
    a_d = np.log2((R['YL'] + C.KAPPA) / (R['YR'] + C.KAPPA))
    dv = a_d.var(axis=2, ddof=0)
    q = (1.0 / (pL + C.KAPPA) + 1.0 / (pR + C.KAPPA)) / LN2 ** 2
    n = pL + pR
    p = (pL + C.KAPPA) / (n + 2 * C.KAPPA)
    use = (pL >= C.EXPRESSIBLE_MIN) & (pR >= C.EXPRESSIBLE_MIN) & (dv > C.EPS)
    g, d = np.nonzero(use)
    df = pd.DataFrame(dict(gene=g, n=n[use], p=p[use], dv=dv[use], v=dv[use] + q[use]))
    df['log_n'], df['log_pq'] = np.log(df.n + 2 * C.KAPPA), np.log(df.p * (1 - df.p))
    df['log_dv'], df['log_v'] = np.log(df.dv), np.log(df.v)
    df['dv_rel'] = df.dv * (df.n + 2 * C.KAPPA) * LN2 ** 2
    genes = df.gene.unique()
    pick = np.random.default_rng(np.random.SeedSequence(C.SEED, spawn_key=(BOOT_KEY,))).integers(0, len(genes), (N_BOOT, len(genes)))
    res = {'pairs': len(df), 'genes': len(genes)}
    for y in ('log_dv', 'log_v'):
        XtX, Xty = gene_moments(df, y)
        est = np.linalg.solve(XtX.sum(0), Xty.sum(0))
        b = np.linalg.solve(XtX[pick].sum(1), Xty[pick].sum(1)[..., None])[..., 0]
        res[y] = dict(slope_log_n=float(est[0]), slope_log_pq=float(est[1]),
                      ci_log_n=[float(x) for x in np.quantile(b[:, 0], [.025, .975])],
                      ci_log_pq=[float(x) for x in np.quantile(b[:, 1], [.025, .975])])
    grid = {}
    dist = (df.p - 0.5).abs()
    for lo, hi in N_BANDS:
        row = {}
        for plo, phi in P_BANDS:
            sel = (df.n >= lo) & (df.n < hi) & (dist >= plo) & (dist < phi)
            row[f'{plo}-{phi}'] = dict(pairs=int(sel.sum()), median_dv_rel=float(df.dv_rel[sel].median()) if sel.any() else None,
                                       law=float(np.median(1.0 / (df.p[sel] * (1 - df.p[sel])))) if sel.any() else None)
        grid[f'{lo}-{hi}'] = row
    res['grid'] = grid
    return res


def part_bc(I, R):
    out = {}
    for sc in ('beta0.0', 'beta0.8'):
        rows = []
        for r in range(1 if sc == 'beta0.0' else 3):
            ds = C.load_dataset(C.DATASETS, sc, r)
            M = G2.move_records(R, ds['perm'], ds['swap'])
            va_real = G2.allelic_variance(M['pL'], M['pR'], M['pL'], M['pR'], M['YL'], M['YR'])
            kept = C.allelic_kept(ds['pL'], ds['pR'], ds['Va'])
            j = np.array([I['vdf'].index.get_loc(v) for v in ds['causal_variant']])
            s = I['xL'][j].astype(float) - I['xR'][j]
            for k in range(len(ds['A'])):
                if sc == 'beta0.8' and ds['is_null'][k]:
                    continue
                h = kept[k] & (s[k] != 0) & (va_real[k] > C.EPS)
                if h.sum() < MIN_HETS:
                    continue
                e = s[k][h] * ds['A'][k][h] * (np.sign(ds['beta'][k]) if sc == 'beta0.8' else 1.0)
                rows.append(dict(gene=k, rep=r, n_het=int(h.sum()),
                                 rho=spearmanr(e, 1.0 / ds['Va'][k][h])[0], rho_real=spearmanr(e, 1.0 / va_real[k][h])[0]))
        df = pd.DataFrame(rows).dropna()
        df['rho_minus_real'] = df.rho - df.rho_real   # paired: the same donors, weights after against before the effect
        genes = df.gene.unique()
        rng = np.random.default_rng(np.random.SeedSequence(C.SEED, spawn_key=(BOOT_KEY, 1, int(sc == 'beta0.8'))))
        gm = df.groupby('gene')[['rho', 'rho_real', 'rho_minus_real']].agg(['sum', 'count'])
        bs = {c: [] for c in ('rho', 'rho_real', 'rho_minus_real')}
        for _ in range(N_BOOT):
            pick = rng.choice(genes, len(genes))
            for c in bs:
                bs[c].append(gm.loc[pick, (c, 'sum')].sum() / gm.loc[pick, (c, 'count')].sum())
        out[sc] = {c: dict(mean=float(df[c].mean()), ci=[float(x) for x in np.quantile(bs[c], [.025, .975])],
                           units=len(df)) for c in bs}
    return out


def main():
    I, R, _ = C.load()
    res = dict(gene_set=C.GENE_SET, a=part_a(R), bc=part_bc(I, R))
    OUT.mkdir(exist_ok=True)
    C.write_json(OUT / f'imbalance_{C.GENE_SET}.json', res)
    a, bc = res['a'], res['bc']
    f = lambda d, k: f'{d[k]:+.3f} [{d["ci_" + k[6:]][0]:+.3f}, {d["ci_" + k[6:]][1]:+.3f}]'   # noqa: E731
    print(f'{C.GENE_SET} A: {a["pairs"]:,} admitted heterozygous donor-gene pairs in {a["genes"]} genes')
    for y, name in (('log_dv', 'Gibbs across-draw variance'), ('log_v', 'shipped allelic variance')):
        print(f'   {name}: slope on log p(1-p) {f(a[y], "slope_log_pq")} (law -1); on log n {f(a[y], "slope_log_n")} (law -1)')
    print('   median Gibbs variance x (n+1) x ln2^2 [law 1/(p(1-p))] by n band (rows) and |p - 0.5| band (columns):')
    for nb, row in a['grid'].items():
        print(f'   n {nb:>9s}: ' + '  '.join(f'{pb}: {v["median_dv_rel"]:.2f} [{v["law"]:.2f}] ({v["pairs"]:,})' if v['pairs'] else f'{pb}: -'
                                          for pb, v in row.items()))
    for sc, d in bc.items():
        print(f'   {C.GENE_SET} {sc}: mean within-gene Spearman(weight, ratio in the effect direction) '
              f'{d["rho"]["mean"]:+.3f} [{d["rho"]["ci"][0]:+.3f}, {d["rho"]["ci"][1]:+.3f}]; weights from the record before the '
              f'effect {d["rho_real"]["mean"]:+.3f} [{d["rho_real"]["ci"][0]:+.3f}, {d["rho_real"]["ci"][1]:+.3f}]; paired '
              f'difference {d["rho_minus_real"]["mean"]:+.3f} [{d["rho_minus_real"]["ci"][0]:+.3f}, {d["rho_minus_real"]["ci"][1]:+.3f}] '
              f'({d["rho"]["units"]} gene units)')
    print(f'wrote {OUT / f"imbalance_{C.GENE_SET}.json"}')


if __name__ == '__main__':
    main()
