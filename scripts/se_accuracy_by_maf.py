"""Standard-error accuracy by minor-allele-frequency bucket, all variants.

The per-gene ratio measured earlier used ONE variant per gene -- that gene's
RASQUAL observed lead -- so it said nothing about how the standard error behaves
at variants with different allele frequencies, and the variants it did use were
selected for carrying signal. This sweeps every tested variant in every gene's
window and buckets by MAF.

Per (gene, variant), over the null draws:

    ratio = mean of the reported slope_se  /  sd of the reported slope

Under the null the true slope is zero, so the denominator is the estimator's
realized error. hapmixQTL reports slope_se directly from map_nominal, so nothing
is derived through a Wald identity here; mixQTL reports beta and se from
mixqtl_scan; RASQUAL's per-variant rows were retained for every null draw, so
that arm costs no new compute -- its se comes from |beta| / sqrt(chi2), the Wald
approximation to its likelihood ratio, which is the only option it offers.

Accumulators (count, sum, sum of squares) are kept per variant rather than the
raw values, so memory does not scale with the number of draws.
"""
import contextlib
import io
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import compare_mixqtl_replication as CM        # noqa: E402
import tensorqtl.mixqtl_replication as MX      # noqa: E402
from tensorqtl.hapmixqtl import map_nominal    # noqa: E402

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
RUN = D / 'rasqual_default_mode_20260923'
SEED = 42
LN2 = np.log(2.0)
BUCKETS = [(0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.50)]


class Acc:
    """Streaming count / sum / sumsq for beta and se, keyed by (gene, variant)."""

    def __init__(self):
        self.d = defaultdict(lambda: np.zeros(5))   # n, sum_b, sumsq_b, sum_se, sum_se2

    def add(self, key, b, se):
        a = self.d[key]
        a[0] += 1; a[1] += b; a[2] += b * b; a[3] += se; a[4] += se * se

    def frame(self, arm, min_n=10):
        rows = []
        for (g, v), a in self.d.items():
            n = a[0]
            if n < min_n:
                continue
            var_b = (a[2] - a[1] ** 2 / n) / (n - 1)
            if var_b <= 0:
                continue
            rows.append(dict(arm=arm, gene=g, variant=v, n=int(n),
                             sd_beta=float(np.sqrt(var_b)),
                             mean_se=float(a[3] / n),
                             rms_se=float(np.sqrt(a[4] / n))))
        return pd.DataFrame(rows)


def rasqual_rows(gene, draw, want):
    """(variant_id, log_afc, chi2) for wanted variants of one gene and draw."""
    f = RUN / 'rasqual_rows' / f'null_{draw:03d}' / f'{gene}.tsv'
    if not f.exists():
        return
    for ln in f.read_text().split('\n'):
        x = ln.split('\t')
        if len(x) < 25:
            continue
        vid = f'{x[2]}_{x[3]}_{x[4]}_{x[5]}'
        if vid not in want:
            continue
        try:
            if int(float(x[22])) != 0:
                continue
            chi2 = float(x[10])
            pi = min(max(float(x[11]), 1e-6), 1 - 1e-6)
        except (ValueError, IndexError):
            continue
        if chi2 > 0:
            yield vid, np.log(pi / (1 - pi)), chi2


def main():
    n_draw = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    null46 = [l.strip() for l in open(D / 'genes_null46_20260923.txt') if l.strip()]
    I = CM.load_inputs(gene_list=str(D / 'genes_59_stratified_20260923.txt'),
                       regions=str(RUN / 'regions.bed'))
    order, keep = I['order'], I['keep']
    N = len(order)
    import run_hapmixqtl_from_salmon as H
    with contextlib.redirect_stdout(io.StringIO()):
        A, T, Va, Vt, _ = H.compute_summaries_from_gibbs(
            I['YL'][:, keep], I['YR'][:, keep], yT=I['YT'][:, keep])
    Y1, Y2, YT = MX.summaries_from_gibbs_posterior_mean(I['YL'], I['YR'], I['YT'])
    Y1, Y2, YT = Y1[:, keep], Y2[:, keep], YT[:, keep]

    sel = [i for i, g in enumerate(I['genes']) if g in set(null46)]
    genes = [I['genes'][i] for i in sel]
    gp = I['gp'].loc[genes][['chr', 'pos']]
    A, T, Va, Vt = A[sel], T[sel], Va[sel], Vt[sel]
    Y1, Y2, YT = Y1[sel], Y2[sel], YT[sel]
    gidx = {g: k for k, g in enumerate(genes)}
    print(f'{len(genes)} genes, {n_draw} draws', flush=True)

    rng = np.random.RandomState(SEED)
    perms = [rng.permutation(N) for _ in range(n_draw)]
    acc = {k: Acc() for k in ('hapmixQTL', 'hapmixQTL_allelic',
                              'hapmixQTL_total', 'mixQTL',
                              'mixQTL_allelic', 'mixQTL_total', 'RASQUAL')}
    want_by_gene, af_map = {}, {}
    tmp = Path(tempfile.mkdtemp())

    for p_i, prm in enumerate(perms):
        cov = pd.DataFrame(I['cov_df'].values[prm], index=order,
                           columns=I['cov_df'].columns)
        mk = lambda M: pd.DataFrame(M[:, prm], index=genes, columns=order)
        gdf = pd.DataFrame(I['dos'], index=I['vdf'].index, columns=order)
        with contextlib.redirect_stdout(io.StringIO()):
            map_nominal(gdf, I['vdf'][['chrom', 'pos']], mk(A), mk(T), mk(Va), mk(Vt),
                        gp, xL_df=pd.DataFrame(I['xL'], index=I['vdf'].index, columns=order),
                        xR_df=pd.DataFrame(I['xR'], index=I['vdf'].index, columns=order),
                        prefix='n', covariates_df=cov, window=1_000_000,
                        output_dir=str(tmp), verbose=False, ase_covariates_df=None)
        for pq in sorted(tmp.glob('n*.parquet')):
            df = pd.read_parquet(pq)
            df = df[np.isfinite(df.slope_se) & (df.slope_se > 0)]
            for g, v, af, sl, se in zip(df.phenotype_id, df.variant_id, df.af,
                                        df.slope, df.slope_se):
                acc['hapmixQTL'].add((g, v), float(sl), float(se))
                af_map.setdefault(v, float(af))
                want_by_gene.setdefault(g, set()).add(v)
            # the two channels, reported per variant by map_nominal alongside
            # the combined statistic
            for chan, bcol, scol in (('hapmixQTL_allelic', 'slope_a', 'slope_a_se'),
                                     ('hapmixQTL_total', 'slope_t', 'slope_t_se')):
                if bcol not in df.columns:
                    continue
                sub = df[np.isfinite(df[scol]) & (df[scol] > 0) & np.isfinite(df[bcol])]
                for g, v, sl, se in zip(sub.phenotype_id, sub.variant_id,
                                        sub[bcol], sub[scol]):
                    acc[chan].add((g, v), float(sl), float(se))
            pq.unlink()

        for g in genes:
            k = gidx[g]
            vsel = CM.gene_variant_index(I, g)
            if not vsel.size:
                continue
            vv = I['vdf'].iloc[I['idx']].iloc[vsel]
            h1 = I['xL'][I['idx']][:, keep][vsel].T.astype(float)
            h2 = I['xR'][I['idx']][:, keep][vsel].T.astype(float)
            o = MX.mixqtl_scan(Y1[k][prm], Y2[k][prm], YT[k][prm],
                               I['lib_size'][prm], h1, h2,
                               covariates=I['cov_df'].values[prm],
                               **MX.PACKAGE_DEFAULT_CUTOFFS)
            for chan, key in (('mixQTL', 'meta'), ('mixQTL_allelic', 'asc'),
                              ('mixQTL_total', 'trc')):
                b, s_ = o[key]['beta'], o[key]['se']
                for vid, bb, ss in zip(map(str, vv.index), b, s_):
                    if np.isfinite(bb) and np.isfinite(ss) and ss > 0:
                        acc[chan].add((g, vid), float(bb) * LN2, float(ss) * LN2)

            for vid, afc, chi2 in rasqual_rows(g, p_i, want_by_gene.get(g, set())):
                acc['RASQUAL'].add((g, vid), afc, abs(afc) / np.sqrt(chi2))
        print(f'  draw {p_i + 1}/{n_draw}', flush=True)
    shutil.rmtree(tmp, ignore_errors=True)

    # a variant must appear in most draws to get a usable sd; with a short
    # smoke-test run the floor drops so the pipeline can still be exercised
    min_n = max(3, min(10, n_draw))
    parts = [a.frame(k, min_n) for k, a in acc.items()]
    parts = [q for q in parts if len(q)]
    if not parts:
        raise SystemExit('no variant reached the minimum draw count')
    t = pd.concat(parts, ignore_index=True)
    t['af'] = [af_map.get(v, np.nan) for v in t.variant]
    t['maf'] = np.minimum(t.af, 1 - t.af)
    t['ratio'] = t.mean_se / t.sd_beta
    t = t[t.maf.notna() & (t.maf >= 0.05)]
    out = D / 'se_accuracy_by_maf_20260924'
    out.mkdir(exist_ok=True)
    t.to_csv(out / 'per_variant.tsv.gz', sep='\t', index=False)

    floor = 1.0 / np.sqrt(2 * (n_draw - 1))
    print(f'\nmean reported se / realized sd of beta, per (gene, variant)')
    print(f'1.00 = the standard error means what it says; '
          f'log noise floor {floor:.3f}\n')
    ARMS = ('hapmixQTL', 'hapmixQTL_allelic', 'hapmixQTL_total',
                    'mixQTL', 'mixQTL_allelic', 'mixQTL_total', 'RASQUAL')
    print(f'{"MAF":>12s} ' + ''.join(f'{a:>20s}' for a in ARMS))
    res = {'n_draw': n_draw, 'log_noise_floor': float(floor), 'buckets': {}}
    for lo, hi in BUCKETS:
        line = f'  {lo:.2f}-{hi:.2f} '
        rec = {}
        for arm in ARMS:
            s = t[(t.arm == arm) & (t.maf >= lo) & (t.maf < hi)]
            if len(s) < 30:
                line += f'{"--":>20s}'; continue
            med = float(s.ratio.median())
            line += f'{med:11.3f} (n={len(s)//1000:3d}k)'
            rec[arm] = dict(n=int(len(s)), median=med,
                            q25=float(s.ratio.quantile(.25)),
                            q75=float(s.ratio.quantile(.75)))
        res['buckets'][f'{lo}-{hi}'] = rec
        print(line)
    print('\noverall, all variants MAF>=0.05:')
    for arm in ARMS:
        s = t[t.arm == arm]
        if len(s):
            print(f'  {arm:10s} n={len(s):7d}  median {s.ratio.median():.3f}   '
                  f'IQR {s.ratio.quantile(.25):.3f}-{s.ratio.quantile(.75):.3f}')
            res[arm] = dict(n=int(len(s)), median=float(s.ratio.median()),
                            q25=float(s.ratio.quantile(.25)),
                            q75=float(s.ratio.quantile(.75)))
    (out / 'summary.json').write_text(json.dumps(res, indent=2, default=float))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
