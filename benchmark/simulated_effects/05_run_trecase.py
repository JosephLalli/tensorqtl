"""TReCASE (asSeq 0.99.501, trecase) on every simulated-effects dataset, nominal only (README: Joint models).

Inputs per dataset (column i is real record perm[i], as in every other arm): Y = thinned totals
pT as they are (the TReC negative binomial goes through lgammafn on doubles); Y1, Y2 = rint(pL),
rint(pR) on the records common.allelic_kept admits, 0 elsewhere (asSeq's beta-binomial adds
lchoose(n, nA), which needs integers, and drops records with Y1 + Y2 < min.AS.reads); X = the
RNA-tied covariates in record order and the genotype PCs in place, no constant column (glmFit
charges the intercept itself); offset = log(eff_lib); Z = 3 xL + xR (0 ref|ref, 1 ref|alt, 3
alt|ref, 4 alt|alt; haplotype 1 = L), one table per chromosome over every tested-set variant with
varying ALT dosage (the 507 at constant dosage have no row), from which asSeq's window (same
chromosome, |ePos - mPos| <= 1e6) picks each gene's variants; ePos = the gene position the other
arms use. asSeq's defaults are kept except p.cut (run_trecase.R). The per-gene call is
run_trecase.R, one Rscript process per gene, JOBS at a time.

Per tested variant with varying dosage: pval_nominal = asSeq's printed final_Pvalue (its rule,
trecase.c:1311-1323: the TReC p when trans_Pvalue < transTestP or is NA, else the joint p; checked
by string equality; a printed trans_Pvalue of 5.00e-02 lies on either side and is resolved by which
column final_Pvalue matches), final_stat ('joint' or 'trec'), slope = b / ln 2 of that statistic
(log2 kappa, ALT over REF), slope_se DERIVED as |slope| / sqrt(Chisq) (asSeq reports none; NaN where
Chisq <= 0 or NA), the three models' pval / slope / se / chisq / df, joint_ok, trans_chisq,
pval_trans, n_trec, n_ase, n_ase_het, nb_od, bb_od. Where asSeq's own dosage model fails it refits
TReC with linear dosage (a log fold per ALT allele, not ln kappa; counted from the trace log as
trec_linear_dosage, not flagged per row).

Output: OUT/<scenario>/trecase/nominal_repNNN.parquet (common.write_parquet, fingerprint of the
dataset and 'trecase', unit log2); OUT/summary.json with the counts per dataset and pooled; the
inputs, asSeq's files and one trace log per gene under WORK/<scenario>/repNNN. A gene whose
<gene>_status.tsv exists (written last by the R script) is not rerun; every skip is printed.
"""
import concurrent.futures as cf
import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

import common as C

OUT, WORK = C.JOINT['trecase'], C.ROOT / 'trecase_work'
R_RUNNER = Path(__file__).resolve().with_name('run_trecase.R')
R_ENV = {'R_LD_LIBRARY_PATH': '/usr/lib/R/lib:/usr/lib/x86_64-linux-gnu',
         'LD_LIBRARY_PATH': '/usr/local/cuda/lib64'}   # CLAUDE.md, "R's BLAS crash is an environment clash"
JOBS = 64                      # genes at once; each R process is one thread and about 700 MB (user, 2026-10-01: 64 at a time; the cap of 16 live processes of 2026-09-27 no longer applies)
TRANS_TEST_P = 0.05            # asSeq transTestP default (R/trecase.R), the rule at trecase.c:1311
TRANS_BORDER = '5.00e-02'      # %.2e of a trans p in [0.04995, 0.05005): either side of TRANS_TEST_P
MIN_AS_READS = 5               # asSeq min.AS.reads default (Y1 + Y2, trecase.c:615-617); counts only
CONVERGE = 5e-5                # asSeq converge default: R/trecase.R stops on a Y column with var < converge, so such a gene is not run (no rows)
MIN_N_HET = 5                  # asSeq min.n.het default; counts only
CHANNELS = {'t': 'TReC', 'a': 'ASE', 'joint': 'Joint'}
FAILS = {'joint_theta': 'fail to estimate theta in joint model',   # trecase.c:1003-1012, printed at trace >= 1
         'ase': 'fail ASE model',                                  # :834-838
         'trec_linear_dosage': 'convSNPj@glmNB ='}                 # :740-752, see the docstring


def chrom_int(c):
    if not (c.startswith('chr') and c[3:].isdigit()):
        raise SystemExit(f'chromosome {c!r} is not chr<integer>; asSeq needs integer chromosomes')
    return int(c[3:])


def write_bin(path, a):
    """float64, C order: an [k, N] array is R's N x k matrix read column-major."""
    C.write_atomic(path, lambda fh: fh.write(np.ascontiguousarray(a, np.float64).tobytes()))


def write_genotypes(S, d):
    """chr<k>.Z.bin and chr<k>.markers.tsv for every chromosome of the genes; variant ids per chromosome."""
    I, vdf, idx = S['I'], S['vdf'], S['I']['idx']
    d.mkdir(parents=True, exist_ok=True)
    dos = I['dos'][idx]
    varying = ~(dos == dos[:, [0]]).all(1)
    ids = {}
    for c in sorted({chrom_int(S['gp'].loc[g, 'chr']) for g in S['genes']}):
        on = (vdf.chrom.values == f'chr{c}') & varying
        xL, xR = I['xL'][idx[on]].astype(np.int64), I['xR'][idx[on]].astype(np.int64)
        write_bin(d / f'chr{c}.Z.bin', 3 * xL + xR)
        m = pd.DataFrame(dict(variant_id=vdf.index[on].astype(str), chr=c, pos=vdf.pos.values[on].astype(int)))
        C.write_atomic(d / f'chr{c}.markers.tsv', lambda fh, m=m: m.to_csv(fh, sep='\t', index=False), 'w')
        ids[c] = m.variant_id.tolist()
    print(f'genotypes: {len(ids)} chromosomes, {sum(map(len, ids.values())):,} tested-set variants with varying ALT '
          f'dosage of {len(vdf):,}', flush=True)
    return ids


def allelic_counts(ds):
    """Y1, Y2 as asSeq gets them: rint(pL), rint(pR) on allelic_kept records, 0 elsewhere; and the kept mask."""
    kept = C.allelic_kept(ds['pL'], ds['pR'], ds['Va'])
    return np.where(kept, np.rint(ds['pL']), 0.0), np.where(kept, np.rint(ds['pR']), 0.0), kept


def write_dataset(S, ds, d, counts=allelic_counts):
    """Y, Y1, Y2, X, offset and genes.tsv of one dataset for the R runner."""
    I = S['I']
    X = np.column_stack([I['cov_df'].values[ds['perm']], I['geno_cov_df'].values])
    Y1, Y2, _ = counts(ds)
    d.mkdir(parents=True, exist_ok=True)
    for name, a in dict(Y=ds['pT'], Y1=Y1, Y2=Y2, X=X.T, offset=np.log(ds['eff_lib'])).items():
        write_bin(d / f'{name}.bin', a)
    tab = pd.DataFrame(dict(gene=S['genes'], chr=[chrom_int(S['gp'].loc[g, 'chr']) for g in S['genes']],
                            pos=[int(S['gp'].loc[g, 'pos']) for g in S['genes']]))
    C.write_atomic(d / 'genes.tsv', lambda fh: tab.to_csv(fh, sep='\t', index=False), 'w')
    return X.shape[1]


def run_gene(ddir, gdir, g, tag):
    """One Rscript process unless its status file exists; (wall seconds, skipped). A failed process stops the run."""
    if Path(f'{tag}_status.tsv').exists():
        return 0.0, True
    t0 = time.perf_counter()
    with open(f'{tag}.log', 'w') as fh:
        rc = subprocess.run(['Rscript', str(R_RUNNER), str(ddir), str(gdir), g, str(tag)], stdout=fh,
                            stderr=subprocess.STDOUT, env={**os.environ, **R_ENV}).returncode
    if rc != 0:
        raise SystemExit(f'{g}: Rscript exited {rc}; {tag}.log:\n' + ''.join(open(f'{tag}.log').readlines()[-15:]))
    return time.perf_counter() - t0, False


def convert(g, tag, markers, expected):
    """asSeq's rows for gene g in the output layout, its status row, and its trace log."""
    raw = pd.read_csv(f'{tag}_eqtl.txt', sep='\t', dtype=str, keep_default_na=False)
    status = pd.read_csv(f'{tag}_status.tsv', sep='\t').iloc[0]
    vid = np.array(markers)[raw.MarkerRowID.astype(int).values - 1]
    if (raw.GeneRowID != '1').any() or len(set(vid)) != len(vid) or set(vid) != expected:
        raise SystemExit(f'{g}: asSeq wrote {len(vid)} rows ({len(set(vid))} variants) against {len(expected)} tested '
                         f'variants with varying dosage')
    num = lambda c: pd.to_numeric(raw[c].replace('NA', np.nan)).values.astype(float)   # noqa: E731
    ch = {}
    for k, name in CHANNELS.items():
        b, c2 = num(f'{name}_b'), num(f'{name}_Chisq')
        s = b / np.log(2.0)
        se = np.full(len(s), np.nan)
        pos = np.isfinite(c2) & (c2 > 0)
        se[pos] = np.abs(s[pos]) / np.sqrt(c2[pos])
        ch[k] = dict(pval=num(f'{name}_Pvalue'), slope=s, se=se, chisq=c2, df=num(f'{name}_df'))
    final, trec, joint = raw.final_Pvalue.values, raw.TReC_Pvalue.values, raw.Joint_Pvalue.values
    pt = num('trans_Pvalue')
    use_joint = np.isfinite(pt) & (pt >= TRANS_TEST_P)       # trecase.c:1311: TReC where trans p < transTestP or NA
    border = raw.trans_Pvalue.values == TRANS_BORDER
    use_joint[border] = final[border] != trec[border]
    if (np.where(use_joint, joint, trec) != final).any():
        raise SystemExit(f'{g}: final_Pvalue is not the p that trecase.c:1311-1323 selects')
    joint_ok = np.isfinite(ch['joint']['pval'])
    pick = lambda c: np.where(use_joint, ch['joint'][c], ch['t'][c])   # noqa: E731
    cols = dict(phenotype_id=np.full(len(raw), g), variant_id=vid, pval_nominal=num('final_Pvalue'),
                slope=pick('slope'), slope_se=pick('se'), chisq=pick('chisq'), df=pick('df'),
                final_stat=np.where(use_joint, 'joint', 'trec'))
    for k in ('a', 't', 'joint'):
        cols.update({(f'slope_{k}_se' if c == 'se' else f'{c}_{k}'): ch[k][c] for c in ('pval', 'slope', 'se', 'chisq', 'df')})
    cols.update(joint_ok=joint_ok, trans_chisq=num('trans_Chisq'), pval_trans=pt, n_trec=num('n_TReC'),
                n_ase=num('n_ASE'), n_ase_het=num('n_ASE_Het'), nb_od=np.where(joint_ok, num('NBod'), np.nan),
                bb_od=np.where(joint_ok, num('BBod'), np.nan))
    return pd.DataFrame(cols), status, Path(f'{tag}.log').read_text()


def dataset_counts(df, ds, S, expected, status, secs, logs, counts=allelic_counts):
    """Counts written per dataset (the keys 08_report.py reads, plus the run's timing)."""
    genes = S['genes']
    cz = pd.DataFrame(dict(phenotype_id=genes, variant_id=ds['causal_variant'].astype(str), is_null=ds['is_null']))
    nn = cz[~cz.is_null]
    ran = np.array([v in expected[g] for g, v in zip(nn.phenotype_id, nn.variant_id)], dtype=bool)
    Cz = nn[ran].merge(df, on=['phenotype_id', 'variant_id'], how='left')
    if Cz.final_stat.isna().any():
        raise SystemExit(f'{int(Cz.final_stat.isna().sum())} causal variants of non-null genes have no asSeq row')
    Y1, Y2, kept = counts(ds)
    text = ''.join(logs)
    trace = {k: text.count(s) for k, s in FAILS.items()}
    trace['theta_fail_abs_gradient_max'] = max([abs(float(x)) for x in re.findall(r'gradience=(\S+), fail=', text)],
                                               default=None)
    return dict(rows=len(df), tested_constant_dosage=int(sum(S['n_tested'][g] - len(S['scanned'][g]) for g in genes)),
                informative_zeroed_not_allelic_kept=int(((ds['pL'] + ds['pR'] > 0) & ~kept).sum()),
                as_records_admitted=int(((Y1 + Y2) >= MIN_AS_READS).sum()),
                trec_na=int(df.pval_t.isna().sum()), ase_na_few_het=int((df.pval_a.isna() & (df.n_ase_het < MIN_N_HET)).sum()),
                joint_na=int((~df.joint_ok).sum()), joint_na_by_trace=trace,
                final_joint=int((df.final_stat == 'joint').sum()), final_trec=int((df.final_stat == 'trec').sum()),
                final_na=int(df.pval_nominal.isna().sum()), final_df_not_1=int((df.df.notna() & (df.df != 1)).sum()),
                causal_not_run=[f'{g} {v}' for g, v in zip(nn.phenotype_id[~ran], nn.variant_id[~ran])],
                causal_nonnull=len(Cz), causal_joint_na=int((~Cz.joint_ok.astype(bool)).sum()),
                causal_final_joint=int((Cz.final_stat == 'joint').sum()), causal_final_trec=int((Cz.final_stat == 'trec').sum()),
                causal_final_na=int(Cz.pval_nominal.isna().sum()), trecase_warnings=int(sum(s.n_warnings for s in status)),
                wall_seconds=[round(s, 2) for s in secs], trecase_seconds=[round(float(s.seconds), 2) for s in status])


def run(S, datasets, out, work, run_list, jobs, arm='trecase', counts=allelic_counts):
    """asSeq on the datasets of run_list [(scenario, r)], `jobs` Rscript processes at a time, every dataset's genes queued
    at once; writes the outputs of `arm` (Y1, Y2 from `counts`)."""
    genes = S['genes']
    ids = write_genotypes(S, work / 'genotypes')
    chrom = {g: chrom_int(S['gp'].loc[g, 'chr']) for g in genes}
    expected = {g: S['scanned'][g] & set(ids[chrom[g]]) for g in genes}
    if any(expected[g] != S['scanned'][g] for g in genes):
        raise SystemExit('a tested variant with varying dosage is missing from its chromosome marker table')
    print(f'{R_RUNNER}; {len(run_list)} datasets x {len(genes)} genes, {jobs} Rscript processes', flush=True)
    t_all, summary, queued = time.perf_counter(), {}, []
    ex = cf.ThreadPoolExecutor(jobs)
    try:
        for sc, r in run_list:
            ds = C.load_dataset(datasets, sc, r)
            ddir = work / sc / f'rep{r:03d}'
            n_cov = write_dataset(S, ds, ddir, counts)
            (ddir / 'out').mkdir(exist_ok=True)
            refused = [g for g, y in zip(genes, ds['pT']) if np.var(y, ddof=1) < CONVERGE]
            if refused:
                print(f'{sc} rep {r:03d}: not run, total variance below asSeq\'s converge {CONVERGE:g} (asSeq stops on it): {refused}',
                      flush=True)
            queued.append((sc, r, ds, ddir, n_cov, refused, {g: ex.submit(run_gene, ddir, work / 'genotypes', g, ddir / 'out' / g)
                                                               for g in genes if g not in refused}))
        for sc, r, ds, ddir, n_cov, refused, futs in queued:
            parts, status, secs, logs, skipped = [], [], [], [], []
            for g, f in futs.items():
                s, skip = f.result()
                secs.append(s)
                skipped += [g] if skip else []
                d, st, log = convert(g, ddir / 'out' / g, ids[chrom[g]], expected[g])
                parts.append(d)
                status.append(st)
                logs.append(log)
            df = pd.concat(parts, ignore_index=True)
            C.write_parquet(df, out / sc / arm / f'nominal_rep{r:03d}.parquet', C.fingerprint(ds, arm), 'log2')
            c = dataset_counts(df, ds, S, {g: set() if g in refused else v for g, v in expected.items()}, status, secs, logs, counts)
            c['genes_skipped_existing_status'] = len(skipped)
            c['genes_not_run_constant_total'] = refused
            summary[f'{sc} rep {r:03d}'] = c
            short = {k: v for k, v in c.items() if not k.endswith('_seconds')}
            print(f'{sc} rep {r:03d} ({n_cov} covariates): {json.dumps(short)}; skipped (status present) '
                  f'{skipped[:5]}{"..." if len(skipped) > 5 else ""}; elapsed {(time.perf_counter() - t_all) / 60:.1f} min',
                  flush=True)
    finally:
        ex.shutdown(cancel_futures=True)   # a failed gene cancels the queue rather than waiting it out
    tot = lambda k: sum(c[k] for c in summary.values())   # noqa: E731
    pooled = dict(tests=tot('rows'), joint_na=tot('joint_na'), joint_na_share=tot('joint_na') / tot('rows'),
                  final_na=tot('final_na'), final_joint=tot('final_joint'), final_trec=tot('final_trec'),
                  joint_fail_theta=sum(c['joint_na_by_trace']['joint_theta'] for c in summary.values()),
                  causal_not_run=[x for c in summary.values() for x in c['causal_not_run']],
                  causal_nonnull=tot('causal_nonnull'), causal_joint_na=tot('causal_joint_na'),
                  causal_joint_na_share=tot('causal_joint_na') / tot('causal_nonnull'),
                  causal_final_joint=tot('causal_final_joint'), causal_final_trec=tot('causal_final_trec'),
                  causal_final_na=tot('causal_final_na'))
    C.write_json(out / 'summary.json', dict(per_dataset=summary, pooled=pooled, jobs=jobs,
                                             wall_minutes=(time.perf_counter() - t_all) / 60, covariates=str(C.COV)))
    print(f'pooled over {len(summary)} datasets: {json.dumps(pooled)}; wrote {out}', flush=True)


def main():
    S = C.setup(C.load()[0])
    meta = json.loads((C.DATASETS / 'meta.json').read_text())
    if S['genes'] != meta['genes'] or S['order'] != meta['donors']:
        raise SystemExit(f'{C.DATASETS / "meta.json"}: genes or donors differ from the loader inputs')
    run(S, C.DATASETS, OUT, WORK, C.runs(meta), JOBS)


if __name__ == '__main__':
    main()
