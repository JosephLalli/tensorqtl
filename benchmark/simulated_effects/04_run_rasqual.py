"""RASQUAL on every simulated-effects dataset, nominal only (README: Joint models).

There are no reads, so each gene gets ONE pseudo feature SNP inside its gene body at which every
donor-gene pair the allelic channel admits (common.allelic_kept on the thinned point estimates)
is 0|1 with AS = (rint(pL), rint(pR)) (parseVCF.c:326 reads integers); every other donor is 0|0,
AS 0,0. The tested variants carry the real phased genotypes xL|xR, with AS 0,0 on every line
(parseCell keeps the previous line's counts when the field is absent). Y = thinned totals pT as
they are (the NB density goes through lgamma on doubles); K = eff_lib / mean(eff_lib) for every
gene (makeOffset.R builds a gene-constant size factor; main.c:347-351 divides by the row mean);
-x = the RNA-tied covariates in record order and the genotype PCs in place. Column i of every
array is real record perm[i], as in every other arm. Options: the command line of rasqual_arm in the
retired compare_pipelines.py (archived in brainvar_hapmix_deploy/retired_scripts_20261001/) with -h 0 (the rSNP Hardy-Weinberg filter, default 1e-8, removed 6,318 tested variants and
9 of 450 causal variants; these genotypes are the truth and no other arm filters on it).

Per tested variant with a converged row: chisq (field 11), pval_nominal = chi2.sf(chisq, 1),
slope = log2(pi / (1 - pi)) (field 12; nbem.c:1058 scales expression by 2(1 - pi), 1, 2 pi at
ALT dosage 0, 1, 2), slope_se DERIVED as |slope| / sqrt(chisq) (RASQUAL reports no standard
error; NaN where chisq <= 0), delta, phi, theta, n_feature_snps, r2_rsnp. Excluded and counted:
the pseudo fSNP's own row, non-converged rows (field 23 != 0), absent tested variants.

Output: OUT/<scenario>/rasqual/nominal_repNNN.parquet (common.write_parquet, fingerprint of the
dataset and 'rasqual', unit log2), the Y / K / X binaries in inputs_repNNN, RASQUAL's raw rows
per gene in raw_repNNN (the checkpoint: a gene whose raw file exists is not rerun; every skip is
printed), and OUT/summary.json with the counts per dataset and pooled.
"""
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.stats import chi2

import common as C

RASQUAL = '/mnt/ssd/lalli/usr/local/rasqual/bin/rasqual'   # the build the 2026-09-26 run used (copied 2026-10-01 from its worktree); run() checks its sha256
RASQUAL_SHA256 = 'ac3bd0563862fb9e3c3fdc355f6c588ed42247736eb0e3cde55e94e2820cb363'
OUT = C.JOINT['rasqual']
JOBS = 15                      # genes in parallel, one RASQUAL process each: 15 plus this driver = the shared host's cap of 16 live processes (2026-09-27)
MAF = C.MAF                 # 0.05, the tested set's MAF floor, passed as -a
MIN_COVERAGE = 0.05            # -d, RASQUAL's default (usage.c:46)
HWE_P = 0.0                    # -h; 0 turns the rSNP HWE filter off (default 1e-8, main.c:387/498), see the docstring
GT = np.array(['0|0:0,0', '0|1:0,0', '1|0:0,0', '1|1:0,0'])   # index 2 xL + xR
NUM = ['chisq', 'effect_size_pi', 'error_rate_delta', 'ref_mapping_bias_phi', 'overdispersion_theta',
       'n_feature_snps', 'convergence', 'r2_prior_posterior_rsnp']


def rsnp_text(S, g):
    """The gene's tested variants as VCF data lines, real phased GT, AS 0,0."""
    I, rows = S['I'], S['tested_rows'][g]
    v = I['vdf'].iloc[rows]
    fields = GT[I['xL'][rows] * 2 + I['xR'][rows]]
    return ''.join(f'{c}\t{p}\t{i}\t{r}\t{a}\t.\tPASS\t.\tGT:AS\t' + '\t'.join(x) + '\n'
                   for c, p, i, r, a, x in zip(v.chrom, v.pos, v.index, v.ref, v.alt, fields))


def pseudo_site(S, g):
    """(chrom, pos, body start, body end): pos inside the body at no biallelic SNP of the VCF read."""
    r, vdf = S['I']['gp'].loc[g], S['I']['vdf']
    taken = set(vdf.pos.values[vdf.chrom.values == r['chr']])
    pos = (int(r['start']) + int(r['end'])) // 2
    while pos in taken:
        pos += 1
    if pos > int(r['end']):
        raise SystemExit(f'{g}: no free position in the upper half of the gene body')
    return r['chr'], pos, int(r['start']), int(r['end'])


def pseudo_line(g, site, pL, pR, kept):
    a, b = np.rint(pL).astype(np.int64), np.rint(pR).astype(np.int64)
    fields = np.where(kept, [f'0|1:{x},{y}' for x, y in zip(a, b)], '0|0:0,0')
    return f'{site[0]}\t{site[1]}\t{g}_pseudo_fsnp\tA\tC\t.\tPASS\t.\tGT:AS\t' + '\t'.join(fields) + '\n'


def write_bins(S, ds, d):
    """Y, K and X of one dataset, float64, as main.c reads them."""
    I = S['I']
    X = np.column_stack([I['cov_df'].values[ds['perm']], I['geno_cov_df'].values])
    arrays = dict(Y=ds['pT'], K=np.tile(ds['eff_lib'] / ds['eff_lib'].mean(), (len(ds['pT']), 1)), X=X.T)
    d.mkdir(parents=True, exist_ok=True)
    for name, a in arrays.items():
        C.write_atomic(d / f'{name}.bin', lambda fh, a=a: fh.write(np.ascontiguousarray(a, np.float64).tobytes()))
    return {n: str(d / f'{n}.bin') for n in arrays}, X.shape[1]


def run_gene(k, g, site, text, bins, n, raw):
    """RASQUAL on one gene, its output checkpointed at `raw`; (rows as strings, wall seconds, skipped)."""
    if raw.exists():
        out, secs, skipped = raw.read_text(), 0.0, True
    else:
        cmd = [RASQUAL, '-y', bins['Y'], '-k', bins['K'], '-n', str(n), '-j', str(k + 1), '-l', str(text.count('\n')),
               '-m', '1', '-s', str(site[2]), '-e', str(site[3]), '-f', g, '-z', '-d', str(MIN_COVERAGE), '-a', str(MAF),
               '-h', str(HWE_P), '-x', bins['X'], '--n-threads', '1']
        t0 = time.perf_counter()
        out = subprocess.run(cmd, input=text, stdout=subprocess.PIPE, text=True, check=True).stdout
        secs, skipped = time.perf_counter() - t0, False
        C.write_atomic(raw, lambda fh: fh.write(out), 'w')
    rows = [ln.split('\t') for ln in out.splitlines()]
    bad = [r for r in rows if len(r) != len(C.RASQUAL_FIELDS) or r[0] != g or r[1] == 'SKIPPED']
    if bad or not rows:
        raise SystemExit(f'{g}: {len(rows)} RASQUAL rows, {len(bad)} malformed or SKIPPED, e.g. {bad[:1]}')
    return pd.DataFrame(rows, columns=C.RASQUAL_FIELDS), secs, skipped


def assemble(g, raw, tested, causal):
    """Converged tested rows in the output layout, and the counts of what was excluded; causal None if null."""
    d = raw.astype({c: float for c in NUM})
    pseudo = d.rs_id == f'{g}_pseudo_fsnp'
    if (~pseudo & ~d.rs_id.isin(tested)).any():
        raise SystemExit(f'{g}: RASQUAL rows for untested ids')
    t = d[~pseudo]
    ok = t[t.convergence == 0]
    pi = ok.effect_size_pi.values
    if not ((pi > 0) & (pi < 1)).all():
        raise SystemExit(f'{g}: converged pi outside (0, 1)')
    slope, c2 = np.log2(pi / (1 - pi)), ok.chisq.values
    se = np.full(len(ok), np.nan)
    se[c2 > 0] = np.abs(slope[c2 > 0]) / np.sqrt(c2[c2 > 0])
    out = pd.DataFrame(dict(phenotype_id=g, variant_id=ok.rs_id.astype(str).values, slope=slope, slope_se=se,
                            pval_nominal=chi2.sf(c2, 1), chisq=c2, pi=pi, delta=ok.error_rate_delta.values,
                            phi=ok.ref_mapping_bias_phi.values, theta=ok.overdispersion_theta.values,
                            n_feature_snps=ok.n_feature_snps.values.astype(int), r2_rsnp=ok.r2_prior_posterior_rsnp.values))
    return out, dict(pseudo=int(pseudo.sum()), nonconv=int(len(t) - len(ok)), absent=len(tested) - len(t),
                     chisq_le0=int((c2 <= 0).sum()), no_fsnp=int((d.n_feature_snps == 0).all()),
                     causal_nonconv=int(causal in set(t.rs_id[t.convergence != 0])),
                     causal_absent=int(causal is not None and causal not in set(t.rs_id)))


def run(S, datasets, out, run_list, jobs):
    """RASQUAL on the datasets of run_list [(scenario, r)], `jobs` genes at a time; writes the outputs (docstring)."""
    if not os.access(RASQUAL, os.X_OK):
        raise SystemExit(f'{RASQUAL}: missing or not executable')
    if hashlib.sha256(open(RASQUAL, 'rb').read()).hexdigest() != RASQUAL_SHA256:
        raise SystemExit(f'{RASQUAL}: sha256 differs from RASQUAL_SHA256 {RASQUAL_SHA256}')
    print(f'{RASQUAL} sha256 {RASQUAL_SHA256}', flush=True)
    genes = S['genes']
    text, sites = {g: rsnp_text(S, g) for g in genes}, {g: pseudo_site(S, g) for g in genes}
    print(f'{len(run_list)} datasets x {len(genes)} genes, {jobs} jobs; tested variants {int(S["n_tested"].sum()):,}',
          flush=True)
    summary, secs = {}, []
    ex = cf.ThreadPoolExecutor(jobs)
    try:
        for sc, r in run_list:
            ds = C.load_dataset(datasets, sc, r)
            kept = C.allelic_kept(ds['pL'], ds['pR'], ds['Va'])
            d = out / sc / 'rasqual'
            bins, n_cov = write_bins(S, ds, d / f'inputs_rep{r:03d}')
            (d / f'raw_rep{r:03d}').mkdir(exist_ok=True)
            futs = [ex.submit(run_gene, k, g, sites[g], pseudo_line(g, sites[g], ds['pL'][k], ds['pR'][k], kept[k])
                              + text[g], bins, len(S['order']), d / f'raw_rep{r:03d}' / f'{g}.txt')
                    for k, g in enumerate(genes)]
            parts, cnt, skipped = [], {}, []
            for k, (g, f) in enumerate(zip(genes, futs)):
                raw, s, skip = f.result()
                secs.append(s)
                skipped += [g] if skip else []
                df, c = assemble(g, raw, S['tested'][g], None if ds['is_null'][k] else str(ds['causal_variant'][k]))
                parts.append(df)
                for key, v in c.items():
                    cnt[key] = cnt.get(key, 0) + v
            df = pd.concat(parts, ignore_index=True)
            C.write_parquet(df, d / f'nominal_rep{r:03d}.parquet', C.fingerprint(ds, 'rasqual'), 'log2')
            a, b = np.rint(ds['pL']), np.rint(ds['pR'])
            cnt.update(rows=len(df), tests=int(S['n_tested'].sum()), covariates=n_cov, het=int(kept.sum()),
                       informative=int((ds['pL'] + ds['pR'] > 0).sum()), as00=int((kept & (a + b == 0)).sum()),
                       causal_nonnull=int((~ds['is_null']).sum()), genes_skipped_existing_raw=len(skipped))
            summary[f'{sc} rep {r:03d}'] = cnt
            print(f'{sc} rep {r:03d}: {json.dumps(cnt)}; skipped (raw file present) {skipped[:5]}{"..." if len(skipped) > 5 else ""}',
                  flush=True)
    finally:
        ex.shutdown(cancel_futures=True)   # a failed gene cancels the queue; running RASQUAL processes still finish
    tot = lambda k: sum(c[k] for c in summary.values())   # noqa: E731
    pooled = {k: tot(k) for k in ('rows', 'tests', 'nonconv', 'absent', 'chisq_le0', 'no_fsnp', 'causal_nonconv',
                                  'causal_absent', 'causal_nonnull')}
    C.write_json(out / 'summary.json', dict(per_dataset=summary, pooled=pooled, jobs=jobs, rasqual=RASQUAL, rasqual_sha256=RASQUAL_SHA256))
    s = np.array([x for x in secs if x > 0])
    print(f'pooled: {json.dumps(pooled)}; seconds per gene run median {np.median(s) if len(s) else 0:.1f}, '
          f'{len(secs) - len(s)} genes taken from raw files; wrote {out}', flush=True)


def main():
    S = C.setup(C.load()[0])
    meta = json.loads((C.DATASETS / 'meta.json').read_text())
    if S['genes'] != meta['genes'] or S['order'] != meta['donors']:
        raise SystemExit(f'{C.DATASETS / "meta.json"}: genes or donors differ from the loader inputs')
    run(S, C.DATASETS, OUT, C.runs(meta), JOBS)


if __name__ == '__main__':
    main()
