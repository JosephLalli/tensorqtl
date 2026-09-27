"""RASQUAL on every plasmode dataset, nominal only (no -r permutations).

Datasets: make_datasets.py. Tested variants, genotype frames and covariates:
run_arms.setup, so RASQUAL sees the variants and record order the other arms see.

INPUT DESIGN (user-approved 2026-09-26, het set changed on review the same day,
pending the user's confirmation). There are no reads, so:
  fSNP   Each gene gets ONE pseudo feature SNP inside its gene body (gp start..end)
         at a position that is no biallelic SNP of the VCF read (the body midpoint,
         moved up past any such SNP). Every donor-gene pair the allelic channel
         admits (MD.allelic_kept on the dataset's THINNED point estimates and Va,
         as run_arms.arm_variances) is 0|1 there with AS = (round(pL), round(pR))
         as (ref, alt), so haplotype 1 = L; every other donor is 0|0 with AS 0,0.
         The approved set was pL + pR > 0, which also admitted the 950-957 of
         6,981-6,982 informative pairs per dataset with exactly one haplotype below
         0.5 read (Salmon exact zeros, dropped by every other arm) as AS (x, 0),
         maximal imbalance in a random direction. AS must be integers
         (parseVCF.c:326 reads "%ld,%ld"); admission and rounding counts are
         printed per dataset.
  rSNPs  The gene's tested variants with the real phased genotypes xL|xR (L first,
         matching the fSNP phase). Every line carries AS, 0,0 on rSNPs, because
         parseCell fills the AS buffer only when the field is present
         (parseVCF.c:326), so a line without it inherits the previous line's counts.
  Y      Thinned totals pT as they are: the NB density is evaluated through lgamma
         on doubles (nbglm.c:84, nbem.c:780), so fractional Salmon counts need no
         rounding.
  K      eff_lib / mean(eff_lib), the same row for every gene: RASQUAL's
         makeOffset.R:20-24 builds a gene-constant size factor, column sums over the
         complete count table divided by their mean, and README.md:158 requires the
         offset to come from the complete expression data. The edgeR effective
         library size (lib.size x TMM over 41,552 genes, the pipeline's log2 CPM
         library size) stands in for the column sums. main.c:347-351 divides each
         gene's row by its mean, so only the ratios matter.
  -x     The dataset's RNA-tied covariates (cov_df rows in the order perm) and the
         genotype PCs in place, 17 columns written covariate-major, as main.c:299-306
         reads them (it adds the intercept and centres each column).
  Order  As run_arms: column i of Y, K, AS and the RNA-tied covariates is real record
         perm[i] (already applied in the dataset); genotypes and genotype PCs stay.

COMMAND: compare_pipelines.rasqual_arm's line, one thread per process. Options
(usage.c line): -y 16, -k 17, -s 18, -e 19, -j 20 (row of Y and K), -l 21 (every
line fed), -m 22 (1), -n 23, -f 30, -a 36 (0.05, the tested set's own MAF floor),
-d 46 (0.05, RASQUAL's default, rasqual_arm's value), -x 51, --n-threads 63, -z 67
(no RSQ in INFO, so rsq falls back to the accuracy of the GT-derived allelic
probabilities, main.c:470; kept for parity with rasqual_arm). --force is not
needed: (fSNPs + 1) x rSNPs <= 2 x 12,943 = 25,886 < 30,000 (main.c:582).
One deviation from RASQUAL's defaults, -h 0: the default rSNP HWE filter at p 1e-8
(usage.c:37, main.c:387/498) removed 6,318 of 473,144 tested variants and the
causal variant of 9 of 450 non-null causal units, while these genotypes are the
truth and no other arm's tested set has an HWE filter. Kept as defaults: allelic
probabilities truncated to [0.001, 0.999] (usage.c:68, main.c:229); reference-bias
phi estimated (usage.c:60 fixes it); posterior genotype updating on
(README.md:97-101), although these genotypes are the truth by construction.

SMOKE (beta 0.8 rep 000, ASPHD1 / NISCH / ZNF420, 9,219 tested variants) checks
RASQUAL's chisq rank and sign at the causal variant. With the pL + pR > 0 het set
ASPHD1's causal variant ranked 2,086 of 2,753 (its 27 one-haplotype-zero pairs
drove phi to 0.36) and ZNF420's (4 such pairs) 8; NISCH has none. With allelic_kept
and -h 0 (2026-09-26): ranks 3 / 1 / 2 of 2,828 / 3,062 / 3,329, true sign in all
three, every tested variant a converged row; 74-158 s per gene single-threaded,
39.6 ms per tested variant pooled at host load 110-140 of 256 cores (68.6 / 80.6
ms with the old het set), so no whole-gene smoke runs in under a minute.

OUTPUT, per tested variant with a converged row (README.md:48-74 fields, named by
compare_pipelines.RASQUAL_FIELDS): chisq (field 11, 2 x log likelihood ratio of the
joint model), pval_nominal = chi2.sf(chisq, 1), pi (field 12), slope =
log2(pi / (1 - pi)), the log2 aFC ALT over REF (nbem.c:1058 scales expression by
2(1 - pi), 1, 2 pi at ALT dosage 0, 1, 2, so pi is the ALT allele's share),
slope_se DERIVED as |slope| / sqrt(chisq), the Wald back-derivation (RASQUAL reports
no standard error; this equals one only where the Wald approximation holds; NaN
where chisq <= 0, counted), plus delta, phi, theta (fields 13-15), n_feature_snps
(field 17: 0 means RASQUAL did not admit the pseudo fSNP, main.c:535, and fitted
total counts only; counted) and r2_rsnp (field 25). Excluded and counted: the
pseudo fSNP's own row when it passes the rSNP filters, and non-converged rows
(field 23 != 0); non-null causal variants left without a row (non-converged or
absent) are counted separately, since score.py scores a missing p as not detected.

WHAT THIS CANNOT ANSWER. RASQUAL's delta (sequencing/mapping error) and phi
(reference mapping bias) model a read-level process these Salmon haplotype counts
did not pass through, and one pseudo fSNP stands in for per-SNP counts.

Output: OUT/<scenario>/rasqual/nominal_repNNN.parquet (run_arms.write_parquet,
fingerprint run_arms.fingerprint(ds, 'rasqual'), unit log2) and the Y / K / X
binaries RASQUAL read in OUT/<scenario>/rasqual/inputs_repNNN; SMOKE writes under
OUT/smoke.
"""
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import make_datasets as MD                           # noqa: E402
import run_arms as RA                                # noqa: E402
import compare_mixqtl_replication as CM              # noqa: E402
from compare_pipelines import RASQUAL_FIELDS         # noqa: E402

RASQUAL = Path('/mnt/ssd/lalli/tensorqtl/.claude/worktrees/mixqtl-replication/rasqual_src/src/rasqual')  # the build compare_pipelines ran (hapmix-runbook holds a second); sha256 printed
DATASETS = MD.ROOT / 'datasets'
OUT = MD.ROOT / 'results_rasqual'
ARM = 'rasqual'
JOBS = 48                      # genes in parallel, one RASQUAL thread each; the shared 256-core host allows this run at most 48 (64 for the 2026-09-26 run)
MAF = CM.MAF                   # 0.05, the tested set's MAF floor, passed as -a
MIN_COVERAGE = 0.05            # -d, RASQUAL's default (usage.c:46) and rasqual_arm's value
HWE_P = 0.0                    # -h; 0 turns the rSNP HWE filter off (default 1e-8, main.c:387/498), see docstring
SMOKE = False                  # True: beta0.8 rep000, SMOKE_GENES, VCF read over their regions only
SMOKE_GENES = ('ASPHD1', 'NISCH', 'ZNF420')   # non-null at beta 0.8 rep 000, causal variant at the split arm's rank 1, fewest tested variants, both signs
RANK_MAX = 5                   # smoke: causal chisq rank <= this with the true sign in every smoke gene; a misaligned input ranks it uniformly among ~3,000 rows
TIE = 1e5                      # RASQUAL ties chisq values equal after round(x * 1e5) (main.c:722, 728)
GT = np.array(['0|0:0,0', '0|1:0,0', '1|0:0,0', '1|1:0,0'])   # index 2 xL + xR
NUM = ['chisq', 'effect_size_pi', 'error_rate_delta', 'ref_mapping_bias_phi', 'overdispersion_theta',
       'n_feature_snps', 'convergence', 'r2_prior_posterior_rsnp']


def load():
    """run_arms.setup on the loader inputs; SMOKE reads the VCF over the smoke genes' regions only.

    The gene list stays all 100 genes, so every gene body still excludes variants and the
    smoke genes' tested sets equal the full run's.
    """
    regions = MD.REGIONS
    if SMOKE:
        bed = pd.read_csv(MD.REGIONS, sep='\t', header=None)
        sub = bed[bed[3].isin(SMOKE_GENES)]
        if len(sub) != len(SMOKE_GENES):
            raise SystemExit(f'{MD.REGIONS}: {len(sub)} lines for {SMOKE_GENES}')
        regions = OUT / 'smoke' / 'regions.bed'
        regions.parent.mkdir(parents=True, exist_ok=True)
        MD.write_atomic(regions, lambda fh: sub.to_csv(fh, sep='\t', header=False, index=False), 'w')
    return RA.setup(CM.load_point_estimate_inputs(gene_list=str(MD.GENES), regions=str(regions)))


def rsnp_text(S, g):
    """The gene's tested variants as VCF data lines, real phased GT, AS 0,0."""
    I, rows = S['I'], S['tested_rows'][g]
    v = I['vdf'].iloc[rows]
    cells = GT[I['xL'][rows] * 2 + I['xR'][rows]]
    return ''.join(f'{c}\t{p}\t{i}\t{r}\t{a}\t.\tPASS\t.\tGT:AS\t' + '\t'.join(x) + '\n'
                   for c, p, i, r, a, x in zip(v.chrom, v.pos, v.index, v.ref, v.alt, cells))


def pseudo_site(S, g):
    """(chrom, pos, body start, body end): pos inside the body, no biallelic SNP of the VCF read."""
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
    cells = np.where(kept, [f'0|1:{x},{y}' for x, y in zip(a, b)], '0|0:0,0')
    return f'{site[0]}\t{site[1]}\t{g}_pseudo_fsnp\tA\tC\t.\tPASS\t.\tGT:AS\t' + '\t'.join(cells) + '\n'


def admission(pL, pR, kept):
    """What the pseudo fSNP's het set drops and what rounding to the AS field does to what it keeps."""
    a, b = np.rint(pL), np.rint(pR)
    return (f'{int(kept.sum())} het of {int((pL + pR > 0).sum())} informative pairs, '
            f'{int(((pL + pR > 0) & ~kept).sum())} excluded by allelic_kept; among het, AS total changed by '
            f'> 0.5 read {int((kept & (abs(a + b - pL - pR) > 0.5)).sum())}, AS 0,0 {int((kept & (a + b == 0)).sum())}')


def write_bins(S, ds, d):
    """Y, K and X for one dataset, float64, as main.c reads them."""
    I = S['I']
    X = np.column_stack([I['cov_df'].values[ds['perm']], I['geno_cov_df'].values])
    arrays = dict(Y=ds['pT'], K=np.tile(ds['eff_lib'] / ds['eff_lib'].mean(), (len(ds['pT']), 1)), X=X.T)
    d.mkdir(parents=True, exist_ok=True)
    for name, a in arrays.items():
        MD.write_atomic(d / f'{name}.bin', lambda fh, a=a: fh.write(np.ascontiguousarray(a, np.float64).tobytes()))
    return {n: str(d / f'{n}.bin') for n in arrays}, X.shape[1]


def run_gene(k, g, site, text, bins, n):
    """RASQUAL on one gene; its rows as strings and the wall seconds."""
    cmd = [str(RASQUAL), '-y', bins['Y'], '-k', bins['K'], '-n', str(n), '-j', str(k + 1),
           '-l', str(text.count('\n')), '-m', '1', '-s', str(site[2]), '-e', str(site[3]), '-f', g,
           '-z', '-d', str(MIN_COVERAGE), '-a', str(MAF), '-h', str(HWE_P), '-x', bins['X'], '--n-threads', '1']
    t0 = time.perf_counter()
    out = subprocess.run(cmd, input=text, stdout=subprocess.PIPE, text=True, check=True).stdout
    secs = time.perf_counter() - t0
    rows = [ln.split('\t') for ln in out.splitlines()]
    bad = [r for r in rows if len(r) != len(RASQUAL_FIELDS) or r[0] != g or r[1] == 'SKIPPED']
    if bad or not rows:
        raise SystemExit(f'{g}: {len(rows)} RASQUAL rows, {len(bad)} malformed or SKIPPED, e.g. {bad[:1]}')
    return pd.DataFrame(rows, columns=RASQUAL_FIELDS), secs


def assemble(g, raw, tested, causal):
    """Converged tested rows in the output layout, and the counts of what was excluded; causal None if null."""
    d = raw.astype({c: float for c in NUM})
    pseudo = d.rs_id == f'{g}_pseudo_fsnp'
    stray = ~pseudo & ~d.rs_id.isin(tested)
    if stray.any():
        raise SystemExit(f'{g}: RASQUAL rows for untested ids, e.g. {d.rs_id[stray].iloc[0]}')
    t = d[~pseudo]
    ok = t[t.convergence == 0]
    pi = ok.effect_size_pi.values
    if not ((pi > 0) & (pi < 1)).all():
        raise SystemExit(f'{g}: converged pi outside (0, 1): {pi[(pi <= 0) | (pi >= 1)][:3]}')
    slope = np.log2(pi / (1 - pi))
    c2 = ok.chisq.values
    se = np.full(len(ok), np.nan)
    se[c2 > 0] = np.abs(slope[c2 > 0]) / np.sqrt(c2[c2 > 0])
    out = pd.DataFrame(dict(phenotype_id=g, variant_id=ok.rs_id.astype(str).values, slope=slope, slope_se=se,
                            pval_nominal=chi2.sf(c2, 1), chisq=c2, pi=pi, delta=ok.error_rate_delta.values,
                            phi=ok.ref_mapping_bias_phi.values, theta=ok.overdispersion_theta.values,
                            n_feature_snps=ok.n_feature_snps.values.astype(int),
                            r2_rsnp=ok.r2_prior_posterior_rsnp.values))
    return out, dict(pseudo=int(pseudo.sum()), nonconv=int(len(t) - len(ok)), absent=len(tested) - len(t),
                     chisq_le0=int((c2 <= 0).sum()), no_fsnp=int((d.n_feature_snps == 0).all()),
                     causal_nonconv=int(causal in set(t.rs_id[t.convergence != 0])),
                     causal_absent=int(causal is not None and causal not in set(t.rs_id)))


def causal_check(df, ds, S):
    """RASQUAL's chisq rank (ties share the smallest rank) and sign at each smoke gene's causal variant."""
    ok = 0
    for g in SMOKE_GENES:
        k = S['genes'].index(g)
        a = df[df.phenotype_id == g]
        c = a[a.variant_id == ds['causal_variant'][k]]
        if c.empty:
            raise SystemExit(f'{g}: no converged RASQUAL row at the causal variant {ds["causal_variant"][k]}')
        c = c.iloc[0]
        rank = 1 + int((np.round(a.chisq * TIE) > np.round(c.chisq * TIE)).sum())
        good = rank <= RANK_MAX and np.sign(c.slope) == np.sign(ds['beta'][k])
        ok += good
        top = a.loc[a.chisq.idxmax()]
        print(f'  {g}: causal {c.variant_id} beta {ds["beta"][k]:+.1f}: rank {rank} of {len(a)}, chisq {c.chisq:.2f}, '
              f'slope {c.slope:+.3f}, phi {c.phi:.3f}; lead {top.variant_id} chisq {top.chisq:.2f}; pass {good}')
    print(f'causal rank <= {RANK_MAX} with the true sign in {ok} of {len(SMOKE_GENES)} genes (want all)')
    if ok < len(SMOKE_GENES):
        raise SystemExit('smoke causal check failed')


def main():
    if not os.access(RASQUAL, os.X_OK):
        raise SystemExit(f'{RASQUAL}: missing or not executable')
    print(f'{RASQUAL} sha256 {hashlib.sha256(RASQUAL.read_bytes()).hexdigest()}', flush=True)
    S = load()
    meta = json.loads((DATASETS / 'meta.json').read_text())
    if S['genes'] != meta['genes'] or S['order'] != meta['donors']:
        raise SystemExit(f'{DATASETS / "meta.json"}: genes or donors differ from the loader inputs')
    genes = list(SMOKE_GENES) if SMOKE else S['genes']
    runs = [('beta0.8', 0)] if SMOKE else [(f'beta{b}', r) for b in meta['betas']
                                           for r in range(meta['n_datasets'][str(b)])]
    out = OUT / 'smoke' if SMOKE else OUT
    text = {g: rsnp_text(S, g) for g in genes}
    sites = {g: pseudo_site(S, g) for g in genes}
    n_t = S['n_tested'][genes]
    kk = [S['genes'].index(g) for g in genes]
    print(f'{len(runs)} datasets x {len(genes)} genes, {JOBS} jobs; tested variants per gene '
          f'{n_t.min()}-{n_t.max()}, {int(n_t.sum()):,} in all', flush=True)
    secs, jobs = [], {}
    ex = cf.ThreadPoolExecutor(JOBS)
    try:
        for sc, r in runs:
            ds = dict(np.load(DATASETS / sc / f'rep{r:03d}.npz'))
            kept = MD.allelic_kept(ds['pL'], ds['pR'], ds['Va'])
            bins, n_cov = write_bins(S, ds, out / sc / ARM / f'inputs_rep{r:03d}')
            print(f'{sc} rep {r:03d}: {n_cov} covariates; pseudo fSNP '
                  f'{admission(ds["pL"][kk], ds["pR"][kk], kept[kk])}', flush=True)
            jobs[(sc, r)] = (ds, [ex.submit(run_gene, k, g, sites[g],
                                            pseudo_line(g, sites[g], ds['pL'][k], ds['pR'][k], kept[k]) + text[g],
                                            bins, len(S['order'])) for k, g in zip(kk, genes)])
        for (sc, r), (ds, futs) in jobs.items():
            parts, cnt = [], {}
            for k, g, f in zip(kk, genes, futs):
                raw, s = f.result()
                secs.append((s, len(S['tested'][g])))
                df, c = assemble(g, raw, S['tested'][g], None if ds['is_null'][k] else str(ds['causal_variant'][k]))
                parts.append(df)
                for key, v in c.items():
                    cnt[key] = cnt.get(key, 0) + v
            df = pd.concat(parts, ignore_index=True)
            RA.write_parquet(df, out / sc / ARM / f'nominal_rep{r:03d}.parquet', RA.fingerprint(ds, ARM), 'log2')
            print(f'{sc} rep {r:03d}: {len(df):,} rows written; excluded non-converged {cnt["nonconv"]}, pseudo-fSNP '
                  f'rows {cnt["pseudo"]}; tested variants with no RASQUAL row {cnt["absent"]}; chisq <= 0 (slope_se '
                  f'NaN) {cnt["chisq_le0"]}; genes where RASQUAL did not admit the pseudo fSNP {cnt["no_fsnp"]}; '
                  f'non-null causal variants without a row: non-converged {cnt["causal_nonconv"]}, absent '
                  f'{cnt["causal_absent"]} (of {int((~ds["is_null"][kk]).sum())})', flush=True)
    finally:
        ex.shutdown(cancel_futures=True)   # a failed gene cancels the queue; running RASQUAL processes still finish
    s = np.array(secs)
    rate, per = s[:, 0].sum() / s[:, 1].sum(), 1e3 * s[:, 0] / s[:, 1]
    print(f'seconds per gene median {np.median(s[:, 0]):.1f} [{s[:, 0].min():.1f}, {s[:, 0].max():.1f}]; '
          f'{1e3 * rate:.2f} ms per tested variant pooled, per gene {per.min():.1f}-{per.max():.1f}')
    if SMOKE:
        causal_check(df, ds, S)
        n_full = len(pd.read_parquet(RA.RESULTS / 'beta0.8' / 'split' / 'nominal_rep000.parquet', columns=['slope']))
        n_ds = sum(meta['n_datasets'].values())
        print(f'full run: {n_full:,} tested gene-variant pairs x {n_ds} datasets at {1e3 * rate:.2f} ms each over '
              f'{JOBS} jobs = {n_full * n_ds * rate / JOBS / 3600:.2f} h')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
