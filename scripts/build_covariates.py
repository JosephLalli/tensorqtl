#!/usr/bin/env python3
"""
Build the covariate matrix for the hapmixQTL / RASQUAL comparison.

WHY COVARIATES ARE PASSED IN, NOT REGRESSED OUT FIRST
=====================================================
Both methods take covariates natively and both would be wrong if handed a
pre-residualized phenotype.

hapmixQTL projects covariates out INSIDE the weighted space: WeightedResidualizer
scales them by sqrt(w) before the QR, and the allelic and total channels use
DIFFERENT weights (sqrt_wa, sqrt_wt). An unweighted pre-regression projects onto
the wrong subspace -- the residual is not orthogonal to the covariates under the
WLS inner product -- and one pre-regression cannot serve two differently
weighted channels.

RASQUAL fits a GLM on the COUNT scale (-x/--covariates, main.c:287). Residuals
of log counts are not counts, so pre-regression breaks the negative-binomial
model it is built on.

THE ORTHOGONALITY REQUIREMENT
=============================
Expression PCs are computed on expression that has ALREADY been residualized
against the genotype PCs and the metadata covariates. Otherwise the leading
expression PCs simply re-encode age, batch and ancestry -- they are the largest
sources of expression variance in a developmental cohort -- and the design
matrix carries the same signal twice, inflating its condition number and
splitting the effect across collinear columns. Residualizing first makes the
expression PCs orthogonal to the rest by construction, so they capture only
structure nothing else explains.

Run:
  python3 scripts/build_covariates.py --selftest
  python3 scripts/build_covariates.py --metadata v1.4.tsv --pairing pairing.tsv \\
      --salmon salmon.tsv --tx2gene tx2gene.tsv --vcf analysis.vcf.gz \\
      --out cov/
"""

import argparse
import csv
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _open(p):
    return gzip.open(p, 'rt') if str(p).endswith('.gz') else open(p)


def _sex_code(v):
    """1 male, 0 female, NaN unknown. BrainVar writes karyotypes (XX/XY), not
    M/F, and a parser that only knew M/F silently produced an all-NaN column
    that propagated into the QR and made the SVD fail to converge."""
    v = (v or '').strip().upper()
    if v in ('XY', 'M', 'MALE', '1'):
        return 1.0
    if v in ('XX', 'F', 'FEMALE', '0'):
        return 0.0
    return np.nan


def metadata_covariates(metadata, pairing, min_level_n=2):
    """DataFrame indexed by DNA library: age, age^2, RIN, sex, batch."""
    rows = [l.split('\t') for l in Path(pairing).read_text().strip().split('\n')]
    if rows and rows[0][0].startswith('dna'):
        rows = rows[1:]
    dna2rna = {r[0].strip(): r[1].strip() for r in rows if len(r) >= 2}
    meta = {}
    with open(metadata) as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            meta[(r.get('LibraryID') or '').strip()] = r
    out = {}
    for dna, rna in dna2rna.items():
        m = meta.get(rna, {})
        g = lambda k: (m.get(k) or '').strip()
        def num(k):
            try:
                return float(g(k))
            except ValueError:
                return np.nan
        age = num('AgeDays')
        sex = g('SexForAnalysis').upper()
        # Two independent batch axes, and the RNA one is what matters for
        # expression. The DNA suffix (_D1 GeneWiz / _D2 YCGA) is the WGS batch;
        # the RNA suffix (_R1 / _R2) is the RNA-seq batch. A sample can be _D1
        # and _R2, so keying batch off the DNA library -- as this first did --
        # models the wrong technical factor for a phenotype that is expression.
        batch = 1.0 if dna.endswith('_D2') else 0.0
        batch_rna = 1.0 if rna.endswith('_R2') else 0.0
        out[dna] = {
            'age_days': age,
            'age_days_sq': age * age if age == age else np.nan,
            'rin': num('RIN'),
            'sex': _sex_code(sex),
            'batch_bv2': batch,
            'batch_rna2': batch_rna,
        }
    df = pd.DataFrame(out).T
    # mean-impute rather than drop: losing a whole donor to one missing RIN is
    # a worse trade than a slightly attenuated covariate.
    # An all-NaN column cannot be mean-imputed; drop it and say so rather
    # than letting NaN reach the QR.
    dead = [c for c in df.columns if df[c].isna().all()]
    for c in dead:
        print(f'  dropping {c}: no usable values for any sample')
    df = df.drop(columns=dead)
    n_missing = int(df.isna().sum().sum())
    df = df.fillna(df.mean(numeric_only=True))
    if n_missing:
        print(f'  {n_missing} missing metadata values mean-imputed')
    # A constant column is collinear with the intercept: it carries no
    # information and costs a degree of freedom. Here the whole cohort is
    # BrainVar1, so the BV1/BV2 batch term is constant and is dropped.
    # An indicator whose minority level holds fewer than min_level_n samples
    # cannot be estimated: with a single sample it absorbs that sample
    # entirely, fitting its residual exactly, so the sample contributes nothing
    # to the covariate-adjusted channel AND a degree of freedom is spent. Keep
    # the sample, drop the column. The BrainVar cohort here is 91 _R1 against
    # one _R2, which is exactly that case.
    thin = []
    for c in df.columns:
        vc = df[c].value_counts()
        if len(vc) == 2 and vc.min() < min_level_n:
            lone = list(df.index[df[c] == vc.idxmin()])
            thin.append(c)
            print(f'  dropping {c}: minority level has {vc.min()} sample(s) '
                  f'({", ".join(lone[:3])}) -- not estimable; the sample is '
                  'kept, the column is not')
    df = df.drop(columns=thin)
    const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    for c in const:
        print(f'  dropping {c}: constant across all {len(df)} samples '
              '(collinear with the intercept)')
    return df.drop(columns=const)


def genotype_pcs(vcf, samples, n_pc=3, max_variants=60000, seed=0):
    """PCs of the dosage matrix, thinned while streaming so the whole VCF is
    never materialised."""
    want = set(samples)
    keep_idx = order = None
    rows, seen = [], 0
    stride = None
    with _open(vcf) as fh:
        for line in fh:
            if line.startswith('##'):
                continue
            f = line.rstrip('\n').split('\t')
            if line.startswith('#CHROM'):
                vs = f[9:]
                keep_idx = [i for i, s in enumerate(vs) if s in want]
                order = [vs[i] for i in keep_idx]
                continue
            seen += 1
            if stride is None and seen > 200000:
                stride = 1                      # decided below on the fly
            if seen % 37:                       # coprime thinning, no clustering
                continue
            if len(f) < 10 or len(f[3]) != 1 or len(f[4]) != 1:
                continue
            gt_i = f[8].split(':').index('GT') if 'GT' in f[8] else 0
            d = np.empty(len(keep_idx), np.float32)
            ok = True
            for k, i in enumerate(keep_idx):
                g = f[9 + i].split(':')[gt_i].replace('|', '/')
                a, b = g.split('/')[:2]
                if a == '.' or b == '.':
                    ok = False
                    break
                d[k] = (a != '0') + (b != '0')
            if ok:
                rows.append(d)
            if len(rows) >= max_variants:
                break
    if not rows:
        raise SystemExit(f'no usable variants read from {vcf} for genotype PCs')
    G = np.asarray(rows, np.float32)            # [variants, samples]
    G = G[G.std(1) > 0]
    G = (G - G.mean(1, keepdims=True)) / G.std(1, keepdims=True)
    # PCs of the sample covariance; SVD on the standardised matrix
    U, S, Vt = np.linalg.svd(G / np.sqrt(G.shape[0]), full_matrices=False)
    pcs = Vt[:n_pc].T                           # [samples, n_pc]
    df = pd.DataFrame(pcs, index=order,
                      columns=[f'geno_pc{i+1}' for i in range(n_pc)])
    print(f'  genotype PCs from {G.shape[0]} thinned variants')
    return df.loc[list(samples)]


def residualize(Y, C):
    """Project columns of C out of the rows of Y (Y: [features, samples])."""
    X = np.column_stack([np.ones(C.shape[0]), C])
    Q, _ = np.linalg.qr(X)
    return Y - (Y @ Q) @ Q.T


def expression_pcs(expr, base, n_pc=10, min_samples_expressed=0.5, seed=0):
    """PCs of expression AFTER removing `base`, so they are orthogonal to it."""
    Y = np.asarray(expr, float)                 # [genes, samples]
    keep = (Y > 0).mean(1) >= min_samples_expressed
    Y = Y[keep]
    if Y.shape[0] < n_pc:
        raise SystemExit(f'only {Y.shape[0]} genes pass the expression filter; '
                         f'cannot take {n_pc} PCs')
    Y = np.log1p(Y)
    Y = (Y - Y.mean(1, keepdims=True))
    sd = Y.std(1, keepdims=True); sd[sd == 0] = 1.0
    Y = Y / sd
    Yr = residualize(Y, np.asarray(base, float))
    U, S, Vt = np.linalg.svd(Yr / np.sqrt(Yr.shape[0]), full_matrices=False)
    pcs = Vt[:n_pc].T
    var = (S ** 2 / (S ** 2).sum())[:n_pc]
    print(f'  expression PCs from {Y.shape[0]} genes; '
          f'top-{n_pc} residual variance {100*var.sum():.1f}%')
    return pcs


def build(metadata, pairing, salmon, tx2gene, vcf, n_expr_pc=10, n_geno_pc=3,
          hap_suffix=('_L', '_R'), seed=0, min_level_n=2):
    meta = metadata_covariates(metadata, pairing, min_level_n=min_level_n)
    samples = list(meta.index)
    gpc = genotype_pcs(vcf, samples, n_pc=n_geno_pc, seed=seed)
    base = pd.concat([meta, gpc], axis=1).loc[samples]
    sys.path.insert(0, str(Path(__file__).parent))
    from make_rasqual_inputs import read_salmon_totals
    expr = read_salmon_totals(salmon, tx2gene, samples, hap_suffix)
    epc = expression_pcs(expr.values, base.values, n_pc=n_expr_pc, seed=seed)
    epc = pd.DataFrame(epc, index=samples,
                       columns=[f'expr_pc{i+1}' for i in range(n_expr_pc)])
    return pd.concat([base, epc], axis=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--metadata'); ap.add_argument('--pairing')
    ap.add_argument('--salmon'); ap.add_argument('--tx2gene'); ap.add_argument('--vcf')
    ap.add_argument('--n-expr-pc', type=int, default=10)
    ap.add_argument('--n-geno-pc', type=int, default=3)
    ap.add_argument('--min-level-n', type=int, default=2,
                    help='drop an indicator whose minority level has fewer '
                         'than this many samples: it is not estimable and '
                         'absorbs those samples entirely (default 2)')
    ap.add_argument('--hap-suffix', default='_L,_R')
    ap.add_argument('--out', default='cov')
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    for r in ('metadata', 'pairing', 'salmon', 'tx2gene', 'vcf'):
        if not getattr(a, r):
            raise SystemExit(f'--{r} is required (or --selftest)')
    C = build(a.metadata, a.pairing, a.salmon, a.tx2gene, a.vcf,
              a.n_expr_pc, a.n_geno_pc, tuple(a.hap_suffix.split(',')),
              min_level_n=a.min_level_n)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    C.to_csv(out / 'covariates.tsv', sep='\t')
    # RASQUAL's -x is COVARIATE-major, not sample-major: it reads into X and
    # then takes mean(X + N*i, N), i.e. N consecutive doubles are ONE covariate
    # across all samples (main.c:301-307). Writing [samples, covariates] would
    # transpose the design silently and fit the wrong model, so transpose here.
    np.asarray(C.values.T, np.float64).tofile(out / 'covariates.bin')
    (out / 'covariates.n').write_text(f'{C.shape[1]}\n')
    print(f'{C.shape[0]} samples x {C.shape[1]} covariates -> {out}/')
    print('  ' + ', '.join(C.columns))
    print(f'pass --covariates {out}/covariates.tsv to the comparison; it '
          f'feeds hapmixQTL directly and RASQUAL as -x covariates.bin')
    print('  (RASQUAL derives the covariate count from the file itself at '
          'main.c:299; -p is NOT passed, because that parser runs after the '
          'allocation and would overwrite the derived value)')
    return 0


def selftest():
    import tempfile
    rng = np.random.RandomState(0)
    td = Path(tempfile.mkdtemp())
    N, G = 40, 300
    dna = [f'{100+i}_D{1 if i % 4 else 2}' for i in range(N)]
    rna = [f'{100+i}_R1' for i in range(N)]
    (td / 'pair.tsv').write_text('dna_library\trna_library\tbam\n' +
                                 '\n'.join(f'{d}\t{r}\tB{i}'
                                           for i, (d, r) in enumerate(zip(dna, rna))) + '\n')
    age = rng.uniform(50, 300, N)
    with open(td / 'meta.tsv', 'w') as fh:
        fh.write('LibraryID\tLibraryModality\tSexForAnalysis\tAgeDays\tRIN\n')
        for i, r in enumerate(rna):
            fh.write(f'{r}\tbulkRNA\t{"M" if i%2 else "F"}\t{age[i]:.1f}\t'
                     f'{rng.uniform(5,9):.1f}\n')
    meta = metadata_covariates(td / 'meta.tsv', td / 'pair.tsv')
    assert list(meta.index) == dna, meta.index
    assert np.allclose(meta['age_days_sq'].values,
                       meta['age_days'].values ** 2), 'age^2 must be age squared'
    assert meta['batch_bv2'].sum() == sum(1 for d in dna if d.endswith('_D2'))
    # karyotype encoding, and a constant column must be dropped not carried
    assert _sex_code('XY') == 1.0 and _sex_code('XX') == 0.0
    assert _sex_code('M') == 1.0 and _sex_code('female') == 0.0
    assert np.isnan(_sex_code('')) and np.isnan(_sex_code('?'))
    one_batch = [f'{100+i}_D1' for i in range(6)]
    (td / 'p1.tsv').write_text('dna_library\trna_library\tbam\n' +
        '\n'.join(f'{d}\t{100+i}_R1\tB' for i, d in enumerate(one_batch)) + '\n')
    # A _D1 sample whose RNA library is _R2 must register on the RNA axis and
    # NOT the DNA axis. Keying batch off the DNA suffix would have missed it.
    (td / 'p2.tsv').write_text(
        'dna_library\trna_library\tbam\n'
        + '\n'.join(f'{200+i}_D1\t{200+i}_R{2 if i < 3 else 1}\tB'
                     for i in range(6)) + '\n')
    with open(td / 'm2.tsv', 'w') as fh:
        fh.write('LibraryID\tSexForAnalysis\tAgeDays\tRIN\n')
        for i in range(6):
            fh.write(f'{200+i}_R{2 if i < 3 else 1}\tXY\t{120+i}\t7.5\n')
    m2 = metadata_covariates(td / 'm2.tsv', td / 'p2.tsv')
    assert 'batch_rna2' in m2.columns, 'RNA batch must survive when it varies'
    assert m2['batch_rna2'].sum() == 3, m2['batch_rna2'].tolist()
    # ... but a single-sample level is not estimable and must be dropped,
    # WITHOUT dropping the sample
    (td / 'p3.tsv').write_text(
        'dna_library\trna_library\tbam\n'
        + '\n'.join(f'{300+i}_D1\t{300+i}_R{2 if i == 0 else 1}\tB'
                     for i in range(8)) + '\n')
    with open(td / 'm3.tsv', 'w') as fh:
        fh.write('LibraryID\tSexForAnalysis\tAgeDays\tRIN\n')
        for i in range(8):
            fh.write(f'{300+i}_R{2 if i == 0 else 1}\tXY\t{130+i}\t7.0\n')
    m3 = metadata_covariates(td / 'm3.tsv', td / 'p3.tsv')
    assert 'batch_rna2' not in m3.columns, 'singleton level must be dropped'
    assert len(m3) == 8, 'the SAMPLE must be kept, only the column dropped'
    assert 'batch_bv2' not in m2.columns, \
        'DNA batch is constant here and must be dropped, not confused with RNA'
    with open(td / 'm1.tsv', 'w') as fh:
        fh.write('LibraryID\tSexForAnalysis\tAgeDays\tRIN\n')
        for i in range(6):
            fh.write(f'{100+i}_R1\tXX\t{100+i}\t7.0\n')
    m1 = metadata_covariates(td / 'm1.tsv', td / 'p1.tsv')
    assert 'batch_bv2' not in m1.columns, 'constant batch must be dropped'
    assert 'batch_rna2' not in m1.columns, 'constant RNA batch must be dropped'
    assert 'sex' not in m1.columns, 'constant sex must be dropped'
    assert (m1['age_days'].values == np.arange(100, 106)).all()
    # genotype PCs from a fabricated VCF with two ancestry-like clusters
    grp = (np.arange(N) % 2)
    with open(td / 'g.vcf', 'w') as fh:
        fh.write('##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER'
                 '\tINFO\tFORMAT\t' + '\t'.join(dna) + '\n')
        for v in range(4000):
            p = np.where(grp == 0, 0.2, 0.8) if v % 3 == 0 else np.full(N, 0.5)
            gt = (rng.rand(N) < p).astype(int) + (rng.rand(N) < p).astype(int)
            cells = ['|'.join(['1' if x > 0 else '0',
                               '1' if x > 1 else '0']) for x in gt]
            fh.write(f'1\t{v*100+1}\tv{v}\tA\tG\t.\tPASS\t.\tGT\t' + '\t'.join(cells) + '\n')
    gpc = genotype_pcs(td / 'g.vcf', dna, n_pc=3, max_variants=4000)
    assert gpc.shape == (N, 3), gpc.shape
    r = abs(np.corrcoef(gpc['geno_pc1'].values, grp)[0, 1])
    assert r > 0.5, f'PC1 should track the planted structure, got r={r:.2f}'
    # expression PCs must come out orthogonal to the base covariates
    base = pd.concat([meta, gpc], axis=1)
    latent = rng.randn(N)
    expr = (rng.gamma(2, 50, size=(G, N))
            + 40 * np.outer(rng.randn(G), base['age_days'].values / 100.0)
            + 40 * np.outer(rng.randn(G), latent))
    expr = np.clip(expr, 0, None)
    epc = expression_pcs(expr, base.values, n_pc=5)
    B = np.column_stack([np.ones(N), base.values])
    resid = epc - B @ np.linalg.lstsq(B, epc, rcond=None)[0]
    assert np.allclose(resid, epc, atol=1e-8), \
        'expression PCs are not orthogonal to the base covariates'
    worst = max(abs(np.corrcoef(epc[:, k], base.values[:, j])[0, 1])
                for k in range(epc.shape[1]) for j in range(base.shape[1]))
    assert worst < 1e-6, f'max |corr| with a base covariate is {worst:.2e}'
    print('SELF-TEST: covariate construction\n')
    # the binary must be covariate-major for RASQUAL
    import tempfile as _tf
    bt = Path(_tf.mkdtemp()) / 'c.bin'
    demo = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [10.0, 20.0, 30.0]})
    np.asarray(demo.values.T, np.float64).tofile(bt)
    back = np.fromfile(bt, np.float64)
    assert back[:3].tolist() == [1.0, 2.0, 3.0], back
    assert back[3:].tolist() == [10.0, 20.0, 30.0], back
    print('checks: age^2 is age squared; batch read from BOTH the _D1/_D2 (WGS) '
          'and _R1/_R2 (RNA) library suffixes, which vary independently; genotype PC1 recovers planted population structure; and '
          f'expression PCs are orthogonal to genotype PCs + metadata '
          f'(max |corr| {worst:.1e}) rather than re-encoding them; and the '
          'RASQUAL binary is covariate-major as main.c reads it')
    print('SELF-TEST OK')
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
