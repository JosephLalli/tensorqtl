"""
Head-to-head against the REAL RASQUAL binary (Kumasaka et al. 2016).

Sections 7h/7i used `RASQUAL-like`, our own reimplementation. This runs the
actual package, built by `scripts/build_rasqual.sh`.

WHY THE SIMULATOR HAD TO CHANGE
-------------------------------
The earlier harnesses collapsed the allele-specific signal into one aggregate
count per sample. RASQUAL does not work that way, and the difference is the
whole reason its phi is identifiable:

  rSNP  the *regulatory* SNP being tested -- carries the genetic effect
  fSNP  a *feature* SNP inside the gene body -- carries the allele-specific
        READS, and is only informative in samples heterozygous for it

RASQUAL's supplement puts it exactly: the allelic split is
p(Y1_il | Y_il, D_il; theta, pi, phi, delta) "where D_il is determined by the
combination of G_i and G_il" -- the rSNP genotype AND the fSNP genotype.

Samples heterozygous at an fSNP but HOMOZYGOUS at the rSNP carry allelic counts
with no genetic effect. Those samples are what separate reference bias from the
eQTL effect. Simulating a single SNP that is both rSNP and fSNP would leave phi
and kappa confounded -- the same bug found in our reimplementation (docs sec 7i).
So this simulator generates two haplotypes with an rSNP and several fSNPs in LD.

INPUT FORMAT (verified against the authors' bundled data)
---------------------------------------------------------
  Y.bin  feature x sample read counts, float64, row-major, no names
  K.bin  same shape, sample-specific offsets
  VCF    phased GT plus an "AS" subfield holding "ref,alt" counts, e.g. GT:AS
         with 0|1:1,10. fSNPs are the SNPs falling inside the -s/-e intervals.

Run:  python3 tests/ase_rasqual_real.py --reps 100 \
          --rasqual /path/to/rasqual_src/src/rasqual
"""

import argparse
import json
import subprocess
import sys
import tempfile
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from scipy import stats
from ase_external_benchmark import hapmix_pval, trecase_lrt, trec_lrt

CHROM, FEAT_START, FEAT_END = '1', 1000, 3000
RSNP_POS = 5000                     # regulatory SNP, outside the gene body


def simulate_locus(N, rng, kappa=1.0, n_fsnp=3, maf=0.35, mu=200.0,
                   theta=0.2, as_frac=0.25, lib_sd=0.25, phi=0.5, ld=0.6):
    """Two haplotypes carrying one rSNP and several fSNPs in LD.

    Returns everything both RASQUAL and hapmixQTL need from the same truth.
    """
    # rSNP haplotype alleles
    h1r = (rng.rand(N) < maf).astype(int)
    h2r = (rng.rand(N) < maf).astype(int)
    # fSNP alleles correlated with the rSNP on the same haplotype (LD)
    def linked(h):
        keep = rng.rand(N) < ld
        return np.where(keep, h, (rng.rand(N) < maf).astype(int))
    h1f = [linked(h1r) for _ in range(n_fsnp)]
    h2f = [linked(h2r) for _ in range(n_fsnp)]

    # haplotype-specific expression: ALT allele at the rSNP multiplies by kappa
    e1 = np.where(h1r == 1, kappa, 1.0)
    e2 = np.where(h2r == 1, kappa, 1.0)
    lib = np.exp(rng.normal(0, lib_sd, N))
    mean_t = lib * mu * (e1 + e2) / 2.0
    r = 1.0 / max(theta, 1e-8)
    T = rng.negative_binomial(r, r / (r + mean_t)).astype(float)

    n_as_tot = rng.binomial(T.astype(int), as_frac)
    # split the allele-specific reads across fSNPs
    as_ref = np.zeros((n_fsnp, N), int)
    as_alt = np.zeros((n_fsnp, N), int)
    for i in range(N):
        if n_as_tot[i] <= 0:
            continue
        per = rng.multinomial(n_as_tot[i], np.ones(n_fsnp) / n_fsnp)
        for k in range(n_fsnp):
            if per[k] == 0 or h1f[k][i] == h2f[k][i]:
                continue                       # homozygous fSNP: uninformative
            # fraction of reads from haplotype 1, driven by the rSNP effect
            p1 = e1[i] / (e1[i] + e2[i])
            # reference bias favours the REF allele
            p_alt_hap = p1 if h1f[k][i] == 1 else (1 - p1)
            num = (1 - phi) * p_alt_hap
            p_alt = num / max(num + phi * (1 - p_alt_hap), 1e-12)
            a_ = max(p_alt / theta, 1e-6); b_ = max((1 - p_alt) / theta, 1e-6)
            alt = rng.binomial(per[k], rng.beta(a_, b_))
            as_alt[k, i] = alt; as_ref[k, i] = per[k] - alt

    g = h1r + h2r
    s = np.where(h1r != h2r, np.where(h1r == 1, 1.0, -1.0), 0.0)
    # hapmixQTL consumes aggregated haplotype counts
    yL = np.zeros(N); yR = np.zeros(N)
    for k in range(n_fsnp):
        yL += np.where(h1f[k] == 1, as_alt[k], as_ref[k]) * (h1f[k] != h2f[k])
        yR += np.where(h2f[k] == 1, as_alt[k], as_ref[k]) * (h1f[k] != h2f[k])
    het = g == 1
    return dict(g=g.astype(float), s=s, het=het, T=T, lib=lib,
                yL=yL, yR=yR, n_as=(yL + yR),
                y_alt=np.where(s >= 0, yL, yR),
                h1r=h1r, h2r=h2r, h1f=h1f, h2f=h2f,
                as_ref=as_ref, as_alt=as_alt, n_fsnp=n_fsnp)


def write_rasqual_inputs(loci, outdir):
    """Y.bin / K.bin / VCF in RASQUAL's format."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    N = len(loci[0]['g'])
    Y = np.array([d['T'] for d in loci], dtype=np.float64)          # feat x samp
    K = np.array([d['lib'] * d['T'].mean() for d in loci], dtype=np.float64)
    Y.tofile(outdir / 'Y.bin'); K.tofile(outdir / 'K.bin')

    samples = [f'S{i:04d}' for i in range(N)]
    vcf_rows = []
    for li, d in enumerate(loci):
        # each locus gets its own contig so windows never overlap
        contig = str(li + 1)
        # fSNPs inside the feature body
        for k in range(d['n_fsnp']):
            pos = FEAT_START + 200 * (k + 1)
            fields = []
            for i in range(N):
                gt = f"{d['h1f'][k][i]}|{d['h2f'][k][i]}"
                fields.append(f"{gt}:{d['as_ref'][k, i]},{d['as_alt'][k, i]}")
            vcf_rows.append([contig, str(pos), f'f{li}_{k}', 'A', 'G', '100',
                             'PASS', 'RSQ=1.0', 'GT:AS'] + fields)
        # the tested rSNP, outside the body, no AS reads of its own
        fields = [f"{d['h1r'][i]}|{d['h2r'][i]}:0,0" for i in range(N)]
        vcf_rows.append([contig, str(RSNP_POS), f'r{li}', 'A', 'G', '100',
                         'PASS', 'RSQ=1.0', 'GT:AS'] + fields)

    vcf = outdir / 'in.vcf'
    with open(vcf, 'w') as fh:
        fh.write('##fileformat=VCFv4.1\n')
        fh.write('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t'
                 + '\t'.join(samples) + '\n')
        for row in vcf_rows:
            fh.write('\t'.join(row) + '\n')
    return outdir, N


def run_rasqual(binary, outdir, N, loci):
    """One RASQUAL call per locus; returns the chi-square statistic each."""
    stats_out = []
    vcf_lines = {}
    for line in open(Path(outdir) / 'in.vcf'):
        if line.startswith('#'):
            continue
        vcf_lines.setdefault(line.split('\t', 1)[0], []).append(line)
    for li, d in enumerate(loci):
        contig = str(li + 1)
        lines = vcf_lines[contig]
        n_fsnp = d['n_fsnp']
        cmd = [binary, '-y', str(Path(outdir) / 'Y.bin'),
               '-k', str(Path(outdir) / 'K.bin'), '-n', str(N),
               '-j', str(li + 1), '-l', str(len(lines)), '-m', str(n_fsnp),
               '-s', str(FEAT_START), '-e', str(FEAT_END),
               # NOTE: no -t. "-t/--lead-snp  Output only the lead QTL SNP"
               # would give a multiple-testing-inflated lead statistic; every
               # other method here does a single-SNP test at the rSNP.
               '-f', f'gene{li}', '-z']
        try:
            p = subprocess.run(cmd, input=''.join(lines), capture_output=True,
                               text=True, timeout=120)
        except Exception:
            stats_out.append(np.nan); continue
        # Take the statistic AT THE rSNP, not the max over the region: every
        # other method here performs a single-SNP test, so a lead-SNP maximum
        # would be a different (multiple-testing-inflated) statistic.
        # Column 11 (1-indexed) is "Chi square statistic (2 x log Likelihood
        # ratio)"; column 23 is "Convergence status (0=success)".
        val = np.nan
        for line in p.stdout.strip().split('\n'):
            f = line.split('\t')
            if len(f) < 25 or f[1] == 'SKIPPED':
                continue
            if f[1] != f'r{li}':
                continue
            try:
                if int(float(f[22])) != 0:      # did not converge
                    continue
                val = float(f[10])
            except ValueError:
                continue
        stats_out.append(val)
    return np.array(stats_out, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rasqual', required=True, help='path to the rasqual binary')
    ap.add_argument('--reps', type=int, default=100)
    ap.add_argument('--N', type=int, default=100)
    ap.add_argument('--kappas', type=str, default='1.25')
    ap.add_argument('--phis', type=str, default='0.5,0.6')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    res = {'config': vars(args), 'scenarios': {}}

    for phi in [float(x) for x in args.phis.split(',')]:
        print(f"\n{'='*70}\nphi = {phi}   (N={args.N}, {args.reps} loci)\n{'='*70}")
        cells = {}
        for label, kappa in [('null', 1.0)] + [
                (f'aFC {k}', float(k)) for k in args.kappas.split(',')]:
            loci = [simulate_locus(args.N, np.random.RandomState(
                        7000 + int(phi * 100) * 1000 + i),
                        kappa=kappa, phi=phi) for i in range(args.reps)]
            with tempfile.TemporaryDirectory() as td:
                od, N = write_rasqual_inputs(loci, td)
                rq = run_rasqual(args.rasqual, od, N, loci)
            hm = np.array([hapmix_pval(d, np.random.RandomState(i), 'estimate')[1]
                           for i, d in enumerate(loci)])
            tc = np.array([trecase_lrt(d)[1] for d in loci])
            tr = np.array([trec_lrt(d)[1] for d in loci])
            cells[label] = dict(RASQUAL=rq, hapmixQTL=hm, TReCASE=tc, trcQTL=tr)
        # matched empirical FPR 10% off the null cell
        out = {}
        for meth in ('trcQTL', 'TReCASE', 'RASQUAL', 'hapmixQTL'):
            nul = cells['null'][meth]; nul = nul[np.isfinite(nul)]
            if nul.size < 10:
                out[meth] = dict(note='insufficient RASQUAL output'); continue
            thr = np.quantile(nul, 0.90)
            row = dict(n_null=int(nul.size))
            for label in cells:
                if label == 'null':
                    continue
                v = cells[label][meth]; v = v[np.isfinite(v)]
                row[label] = float(np.mean(v > thr)) if v.size else float('nan')
            out[meth] = row
            print(f"  {meth:12s} n_null={row['n_null']:4d}  " +
                  "  ".join(f"{k}: power={row[k]:.3f}"
                            for k in row if k not in ('n_null', 'note')))
        res['scenarios'][str(phi)] = {k: {kk: (vv if not isinstance(vv, np.ndarray)
                                               else None)
                                          for kk, vv in v.items()}
                                      for k, v in out.items()}
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, default=float))
        print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
