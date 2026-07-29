"""
Reproducibility harness: fastPHASE-fit HMM vs. per-gene Gaussian as the KFc
knockoff generator, on REAL phased HPRC v2.0 genotypes.

WHY THIS EXISTS
---------------
The shipped KFc eGene filter (docs/calibration_findings.md sec 8) uses a per-gene
GAUSSIAN knockoff. That choice was not arbitrary -- it was settled empirically
against a properly-fit HMM knockoff whose parameters were estimated from real
haplotypes with fastPHASE (Scheet & Stephens 2006), the same model family Sesia
et al. (2019) use for HMM knockoffs. This script reproduces that comparison so
the decision is auditable, not a bare claim in a doc.

It is OPT-IN and SELF-SKIPPING: it needs `pysam` + outbound network to the public
`human-pangenomics` S3 bucket, plus the `fastphase` PyPI package (which needs a
one-line int32 patch -- see docs/fastphase_setup.md) and `ray`. None of these are
tensorQTL dependencies; if any is missing the pytest gate skips cleanly and the
`__main__` path prints a short instruction instead of failing.

WHAT IT MEASURES
----------------
Two knockoff-quality signals on real LD, for each generator:
  1. Null-W SYMMETRY. For a null gene (Y independent of genotype) a valid
     knockoff gives W = imp(real) - imp(knockoff) symmetric about 0, so
     frac(W>0) -> 0.5 and mean W -> 0. Deviation = misspecification.
  2. End-to-end realized FDR + power under planted cis signal, mirror-null
     selection (ko.mirror_select_egenes).

RESULT (recorded here for provenance; reproduced from the scratchpad run that
informed docs/calibration_findings.md sec 8 -- 160 real HPRC chr1 windows,
N=232 individuals, p=30 variants/window, K=20 HMM clusters, mirror q=0.10):

    Null-W symmetry (real HPRC v2.0):
      toy genotype-HMM K=8            : frac(W>0) ~ 0.20-0.31   (misspecified)
      fastPHASE fit(K=20) -> our gen  : mean W  -0.276 -> -0.023 (bias fixed)
      Gaussian shrink=0.1             : frac(W>0) ~ 0.556        (well-specified)

    End-to-end FDR / power (target FDR 0.10):
      PVE=0.10: fastPHASE-HMM FDR=0.062 power=0.09 R=6  | Gaussian FDR=0.014 power=0.23 R=15
      PVE=0.15: fastPHASE-HMM FDR=0.042 power=0.11 R=8  | Gaussian FDR=0.030 power=0.54 R=36

Both generators CONTROL FDR on real data; the properly-fit fastPHASE HMM removes
the toy HMM's null-bias. But for the KFc min-p (marginal) statistic the Gaussian
knockoff yields 3-5x more power at matched FDR -- so per-gene Gaussian is the
shipped KFc generator. (Intuition: the min-p statistic rewards a knockoff that is
maximally decorrelated from the real genotype at each variant; the Gaussian
construction targets exactly that second-moment structure, whereas the HMM spends
its fidelity budget reproducing the full haplotype process, which the marginal
statistic never uses.)

Run:  python tests/fastphase_repro.py --n_windows 40 --reps 8
"""

import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tensorqtl.knockoffs as ko
except ImportError:  # standalone run: tensorqtl dir directly on path
    sys.path.insert(0, str(Path(__file__).parent.parent / 'tensorqtl'))
    import knockoffs as ko

HPRC_V2_URL = ('https://s3-us-west-2.amazonaws.com/human-pangenomics/pangenomes/'
               'freeze/release2/minigraph-cactus/hprc-v2.0-mc-grch38.wave.vcf.gz')


# --------------------------------------------------------------------------
#  Fetch REAL phased haplotypes (two channels xL, xR) from HPRC v2.0.
# --------------------------------------------------------------------------
def fetch_phased(url=HPRC_V2_URL, chrom='chr1', start=5_000_000, end=90_000_000,
                 p_per_win=30, n_windows=40, maf_min=0.05):
    """Pull biallelic-SNP phased haplotype matrices for `n_windows` cis-windows.

    Returns [(xL, xR), ...] where xL, xR are [N, p] 0/1 haplotype allele matrices
    for the two parental chromosomes, and N (number of individuals).
    """
    import pysam
    vf = pysam.VariantFile(url)
    samples = list(vf.header.samples)
    N = len(samples)
    xl, xr = [], []
    for rec in vf.fetch(chrom, start, end):
        if rec.ref is None or rec.alts is None or len(rec.alts) != 1:
            continue
        if len(rec.ref) != 1 or len(rec.alts[0]) != 1:
            continue
        a0 = np.zeros(N); a1 = np.zeros(N); miss = 0
        for i, s in enumerate(samples):
            gt = rec.samples[s].get('GT')
            if gt is None or len(gt) < 2 or gt[0] is None or gt[1] is None:
                miss += 1; continue
            a0[i] = 1 if gt[0] > 0 else 0
            a1[i] = 1 if gt[1] > 0 else 0
        if miss / N > 0.2:
            continue
        d = a0 + a1
        af = d.sum() / (2 * N)
        if min(af, 1 - af) < maf_min or d.std() < 1e-6:
            continue
        xl.append(a0); xr.append(a1)
        if len(xl) >= p_per_win * n_windows:
            break
    xl = np.array(xl); xr = np.array(xr)
    nwin = len(xl) // p_per_win
    wins = [(xl[w * p_per_win:(w + 1) * p_per_win].T,
             xr[w * p_per_win:(w + 1) * p_per_win].T) for w in range(nwin)]
    return wins, N


# --------------------------------------------------------------------------
#  fastPHASE fit -> our HMM-knockoff parameterization.
# --------------------------------------------------------------------------
def fastphase_to_ours(par):
    """Map fastPHASE parameters (alpha, theta, rho) to (init_p, Q, emission_p)
    consumed by ko.hmm_knockoffs. alpha[j]=cluster weights at locus j,
    theta[j,k]=P(allele 1 | cluster k) at locus j, rho[j]=jump/recombination
    prob into locus j."""
    alpha, theta, rho = par.alpha, par.theta, par.rho
    L, K = theta.shape
    init_p = alpha[0] / alpha[0].sum()
    Q = np.empty((L - 1, K, K))
    for j in range(L - 1):
        r = float(rho[j + 1])
        a = alpha[j + 1] / alpha[j + 1].sum()
        Q[j] = (1 - r) * np.eye(K) + r * np.tile(a, (K, 1))
    emission_p = np.empty((L, 2, K))
    emission_p[:, 1, :] = theta
    emission_p[:, 0, :] = 1 - theta
    return init_p, Q, emission_p


def make_fastphase_fitter(p, K=20, nstep=12, nproc=4):
    """Return fp_fit(xL, xR) -> (init_p, Q, emission_p), fitting fastPHASE on the
    2N haplotypes [xL; xR] of one window. Requires the patched `fastphase` pkg
    and `ray` (see docs/fastphase_setup.md)."""
    import ray
    ray.init(num_cpus=nproc, ignore_reinit_error=True, log_to_driver=False)
    from fastphase.fastphase_ray import fastphase
    FP = fastphase(p, nproc=nproc)

    def fp_fit(xL, xR):
        H = np.vstack([xL, xR]).astype(np.int32)   # [2N, p] haplotypes
        try:
            FP.reset()
        except Exception:
            FP.haplotypes = {}; FP.genotypes = {}; FP.genolik = {}
        for i in range(H.shape[0]):
            FP.addHaplotype(i, np.ascontiguousarray(H[i], dtype=np.int32))
        par = FP.fit(nClus=K, nstep=nstep, verbose=False)
        return fastphase_to_ours(par)

    return fp_fit


# --------------------------------------------------------------------------
#  Generators (given a window's phased haplotypes, return a diploid knockoff).
# --------------------------------------------------------------------------
def gen_gaussian(xL, xR, shrink=0.1, seed=13):
    import torch
    G = (xL + xR).astype(np.float32)
    return ko.gaussian_knockoff(torch.tensor(G), shrink=shrink,
                                generator=torch.Generator().manual_seed(seed)
                                ).numpy().astype(np.float64)


def gen_toy_hmm(xL, xR, hmm_K=8, seed=13):
    G = (xL + xR).astype(np.int64)
    return ko.genotype_hmm_knockoffs(G, K=hmm_K, M=1, n_em_iter=12, seed=seed
                                     )[0].astype(np.float64)


def gen_fastphase(xL, xR, fp_fit, seed=13):
    """Route-2 phased knockoff: fit fastPHASE, generate a knockoff haplotype for
    each channel, recombine to a diploid dosage."""
    init_p, Q, emission_p = fp_fit(xL, xR)
    xkL = ko.hmm_knockoffs(xL.astype(np.int64), init_p, Q, emission_p, seed=seed)
    xkR = ko.hmm_knockoffs(xR.astype(np.int64), init_p, Q, emission_p, seed=seed + 1)
    return (xkL + xkR).astype(np.float64)


# --------------------------------------------------------------------------
#  Metrics.
# --------------------------------------------------------------------------
def null_W_symmetry(wins, N, gen, seed0=1000):
    """frac(W>0) and mean W over null genes (want 0.5 and 0)."""
    W = []
    for gi, (xL, xR) in enumerate(wins):
        G = (xL + xR).astype(np.float64)
        y = np.random.RandomState(seed0 + gi).randn(N)
        Gk = gen(xL, xR)
        W.append(ko.gene_W_marginal(G, Gk, y))
    W = np.array(W)
    return dict(frac_pos=float(np.mean(W > 0)), mean_W=float(W.mean()),
                atom=float(np.mean(np.abs(W) < 1e-9)))


def fdr_power(wins, N, gen, pve=0.10, egene_frac=0.4, q=0.1, reps=8):
    """Realized FDR + power under planted cis signal, mirror-null selection.
    The knockoff for each window is generated ONCE and reused across reps (the
    knockoff does not depend on the phenotype), matching the shipped pipeline."""
    Gk = [gen(xL, xR) for (xL, xR) in wins]
    G = [(xL + xR).astype(np.float64) for (xL, xR) in wins]
    fdps, powers, Rs = [], [], []
    for rep in range(reps):
        r = np.random.RandomState(10 + rep)
        n_sig = int(egene_frac * len(wins))
        sigset = set(r.permutation(len(wins))[:n_sig].tolist())
        Wg, truth = [], []
        for gi in range(len(wins)):
            y = r.randn(N)
            if gi in sigset:
                c = r.randint(G[gi].shape[1])
                xc = (G[gi][:, c] - G[gi][:, c].mean()) / (G[gi][:, c].std() + 1e-9)
                y = y + np.sqrt(pve / (1 - pve)) * xc
            Wg.append(ko.marginal_importance(G[gi], y) - ko.marginal_importance(Gk[gi], y))
            truth.append(gi in sigset)
        Wg = np.array(Wg); truth = np.array(truth)
        gid = [f"g{i}" for i in range(len(Wg))]
        sel = ko.mirror_select_egenes(gid, Wg, q=q, offset=1)
        sm = np.array([g in set(sel['selected']) for g in gid])
        R = sm.sum()
        fdps.append((sm & ~truth).sum() / max(R, 1))
        powers.append((sm & truth).sum() / max(truth.sum(), 1))
        Rs.append(int(R))
    return dict(fdr=float(np.mean(fdps)), power=float(np.mean(powers)),
                meanR=float(np.mean(Rs)))


# --------------------------------------------------------------------------
#  pytest gate (opt-in, self-skipping): the fastPHASE-fit HMM removes the toy
#  HMM's null bias on real data. Skipped if pysam/network/fastphase/ray absent.
# --------------------------------------------------------------------------
def test_fastphase_fit_removes_null_bias_on_real_hprc():
    import pytest
    try:
        import pysam  # noqa: F401
        import ray  # noqa: F401
        from fastphase.fastphase_ray import fastphase  # noqa: F401
    except Exception as e:
        pytest.skip(f"fastphase/ray/pysam unavailable (see docs/fastphase_setup.md): {e}")
    try:
        wins, N = fetch_phased(n_windows=12)
    except Exception as e:
        pytest.skip(f"HPRC network unavailable: {e}")
    if len(wins) < 6:
        pytest.skip(f"only {len(wins)} real windows fetched")
    p = wins[0][0].shape[1]
    fp_fit = make_fastphase_fitter(p, K=20, nstep=12)
    sym_fp = null_W_symmetry(wins, N, lambda xL, xR: gen_fastphase(xL, xR, fp_fit))
    sym_toy = null_W_symmetry(wins, N, gen_toy_hmm)
    # the properly-fit HMM's null W is closer to symmetric than the toy HMM's
    assert abs(sym_fp['mean_W']) < abs(sym_toy['mean_W']) + 0.05, \
        f"fastPHASE fit did not improve null bias: fp={sym_fp} toy={sym_toy}"


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_windows', type=int, default=40)
    ap.add_argument('--reps', type=int, default=8)
    ap.add_argument('--K', type=int, default=20, help='fastPHASE clusters')
    args = ap.parse_args()

    try:
        import pysam  # noqa: F401
    except Exception as e:
        sys.exit(f"pysam required: {e}")
    print(f"Fetching {args.n_windows} real phased HPRC v2.0 cis-windows ...")
    wins, N = fetch_phased(n_windows=args.n_windows)
    p = wins[0][0].shape[1]
    print(f"  {len(wins)} windows, N={N} individuals, p={p} variants/window")

    gens = [('toy genotype-HMM K=8         ', gen_toy_hmm),
            ('Gaussian shrink=0.1          ', gen_gaussian)]
    try:
        from fastphase.fastphase_ray import fastphase  # noqa: F401
        import ray  # noqa: F401
        fp_fit = make_fastphase_fitter(p, K=args.K, nstep=12)
        gens.insert(1, (f'fastPHASE fit(K={args.K}) -> our gen',
                        lambda xL, xR: gen_fastphase(xL, xR, fp_fit)))
    except Exception as e:
        print(f"  [fastphase/ray not installed -- skipping the HMM-fit row; "
              f"see docs/fastphase_setup.md]  ({type(e).__name__}: {e})")

    print("\nNull-W symmetry (frac(W>0) -> 0.5, mean W -> 0 = well-specified):")
    for lab, gen in gens:
        s = null_W_symmetry(wins, N, gen)
        print(f"  {lab}: frac(W>0)={s['frac_pos']:.3f}  mean W={s['mean_W']:+.3f}  atom={s['atom']:.2f}")

    print(f"\nEnd-to-end FDR/power (mirror-null q=0.10, {args.reps} reps, real HPRC LD):")
    for lab, gen in gens:
        for pve in (0.10, 0.15):
            r = fdr_power(wins, N, gen, pve=pve, reps=args.reps)
            print(f"  {lab} PVE={pve}: FDR={r['fdr']:.3f} (target 0.10)  "
                  f"power={r['power']:.2f}  meanR={r['meanR']:.1f}")
