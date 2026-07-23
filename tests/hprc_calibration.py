"""
REAL-DATA (HPRC) calibration harness for the KFc knockoff eGene filter.

Synthetic HMM-simulated genotypes flatter the HMM knockoff (the data are Markov).
This harness validates on REAL human LD: it pulls cis-window-sized genotype
matrices from the public HPRC (Human Pangenome Reference Consortium) v2.0
minigraph-cactus phased VCF on the `human-pangenomics` S3 bucket via pysam remote
tabix (only the queried regions are downloaded, not the ~4 GB file), and measures
the KFc filter's calibration on real LD.

Metric of knockoff quality: null-W SYMMETRY. For a null gene (Y independent of
genotype) a valid knockoff gives W = imp(real) - imp(knockoff) symmetric about 0,
so frac(W>0) -> 0.5; deviation = misspecification. Then realized FDR + power under
planted signal.

Findings recorded in docs/calibration_findings.md sec 8: on real HPRC v2.0
(N=232), per-gene Gaussian (shrink~0.1) is well-specified (frac~0.556, FDR
0.06-0.07 at target 0.10, power 0.31-0.55); our toy HMM knockoff is misspecified
(frac~0.2-0.31, ~0 power) -- an under-fitting issue (see the caveat in the doc).

Requires: pysam + outbound network to the HPRC S3 bucket. Opt-in / slow.
Run:  python tests/hprc_calibration.py --n_windows 60 --reps 8
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


def fetch_windows(url=HPRC_V2_URL, chrom='chr1', start=5_000_000, end=60_000_000,
                  p_per_win=30, n_windows=60, maf_min=0.05):
    """Pull biallelic-SNP dosage matrices [N, p] for `n_windows` real cis-windows."""
    import pysam
    vf = pysam.VariantFile(url)
    samples = list(vf.header.samples)
    N = len(samples)
    cols = []
    for rec in vf.fetch(chrom, start, end):
        if rec.ref is None or rec.alts is None or len(rec.alts) != 1:
            continue
        if len(rec.ref) != 1 or len(rec.alts[0]) != 1:
            continue
        d = np.full(N, np.nan)
        for i, s in enumerate(samples):
            gt = rec.samples[s].get('GT')
            if gt is None:
                continue
            al = [a for a in gt if a is not None]
            if al:
                d[i] = sum(1 for a in al if a and a > 0)
        if np.isnan(d).mean() > 0.2:
            continue
        m = np.nanmean(d)
        d = np.where(np.isnan(d), m, d)
        af = d.sum() / (2 * N)
        if min(af, 1 - af) < maf_min or d.std() < 1e-6:
            continue
        cols.append(d)
        if len(cols) >= p_per_win * n_windows:
            break
    cols = np.array(cols)
    nw = len(cols) // p_per_win
    wins = [cols[w * p_per_win:(w + 1) * p_per_win].T for w in range(nw)]  # [N,p]
    return wins, N


def _knockoff(G, kind, shrink=0.1, hmm_K=8, seed=0):
    if kind == 'gaussian':
        import torch
        Xt = torch.tensor(G, dtype=torch.float32)
        return ko.gaussian_knockoff(Xt, shrink=shrink,
                                    generator=torch.Generator().manual_seed(seed)).numpy().astype(np.float64)
    elif kind == 'hmm':
        Gi = np.rint(G).clip(0, 2).astype(np.int64)
        return ko.genotype_hmm_knockoffs(Gi, K=hmm_K, M=1, n_em_iter=12, seed=seed)[0].astype(np.float64)
    raise ValueError(kind)


def null_W_symmetry(wins, kind='gaussian', shrink=0.1, hmm_K=8, seed0=0):
    """frac(W>0) and mean W over null genes (want 0.5 and 0)."""
    W = []
    for gi, G in enumerate(wins):
        rng = np.random.RandomState(1000 + gi)
        y = rng.randn(G.shape[0])
        Gk = _knockoff(G, kind, shrink, hmm_K, seed=seed0 + gi)
        W.append(ko.gene_W_marginal(G.astype(np.float64), Gk, y))
    W = np.array(W)
    return dict(frac_pos=float(np.mean(W > 0)), mean_W=float(W.mean()),
                atom=float(np.mean(np.abs(W) < 1e-9)), W=W)


def fdr_power(wins, kind='gaussian', shrink=0.1, hmm_K=8, pve=0.10,
              egene_frac=0.4, q=0.1, reps=8):
    """Realized FDR + power under planted signal, mirror-null selection."""
    N = wins[0].shape[0]
    fdps, powers, Rs = [], [], []
    for rep in range(reps):
        r = np.random.RandomState(10 + rep)
        n_sig = int(egene_frac * len(wins))
        sigset = set(r.permutation(len(wins))[:n_sig].tolist())
        Wg, truth = [], []
        for gi, G in enumerate(wins):
            y = r.randn(N)
            sig = gi in sigset
            if sig:
                c = r.randint(G.shape[1])
                xc = (G[:, c] - G[:, c].mean()) / (G[:, c].std() + 1e-9)
                y = y + np.sqrt(pve / (1 - pve)) * xc
            Gk = _knockoff(G, kind, shrink, hmm_K, seed=500 + rep * 1000 + gi)
            Wg.append(ko.gene_W_marginal(G.astype(np.float64), Gk, y))
            truth.append(sig)
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
#  pytest gate (opt-in): the shipped real-data claim -- Gaussian KFc controls
#  FDR on real HPRC LD. Skipped if pysam or network is unavailable.
# --------------------------------------------------------------------------
def test_gaussian_kfc_controls_fdr_on_real_hprc():
    import pytest
    try:
        import pysam  # noqa: F401
        wins, N = fetch_windows(n_windows=60)
    except Exception as e:
        pytest.skip(f"HPRC/pysam unavailable: {e}")
    if len(wins) < 30:
        pytest.skip(f"only {len(wins)} real windows fetched")
    sym = null_W_symmetry(wins, kind='gaussian', shrink=0.1)
    assert sym['atom'] == 0.0, "KFc statistic must have no atom on real data"
    assert 0.4 <= sym['frac_pos'] <= 0.65, \
        f"Gaussian null-W not symmetric on real LD: frac_pos={sym['frac_pos']:.3f}"
    res = fdr_power(wins, kind='gaussian', shrink=0.1, pve=0.15, reps=6)
    assert res['fdr'] <= 0.1 + 0.05, f"realized FDR={res['fdr']:.3f}"
    assert res['power'] >= 0.2, f"power={res['power']:.2f}"


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_windows', type=int, default=60)
    ap.add_argument('--reps', type=int, default=8)
    ap.add_argument('--hmm_K', type=int, default=8)
    args = ap.parse_args()
    print(f"Fetching {args.n_windows} real HPRC v2.0 cis-windows ...")
    wins, N = fetch_windows(n_windows=args.n_windows)
    print(f"  {len(wins)} windows, N={N} individuals, p={wins[0].shape[1]}")
    print("\nNull-W symmetry (frac(W>0) -> 0.5 = well-specified):")
    for kind, kw in [('gaussian shrink=0.05', dict(kind='gaussian', shrink=0.05)),
                     ('gaussian shrink=0.1', dict(kind='gaussian', shrink=0.1)),
                     ('gaussian shrink=0.2', dict(kind='gaussian', shrink=0.2)),
                     (f'toy HMM K={args.hmm_K}', dict(kind='hmm', hmm_K=args.hmm_K))]:
        s = null_W_symmetry(wins, **kw)
        print(f"  {kind:22s}: frac(W>0)={s['frac_pos']:.3f}  mean W={s['mean_W']:+.3f}  atom={s['atom']:.2f}")
    print("\nFDR/power (Gaussian shrink=0.1, mirror-null q=0.1, real HPRC LD):")
    for pve in (0.10, 0.15):
        r = fdr_power(wins, kind='gaussian', shrink=0.1, pve=pve, reps=args.reps)
        print(f"  PVE={pve}: realized FDR={r['fdr']:.3f} (target 0.10)  power={r['power']:.2f}  meanR={r['meanR']:.1f}")
