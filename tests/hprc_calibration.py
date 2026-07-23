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
#  Permutation-null tail-symmetry experiment (the oracle-specified next step).
#
#  Question: is the Gaussian generator's null W a pure LOCATION shift (mean<0)
#  or a left-SKEW (heavy negative tail, bulk>0)? Barber-Candes control depends
#  on TAIL sign-symmetry at the operating threshold tau, not the mean. We
#  measure A(t)=#{W<=-t}/#{W>=t} across the tau region on a contamination-free
#  permutation null (Y independent of genotype -- no signal to bias the null
#  center, curing the circularity that sank naive median-recentering), then test
#  whether centering by the permutation-null mean recovers FDR toward target
#  WITHOUT breaking control (upper-CI realized FDR <= q).
# --------------------------------------------------------------------------

def permutation_null_W(wins, kind='gaussian', shrink=0.1, hmm_K=8,
                       draws_per_window=15, seed0=0):
    """Contamination-free null W: for each window draw `draws_per_window`
    independent phenotypes Y ~ N(0,I) (Y _|_ genotype) and an independent
    knockoff; return the flat array of W = imp(real) - imp(knockoff)."""
    W = []
    for gi, G in enumerate(wins):
        Gf = G.astype(np.float64)
        for d in range(draws_per_window):
            rng = np.random.RandomState(90000 + seed0 + gi * 1000 + d)
            y = rng.randn(G.shape[0])
            Gk = _knockoff(G, kind, shrink, hmm_K, seed=70000 + seed0 + gi * 1000 + d)
            W.append(ko.gene_W_marginal(Gf, Gk, y))
    return np.asarray(W)


def collect_signal_W(wins, kind='gaussian', shrink=0.1, hmm_K=8, pve=0.10,
                     egene_frac=0.4, reps=30, seed=0):
    """Per-rep (Wg, truth) under planted signal -- knockoffs computed ONCE per
    rep so multiple selection centers can be evaluated without recompute."""
    N = wins[0].shape[0]
    per_rep = []
    for rep in range(reps):
        r = np.random.RandomState(10 + seed + rep)
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
            Gk = _knockoff(G, kind, shrink, hmm_K, seed=500 + (seed + rep) * 100000 + gi)
            Wg.append(ko.gene_W_marginal(G.astype(np.float64), Gk, y))
            truth.append(sig)
        per_rep.append((np.asarray(Wg), np.asarray(truth)))
    return per_rep


def eval_selection(per_rep, center=0.0, q=0.1):
    """Realized FDP/power/tau over reps for a given additive center on W
    (select on W - center; center = null-mean shifts a left-biased null to 0)."""
    fdps, powers, taus, Rs = [], [], [], []
    for Wg, truth in per_rep:
        gid = [f"g{i}" for i in range(len(Wg))]
        sel = ko.mirror_select_egenes(gid, Wg - center, q=q, offset=1)
        sm = np.array([g in set(sel['selected']) for g in gid])
        R = int(sm.sum())
        fdps.append((sm & ~truth).sum() / max(R, 1))
        powers.append((sm & truth).sum() / max(truth.sum(), 1))
        taus.append(sel['tau']); Rs.append(R)
    return np.array(fdps), np.array(powers), np.array(taus), np.array(Rs)


def _boot_ci(vals, reps=4000, alpha=0.05, seed=1):
    """95% bootstrap CI for the mean of `vals`."""
    vals = np.asarray(vals, float)
    rng = np.random.RandomState(seed)
    means = vals[rng.randint(0, len(vals), size=(reps, len(vals)))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(vals.mean()), float(lo), float(hi)


def run_permutation_null_experiment(wins, reps=30, q=0.1, seed=0):
    """Report null-W location vs skew (A(t) across the tau region) and the
    mean-centering FDR/power counterfactual with bootstrap CIs. Records evidence
    for whether centering is a valid de-bias; does NOT change any shipped path."""
    print(f"\n{'='*72}\nPermutation-null tail-symmetry experiment "
          f"({len(wins)} windows, N={wins[0].shape[0]}, reps={reps}, q={q})\n{'='*72}")

    Wn = permutation_null_W(wins, kind='gaussian', shrink=0.1)
    c_mean, c_med = float(Wn.mean()), float(np.median(Wn))
    print(f"\nNull W (Gaussian shrink=0.1, {len(Wn)} contamination-free draws):")
    print(f"  mean={c_mean:+.4f}   median={c_med:+.4f}   frac(W>0)={np.mean(Wn > 0):.3f}")
    print("  (mean<0 AND median>0 => LEFT-SKEW, not a location shift)")

    per_rep_10 = collect_signal_W(wins, pve=0.10, reps=reps, seed=seed)
    _, _, taus10, _ = eval_selection(per_rep_10, center=0.0, q=q)
    finite = taus10[np.isfinite(taus10)]
    tau_med = float(np.median(finite)) if finite.size else float('nan')
    tau_lo = float(np.percentile(finite, 25)) if finite.size else float('nan')
    tau_hi = float(np.percentile(finite, 75)) if finite.size else float('nan')
    print(f"\nOperating threshold tau (PVE 0.10 runs): median={tau_med:.3f} "
          f"(IQR {tau_lo:.3f}-{tau_hi:.3f})")

    qs = np.percentile(np.abs(Wn), [80, 90, 95, 97.5, 99])
    extra = [tau_med] if np.isfinite(tau_med) else []
    ts = np.unique(np.round(np.concatenate([qs, extra]), 3))
    print("\nA(t) = #{W<=-t}/#{W>=t}  (1.0 = tail-symmetric; >1 = tail biased toward"
          " nulls => conservative)")
    print(f"  {'t':>7} {'raw neg/pos':>16} {'A_raw':>8}   {'centered neg/pos':>18} {'A_ctr':>8}")
    Wc = Wn - c_mean
    for t in ts:
        n0, p0 = int(np.sum(Wn <= -t)), int(np.sum(Wn >= t))
        n1, p1 = int(np.sum(Wc <= -t)), int(np.sum(Wc >= t))
        a0 = (n0 / p0) if p0 else float('inf')
        a1 = (n1 / p1) if p1 else float('inf')
        mark = ' <- tau' if np.isfinite(tau_med) and abs(t - round(tau_med, 3)) < 1e-6 else ''
        print(f"  {t:>7.3f} {f'{n0}/{p0}':>16} {a0:>8.2f}   {f'{n1}/{p1}':>18} {a1:>8.2f}{mark}")

    print(f"\nRealized FDR / power (mirror q={q}, offset=1); 95% bootstrap CI over {reps} reps:")
    print(f"  {'PVE':>5} {'center':>10} {'FDR (95% CI)':>26} {'power (95% CI)':>24} {'ctrl?':>6}")
    per_rep_15 = collect_signal_W(wins, pve=0.15, reps=reps, seed=seed)
    for pve, per_rep in [(0.10, per_rep_10), (0.15, per_rep_15)]:
        for label, center in [('none', 0.0), ('null-mean', c_mean)]:
            fdps, powers, _, _ = eval_selection(per_rep, center=center, q=q)
            fm, flo, fhi = _boot_ci(fdps)
            pm, plo, phi = _boot_ci(powers)
            ctrl = 'OK' if fhi <= q + 1e-9 else 'VIOL'
            print(f"  {pve:>5} {label:>10} {f'{fm:.3f} [{flo:.3f},{fhi:.3f}]':>26} "
                  f"{f'{pm:.2f} [{plo:.2f},{phi:.2f}]':>24} {ctrl:>6}")

    print("\nInterpretation:")
    print("  - If A(t) stays >1 in the TAIL even after centering, the null is SKEWED")
    print("    (not a location shift) and centering the mean is NOT a valid de-bias.")
    print("  - Centering is shippable ONLY if the centered upper-CI FDR <= q at every")
    print("    PVE AND power strictly rises. Otherwise the shipped (uncentered) path,")
    print("    which controls FDR conservatively, stands -- pending a generator fix.")
    return dict(null_mean=c_mean, null_median=c_med, tau_med=tau_med)


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


def _load_windows(n_windows, cache=None):
    """Fetch windows, optionally caching to/from an .npz (windows are all [N,p])."""
    import os
    if cache and os.path.exists(cache):
        z = np.load(cache)
        wins = [z[k] for k in sorted(z.files, key=lambda s: int(s.split('_')[1]))]
        print(f"  loaded {len(wins)} cached windows from {cache}")
        return wins, wins[0].shape[0]
    print(f"Fetching {n_windows} real HPRC v2.0 cis-windows ...")
    wins, N = fetch_windows(n_windows=n_windows)
    print(f"  {len(wins)} windows, N={N} individuals, p={wins[0].shape[1]}")
    if cache:
        np.savez(cache, **{f"w_{i}": w for i, w in enumerate(wins)})
        print(f"  cached windows to {cache}")
    return wins, N


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_windows', type=int, default=60)
    ap.add_argument('--reps', type=int, default=8)
    ap.add_argument('--hmm_K', type=int, default=8)
    ap.add_argument('--permutation_null', action='store_true',
                    help='Run the permutation-null tail-symmetry experiment (defaults to 300 windows, reps=30)')
    ap.add_argument('--cache', default=None, help='Path to .npz to cache/reuse fetched windows')
    args = ap.parse_args()

    if args.permutation_null:
        nw = args.n_windows if args.n_windows != 60 else 300
        reps = args.reps if args.reps != 8 else 30
        wins, N = _load_windows(nw, cache=args.cache)
        if len(wins) < 30:
            raise SystemExit(f"only {len(wins)} windows fetched; need >= 30")
        run_permutation_null_experiment(wins, reps=reps, q=0.1)
        sys.exit(0)

    wins, N = _load_windows(args.n_windows, cache=args.cache)
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
