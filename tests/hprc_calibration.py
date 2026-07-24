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


def eval_selection(per_rep, center=0.0, q=0.1, offset=1):
    """Realized FDP/power/tau over reps for a given additive center on W
    (select on W - center; center = null-mean shifts a left-biased null to 0)."""
    fdps, powers, taus, Rs = [], [], [], []
    for Wg, truth in per_rep:
        gid = [f"g{i}" for i in range(len(Wg))]
        sel = ko.mirror_select_egenes(gid, Wg - center, q=q, offset=offset)
        sm = np.array([g in set(sel['selected']) for g in gid])
        R = int(sm.sum())
        fdps.append((sm & ~truth).sum() / max(R, 1))
        powers.append((sm & truth).sum() / max(truth.sum(), 1))
        taus.append(sel['tau']); Rs.append(R)
    return np.array(fdps), np.array(powers), np.array(taus), np.array(Rs)


def _fast_threshold(W, q=0.1, offset=1):
    """Vectorized mirror knockoff+ threshold; verified == ko.knockoff_threshold.
    Used to make the gene-count scaling sweep tractable (ko.mirror_select_egenes
    computes per-gene q-values in O(m^2), too slow at m~2000)."""
    W = np.asarray(W, dtype=np.float64)
    ws = np.sort(W)
    cand = np.unique(np.abs(W[W != 0.0]))
    if cand.size == 0:
        return np.inf
    ge = ws.size - np.searchsorted(ws, cand, side='left')
    le = np.searchsorted(ws, -cand, side='right')
    fdp = (offset + le) / np.maximum(1, ge)
    ok = np.where(fdp <= q)[0]
    return cand[ok[0]] if ok.size else np.inf


def _fast_select_mask(W, q=0.1, offset=1, center=0.0):
    Wc = np.asarray(W, dtype=np.float64) - center
    tau = _fast_threshold(Wc, q=q, offset=offset)
    return (Wc >= tau) if np.isfinite(tau) else np.zeros(Wc.shape[0], bool)


def run_calibration_scaling_experiment(wins, ms=(150, 300, 600, 1200, 2000),
                                       reps=15, q=0.1, seed=0):
    """Realized FDR (centered) vs gene count m, on the real HPRC per-window W
    distribution bootstrap-pooled to larger m. Isolates the small-m +1-offset
    conservatism from the generator bias: with the location bias removed by
    centering, realized FDR should climb toward q as m grows (the +1 becomes
    negligible), i.e. CALIBRATION requires centering AND scale, not just control.
    Also reports the offset=0 ceiling (calibrated but only asymptotically
    controlled)."""
    # verify the fast selector reproduces the shipped threshold exactly
    rng = np.random.RandomState(0)
    for _ in range(25):
        w = rng.randn(400)
        for off in (0, 1):
            a = _fast_threshold(w, q=q, offset=off)
            b = ko.knockoff_threshold(w, q=q, offset=off)
            assert (np.isinf(a) and np.isinf(b)) or np.isclose(a, b), (a, b, off)

    c = float(permutation_null_W(wins, kind='gaussian', shrink=0.1).mean())
    m_max = max(ms)
    idx = np.random.RandomState(seed).randint(0, len(wins), size=m_max)
    exp = [wins[i] for i in idx]   # real windows resampled w/ replacement to m_max
    print(f"\n{'='*72}\nCalibration-vs-gene-count scaling "
          f"(real HPRC windows bootstrap-pooled; base={len(wins)}, N={wins[0].shape[0]}, "
          f"reps={reps}, q={q})\n{'='*72}")
    print(f"null-mean center c = {c:+.4f}   (each real window reused ~{m_max/len(wins):.1f}x "
          f"with independent knockoffs/Y)")
    print("  columns: realized FDR [95% CI] (power) for each configuration\n")
    print(f"  {'PVE':>4} {'m':>6}  {'shipped off=1':>22} {'centered off=1':>22} {'centered off=0':>22}")
    for pve in (0.10, 0.15):
        per_rep = collect_signal_W(exp, pve=pve, reps=reps, seed=seed)
        for m in ms:
            cells = []
            for center, offset in [(0.0, 1), (c, 1), (c, 0)]:
                fdps, powers = [], []
                for Wg, truth in per_rep:
                    mask = _fast_select_mask(Wg[:m], q=q, offset=offset, center=center)
                    R = int(mask.sum())
                    fdps.append((mask & ~truth[:m]).sum() / max(R, 1))
                    powers.append((mask & truth[:m]).sum() / max(truth[:m].sum(), 1))
                fm, flo, fhi = _boot_ci(fdps)
                pm, _, _ = _boot_ci(powers)
                cells.append(f"{fm:.3f}[{flo:.2f},{fhi:.2f}]({pm:.2f})")
            print(f"  {pve:>4} {m:>6}  {cells[0]:>22} {cells[1]:>22} {cells[2]:>22}")
    print("\nRead: with the location bias removed (centered), realized FDR should rise")
    print("toward q as m grows -- the residual small-m gap is the +1 knockoff-offset")
    print("(finite-sample control), not the generator. offset=0 removes the +1 (near-")
    print("exact calibration) but controls FDR only asymptotically -- watch for >q at small m.")
    print("Caveat: bootstrap-pooling assumes independent genes; real genome-wide LD adds")
    print("PRDS dependence, so treat the large-m numbers as the independent-gene ideal.")
    return dict(center=c)


def permnull_storey_select(Wg, Wnull_sorted_centered, center, q=0.1, lam=0.5, pi0=None):
    """CALIBRATED alternative to the mirror: right-tail p-value of each (centered)
    gene W against the clean permutation null, Storey pi0, BH-Storey q-values,
    select q <= target. Unlike the mirror this uses the UNCONTAMINATED null (no
    weak-signal negatives), so realized FDR tracks the target instead of sitting
    well below it -- at the cost of the mirror's distribution-free guarantee (this
    is BH-Storey, valid under PRDS). Returns (mask, pi0)."""
    Wc = np.asarray(Wg, dtype=np.float64) - center
    n0 = len(Wnull_sorted_centered)
    p = (n0 - np.searchsorted(Wnull_sorted_centered, Wc, side='left') + 1) / (n0 + 1)
    p = np.clip(p, 1.0 / (n0 + 1), 1.0)
    if pi0 is None:
        pi0 = min(1.0, float(np.mean(p >= lam) / (1.0 - lam)))
    m = len(p); order = np.argsort(p); ps = p[order]
    qs = pi0 * ps * m / np.arange(1, m + 1)
    qs = np.minimum.accumulate(qs[::-1])[::-1]
    qv = np.empty(m); qv[order] = np.clip(qs, 0, 1)
    return qv <= q, pi0


def run_selector_comparison(wins, ms=(300, 1200), reps=25, q=0.1, seed=0):
    """Mirror (shipped, conservative) vs permutation-null Storey (calibrated) on
    the real HPRC per-window W, bootstrap-pooled to gene count m. Shows that
    calibration (realized FDR ~ target) is achievable, and the associated
    control/robustness trade-off, with bootstrap CIs."""
    Wnull = permutation_null_W(wins, kind='gaussian', shrink=0.1)
    center = float(Wnull.mean())
    null_c = np.sort(Wnull - center)
    print(f"\n{'='*72}\nSelector comparison: mirror (control) vs permutation-null Storey "
          f"(calibration)\n(real HPRC bootstrap-pooled, N={wins[0].shape[0]}, reps={reps}, "
          f"q={q}, center={center:+.4f})\n{'='*72}")
    for pve in (0.10, 0.15):
        idx = np.random.RandomState(seed).randint(0, len(wins), size=max(ms))
        per_rep = collect_signal_W([wins[i] for i in idx], pve=pve, reps=reps, seed=seed)
        for m in ms:
            mir_f, mir_p, sto_f, sto_p, pi0s = [], [], [], [], []
            for Wg, truth in per_rep:
                W, tr = Wg[:m], truth[:m]
                sel = ko.mirror_select_egenes([str(i) for i in range(m)], W - center, q=q, offset=1)
                mm = np.zeros(m, bool); mm[[int(g) for g in sel['selected']]] = True
                sm, pi0 = permnull_storey_select(W, null_c, center, q=q); pi0s.append(pi0)
                for mask, fl, pl in ((mm, mir_f, mir_p), (sm, sto_f, sto_p)):
                    R = int(mask.sum())
                    fl.append((mask & ~tr).sum() / max(R, 1))
                    pl.append((mask & tr).sum() / max(tr.sum(), 1))
            mf, mlo, mhi = _boot_ci(mir_f); mpw, _, _ = _boot_ci(mir_p)
            sf, slo, shi = _boot_ci(sto_f); spw, _, _ = _boot_ci(sto_p)
            print(f"\nPVE={pve}  m={m}  (est pi0={np.mean(pi0s):.2f}, true pi0={float(np.mean(~per_rep[0][1])):.2f})")
            print(f"  mirror  (control) : FDR {mf:.3f} [{mlo:.3f},{mhi:.3f}]  power {mpw:.2f}  "
                  f"{'OK' if mhi <= q + 1e-9 else 'over'}")
            print(f"  storey  (calibr.) : FDR {sf:.3f} [{slo:.3f},{shi:.3f}]  power {spw:.2f}  "
                  f"{'OK' if shi <= q + 1e-9 else 'over'}")
    print("\nStorey reaches realized FDR ~ target (calibrated) with more power, and controls")
    print("at scale; mild overshoot at small m. Trade-off: BH-Storey (PRDS-valid) replaces")
    print("the mirror's distribution-free guarantee, and needs a per-dataset permutation pass.")
    return dict(center=center)


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
    ap.add_argument('--scaling', action='store_true',
                    help='Run the calibration-vs-gene-count scaling experiment (centered FDR as m grows)')
    ap.add_argument('--selector_comparison', action='store_true',
                    help='Compare the mirror (control) vs permutation-null Storey (calibration) selectors')
    ap.add_argument('--cache', default=None, help='Path to .npz to cache/reuse fetched windows')
    args = ap.parse_args()

    if args.permutation_null or args.scaling or args.selector_comparison:
        nw = args.n_windows if args.n_windows != 60 else 300
        reps = args.reps if args.reps != 8 else 30
        wins, N = _load_windows(nw, cache=args.cache)
        if len(wins) < 30:
            raise SystemExit(f"only {len(wins)} windows fetched; need >= 30")
        if args.permutation_null:
            run_permutation_null_experiment(wins, reps=reps, q=0.1)
        if args.scaling:
            run_calibration_scaling_experiment(wins, reps=min(reps, 15), q=0.1)
        if args.selector_comparison:
            run_selector_comparison(wins, reps=min(reps, 25), q=0.1)
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
