"""
Reference mapping bias: how RASQUAL fares on UNFILTERED data, and whether our
diagnostic catches what hapmixQTL cannot model.

Section 7h showed that at a uniform phi = 0.60, every method that does not model
reference bias breaks (type-I 0.52-0.64) while a phi-fitting model is untouched.
But phi = 0.60 applied identically to every gene and every site is the *easiest*
possible bias for a single-phi model to absorb. Real unfiltered data is not like
that. Two harder regimes:

  ACROSS GENES.   Most genes are near-unbiased; a minority are badly biased
                  (low-mappability regions, repeats, segmental duplications).
                  RASQUAL fits phi per feature, so in principle it adapts --
                  this tests whether it actually does.

  WITHIN A GENE.  A gene's allele-specific counts are summed over several
                  het sites, each with its OWN bias. RASQUAL fits ONE phi per
                  feature, so within-gene heterogeneity is the case its model
                  cannot represent. This is the sharpest test of its robustness.

Direction matters: reference bias is mechanistically *directional* -- reads are
aligned to the reference genome, so the reference allele is preferentially
captured. Heterogeneous but one-sided bias does NOT average away, unlike
symmetric noise. All scenarios below draw phi >= 0.5.

PART B validates `hapmixqtl.reference_bias_diagnostic`: its power to detect bias
at each severity, and -- more important -- its specificity, i.e. that genuine
allele-specific expression does not trigger it. The diagnostic's whole premise is
that real cis effects have arbitrary sign with respect to the reference allele
and cancel when pooled across genes, while bias accumulates.

Run:  python3 tests/ase_reference_bias.py --reps 300
"""

import argparse
import json
import sys
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from scipy import stats
from ase_external_benchmark import trec_lrt, trecase_lrt, hapmix_pval, trec_multiplier
from ase_rasqual_comparison import shared_theta_lrt, observed_pi

try:
    from tensorqtl.hapmixqtl import reference_bias_diagnostic
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'tensorqtl'))
    from hapmixqtl import reference_bias_diagnostic


def draw_phi(mode, rng, n_sites, sev):
    """Draw per-site phi >= 0.5. `sev` is the mean excess bias above 0.5."""
    if mode == 'none':
        return np.full(n_sites, 0.5)
    if mode == 'uniform':
        return np.full(n_sites, 0.5 + sev)
    if mode == 'across_genes':
        # a minority of genes badly biased, most nearly clean (one phi per gene)
        bad = rng.rand() < 0.25
        return np.full(n_sites, 0.5 + (4.0 * sev if bad else 0.0))
    if mode == 'within_gene':
        # each site its own bias; one-sided, so it does not average away
        return 0.5 + np.clip(rng.exponential(sev, n_sites), 0, 0.45)
    raise ValueError(mode)


def simulate(N, rng, kappa=1.0, maf=0.3, mu=200.0, theta=0.2, as_frac=0.25,
             lib_sd=0.25, phi_mode='none', sev=0.10, n_sites=4, delta=0.0):
    """Counts summed over `n_sites` het sites, each with its own phi."""
    g = rng.binomial(2, maf, size=N)
    s = np.zeros(N); het = g == 1
    s[het] = rng.choice([-1.0, 1.0], size=int(het.sum()))
    lib = np.exp(rng.normal(0, lib_sd, N))
    mean_t = lib * mu * trec_multiplier(g, kappa)
    r = 1.0 / max(theta, 1e-8)
    T = rng.negative_binomial(r, r / (r + mean_t)).astype(float)
    n_as_tot = rng.binomial(T.astype(int), as_frac).astype(float)

    phis = draw_phi(phi_mode, rng, n_sites, sev)
    pi_alt = kappa / (1.0 + kappa)
    y_alt = np.zeros(N); n_as = np.zeros(N)
    for i in range(N):
        if n_as_tot[i] <= 0:
            continue
        # split this sample's AS reads across the sites
        per = rng.multinomial(int(n_as_tot[i]), np.ones(n_sites) / n_sites)
        pi_i = pi_alt if het[i] else 0.5
        for k in range(n_sites):
            if per[k] == 0:
                continue
            po = observed_pi(pi_i, delta, phis[k])
            a_ = max(po / theta, 1e-6); b_ = max((1 - po) / theta, 1e-6)
            y_alt[i] += rng.binomial(per[k], rng.beta(a_, b_))
        n_as[i] = per.sum()

    y_hi, y_lo = y_alt, n_as - y_alt
    yL = np.where(s >= 0, y_hi, y_lo); yR = np.where(s >= 0, y_lo, y_hi)
    return dict(g=g.astype(float), s=s, het=het, T=T, n_as=n_as,
                y_alt=y_alt, yL=yL, yR=yR, lib=lib)


METHODS = {
    'trcQTL': lambda d, r: trec_lrt(d),
    'TReCASE': lambda d, r: trecase_lrt(d),
    'RASQUAL-like (fits phi)': lambda d, r: shared_theta_lrt(d, True, True),
    'hapmixQTL': lambda d, r: hapmix_pval(d, r, 'estimate'),
}


def run(reps, N, kappa, seed0, **kw):
    out = {k: [] for k in METHODS}
    for r in range(reps):
        rng = np.random.RandomState(seed0 + r)
        d = simulate(N, rng, kappa=kappa, **kw)
        for name, fn in METHODS.items():
            _, st = fn(d, np.random.RandomState(seed0 + 313000 + r))
            out[name].append(st)
    return {k: np.array(v) for k, v in out.items()}


def part_A(reps, N, kappa_alt):
    print("\n=== A. RASQUAL on UNFILTERED data: how far does fitting phi carry? ===")
    scen = [
        ('clean (no bias)', dict(phi_mode='none')),
        ('uniform  phi=0.60', dict(phi_mode='uniform', sev=0.10)),
        ('uniform  phi=0.75', dict(phi_mode='uniform', sev=0.25)),
        ('across genes (25% badly biased)', dict(phi_mode='across_genes', sev=0.06)),
        ('WITHIN gene (per-site bias)', dict(phi_mode='within_gene', sev=0.10)),
        ('WITHIN gene, severe', dict(phi_mode='within_gene', sev=0.25)),
    ]
    res = {}
    print(f"  {'scenario':>34s} | " +
          " | ".join(f"{k[:22]:>22s}" for k in METHODS))
    print("  " + "-" * 34 + "-+-" + "-+-".join(["-" * 22] * len(METHODS)))
    for label, kw in scen:
        null = run(reps, N, 1.0, 21_000, **kw)
        thr = {k: np.quantile(v[np.isfinite(v)], 0.90) for k, v in null.items()}
        alt = run(reps, N, kappa_alt, 88_000, **kw)
        nom, pw = {}, {}
        for k in METHODS:
            v = null[k][np.isfinite(null[k])]
            p = (stats.chi2.sf(v, 1) if 'hapmix' not in k
                 else 2 * stats.t.sf(np.sqrt(np.clip(v, 0, None)), N - 2))
            nom[k] = float(np.mean(p < 0.05))
            a = alt[k][np.isfinite(alt[k])]
            pw[k] = float(np.mean(a > thr[k])) if a.size else float('nan')
        res[label] = dict(nominal_typeI=nom, power_matched=pw)
        print(f"  {label:>34s} | " +
              " | ".join(f"t1={nom[k]:.3f} pw={pw[k]:.3f}" for k in METHODS))
    print("\n  t1 = nominal type-I @0.05 (target 0.05); pw = power at matched"
          f"\n  empirical FPR 10%, aFC={kappa_alt}. A method that is calibrated"
          "\n  (t1 ~ 0.05) AND retains power has genuinely handled the bias.")
    return res


def part_B(reps, N):
    print("\n=== B. Does the shipped diagnostic catch it? ===")
    print("  hapmixqtl.reference_bias_diagnostic pools across genes: real ASE has")
    print("  arbitrary sign w.r.t. the reference allele and cancels; bias accumulates.")
    print(f"  {'scenario':>34s} | {'ref frac':>9s} | {'detect rate':>11s}")
    print("  " + "-" * 62)
    scen = [
        ('clean, null genes', dict(phi_mode='none'), 1.0),
        ('clean, STRONG real ASE (aFC 2.0)', dict(phi_mode='none'), 2.0),
        ('uniform phi=0.55', dict(phi_mode='uniform', sev=0.05), 1.0),
        ('uniform phi=0.60', dict(phi_mode='uniform', sev=0.10), 1.0),
        ('uniform phi=0.70', dict(phi_mode='uniform', sev=0.20), 1.0),
        ('across genes (25% biased)', dict(phi_mode='across_genes', sev=0.06), 1.0),
        ('WITHIN gene (per-site)', dict(phi_mode='within_gene', sev=0.10), 1.0),
        ('phi=0.60 AND strong real ASE', dict(phi_mode='uniform', sev=0.10), 2.0),
    ]
    res = {}
    n_panels, genes_per_panel = 24, 40
    for label, kw, kap in scen:
        fracs, flags = [], []
        for panel in range(n_panels):
            YL, YR, S = [], [], []
            for gi in range(genes_per_panel):
                rng = np.random.RandomState(50_000 + panel * 1000 + gi)
                # real ASE: the increasing allele is ALT or REF at random,
                # exactly the symmetry the diagnostic relies on
                k = kap if rng.rand() < 0.5 else (1.0 / kap)
                d = simulate(N, rng, kappa=k, **kw)
                YL.append(d['yL']); YR.append(d['yR']); S.append(d['s'])
            out = reference_bias_diagnostic(np.array(YL), np.array(YR), np.array(S))
            fracs.append(out['ref_fraction']); flags.append(out['flag'])
        res[label] = dict(mean_ref_fraction=float(np.mean(fracs)),
                          detect_rate=float(np.mean(flags)))
        print(f"  {label:>34s} | {np.mean(fracs):9.4f} | {np.mean(flags):11.3f}")
    print("\n  Rows 1-2 are SPECIFICITY (want detect ~0, fraction ~0.5); the rest")
    print("  are POWER. Row 2 is the key control: strong genuine ASE must not")
    print("  trigger it. Row 8 checks bias is still caught when real ASE coexists.")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=300)
    ap.add_argument('--N', type=int, default=100)
    ap.add_argument('--kappa', type=float, default=1.25)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    res = {'config': vars(args)}
    res['A_unfiltered'] = part_A(args.reps, args.N, args.kappa)
    res['B_diagnostic'] = part_B(args.reps, args.N)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, default=float))
        print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
