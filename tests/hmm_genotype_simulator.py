"""
fastPHASE-style HMM genotype simulator for knockoff validation.

Every prior calibration test used Gaussian-factor dosages -- the FAVORABLE case
for second-order (Gaussian) knockoffs, since the data matches their assumptions.
Real genotypes are discrete and haplotypic with rare variants and sharp LD
boundaries at recombination hotspots. This module simulates genotypes from the
same generative model SNPknock's HMM knockoffs assume (fastPHASE: K ancestral
haplotype clusters, a Markov chain along the chromosome with per-interval
recombination, Bernoulli emissions), so we can test whether the Gaussian
knockoff generator's FDR calibration survives realistic LD -- the central
real-data concern raised in review.

Model (per haplotype, marching along L SNPs):
  - hidden state z_j in {1..K}: ancestral cluster at SNP j.
  - transition j-1 -> j: with prob (1 - exp(-rho_j)) the chain "jumps" and
    redraws z_j ~ alpha (cluster frequencies), else z_j = z_{j-1}. rho_j is the
    per-interval recombination weight; a genetic map (cM) sets rho_j = c * d_j,
    and hotspots are large rho_j (sharp LD breaks).
  - emission: allele 1 at SNP j given cluster k with prob theta[j, k].
Diploid: two independent haplotypes per individual summed to a dosage {0,1,2};
optionally the phased haplotype alleles (xL, xR) are returned for the two-channel
hapmixQTL knockoff work.

LD, allele-frequency spectrum, block structure, and hotspots all emerge from
(K, alpha, theta, rho). Rare variants are produced by a skewed theta prior.

Pure numpy, seeded, small enough for the calibration harness.
"""

import numpy as np


def make_recombination_map(n_snps, rng, base_rate=0.05, n_hotspots=3,
                           hotspot_strength=8.0, hotspot_width=3):
    """
    Per-interval recombination weights rho_j (length n_snps; rho_0 unused).

    base_rate is the background per-interval jump weight; hotspots are a few
    intervals with rho multiplied by hotspot_strength, creating sharp LD breaks.
    In real use rho_j would come from a genetic map (rho_j = c * cM_distance_j);
    here we synthesize a map with tunable hotspots.

    Returns rho: array [n_snps], rho[0] = 0 (no interval before the first SNP).
    """
    rho = np.full(n_snps, base_rate, dtype=np.float64)
    rho[0] = 0.0
    if n_hotspots > 0:
        centers = rng.choice(np.arange(1, n_snps), size=min(n_hotspots, n_snps - 1),
                             replace=False)
        for c in centers:
            lo, hi = max(1, c - hotspot_width), min(n_snps, c + hotspot_width + 1)
            rho[lo:hi] *= hotspot_strength
    return rho


def _sample_hmm_haplotypes(n_hap, n_snps, K, alpha, theta, rho, rng):
    """
    Sample n_hap haplotypes [n_hap, n_snps] of 0/1 alleles from the HMM.
    """
    # hidden states
    Z = np.empty((n_hap, n_snps), dtype=np.int32)
    Z[:, 0] = rng.choice(K, size=n_hap, p=alpha)
    for j in range(1, n_snps):
        jump = rng.random(n_hap) < (1.0 - np.exp(-rho[j]))
        redraw = rng.choice(K, size=n_hap, p=alpha)
        Z[:, j] = np.where(jump, redraw, Z[:, j - 1])
    # emissions: allele 1 w.p. theta[j, z]
    probs = theta[np.arange(n_snps)[None, :], Z]      # [n_hap, n_snps]
    H = (rng.random((n_hap, n_snps)) < probs).astype(np.int8)
    return H


def simulate_hmm_genotypes(n_snps, N, seed, K=5, rho=None, base_rate=0.02,
                           n_hotspots=3, hotspot_strength=8.0,
                           rare_variant_skew=0.7, return_phased=False):
    """
    Simulate a diploid genotype matrix from a fastPHASE-style HMM.

    Args:
        n_snps, N: number of SNPs and individuals.
        seed: RNG seed.
        K: number of ancestral clusters.
        rho: optional precomputed per-interval recombination map [n_snps]; if
            None one is synthesized (base_rate + hotspots).
        base_rate, n_hotspots, hotspot_strength: recombination-map params (used
            only if rho is None).
        rare_variant_skew: in (0,1). Larger -> theta drawn from a more skewed
            Beta, producing more low-MAF variants (the regime Gaussian knockoffs
            handle worst). 0.5 gives roughly uniform allele frequencies.
        return_phased: also return the two phased haplotype-allele matrices.

    Returns:
        geno: [n_snps, N] dosage 0/1/2 (float32).
        pos:  [n_snps] integer positions (5 kb spacing scaled by local rho, so
              hotspots sit at wider gaps -- purely cosmetic for windowing).
        info: dict with 'rho', 'maf', and (if return_phased) 'xL','xR' each
              [n_snps, N] of 0/1 alleles.
    """
    rng = np.random.default_rng(seed)
    if rho is None:
        rho = make_recombination_map(n_snps, rng, base_rate=base_rate,
                                     n_hotspots=n_hotspots,
                                     hotspot_strength=hotspot_strength)
    # Cluster frequencies are SKEWED (one dominant ancestral background) so most
    # haplotypes share the common cluster -> low-MAF variants. This decouples the
    # allele-frequency spectrum from the LD strength.
    conc = np.full(K, 0.4)
    conc[0] = 4.0
    alpha = rng.dirichlet(conc)
    # theta[j,k]: allele-1 prob, drawn NEAR 0 or 1 (Beta(s,s), small s) so cluster
    # identity strongly determines the allele -> real LD (nearby SNPs sharing a
    # cluster path are correlated). rare_variant_skew (in (0,1)) controls how
    # deterministic: larger -> more bimodal theta -> stronger LD.
    s = max(0.05, 0.35 * (1.0 - rare_variant_skew) + 0.1)
    theta = rng.beta(s, s, size=(n_snps, K))

    # two haplotypes per individual
    HL = _sample_hmm_haplotypes(N, n_snps, K, alpha, theta, rho, rng)  # [N, n_snps]
    HR = _sample_hmm_haplotypes(N, n_snps, K, alpha, theta, rho, rng)
    xL = HL.T.astype(np.int8)                                  # [n_snps, N]
    xR = HR.T.astype(np.int8)
    geno = (xL + xR).astype(np.float32)                        # dosage 0/1/2

    af = geno.sum(1) / (2 * N)
    maf = np.minimum(af, 1 - af)

    # positions: cumulative, widened at high-rho intervals (visual/hotspot-aware)
    gaps = 5000 * (1.0 + rho / max(base_rate, 1e-9))
    pos = (10_000 + np.cumsum(gaps)).astype(int)

    info = {'rho': rho, 'maf': maf, 'alpha': alpha}
    if return_phased:
        info['xL'] = xL
        info['xR'] = xR
    return geno, pos, info


def ld_decay(geno, pos, max_dist=200_000, n_bins=20):
    """
    Diagnostic: mean r^2 vs physical distance, to confirm realistic LD decay.
    Returns (bin_centers, mean_r2).
    """
    g = geno - geno.mean(1, keepdims=True)
    sd = g.std(1, keepdims=True)
    sd[sd == 0] = 1
    gs = g / sd
    n = geno.shape[0]
    dists, r2s = [], []
    for i in range(n):
        for j in range(i + 1, min(i + 40, n)):
            d = abs(pos[j] - pos[i])
            if d > max_dist:
                break
            r = float((gs[i] * gs[j]).mean())
            dists.append(d); r2s.append(r * r)
    dists = np.array(dists); r2s = np.array(r2s)
    edges = np.linspace(0, max_dist, n_bins + 1)
    centers, means = [], []
    for b in range(n_bins):
        m = (dists >= edges[b]) & (dists < edges[b + 1])
        if m.any():
            centers.append(0.5 * (edges[b] + edges[b + 1]))
            means.append(r2s[m].mean())
    return np.array(centers), np.array(means)


if __name__ == '__main__':
    # quick self-check: LD decays, hotspots break LD, rare variants present
    geno, pos, info = simulate_hmm_genotypes(400, 300, seed=1, return_phased=True)
    print(f"geno {geno.shape}, MAF: min={info['maf'].min():.3f} "
          f"median={np.median(info['maf']):.3f} frac<0.05={np.mean(info['maf']<0.05):.2f}")
    c, r2 = ld_decay(geno, pos)
    print("LD decay r^2 by distance bin:")
    for cc, rr in zip(c[:8], r2[:8]):
        print(f"  {cc/1000:5.0f} kb: r2={rr:.3f}")
    assert (geno.max() <= 2) and (geno.min() >= 0)
    assert info['xL'].shape == geno.shape and info['xR'].shape == geno.shape
    assert np.allclose(info['xL'] + info['xR'], geno)
    print("phased xL+xR == dosage: OK")
    print("OK")
