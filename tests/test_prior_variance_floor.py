"""
Tests for susie(prior_variance_floor=...) -- the optional positive floor on the
single-effect prior variance.

Background. By default SuSiE snaps a single effect's estimated prior variance to
EXACTLY 0 when there is no signal (check_null_threshold parsimony). That makes
every per-variant Bayes factor identical -> alpha exactly uniform -> PIP
identically 0 for a null effect. Harmless for plain fine-mapping, but it
manufactures a degenerate atom at W=0 for any variant-contrast statistic (e.g. a
knockoff maxPIP(orig)-maxPIP(knockoff)), which breaks the statistic's null
distribution. `prior_variance_floor > 0` clamps V to the floor instead of
snapping to 0, keeping a continuous, finite-sample-informative posterior.

These tests assert: (1) the default (0.0) is a no-op (unchanged behavior); (2) a
positive floor removes the null atom (continuous W); (3) signal is still
detected; (4) the floor does not perturb a clearly-signalled fit (V is only
clamped when it would otherwise collapse).
"""

import numpy as np
import torch
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorqtl.susie as susie
import tensorqtl.knockoffs as ko


def _corr_genotypes(N, p, seed):
    rng = np.random.RandomState(seed)
    Z = rng.randn(N, 5)
    load = np.zeros((p, 5))
    for j in range(p):
        load[j, j % 5] = 1.0
    Xd = np.clip(np.round((Z @ load.T + 0.4 * rng.randn(N, p)) - (Z @ load.T).min(0)), 0, 2)
    return Xd.astype(np.float32), rng


def _null_W(Xt, floor, M=12, seed=0):
    p = Xt.shape[1]
    rng = np.random.RandomState(seed)
    W = []
    for m in range(M):
        Xk = ko.gaussian_knockoff(Xt, shrink=0.1, generator=torch.Generator().manual_seed(m))
        y = torch.tensor(rng.randn(Xt.shape[0]), dtype=torch.float32).reshape(-1, 1)
        Xaug = torch.cat([Xt, Xk], 1)
        res = susie.susie(Xaug, y, L=5, estimate_prior_variance=True,
                          prior_variance_floor=floor, max_iter=100)
        pip = np.asarray(res['pip'])
        W.append(pip[:p].max() - pip[p:].max())
    return np.array(W)


def test_default_is_noop_atom_present():
    """Default prior_variance_floor=0.0 keeps the exact-0 snap -> null W is a
    point mass at 0 (the atom)."""
    Xd, _ = _corr_genotypes(150, 30, seed=0)
    W = _null_W(torch.tensor(Xd), floor=0.0)
    assert np.mean(np.abs(W) < 1e-9) > 0.8, "default should retain the null atom"


def test_floor_removes_atom():
    """A positive floor makes the null W continuous (no exact ties)."""
    Xd, _ = _corr_genotypes(150, 30, seed=0)
    W = _null_W(torch.tensor(Xd), floor=1e-2)
    assert np.mean(np.abs(W) < 1e-9) < 0.1, "floor should remove the atom"
    assert np.unique(np.round(W, 8)).size >= 8, "W should be continuous/varied"


def test_floor_still_detects_signal():
    """With the floor on, a real cis signal is still fine-mapped (high maxPIP)."""
    Xd, rng = _corr_genotypes(200, 30, seed=1)
    c = 5
    xc = (Xd[:, c] - Xd[:, c].mean()) / (Xd[:, c].std() + 1e-9)
    y = torch.tensor(np.sqrt(0.15 / 0.85) * xc + rng.randn(200),
                     dtype=torch.float32).reshape(-1, 1)
    res = susie.susie(torch.tensor(Xd), y, L=5, estimate_prior_variance=True,
                      prior_variance_floor=1e-2, max_iter=100)
    assert np.asarray(res['pip']).max() > 0.3, "signal should still be detected"


def test_floor_keeps_causal_top_but_adds_baseline():
    """The floor is NOT a free no-op for plain fine-mapping: keeping the empty
    L-1 effects 'on' (V=floor instead of 0) adds a small BASELINE PIP to every
    variant. The causal variant must still dominate (top rank, high PIP), but the
    non-causal baseline is nonzero (and larger than the unfloored ~0 baseline).
    This trade -- atom removed, small symmetric baseline gained -- is exactly why
    the floor is OFF by default and is intended for variant-contrast statistics
    (knockoffs), where the baseline is symmetric across original/knockoff and the
    signal still dominates the max."""
    Xd, rng = _corr_genotypes(300, 25, seed=2)
    c = 3
    xc = (Xd[:, c] - Xd[:, c].mean()) / (Xd[:, c].std() + 1e-9)
    y = torch.tensor(np.sqrt(0.25 / 0.75) * xc + rng.randn(300),
                     dtype=torch.float32).reshape(-1, 1)
    r0 = susie.susie(torch.tensor(Xd), y, L=5, estimate_prior_variance=True,
                     prior_variance_floor=0.0, max_iter=200)
    rf = susie.susie(torch.tensor(Xd), y, L=5, estimate_prior_variance=True,
                     prior_variance_floor=1e-2, max_iter=200)
    p0 = np.asarray(r0['pip']); pf = np.asarray(rf['pip'])
    # causal variant still the top signal under both
    assert p0.argmax() == c and pf.argmax() == c
    assert pf[c] > 0.8, "causal PIP must stay high with the floor"
    # unfloored non-causal baseline ~0; floored baseline is nonzero (the trade)
    noncausal = np.delete(np.arange(len(pf)), c)
    assert p0[noncausal].max() < 0.05, "unfloored non-causal PIPs ~ 0"
    assert pf[noncausal].mean() > 0.05, "floored empty effects add a baseline PIP"


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
