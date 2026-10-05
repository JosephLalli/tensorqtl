"""The DEPRECATED per-gene moment estimator of `tau`, and its denominator.

Moved here from `tests/test_hapmixqtl.py` on 2026-09-23 to finish the
segregation begun by the quarantine: `_estimate_tau` lives in
`tensorqtl/fitted_variance.py` and is reached from `hapmixqtl` only through the
`DeprecationWarning`-raising module `__getattr__`. Neither shipped mode calls
it -- default mode is `tau_mode='zero'`, under which no `tau` exists to
estimate, and mixQTL mode fits its own dispersion at every variant.

What it computed: `tau` is the extra variance a gene carries beyond the
per-observation inferential variance `v_i`, estimated by the method of moments
from the gene's own weighted residuals -- the circularity that deprecated the
whole family. `TestTauDenominator` pins the specific correction that the
denominator must be DerSimonian and Laird's `sum w - sum w^2 / sum w` (the
random-effects moment estimator of between-study variance, here between-donor),
not `(n - q) * mean(w)`, which agrees with it only when the weights are equal.
"""
import numpy as np
import torch

from test_hapmixqtl import _make_gaussian_seed
from tensorqtl.hapmixqtl import _estimate_tau


class TestTauEstimation:

    def test_tau_nonnegative(self, device):
        """Estimated tau is clamped to be non-negative."""
        N = 100
        rng = _make_gaussian_seed(20)
        # Small inferential variance, extra biological dispersion present
        v_inf = np.full(N, 0.1)
        y = rng.normal(0, 1.0, N)  # variance >> v_inf -> positive tau
        y_t = torch.tensor(y, dtype=torch.float64, device=device)
        v_t = torch.tensor(v_inf, dtype=torch.float64, device=device)
        tau = _estimate_tau(y_t, v_t, None, device)
        assert tau.item() >= 0.0

    def test_tau_zero_when_overweighted(self, device):
        """If residual variance is below the weighting scale, tau clamps to 0."""
        N = 100
        rng = _make_gaussian_seed(21)
        # Huge v_inf -> weighted residual variance tiny -> tau=0
        v_inf = np.full(N, 100.0)
        y = rng.normal(0, 0.01, N)
        y_t = torch.tensor(y, dtype=torch.float64, device=device)
        v_t = torch.tensor(v_inf, dtype=torch.float64, device=device)
        tau = _estimate_tau(y_t, v_t, None, device)
        assert tau.item() == 0.0


class TestTauDenominator:

    def test_intercept_only_tau_is_exactly_dersimonian_laird(self, device):
        """With an intercept alone the leverage is h_i = w_i / sum_j w_j, so the
        moment estimator's denominator sum_i w_i(1-h_i) equals DerSimonian and
        Laird's sum w - sum w^2 / sum w. The estimator must match that closed
        form exactly. Dividing by (n-q)*mean(w) instead, as an earlier version
        did, agrees only under equal weights."""
        rng = np.random.RandomState(5)
        n = 60
        v = rng.uniform(0.02, 2.0, n)                       # 100x spread in precision
        y = rng.normal(0, np.sqrt(v + 0.4), n)              # true tau = 0.4
        y_t = torch.tensor(y, dtype=torch.float64, device=device)
        v_t = torch.tensor(v, dtype=torch.float64, device=device)
        tau = float(_estimate_tau(y_t, v_t, None, device))

        w = 1.0 / v
        sw = np.sqrt(w); q = sw / np.linalg.norm(sw)
        rss = float(((y * sw) - q * (q @ (y * sw))) @ ((y * sw) - q * (q @ (y * sw))))
        dl_denom = w.sum() - (w ** 2).sum() / w.sum()
        assert np.isclose(dl_denom, (w * (1 - q ** 2)).sum())      # the identity itself
        assert np.isclose(tau, max(0.0, (rss - (n - 1)) / dl_denom), rtol=1e-6)

        old = max(0.0, (rss / (n - 1) - 1.0) / w.mean())           # the superseded form
        assert not np.isclose(tau, old, rtol=1e-3), (tau, old)

    def test_equal_weights_make_the_two_denominators_agree(self, device):
        """The superseded form was not wrong everywhere: under equal weights the
        two denominators coincide, which is why the error stayed small."""
        rng = np.random.RandomState(6)
        n = 60
        v = np.full(n, 0.25)
        y = rng.normal(0, np.sqrt(v[0] + 0.4), n)
        tau = float(_estimate_tau(torch.tensor(y, dtype=torch.float64, device=device),
                                  torch.tensor(v, dtype=torch.float64, device=device), None, device))
        w = 1.0 / v
        sw = np.sqrt(w); q = sw / np.linalg.norm(sw)
        r = (y * sw) - q * (q @ (y * sw)); rss = float(r @ r)
        old = max(0.0, (rss / (n - 1) - 1.0) / w.mean())
        assert np.isclose(tau, old, rtol=1e-6)
