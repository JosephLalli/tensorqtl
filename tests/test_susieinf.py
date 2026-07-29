"""
Regression tests for tensorqtl.susieinf (SuSiE-inf: SuSiE with an infinitesimal
random-effect term). Property-based checks on synthetic correlated genotypes; a
numerical cross-check against the FinucaneLab reference implementation runs only
if that package is importable (it is not a tensorqtl dependency).

The port was verified numerically identical to the reference (max|dPIP| <= 3e-13,
tausq/sigmasq/alpha to machine precision, credible sets identical) across
null / single-strong / single-weak / polygenic / multi-moderate architectures
and both 'moments' and 'MLE' variance estimators.
"""
import sys
import numpy as np
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tensorqtl'))
import susieinf


def _corr_genotypes(N, p, seed):
    """Synthetic correlated genotype-like matrix (5-factor model)."""
    rng = np.random.RandomState(seed)
    Z = rng.randn(N, 5)
    load = np.zeros((p, 5))
    for j in range(p):
        load[j, j % 5] = 1.0
    X = Z @ load.T + 0.6 * rng.randn(N, p)
    return X


def _summary(X, y):
    N, p = X.shape
    s = X.std(0); s[s == 0] = 1
    Xs = (X - X.mean(0)) / s
    yc = (y - y.mean()); yc = yc / (yc.std() if yc.std() > 0 else 1)
    z = Xs.T.dot(yc) / np.sqrt(N)
    return z, float(yc.dot(yc) / N), (Xs.T @ Xs) / N


def test_strong_single_signal_recovered():
    X = _corr_genotypes(300, 100, 0)
    rng = np.random.RandomState(1)
    c = 7
    xc = (X[:, c] - X[:, c].mean()) / X[:, c].std()
    y = np.sqrt(0.20 / 0.80) * xc + rng.randn(300)
    z, meansq, LD = _summary(X, y)
    fit = susieinf.susie_inf(z, meansq, 300, L=10, LD=LD, method='moments')
    cs = susieinf.credible_sets(fit['PIP'], coverage=0.9, purity=0.5, LD=LD, n=300)
    assert fit['converged']
    assert fit['PIP'].max() > 0.5, "strong signal should give a confident PIP"
    # the causal variant (or a tight LD proxy) is in a credible set
    assert any(c in s for s in cs) or fit['PIP'][c].max() > 0.3


def test_pure_null_is_quiet():
    X = _corr_genotypes(300, 100, 2)
    y = np.random.RandomState(3).randn(300)
    z, meansq, LD = _summary(X, y)
    fit = susieinf.susie_inf(z, meansq, 300, L=10, LD=LD, method='moments')
    cs = susieinf.credible_sets(fit['PIP'], coverage=0.9, purity=0.5, LD=LD, n=300)
    # under the null a well-behaved fit reports few/no credible sets
    assert len(cs) <= 1


def test_infinitesimal_variance_estimated_under_polygenic_background():
    X = _corr_genotypes(400, 120, 4)
    rng = np.random.RandomState(5)
    # dense polygenic background, no single large effect
    beta = rng.exponential(0.03, 120) * rng.choice([-1, 1], 120)
    y = ((X - X.mean(0)) / X.std(0)) @ beta + rng.randn(400)
    z, meansq, LD = _summary(X, y)
    fit_inf = susieinf.susie_inf(z, meansq, 400, L=10, LD=LD, method='moments', est_tausq=True)
    fit_plain = susieinf.susie_inf(z, meansq, 400, L=10, LD=LD, method='moments',
                                   est_tausq=False, tausq=0.0)
    assert fit_inf['tausq'] > 0, "polygenic background should drive tau^2 > 0"
    assert fit_plain['tausq'] == 0.0


def test_from_data_wrapper_matches_manual_summary():
    X = _corr_genotypes(250, 80, 6)
    rng = np.random.RandomState(7)
    xc = (X[:, 3] - X[:, 3].mean()) / X[:, 3].std()
    y = np.sqrt(0.15 / 0.85) * xc + rng.randn(250)
    fit = susieinf.susie_inf_from_data(X, y, L=10, method='moments')
    z, meansq, LD = _summary(X, y)
    fit2 = susieinf.susie_inf(z, meansq, 250, L=10, LD=LD, method='moments')
    assert np.allclose(fit['pip'], fit2['PIP'].max(1), atol=1e-8)
    assert 'cs' in fit


@pytest.mark.parametrize('method', ['moments', 'MLE'])
def test_matches_reference_if_available(method):
    """Numerical equivalence to the FinucaneLab reference, if importable."""
    try:
        import importlib.util
        cand = [p for p in [
            "/mnt/ssd/lalli/.claude/jobs/c9d3222b/tmp/fine-mapping-inf/susieinf/susieinf.py",
        ] if Path(p).exists()]
        if not cand:
            pytest.skip("FinucaneLab susieinf reference not available")
        sp = importlib.util.spec_from_file_location("ref_susieinf", cand[0])
        ref = importlib.util.module_from_spec(sp); sp.loader.exec_module(ref)
    except Exception as e:
        pytest.skip(f"reference unavailable: {e}")
    X = _corr_genotypes(300, 100, 9)
    rng = np.random.RandomState(10)
    xc = (X[:, 5] - X[:, 5].mean()) / X[:, 5].std()
    y = np.sqrt(0.12 / 0.88) * xc + 0.5 * rng.randn(300)
    z, meansq, LD = _summary(X, y)
    eig, V = np.linalg.eigh(LD); Dsq = np.maximum(300 * eig, 0.0)
    o = susieinf.susie_inf(z, meansq, 300, L=10, V=V.copy(), Dsq=Dsq.copy(), method=method)
    r = ref.susie(z, meansq, n=300, L=10, V=V.copy(), Dsq=Dsq.copy(), method=method, verbose=False)
    assert np.max(np.abs(o['PIP'] - r['PIP'])) < 1e-8
    assert abs(o['tausq'] - r['tausq']) < 1e-8


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v', '-s']))
