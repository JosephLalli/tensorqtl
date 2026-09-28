"""Meier's correction of the combined standard error for estimated channel
weights (default mode; shipped 2026-09-27, user decision).

The combined slope is the inverse-variance weighted mean of the two channel
slopes with weights w_k = 1/se_k^2 estimated from the same residuals it
combines (a Graybill-Deal mean), so its plug-in variance 1/(w_a + w_t) is too
small. Meier (1953) gives, to first order in 1/nu_k, true over expected
reported variance M = 1 + 4 f_a f_t (1/nu_a + 1/nu_t), f_k = w_k/(w_a + w_t)
the weight shares and nu_k the channels' residual df (_meier_factor). The
combined SE becomes se_c sqrt(M) and the combined t falls by sqrt(M) on the
unchanged Welch-Satterthwaite dof, in map_nominal, map_cis's observed scan,
every permutation and the lead alike.

Two pins, through the shipped code:
  (a) known answer against the stored exact-model run
      (combined_reference_exact_model_20260927, written under the
      uncorrected code): on records its seeded streams regenerate, the
      script's uncorrected() recovers the stored t, and the shipped SE and p
      are the stored statistic corrected by M from the stored shares and df;
  (b) where fewer than two channels carry weight, or no scale is fitted,
      the combined SE is the single channel's SE or the inverse-variance
      value computed here from the reported channel SEs.
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tensorqtl.core import get_t_pval
from tests.test_hapmixqtl_allelic_df import _bank, _nominal

SEED = 42
REPO = Path(__file__).resolve().parents[1]
EXACT_MODEL = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy/combined_reference_exact_model_20260927')


# (a) -----------------------------------------------------------------------

@pytest.fixture(scope='module')
def exact_model():
    """The exact-model run's inputs, through its own script (1-2 min on the
    GPU); skipped without the GPU, the script's inputs or the stored run."""
    if not torch.cuda.is_available():
        pytest.skip('the exact-model record needs the GPU')
    if not (EXACT_MODEL / 'genes').is_dir() or not (EXACT_MODEL / 'gate.json').exists():
        pytest.skip(f'stored run not found at {EXACT_MODEL}')
    sys.path.insert(0, str(REPO / 'scripts'))
    try:
        import combined_reference_exact_model as X
    except Exception as e:                        # a missing deployment input
        pytest.skip(f'combined_reference_exact_model not importable here: {e!r}')
    assert Path(X.OUT).resolve() == EXACT_MODEL.resolve()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            S = X.load()
    except (SystemExit, FileNotFoundError, OSError) as e:
        pytest.skip(f'exact-model inputs not loadable here: {e!r}')
    return X, S


def _record(X, S, gene, level, config):
    """The stored record's inputs (gene, replicate 0, level, configuration),
    rebuilt from the script's seeded streams, and its stored fit."""
    k = S['genes'].index(gene)
    ci, j = list(X.CONFIGS).index(config), X.LEVELS.index(level)
    Z = np.load(X.OUT / 'genes' / f'{gene}.npz')
    assert j in Z['levels'].tolist()
    inf = np.where(S['V']['split'][0][k] > X.EPS)[0]
    assert len(inf) == int(Z['n_a'])
    n = len(inf) if level == 'all' else level
    I, sel = S['I'], S['variants'][gene]
    G_t = torch.tensor(I['dos'][sel], dtype=torch.float32, device=S['dev'])
    X_t = torch.tensor(I['xL'][sel].astype(np.float32) - I['xR'][sel], device=S['dev'])
    rec = X.inputs(S, k, config, n, *X.draw(k, 0, inf, len(I['order'])))
    stored = dict(t=Z['t'][ci, j, 0], t_a=Z['t_a'][ci, j, 0], t_t=Z['t_t'][ci, j, 0],
                  f_a=Z['f_a'][ci, j, 0], nu_ws=Z['nu_ws'][ci, j, 0],
                  nu_a=np.full(X.N_VAR, float(Z['dof_a'][ci, j, 0])),
                  nu_t=np.full(X.N_VAR, float(Z['dof_t'][ci, j, 0])))
    return (G_t, X_t) + tuple(rec), stored, n


@pytest.mark.parametrize('gene,level', [('ASPHD1', 15), ('ABCC4', 'all')])
@pytest.mark.parametrize('config', ['split', 'unit'])
def test_known_answer_against_the_stored_exact_model_run(exact_model, gene, level, config):
    """Replicate 0 of one gene at n_a = 15 and one at all donors, both
    weightings, through the run's own calculate_hapmixqtl_nominal call on
    the record its seeded streams regenerate. The script's uncorrected()
    (t sqrt(M), from the shipped fit's info) returns the stored uncorrected
    t; the shipped M is the one from the stored allelic share and channel
    df; the shipped slope_se is the stored statistic's SE (|slope| / |t|,
    the slope being untouched by the correction) times sqrt(M); and the
    shipped pval_nominal is the script's own Meier reference (pvals).
    PRECISION: the stored t is float32 and the shipped SE arithmetic is
    float32 (se_c and sqrt(M) are each rounded to float32 before they are
    multiplied), so agreement is at float32 resolution (eps 1.2e-7), not
    1e-9. Measured on GPU 1, 2026-09-27, maxima over each record's 200
    variants: recovered t 1.4e-7 to 1.7e-7 relative (ASPHD1 1.4e-7), M
    1.3e-9 to 3.0e-9, dof_nominal 4.4e-8 to 5.2e-8 (nu_WS is stored in
    float32), slope_se 1.2e-7 to 1.4e-7, p 3.9e-7 to 7.4e-7 (|t| times the
    log-p slope times the float32 rounding of t). Asserted at 5e-7, 1e-8,
    2e-7, 5e-7 and 5e-6."""
    X, S = exact_model
    args, st, n = _record(X, S, gene, level, config)
    o, info = X.fit(S, *args)                        # shipped: corrected t and SE
    u, _ = X.uncorrected(o, info)
    assert info['allelic_admitted'] and info['dof_a'] == n - 1 and info['dof_t'] == X.OLD_DOF
    p_ref, ref = X.pvals(st)
    M_ref = ref['meier_factor']
    both = st['f_a'] > 0
    assert (M_ref[both] > 1).all() and (M_ref[~both] == 1).all() and both.any()
    t_st = st['t'].astype(np.float64)
    dof = info['dof_nominal'].cpu().numpy()
    p = get_t_pval(o[0], dof)                        # map_nominal's conversion
    d_t = float(np.max(np.abs(u[0] / t_st - 1)))
    d_M = float(np.max(np.abs(info['meier_factor'].cpu().numpy() / M_ref - 1)))
    d_nu = float(np.max(np.abs(dof / st['nu_ws'] - 1)))
    d_se = float(np.max(np.abs(o[2] / (np.abs(o[1]) / np.abs(t_st) * np.sqrt(M_ref)) - 1)))
    d_p = float(np.max(np.abs(p / p_ref['Meier'] - 1)))
    print(f'\nknown answer {gene} n_a={n} {config}: recovered t rel {d_t:.1e}; M rel {d_M:.1e}; '
          f'dof_nominal rel {d_nu:.1e}; slope_se rel {d_se:.1e}; p rel {d_p:.1e}; median M {np.median(M_ref):.4f}')
    assert d_t < 5e-7 and d_M < 1e-8 and d_nu < 2e-7 and d_se < 5e-7 and d_p < 5e-6


# (b) -----------------------------------------------------------------------

@pytest.mark.parametrize('case', ['allelic_off', 'no_phase', 'below_floor', 'allelic_only', 'model', 'robust'])
def test_no_factor_without_two_fitted_channels(tmp_path, case):
    """allelic_off (keep_a_df all False) and below_floor (n_a = 10 against
    the floor of 15): the combination is the total channel, taken verbatim.
    no_phase: the allelic channel is admitted but carries no weight.
    allelic_only (keep_t_df all False): the combination is the allelic
    channel. se_mode model / robust: no fitted scale, so the combined SE is
    the inverse-variance 1/sqrt(1/se_a^2 + 1/se_t^2) of the reported
    channel SEs, computed here; with two channels M would be above 1."""
    d = _bank(SEED + 21, N=50, V=5, R=2, n_a=10 if case == 'below_floor' else 30, n_cov=2)
    off = pd.DataFrame(False, index=d['A_df'].index, columns=d['A_df'].columns)
    if case == 'no_phase':
        d = dict(d, xL_df=None, xR_df=None)
    kw = dict(allelic_off=dict(keep_a_df=off), allelic_only=dict(keep_t_df=off),
              model=dict(se_mode='model'), robust=dict(se_mode='robust')).get(case, {})
    with pytest.warns(RuntimeWarning) if case in ('model', 'robust') else contextlib.nullcontext():
        res = _nominal(d, tmp_path, **kw)
    se, se_a, se_t = (res[c].astype(np.float64).values for c in ('slope_se', 'slope_a_se', 'slope_t_se'))
    if case in ('allelic_off', 'below_floor'):
        assert not res['allelic_admitted'].any()
        assert np.array_equal(se, se_t) and np.array_equal(res['pval_nominal'], res['pval_t'])
    elif case == 'no_phase':
        assert res['allelic_admitted'].all()
        assert np.allclose(se, se_t, rtol=1e-6, atol=0)
    elif case == 'allelic_only':
        ok = np.isfinite(se_a)
        assert ok.any() and np.allclose(se[ok], se_a[ok], rtol=1e-6, atol=0)
    else:
        w_a = np.where(np.isfinite(se_a) & (se_a > 0), 1 / se_a ** 2, 0.0)
        assert (w_a > 0).any()
        assert np.allclose(se, 1 / np.sqrt(w_a + 1 / se_t ** 2), rtol=1e-6, atol=0)
