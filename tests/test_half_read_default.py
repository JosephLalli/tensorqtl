"""Production half-read split must reproduce the evaluated input contract."""
import numpy as np
import pandas as pd
import pytest

from tensorqtl import hapmixqtl as hm
from tests.test_hapmixqtl_point_estimates import _toy


def test_half_read_transform_retains_zero_counts_and_count_unit_offset():
    counts = np.array([[0., 0., 0.], [.5, .5, .5], [1.25, 25., 2500.]])
    libraries = np.array([1., 1e6, 1e12])
    actual = hm.half_read_log_cpm(counts, libraries)
    np.testing.assert_array_equal(actual, np.log2((counts + .5) / (libraries + 1) * 1e6))
    np.testing.assert_allclose(actual[1] - actual[0], 1., atol=1e-14)
    assert np.isfinite(actual).all()


@pytest.mark.parametrize('count_noise', [False, True])
def test_ase_parity_and_admission_boundaries(count_noise):
    pL = np.array([[0., 0., .49, .5, .2, 2., .5]])
    pR = np.array([[0., 2., .5, .5, .3, 0., .49]])
    pT = pL + pR + 10
    pT[0, 0] = 0  # Total channel still admits a zero-read donor.
    yL = pL[..., None] + np.array([0., 1., 2.])
    yR = pR[..., None] + np.array([1., 0., 2.])
    libraries = np.full(7, 2e7)
    old = hm.summaries_from_point_estimates(
        pL, pR, pT, libraries, yL, yR, yL + yR, count_noise=count_noise)
    new = hm.prepare_default_inputs(pL, pR, pT, libraries, yL, yR, count_noise=count_noise)
    np.testing.assert_array_equal(new[0], old[0])
    keep = np.array([[False, False, False, True, True, False, False]])
    np.testing.assert_array_equal(new[2], np.where(keep, old[2], 0))
    assert (new[2][keep] > 0).all()
    np.testing.assert_array_equal(new[3], np.ones_like(pT))
    assert np.isfinite(new[1]).all()


def test_draws_change_only_ase_weights_and_constant_draws_can_disable_ase():
    pL, pR, pT, yL, yR, _, libraries = _toy()
    first = hm.prepare_default_inputs(pL, pR, pT, libraries, yL, yR)
    second = hm.prepare_default_inputs(pL, pR, pT, libraries, yL * 2, yR)
    for i in (0, 1, 3):
        np.testing.assert_array_equal(first[i], second[i])
    assert not np.allclose(first[2], second[2])
    constant = hm.prepare_default_inputs(pL, pR, pT, libraries,
                                        pL[..., None], pR[..., None], count_noise=False)
    assert not constant[2].any()


@pytest.mark.parametrize('field,value', [
    ('pL', -1.), ('pR', np.inf), ('pT', np.nan),
    ('yL', -1.), ('yR', np.nan), ('eff_lib_size', 0.),
    ('eff_lib_size', np.inf), ('kappa', 0.), ('kappa', np.nan),
])
def test_invalid_domains(field, value):
    pL, pR, pT, yL, yR, _, libraries = _toy()
    args = dict(pL=pL, pR=pR, pT=pT, eff_lib_size=libraries, yL=yL, yR=yR, kappa=.5)
    if field == 'kappa':
        args[field] = value
    else:
        args[field] = args[field].copy()
        args[field].flat[0] = value
    with pytest.raises(ValueError):
        hm.prepare_default_inputs(**args)


@pytest.mark.parametrize('field,change', [
    ('pL', lambda x: x[0]), ('pT', lambda x: x[:, :-1]),
    ('yL', lambda x: x[..., 0]), ('yR', lambda x: x[..., :-1]),
    ('yL', lambda x: x[..., :0]), ('eff_lib_size', lambda x: x[:-1]),
])
def test_invalid_shapes(field, change):
    pL, pR, pT, yL, yR, _, libraries = _toy()
    args = dict(pL=pL, pR=pR, pT=pT, eff_lib_size=libraries, yL=yL, yR=yR)
    args[field] = change(args[field])
    with pytest.raises(ValueError):
        hm.prepare_default_inputs(**args)


def test_bed_reader_defaults_to_unit_total_and_preserves_explicit_override(monkeypatch):
    frames = {name: pd.DataFrame([[value, 0.], [2., 3.]], index=['g1', 'g2'], columns=['s1', 's2'])
              for name, value in [('a', 1.), ('t', 2.), ('va', 3.), ('vt', 4.), ('cat', 5.)]}
    pos = pd.DataFrame({'chr': ['chr1', 'chr1'], 'pos': [1, 2]}, index=['g1', 'g2'])
    monkeypatch.setattr(hm, 'read_phenotype_bed', lambda path: (frames[path], pos))
    default = hm.read_hapmixqtl_inputs('a', 't', 'va')
    pd.testing.assert_frame_equal(default[3], pd.DataFrame(1., index=frames['t'].index, columns=frames['t'].columns))
    custom = hm.read_hapmixqtl_inputs('a', 't', 'va', 'vt', 'cat')
    assert custom[3] is frames['vt'] and custom[4] is frames['cat']
    frames['t'] = frames['t'].iloc[:, ::-1]
    with pytest.raises(AssertionError, match='Sample IDs'):
        hm.read_hapmixqtl_inputs('a', 't', 'va')


def test_mapping_matches_manual_benchmark_arm(tmp_path):
    from tests.test_hapmixqtl import _make_dataset
    d = _make_dataset(seed=829, n_samples=48, n_variants=12, n_phenotypes=2)
    pL, pR, pT, yL, yR, yT, libraries = _toy(seed=14, F=2, N=48, D=12)
    pL[:, :2], pR[:, :2] = 0., 3.  # Exercise ASE admission in actual mapping.
    pL[:, 2], pR[:, 2], pT[:, 2] = 0., 0., 0.
    actual = hm.prepare_default_inputs(pL, pR, pT, libraries, yL, yR)
    A, _, Va, _, _ = hm.summaries_from_point_estimates(pL, pR, pT, libraries, yL, yR, yT)
    Va = np.where((Va > 1e-12) & ~((pL < .5) ^ (pR < .5)), Va, 0.)
    manual = (A, np.log2((pT + .5) / (libraries + 1) * 1e6), Va, np.ones_like(pT))
    common = dict(xL_df=d['xL_df'], xR_df=d['xR_df'], verbose=False)
    cov = pd.DataFrame(np.random.RandomState(31).normal(size=(48, 2)), index=d['A_df'].columns)
    nominal, permuted = [], []
    for name, arrays in [('production', actual), ('manual', manual)]:
        frames = [pd.DataFrame(x, index=d['A_df'].index, columns=d['A_df'].columns) for x in arrays]
        args = (d['genotype_df'], d['variant_df'], *frames, d['pos_df'])
        hm.map_nominal(*args, prefix=name, output_dir=str(tmp_path), covariates_df=cov, **common)
        nominal.append(pd.read_parquet(tmp_path / f'{name}.hapmixqtl_pairs.chr1.parquet'))
        permuted.append(hm.map_cis(*args, nperm=40, seed=19, covariates_df=cov, **common))
    # Full frames include effects, SEs, p-values, df, admission and permutation fits.
    pd.testing.assert_frame_equal(nominal[0], nominal[1], check_exact=True)
    pd.testing.assert_frame_equal(permuted[0], permuted[1], check_exact=True)


def test_half_read_total_gibbs_variance_known_answer():
    """Two draws 8 and 12 around a point estimate of 10 reads: the library cancels from the across-draw variance."""
    pT, L, yT = np.array([[10.0]]), np.array([2.0e6]), np.array([[[8.0, 12.0]]])
    draw_var = ((np.log2(12.5) - np.log2(8.5)) / 2) ** 2
    assert hm.half_read_total_gibbs_variance(pT, L, yT, count_noise=False)[0, 0] == pytest.approx(draw_var, rel=1e-12)
    assert hm.half_read_total_gibbs_variance(pT, L, yT)[0, 0] == pytest.approx(draw_var + 1 / (10.5 * np.log(2) ** 2), rel=1e-12)
