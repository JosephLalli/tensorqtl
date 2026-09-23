"""The fitted-variance quarantine holds.

The deprecated variance machinery moved to tensorqtl/fitted_variance.py on
2026-09-23. These tests pin the three properties that make that a quarantine
rather than a rename:

  1. A default-mode run never imports it. This is the one that matters: it is
     what lets the module be deleted later without touching either shipped
     mode, and what stops the deprecated estimators drifting back into the
     default path by accident.
  2. The deprecated names still resolve from `hapmixqtl`, with a
     DeprecationWarning, so historical scripts and the quarantined tests keep
     working unchanged.
  3. The live gates are not shadowed by that resolution hook and do not warn
     on their shipped fast paths.

Property 1 is checked in a SUBPROCESS on purpose: import state is global and
another test in the same session may already have imported the quarantine, so
an in-process check would silently pass for the wrong reason.
"""
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

try:
    from tensorqtl import hapmixqtl
except ImportError:  # flat layout
    sys.path.insert(0, str(REPO / 'tensorqtl'))
    import hapmixqtl


def _run(code):
    """Run `code` in a fresh interpreter with the repo importable."""
    r = subprocess.run([sys.executable, '-c', code], cwd=str(REPO),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout.strip()


def test_importing_hapmixqtl_does_not_import_the_quarantine():
    out = _run(
        "import sys; sys.path.insert(0, 'tensorqtl');"
        "import hapmixqtl;"
        "print('fitted_variance' in sys.modules)")
    assert out == 'False', 'importing hapmixqtl pulled in the quarantine'


def test_a_default_mode_map_cis_never_imports_the_quarantine():
    """The property that makes this a quarantine: a full default-mode scan
    (tau_mode='zero' + se_mode='fitted') touches none of it."""
    out = _run(
        "import sys; sys.path.insert(0, 'tensorqtl'); sys.path.insert(0, '.');"
        "from tests.test_hapmixqtl import _make_dataset;"
        "import hapmixqtl;"
        "d = _make_dataset(seed=7);"
        "r = hapmixqtl.map_cis(d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'],"
        "                      d['Va_df'], d['Vt_df'], d['pos_df'], xL_df=d['xL_df'],"
        "                      xR_df=d['xR_df'], nperm=20, window=1000000, seed=5,"
        "                      verbose=False);"
        "assert len(r) > 0;"
        "print('fitted_variance' in sys.modules)")
    assert out.splitlines()[-1] == 'False', \
        'a default-mode map_cis imported the deprecated fitted-variance module'


@pytest.mark.parametrize('name', sorted(hapmixqtl._QUARANTINED_NAMES))
def test_deprecated_names_resolve_with_a_warning(name):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        obj = getattr(hapmixqtl, name)
    assert obj is not None
    assert len(w) == 1 and issubclass(w[0].category, DeprecationWarning)
    assert 'fitted_variance' in str(w[0].message)


def test_unknown_attributes_still_raise():
    with pytest.raises(AttributeError):
        hapmixqtl.definitely_not_a_real_name


def test_live_gates_are_not_shadowed_and_do_not_warn():
    """_check_variance_model, _library_factor_tensor and _prior_tuple are LIVE
    gates in hapmixqtl, not quarantined names; the resolution hook must not
    intercept them and their shipped fast paths must be silent."""
    for n in ('_check_variance_model', '_library_factor_tensor', '_prior_tuple'):
        assert n not in hapmixqtl._QUARANTINED_NAMES
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        assert hapmixqtl._library_factor_tensor(None, None, 'cpu') is None
        assert hapmixqtl._prior_tuple(None, 'g') is None
        hapmixqtl._check_variance_model('additive', 'zero', None, None)
    assert [x for x in w if issubclass(x.category, DeprecationWarning)] == []


def test_the_quarantine_still_works_when_asked_for():
    """Deprecated does not mean broken: the module must still reproduce
    historical results, which is the only reason it is retained."""
    try:
        from tensorqtl import fitted_variance as fv
    except ImportError:
        import fitted_variance as fv
    assert fv.VARIANCE_MODELS == ('additive', 'two_component', 'library_scaled')
    assert fv.PRIOR_METHODS == ('deciles', 'trend')
    for n in ('_estimate_tau', '_estimate_c_tau', 'estimate_variance_priors',
              'estimate_library_factors', '_channel_weights_estimated'):
        assert callable(getattr(fv, n)), n
