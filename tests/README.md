# Tests

Run tests from the repository root so Python imports this checkout rather than
an installed upstream `tensorqtl` package.

## Portable hapmixQTL checks

This focused command exercises the default estimator, channel combination,
Meier correction, half-read input preparation, and mixQTL comparator using
fabricated inputs:

```bash
pytest \
  tests/test_hapmixqtl.py \
  tests/test_hapmixqtl_allelic_df.py \
  tests/test_hapmixqtl_meier.py \
  tests/test_half_read_default.py \
  tests/test_mixqtl_replication.py \
  -q
```

The broader method suite also checks permutation behavior, calibration under
simulated nulls, point-estimate input rules, the command line, the Salmon
runner, and quarantine of deprecated variance models:

```bash
pytest \
  tests/test_hapmixqtl.py \
  tests/test_hapmixqtl_allelic_df.py \
  tests/test_hapmixqtl_calibration.py \
  tests/test_hapmixqtl_meier.py \
  tests/test_hapmixqtl_perm_scheme.py \
  tests/test_hapmixqtl_point_estimates.py \
  tests/test_half_read_default.py \
  tests/test_half_read_runner.py \
  tests/test_fitted_variance_quarantine.py \
  tests/fitted_variance/ \
  tests/test_cli.py \
  tests/test_mixqtl_replication.py \
  -q
```

The Meier checks use fixed numeric weights and deterministic fabricated
phenotypes. They do not depend on a stored cohort run, external files, or a
GPU.

## Salmon runner self-test

The runner can build and analyze a temporary miniature dataset:

```bash
python3 scripts/run_hapmixqtl_from_salmon.py --selftest
```

This check requires `Rscript` with edgeR. It creates its inputs under a
temporary directory and reports `SELF-TEST OK` on success.

## Upstream tensorQTL tests

The remaining files exercise upstream tensorQTL modules and their bundled
public or synthetic fixtures. Pytest markers can exclude slower groups:

```bash
pytest tests/ -m "not slow and not integration" -q
```

See `pytest.ini` and `tests/conftest.py` for the available markers and shared
fixtures.
