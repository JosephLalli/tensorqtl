# fastPHASE HMM knockoff — setup & reproducibility

This documents how to install and use the **fastPHASE-fit HMM knockoff** path
used to settle the KFc generator choice (per-gene Gaussian vs. a properly-fit
HMM). The comparison harness is `tests/fastphase_repro.py`; the findings are in
`docs/calibration_findings.md` §8.

> **The shipped method does NOT need any of this.** The production KFc eGene
> filter uses a per-gene **Gaussian** knockoff (`gaussian_knockoff`), which has
> no extra dependencies. fastPHASE + `ray` are only needed to *reproduce the
> generator comparison*, and both `tests/fastphase_repro.py` and
> `tests/hprc_calibration.py` self-skip if they are absent.

## Why fastPHASE

fastPHASE (Scheet & Stephens 2006, *Am. J. Hum. Genet.*) fits the haplotype-cluster
HMM — the same model family Sesia et al. (2019, *Biometrika*) use to construct
HMM knockoffs. Fitting its parameters `(alpha, theta, rho)` on **real** haplotypes
and mapping them into `ko.hmm_knockoffs` gives a *properly-specified* HMM knockoff,
as opposed to our toy `genotype_hmm_knockoffs` EM fit (which under-fits at the
window scale and leaves a null-W bias). This is the fair HMM competitor for the
"is Gaussian actually the right generator?" question.

## Install

```bash
pip install ray
pip install fastphase        # https://pypi.org/project/fastphase/
```

### Required patch (int32 buffer bug)

The PyPI `fastphase` package has a 64-bit integer bug: `calc_func.pyx` declares
a Cython `int32_t` buffer but allocates the array with NumPy's default integer
dtype, which is `int64` on 64-bit platforms. On import/run you get:

```
ValueError: Buffer dtype mismatch, expected 'int32_t' but got 'long'
```

Fix — force the allocation to `int32` in the source and rebuild from the sdist:

```bash
# locate the installed source (or download the sdist and patch there)
python - <<'PY'
import fastphase, pathlib
print(pathlib.Path(fastphase.__file__).parent)
PY

# in calc_func.pyx, change the betaScale allocation:
#   -    betaScale = np.zeros(nLoc, dtype=int)
#   +    betaScale = np.zeros(nLoc, dtype=np.int32)
```

Then rebuild by reinstalling from the patched sdist (a `pip install .` of the
patched source tree — an in-place `cythonize` + hand `cc` compile fails with an
`undefined symbol: PyType_FromMetaclass` link error, so use pip's build):

```bash
pip install /path/to/patched-fastphase-sdist
```

Verify:

```bash
python -c "from fastphase.fastphase_ray import fastphase; print('ok')"
```

## Run the comparison

```bash
# needs pysam + outbound network to the human-pangenomics S3 bucket
python tests/fastphase_repro.py --n_windows 40 --reps 8
```

or via pytest (self-skips if deps/network absent):

```bash
pytest tests/fastphase_repro.py -v
```

## Reference results (provenance)

160 real HPRC v2.0 chr1 windows, N=232 individuals, p=30 variants/window, K=20
HMM clusters, mirror-null selection q=0.10:

**Null-W symmetry** (`frac(W>0) → 0.5`, `mean W → 0` = well-specified):

| generator                     | signal                          |
|-------------------------------|---------------------------------|
| toy genotype-HMM K=8          | `frac(W>0) ≈ 0.20–0.31` (misspecified) |
| fastPHASE fit(K=20) → our gen | `mean W −0.276 → −0.023` (bias fixed) |
| Gaussian shrink=0.1           | `frac(W>0) ≈ 0.556` (well-specified) |

**End-to-end FDR / power** (target FDR 0.10):

| PVE  | fastPHASE-HMM            | Gaussian                 |
|------|-------------------------|--------------------------|
| 0.10 | FDR 0.062, power 0.09, R=6 | FDR 0.014, power 0.23, R=15 |
| 0.15 | FDR 0.042, power 0.11, R=8 | FDR 0.030, power 0.54, R=36 |

## Conclusion

Both generators **control FDR** on real LD, and the fastPHASE fit removes the toy
HMM's null bias — so the HMM knockoff is *valid* when properly fit. But for the
KFc min-p (marginal) statistic the **Gaussian** knockoff delivers **3–5× more
power at matched FDR**. Intuition: the min-p statistic only ever uses each
variant's marginal association, so it rewards a knockoff that is maximally
decorrelated from the real genotype variant-by-variant — exactly the
second-moment target the Gaussian construction optimizes — whereas the HMM spends
its fidelity budget faithfully reproducing the full haplotype process, which the
marginal statistic never reads. **Per-gene Gaussian is therefore the shipped KFc
generator.** The fastPHASE path remains available (and documented here) for
statistics that *do* use joint haplotype structure.
