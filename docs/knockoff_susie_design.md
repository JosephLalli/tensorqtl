# Knockoff-calibrated SuSiE for tensorQTL — integration spec

Status: design. Target branch: `claude/susie-knockoff-calibration-IQ6Za`.
Depends on: `tensorqtl/knockoffs.py`.

---

## ⚠️ REVISION NOTICE (supersedes the CS-level design below)

External expert review found the **credible-set-level FDR** procedure originally
specified here (§§2–4 below) to be **not a valid knockoff procedure**. The core
defect, now confirmed empirically
(`tests/test_knockoffs.py::TestSwapEquivariance`):

> The CS-level statistic extracts credible sets from the **original columns
> only** of an augmented `[X, X̃]` fit. Under the model-X swap (exchange every
> `X_j` with `X̃_j`), a valid statistic must map to its negation via a
> deterministic hypothesis correspondence. This one does **not** — the credible
> set for a real signal *disappears* into the "knockoff" block rather than
> negating. So the pooled negatives are **not** valid negative controls, and
> `knockoff_qval ≤ fdr` does **not** control FDR.

Also invalid as originally written: mean-W derandomization (must use Ren–Barber
e-values), the calibration estimator (`ΣV/ΣR` → must be mean per-replicate FDP),
doubling `L`, and describing KFc as rigorous CS-level precedent.

**What is valid and now implemented as v1 — Path A, eGene-level FDR.** The
gene-level statistic `W_g = max PIP(orig block) − max PIP(knockoff block)` tests
the FIXED hypothesis "gene g has no cis signal" and IS swap-antisymmetric
(verified). Genes are selected at genome-wide FDR via knockoff+ with e-value
derandomization; ordinary SuSiE then localizes signal *within* selected genes.
The reported unit is the **gene**, not the credible set. Implemented in
`susie.map_egenes_knockoffs`.

The CS-level code (`knockoffs.cs_level_W`, `pooled_cs_qvalues`,
`susie.map_knockoffs`) is **retained but relabeled experimental / NOT
FDR-controlled** — an exploratory calibration score only.

Corrections applied to the shared core regardless of path:
- `augmented_susie_fit`: pass `L` (not `2L`).
- `select_egenes`: Ren–Barber e-value derandomization (no W-averaging).
- `calibration_report`: mean per-replicate FDP `(1/B)Σ Vb/max(Rb,1)` + the
  complete-null `P(R>0)`.
- Gaussian generator explicitly documented as *approximate* (pipeline
  development / benchmarking), not a basis for exact FDR on genotype data; an
  HMM / reference-panel haplotype generator is likely required for real-data
  credibility.

### Validity status (reviewer-agreed wording)

> The gene-level statistics satisfy the required deterministic antisymmetry
> within each gene. Formal genome-wide knockoff+ FDR control additionally
> requires a joint null sign-flip property across genes. Independent per-gene
> knockoff generation does not obviously provide that property when cis windows
> overlap, and independent generator randomness alone is insufficient to prove
> it. We therefore compare independent and shared knockoff constructions in
> simulations with overlapping windows and correlated phenotypes while pursuing
> a formal reduction for the overlapping gene-level hypotheses. Until that issue
> is resolved, results are described as **empirically calibrated**, not
> theorem-backed genome-wide FDR control.

Key correction from the second review: independent per-gene knockoff *randomness*
does NOT establish the joint sign property — the property comes from
distributional exchangeability of `(X, X̃, Y)`, not from independent RNG streams.
Under independent per-gene knockoffs a shared SNP has *different, incompatible*
knockoffs across the genes whose windows contain it, so "swap gene A" is not
induced by any coherent swap of a single `(X, X̃)` object. A **shared
chromosome-wide (or LD-block-wide) knockoff** — one `X̃` reused by every gene —
restores coherence and is the construction we test; but even it does not by
itself invoke a standard knockoff theorem, because the discovery units are
phenotype-specific *union* nulls over overlapping feature sets, which needs its
own reduction (closer to simultaneous/union-null knockoffs than to standard
group-sparse multitask knockoffs; Dai–Barber is related but not an exact match).
Simulations can falsify or reveal inflation and support an empirical-calibration
claim; they cannot substitute for that reduction.

### Calibration harness (Freedman–Lane, reviewer-agreed wording)

> Null calibration uses a Freedman–Lane-style residual permutation conditional
> on nuisance covariates. A **common sample permutation** is applied to the full
> gene-by-sample residual matrix so that cross-gene residual correlation is
> preserved. Restricted exchangeability blocks will be supported where required.
> Expression-derived latent covariates (PEER/expression PCs) will be either held
> fixed or re-estimated according to an explicitly reported calibration mode.

Implemented (`susie.map_egenes_knockoffs`, `permute_null=True`): one shared row
permutation of the covariate-residualized phenotype across all genes, in residual
space. NOT yet implemented: restricted exchangeability blocks (relatedness,
batch, strata), and the `covariate_mode = {"fixed","refit_latent"}` policy for
expression-derived factors — currently equivalent to "fixed" (covariates given).
The overlap-stress harness (`tests/overlap_calibration_harness.py`) builds
overlapping windows + tunable cross-gene phenotype correlation
`Y_B = ρ Y_A + √(1−ρ²)ε` (near-duplicated as ρ→1) and compares per-gene vs shared
knockoffs, reporting mean per-replicate FDP, `P(R>0)`, and Monte-Carlo intervals.

Still-open items: the overlapping-gene joint-sign *reduction* (the central
unresolved inferential problem), restricted-exchangeability permutation, the
latent-covariate calibration policy, real wall-time/convergence benchmarking, and
a scalable shared-knockoff generator (block-Gaussian or HMM) for true
chromosome scale. Path C (sample-splitting) is dropped per project scope.

The sections below are **retained as the record of the original CS-level design
and why it failed**; they do not describe a valid procedure.

---

## 1. What we are controlling, and why the aggregation is what it is

**Goal (user's words):** report the set of credible sets whose discovery is
estimated to hold a target FDR (e.g. 5%) — "q=0.05 really means 5% of my calls
are false."

**Two facts force the architecture:**

1. **Knockoffs are generated per gene.** Each cis-window has its own LD, so `X̃`
   is built from that window's genotypes, one gene at a time. Not in dispute.

2. **FDR must be controlled by pooling W across genes, not per gene.** The
   knockoff+ filter has a *detection floor*: with `k` discoveries the smallest
   certifiable FDR is `1/k` (the `+1` offset over `k` positives). A single gene
   yields ~1–3 credible sets, so per-gene CS-level q=0.05 is arithmetically
   impossible (a 2-CS gene floors at 0.5). The only way "q=0.05 means 5%" is
   achievable is to pool the per-CS statistics across all genes and threshold
   once — thousands of discoveries, floor ≈ 1/(#CS genome-wide) ≪ 0.05.

**Validity under W heterogeneity (the reviewed concern).** Pooling is valid even
though W magnitudes vary wildly across genes, because the knockoff guarantee
depends only on the **sign symmetry** of null statistics, not on their scale.
Each gene's null CSs are independently sign-symmetric (`±|W|` equally likely);
the pooled null set is therefore sign-symmetric at every threshold, so the
pooled knockoff+ estimate `FDP̂(t) = (1 + #{W ≤ −t}) / #{W ≥ t}` controls FDR
regardless of cross-gene scale differences.

What pooling costs (not a validity failure, but must be documented):
- **Global ≠ conditional FDR.** 5% overall does *not* mean each gene is at 5%;
  false discoveries may concentrate in a stratum. Same property as BH q-values.
- **Power concentrates in strong loci.** High-W genes dominate the selection.
  For eQTL this is usually acceptable (strong eGenes are the trustworthy ones);
  a v2 within-gene standardization knob (§6) can rebalance if needed.

**The real risk pooling is vulnerable to is knockoff *quality*, not W scale.** If
one gene's `Σ` is badly estimated, its null CSs get systematically positive W
(knockoff too easy to beat) → its nulls are no longer sign-symmetric → pooling
imports anti-conservative bias. Mitigations: mandatory shrinkage, optional HMM
generator, and the null-permutation calibration gate (§5). This is the small-N
`Σ`-estimation problem, and it is the thing the calibration report must catch.

**Precedent.** KFc (Wang et al. 2023) already estimates its gene-level FDP by a
sliding window over the *genome-wide* W distribution — i.e. it pools. Our
approach is the same move done at the credible-set level on individual-level
SuSiE output, emitting a rigorous per-CS knockoff q-value rather than KFc's
`PIP × (1 − FDP)` heuristic multiply.

---

## 2. Output object: per-CS knockoff q-value (not a modified PIP)

We do **not** overwrite SuSiE's PIP (a Bayesian posterior) with a frequentist
factor. We report, per credible set, both numbers:

| column | meaning |
| --- | --- |
| `phenotype_id`, `variant_id` | as in `susie.map` today |
| `pip`, `af`, `cs_id` | as in `susie.map` today (SuSiE's Bayesian output, unchanged) |
| `cs_W` | the CS-level knockoff statistic for this CS (from the gene's fit) |
| `knockoff_qval` | the pooled knockoff q-value for this CS's discovery |
| `selected` | bool: `knockoff_qval <= fdr` |

The **per-CS knockoff q-value** is, for a CS with statistic `W_k` drawn from the
pooled set `W_all`:

```
qval(W_k) = min over thresholds t <= W_k of  FDP̂(t; W_all)
          = min_{t <= W_k}  (1 + #{W_all <= -t}) / max(1, #{W_all >= t})
```

(monotonized so q is non-increasing in W). Reporting `knockoff_qval <= fdr`
yields a set of CSs with the target genome-wide FDR. `cs_W` and `knockoff_qval`
sit *alongside* `pip`, giving each CS a calibrated frequentist confidence next to
its Bayesian one.

---

## 3. Control flow (two-pass: per-gene fit, then global threshold)

Because the threshold is global, we cannot decide a gene's CSs until all genes'
W's are in hand. So the driver is **two-pass**:

```
PASS 1 (per gene, parallelizable):
  for each gene:
    X   = residualized dosage [N, p]              # exactly as susie.map today
    for r in 1..M_knockoffs:                       # M draws for derandomization
        Xk  = knockoffs.gaussian_knockoff(X, shrink, seed=r)   # or hmm
        res = susie.susie([X, Xk], y, L=2L, intercept=..., ...) # augmented fit
        cs_r = knockoffs.cs_level_W(res, p, stat='pip')         # per-CS W (orig cols only)
    aggregate the M draws -> one (cs_id -> W, member variant_ids, pip) record per CS
    stash gene's CS records; also stash SuSiE's normal per-CS output

PASS 2 (global):
  W_all = concat of cs_W across ALL genes
  for each CS: knockoff_qval = qval(cs_W; W_all)      # §2 formula
  selected = knockoff_qval <= fdr
  return the summary table with knockoff_qval + selected columns
```

`M_knockoffs` (derandomization) reduces run-to-run selection variance; the M
draws are aggregated per gene via Ren–Barber e-values (already in
`knockoffs.derandomize_cs`) OR, simplest for v1, by averaging each CS's W across
draws (matched by original-member set). Start with mean-W aggregation; upgrade to
e-value aggregation if stability is inadequate.

**Cost:** `M × (SuSiE fit on 2p columns)` per gene. With `M=5` and 2× columns,
~10× a plain SuSiE fit — still far below the `nperm=10000` cis permutation pass
tensorQTL already runs per gene, so affordable.

---

## 4. Function signatures

### 4.1 New driver in `susie.py` (or a thin wrapper module)

```python
def map_knockoffs(
    genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df,
    # --- knockoff calibration params ---
    fdr=0.05,                     # target genome-wide CS-level FDR
    n_knockoffs=5,                # M derandomization draws per gene
    knockoff='gaussian',          # 'gaussian' | 'hmm' (hmm: later phase)
    shrink=0.05,                  # covariance shrinkage (Gaussian generator)
    w_stat='pip',                 # importance: 'pip' | 'max_alpha'
    knockoff_offset=1,            # knockoff+ (1) vs knockoff (0)
    seed=0,
    emit_diagnostics=True,        # attach per-gene W spread, est. FDP, etc.
    # --- passthrough to susie.map / susie.susie (unchanged semantics) ---
    paired_covariate_df=None, L=10, scaled_prior_variance=0.2,
    estimate_residual_variance=True, estimate_prior_variance=True,
    tol=1e-3, coverage=0.95, min_abs_corr=0.5,
    maf_threshold=0, max_iter=200, window=1000000,
    logger=None, verbose=True, warn_monomorphic=False,
):
    """
    SuSiE fine-mapping with knockoff-calibrated credible-set FDR control.

    Returns:
        summary_df: one row per (credible set member), as susie.map's summary,
            plus columns: cs_W, knockoff_qval, selected.
        diagnostics: dict (per-gene W distribution, calibration inputs) if
            emit_diagnostics, else None.
    """
```

Design choices baked into the signature:
- **Mirrors `susie.map` exactly** for every shared argument, so it is a drop-in
  superset. A caller who ignores the knockoff args gets standard behavior plus
  the calibration columns.
- **`fdr` is the sensitivity dial.** Lower = stricter/fewer CSs.
- **`knockoff='gaussian'` default, `'hmm'` reserved.** Gaussian is dependency-
  free and validates the whole pipeline; HMM (SNPknock, GPLv3, optional runtime
  dep) is a swappable generator added later behind the same interface.

### 4.2 Augmented-fit helper (new, in `knockoffs.py`)

```python
def augmented_susie_fit(susie_module, X_t, y_t, Xk_t, L, **susie_kwargs):
    """
    Fit SuSiE on [X, Xk] and return (res, p). Sets L -> handles the doubled
    column space. Purity/CS reporting must use ORIGINAL columns only (caller
    slices res['alpha'][:, :p] for genotype-LD purity, as hapmixqtl.map_susie
    already does).
    """
```

### 4.3 Pooling + q-value (new, in `knockoffs.py`)

```python
def pooled_cs_qvalues(per_gene_cs, fdr, offset=1):
    """
    per_gene_cs: list of per-CS records across ALL genes, each with a scalar
        'W' (aggregated over knockoff draws), plus carry-through fields
        (phenotype_id, cs_id, member variant indices/ids, pip).
    Returns the same records with 'knockoff_qval' and 'selected' filled in,
    using the pooled W distribution (§2 formula).
    """
```

`knockoffs.py` already has `gaussian_knockoff`, `cs_level_W`,
`filter_credible_sets`, `knockoff_threshold`, `derandomize_cs`,
`calibration_report`. This spec adds `augmented_susie_fit` and
`pooled_cs_qvalues`; `filter_credible_sets` becomes the per-gene special case and
`pooled_cs_qvalues` is the genome-wide aggregator.

---

## 5. Calibration gate (must pass before the target FDR is trusted)

Model-X guarantees FDR only with a correctly estimated knockoff distribution;
at N≈200–500 that is not automatic, so realized FDR is validated empirically:

```
NULL RUN:
  permute y within each gene (destroy all genotype-phenotype association)
  run map_knockoffs at fdr=q
  every selected CS is by definition false
  empirical_FDR = (# selected CSs) / (# would-be discoveries)  -> compare to q
```

`knockoffs.calibration_report` consumes the per-gene false/total counts and
reports `empirical_fdr`, `target_fdr`, and a `calibrated` bool. **Acceptance
criterion for the whole feature: empirical FDR ≤ q (within noise) on a null
permutation of the real data at the intended N.** If it fails, that is the
small-N generator limit asserting itself, and the fix is more shrinkage / the
HMM generator / more knockoff draws — diagnosed before any real-data use.

A SPIKE-IN power run (plant 1–3 causal variants per gene at known PVE, measure
fraction recovered in a selected CS vs realized FDP) accompanies the null run to
quantify the power cost of calibration.

---

## 6. Deferred / v2 knobs (noted, not built in v1)

- **Within-gene W standardization before pooling.** Draw M knockoffs, estimate
  each gene's null-W spread, convert each CS's W to a within-gene z-score/rank,
  pool the standardized statistics. Equalizes null scale across genes so a clear
  top signal in a weak gene isn't drowned by a strong gene's noise band —
  directly addresses the power-concentration cost in §1. Costs the M draws we
  already take. Build only if the plain pool shows a problematic power skew.
- **HMM generator** (`knockoff='hmm'`), optionally with an external genetic
  recombination map to set the HMM transition rates from real local
  recombination instead of a data estimate (strong prior at small N). SNPknock,
  GPLv3, optional runtime dependency, never vendored. Higher real-data fidelity
  than Gaussian, especially at low MAF / across hotspots.
- **e-value derandomization** (`derandomize_cs`) instead of mean-W aggregation,
  if selection stability across seeds is inadequate.

---

## 7. CLI

New mode `cis_susie_knockoffs` in `tensorqtl.py`, mirroring `cis_susie`:

```
python -m tensorqtl ${plink} ${expr_bed} ${prefix} \
    --covariates ${cov} --mode cis_susie_knockoffs \
    --fdr 0.05 --n_knockoffs 5 --knockoff gaussian
```

Writes `${prefix}.SuSiE_knockoff_summary.parquet` (the §2 table) and, if
`--emit_diagnostics`, `${prefix}.SuSiE_knockoff_diagnostics.pickle`.

---

## 8. Build order

1. `augmented_susie_fit` + `pooled_cs_qvalues` in `knockoffs.py` (+ unit tests).
2. `map_knockoffs` in `susie.py` wiring PASS 1 / PASS 2 (reuse `susie.map`'s
   per-gene setup verbatim; only the fit + CS extraction change).
3. **Null-permutation calibration test on synthetic data** — the gate. Do not
   proceed to real-data / CLI until empirical FDR ≈ q here.
4. Spike-in power test.
5. CLI mode.
6. (later) hapmixQTL two-channel path; HMM generator; v2 knobs.

Open question flagged for the two-channel phase (not v1): the augmented fit
doubles a design that, in hapmixQTL, is already a 2N-row whitened stack — the
exchangeability argument there needs the haplotype-level knockoffs (`s̃`), per
the earlier design discussion.
```
