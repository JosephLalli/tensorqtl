#!/usr/bin/env Rscript

# Generate the retained individual-data slot-prior oracle for TensorQTL.
#
# The source checkout and installed package must both come from the pinned
# susieR commit below. Run from the TensorQTL repository root with:
#
# oracle_tmp=$(mktemp -d /tmp/tensorqtl-susier-slot-oracle.XXXXXX)
# cp -a /mnt/ssd/lalli/tmp_research/susieR "$oracle_tmp/susieR"
# mkdir "$oracle_tmp/lib"
# env -u LD_LIBRARY_PATH -u R_LD_LIBRARY_PATH R CMD INSTALL --preclean \
#   --library="$oracle_tmp/lib" --no-multiarch --no-test-load "$oracle_tmp/susieR"
# env -u LD_LIBRARY_PATH -u R_LD_LIBRARY_PATH \
#   SUSIER_SOURCE=/mnt/ssd/lalli/tmp_research/susieR \
#   SUSIER_ORACLE_LIB="$oracle_tmp/lib" \
#   Rscript tests/oracles/susier_slot_prior_reference.R \
#   tests/fixtures/susier_slot_prior_reference.json

required_commit <- "dd9d9ce4693573e9dcd1ec8b3df94b63bac467d2"
source_repository <- Sys.getenv(
  "SUSIER_SOURCE",
  unset = "/mnt/ssd/lalli/tmp_research/susieR"
)
oracle_library <- Sys.getenv("SUSIER_ORACLE_LIB")
output_path <- if (length(commandArgs(trailingOnly = TRUE)) > 0) {
  commandArgs(trailingOnly = TRUE)[[1]]
} else {
  "tests/fixtures/susier_slot_prior_reference.json"
}

if (!dir.exists(source_repository)) {
  stop("SUSIER_SOURCE does not exist: ", source_repository)
}
if (!nzchar(oracle_library) || !dir.exists(oracle_library)) {
  stop("SUSIER_ORACLE_LIB must identify the temporary susieR installation")
}

git_output <- function(...) {
  result <- system2(
    "git",
    c("-C", source_repository, ...),
    stdout = TRUE,
    stderr = TRUE
  )
  status <- attr(result, "status")
  if (!is.null(status) && status != 0) {
    stop("git command failed: ", paste(result, collapse = "\n"))
  }
  result
}

source_commit <- trimws(git_output("rev-parse", "HEAD")[[1]])
if (!identical(source_commit, required_commit)) {
  stop("Expected susieR commit ", required_commit, "; found ", source_commit)
}
if (length(git_output("status", "--porcelain", "--untracked-files=no")) != 0) {
  stop("Pinned susieR source has tracked modifications")
}

.libPaths(c(normalizePath(oracle_library), .libPaths()))
suppressPackageStartupMessages(library(susieR))
suppressPackageStartupMessages(library(jsonlite))
loaded_package <- normalizePath(find.package("susieR"))
expected_package <- normalizePath(file.path(oracle_library, "susieR"))
if (!identical(loaded_package, expected_package)) {
  stop("susieR was not loaded from SUSIER_ORACLE_LIB: ", loaded_package)
}

# The fixture is algebraic rather than RNG-generated so it is identical across
# R versions and can be retained directly in JSON.
n <- 36L
p <- 7L
index <- seq_len(n)
X <- vapply(
  seq_len(p),
  function(j) {
    sin(index * (j + 0.5) / 4.7) +
      cos((index + 2 * j) / (3.1 + j / 4)) +
      ((index * j) %% 11) / 13
  },
  numeric(n)
)
X[, 2] <- 0.72 * X[, 1] + 0.28 * X[, 2]
X[, 5] <- -0.55 * X[, 3] + 0.45 * X[, 5]
beta <- c(1.15, -0.45, 0, 0.8, 0, -0.6, 0.25)
y <- drop(
  X %*% beta +
    0.17 * sin(index / 2.3) -
    0.09 * cos(index / 4.1)
)

common_args <- list(
  X = X,
  y = y,
  L = 5L,
  scaled_prior_variance = 0.25,
  estimate_prior_variance = FALSE,
  estimate_residual_variance = FALSE,
  coverage = NULL,
  max_iter = 8L,
  tol = 0,
  verbose = FALSE,
  track_fit = TRUE
)

case_specs <- list(
  beta_binomial = list(
    prior = list(
      family = "beta_binomial",
      a_beta = 1,
      b_beta = 2,
      skip_threshold_multiplier = 0
    ),
    object = slot_prior_betabinom(a_beta = 1, b_beta = 2)
  ),
  gamma_poisson_sequential = list(
    prior = list(
      family = "gamma_poisson",
      C = 2.5,
      nu = 8,
      update_schedule = "sequential",
      skip_threshold_multiplier = 0
    ),
    object = slot_prior_poisson(
      C = 2.5,
      nu = 8,
      update_schedule = "sequential"
    )
  ),
  gamma_poisson_batch = list(
    prior = list(
      family = "gamma_poisson",
      C = 2.5,
      nu = 8,
      update_schedule = "batch",
      skip_threshold_multiplier = 0
    ),
    object = slot_prior_poisson(
      C = 2.5,
      nu = 8,
      update_schedule = "batch"
    )
  ),
  beta_binomial_warm_start = list(
    prior = list(
      family = "beta_binomial",
      a_beta = 1,
      b_beta = 2,
      c_hat_init = c(0.85, 0.55, 0.25, 0.1, 0.05),
      skip_threshold_multiplier = 0
    ),
    object = slot_prior_betabinom(
      a_beta = 1,
      b_beta = 2,
      c_hat_init = c(0.85, 0.55, 0.25, 0.1, 0.05)
    )
  ),
  gamma_poisson_batch_skip = list(
    prior = list(
      family = "gamma_poisson",
      C = 1.5,
      nu = 8,
      update_schedule = "batch",
      skip_threshold_multiplier = 1.25
    ),
    object = slot_prior_poisson(
      C = 1.5,
      nu = 8,
      update_schedule = "batch",
      skip_threshold_multiplier = 1.25
    )
  )
)

as_numeric_matrix <- function(x) {
  matrix(as.numeric(x), nrow = nrow(x), ncol = ncol(x))
}

fit_case <- function(spec) {
  fit <- suppressWarnings(suppressMessages(
    do.call(susie, c(common_args, list(slot_prior = spec$object)))
  ))

  center <- colMeans(X)
  scale <- apply(X, 2, sd)
  weighted_alpha <- sweep(fit$alpha, 1, fit$c_hat, `*`)
  sparse_standardized <- colSums(weighted_alpha * fit$mu)
  sparse_raw <- sparse_standardized / scale
  fitted_identity <- drop(
    mean(y) + scale(X, center = center, scale = scale) %*%
      sparse_standardized
  )
  pip_identity <- 1 - apply(1 - weighted_alpha, 2, prod)

  if (max(abs(fit$fitted - fitted_identity)) > 1e-10) {
    stop("susieR fitted-value identity failed")
  }
  if (max(abs(fit$pip - pip_identity)) > 1e-12) {
    stop("susieR slot-weighted PIP identity failed")
  }
  if (abs(fit$C_hat - sum(fit$c_hat)) > 1e-12) {
    stop("susieR C_hat identity failed")
  }

  trace_effect <- fit$trace$effect
  slot_weight_history <- matrix(
    trace_effect$slot_weight,
    nrow = length(unique(trace_effect$iteration)),
    byrow = TRUE
  )

  list(
    prior = spec$prior,
    expected = list(
      niter = fit$niter,
      converged = isTRUE(fit$converged),
      sigma2 = as.numeric(fit$sigma2),
      c_hat = as.numeric(fit$c_hat),
      C_hat = as.numeric(fit$C_hat),
      alpha = as_numeric_matrix(fit$alpha),
      mu = as_numeric_matrix(fit$mu),
      V = as.numeric(fit$V),
      lbf = as.numeric(fit$lbf),
      fitted = as.numeric(fit$fitted),
      pip = as.numeric(fit$pip),
      sparse_effect_standardized = as.numeric(sparse_standardized),
      sparse_effect_raw = as.numeric(sparse_raw),
      slot_weight_history = as_numeric_matrix(slot_weight_history)
    )
  )
}

generation_command <- paste(
  c(
    "oracle_tmp=$(mktemp -d /tmp/tensorqtl-susier-slot-oracle.XXXXXX)",
    "cp -a /mnt/ssd/lalli/tmp_research/susieR \"$oracle_tmp/susieR\"",
    "mkdir \"$oracle_tmp/lib\"",
    paste(
      "env -u LD_LIBRARY_PATH -u R_LD_LIBRARY_PATH R CMD INSTALL --preclean",
      "--library=\"$oracle_tmp/lib\" --no-multiarch --no-test-load",
      "\"$oracle_tmp/susieR\""
    ),
    paste(
      "env -u LD_LIBRARY_PATH -u R_LD_LIBRARY_PATH",
      "SUSIER_SOURCE=/mnt/ssd/lalli/tmp_research/susieR",
      "SUSIER_ORACLE_LIB=\"$oracle_tmp/lib\"",
      "Rscript tests/oracles/susier_slot_prior_reference.R",
      "tests/fixtures/susier_slot_prior_reference.json"
    )
  ),
  collapse = "\n"
)

fixture <- list(
  schema_version = 1L,
  provenance = list(
    upstream = "stephenslab/susieR",
    source_repository = "/mnt/ssd/lalli/tmp_research/susieR",
    source_commit = source_commit,
    package_version = as.character(packageVersion("susieR")),
    R_version = R.version.string,
    generation_command = generation_command
  ),
  numeric_contract = list(
    tensorqtl_dtype = "float32",
    rtol = 3e-5,
    atol = 5e-6
  ),
  input = list(
    X = as_numeric_matrix(X),
    y = as.numeric(y),
    scaled_center = as.numeric(colMeans(X)),
    scaled_scale = as.numeric(apply(X, 2, sd))
  ),
  common_args = list(
    L = common_args$L,
    scaled_prior_variance = common_args$scaled_prior_variance,
    residual_variance = NULL,
    estimate_prior_variance = common_args$estimate_prior_variance,
    estimate_residual_variance = common_args$estimate_residual_variance,
    coverage = NULL,
    max_iter = common_args$max_iter,
    tol = common_args$tol,
    standardize = TRUE,
    intercept = TRUE
  ),
  cases = lapply(case_specs, fit_case)
)

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write_json(
  fixture,
  path = output_path,
  auto_unbox = TRUE,
  digits = 17,
  pretty = TRUE,
  null = "null"
)
