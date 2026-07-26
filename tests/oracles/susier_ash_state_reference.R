#!/usr/bin/env Rscript

# Generate deterministic SuSiE-ash state-transition oracles from pinned susieR.
#
# Run from the TensorQTL repository root after installing the pinned checkout:
#
# oracle_tmp=$(mktemp -d /tmp/tensorqtl-susier-ash-oracle.XXXXXX)
# cp -a /mnt/ssd/lalli/tmp_research/susieR "$oracle_tmp/susieR"
# mkdir "$oracle_tmp/lib"
# env -u LD_LIBRARY_PATH -u R_LD_LIBRARY_PATH R CMD INSTALL --preclean \
#   --library="$oracle_tmp/lib" --no-multiarch --no-test-load "$oracle_tmp/susieR"
# env -u LD_LIBRARY_PATH -u R_LD_LIBRARY_PATH \
#   SUSIER_SOURCE=/mnt/ssd/lalli/tmp_research/susieR \
#   SUSIER_ORACLE_LIB="$oracle_tmp/lib" \
#   Rscript tests/oracles/susier_ash_state_reference.R \
#   tests/fixtures/susier_ash_state_reference.json

required_commit <- "dd9d9ce4693573e9dcd1ec8b3df94b63bac467d2"
source_repository <- Sys.getenv(
  "SUSIER_SOURCE",
  unset = "/mnt/ssd/lalli/tmp_research/susieR"
)
oracle_library <- Sys.getenv("SUSIER_ORACLE_LIB")
output_path <- if (length(commandArgs(trailingOnly = TRUE)) > 0) {
  commandArgs(trailingOnly = TRUE)[[1]]
} else {
  "tests/fixtures/susier_ash_state_reference.json"
}

if (!nzchar(oracle_library) || !dir.exists(oracle_library)) {
  stop("SUSIER_ORACLE_LIB must identify the temporary susieR installation")
}

git_output <- function(...) {
  result <- system2(
    "git", c("-C", source_repository, ...), stdout = TRUE, stderr = TRUE
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
if (!identical(
  normalizePath(find.package("susieR")),
  normalizePath(file.path(oracle_library, "susieR"))
)) {
  stop("susieR was not loaded from SUSIER_ORACLE_LIB")
}

update_ash <- getFromNamespace(
  "update_ash_variance_components", "susieR"
)

# A fixed positive-definite correlation matrix with:
#   1--2: collision/tight LD; 1--3: uncertain/moderate LD;
#   5--6: a second tight-LD block; other pairs: independent.
p <- 8L
L <- 3L
Xcorr <- diag(p)
Xcorr[1, 2] <- Xcorr[2, 1] <- 0.99
Xcorr[1, 3] <- Xcorr[3, 1] <- 0.35
Xcorr[2, 3] <- Xcorr[3, 2] <- 0.34
Xcorr[5, 6] <- Xcorr[6, 5] <- 0.99
if (min(eigen(Xcorr, symmetric = TRUE, only.values = TRUE)$values) <= 0) {
  stop("Oracle correlation matrix must be positive definite")
}
n <- 50L
data <- structure(
  list(
    XtX = Xcorr * (n - 1),
    X = NULL,
    Xty = rep(0, p),
    yty = 1,
    n = n,
    p = p,
    X_colmeans = rep(0, p),
    y_mean = 0
  ),
  class = "ss"
)

alpha_row <- function(indices, masses) {
  value <- rep((1 - sum(masses)) / (p - length(indices)), p)
  value[indices] <- masses
  value
}

alpha_uncertain <- alpha_row(c(1, 3), c(0.50, 0.45))
alpha_confident_a <- alpha_row(c(1, 2), c(0.50, 0.49))
alpha_confident_b <- alpha_row(c(5, 6), c(0.50, 0.49))
alpha_diffuse <- alpha_row(c(7, 8), c(0.45, 0.45))
alpha_collision_b <- alpha_row(c(2, 4), c(0.50, 0.45))
alpha_uniform <- rep(1 / p, p)
mu <- rbind(
  seq(0.2, 0.9, length.out = p),
  seq(-0.8, -0.1, length.out = p),
  rep(0.35, p)
)
c_hat <- c(0.4, 0.8, 0.2)

new_model <- function() {
  list(
    alpha = matrix(alpha_uniform, nrow = L, ncol = p, byrow = TRUE),
    mu = mu,
    mu2 = mu^2 + 0.1,
    V = rep(1, L),
    lbf = rep(0, L),
    sigma2 = 1,
    theta = rep(0, p),
    ash_pi = NULL,
    tau2 = 0,
    ash_iter = 0L,
    slot_weights = c_hat
  )
}

carry_state <- function(model, result) {
  fields <- c(
    "ash_iter", "prev_case", "prev_sentinel", "ever_diffuse",
    "diffuse_iter_count", "masked", "ever_unmasked",
    "unmask_candidate_iters", "force_exposed_iter",
    "second_chance_used", ".diag_env"
  )
  for (field in fields) model[[field]] <- result[[field]]
  model$theta <- result$theta
  model$sigma2 <- result$sigma2
  model$tau2 <- result$tau2
  model$ash_pi <- result$ash_pi
  model
}

snapshot <- function(result) {
  history <- result$.diag_env$history
  diagnostic <- history[[length(history)]]
  list(
    ash_iter = as.integer(result$ash_iter),
    current_case = as.integer(diagnostic$concentration),
    sentinels = as.integer(diagnostic$sentinel - 1L),
    effect_purity = as.numeric(diagnostic$smoothness),
    collision = as.logical(diagnostic$collision),
    ever_diffuse = as.integer(result$ever_diffuse),
    diffuse_iter_count = as.integer(result$diffuse_iter_count),
    masked = as.logical(result$masked),
    ever_unmasked = as.logical(result$ever_unmasked),
    unmask_candidate_iters = as.integer(result$unmask_candidate_iters),
    force_exposed_iter = as.integer(result$force_exposed_iter),
    second_chance_used = as.logical(result$second_chance_used),
    b_confident_ss = as.numeric(diagnostic$b_conf_ss[[1]]),
    b_confident_max = as.numeric(diagnostic$b_conf_max[[1]])
  )
}

run_sequence <- function(alpha_sequence, initial_masked = NULL) {
  model <- new_model()
  if (!is.null(initial_masked)) model$masked <- initial_masked
  expected <- vector("list", length(alpha_sequence))
  for (i in seq_along(alpha_sequence)) {
    model$alpha <- alpha_sequence[[i]]
    result <- update_ash(
      data, model,
      list(
        verbose = FALSE,
        estimate_residual_variance = TRUE,
        slot_prior = NULL
      )
    )
    expected[[i]] <- snapshot(result)
    model <- carry_state(model, result)
  }
  expected
}

base_alpha <- rbind(
  alpha_uncertain,
  alpha_confident_b,
  alpha_diffuse
)
wait_expose_second_chance <- rep(list(base_alpha), 7)

oscillation <- list(
  base_alpha,
  rbind(alpha_confident_a, alpha_confident_b, alpha_diffuse)
)

collision <- list(
  rbind(alpha_uncertain, alpha_collision_b, alpha_confident_b)
)

all_cases <- list(
  rbind(alpha_diffuse, alpha_uncertain, alpha_confident_b)
)

delayed_unmask <- list(base_alpha, base_alpha)
delayed_unmask_initial <- rep(FALSE, p)
delayed_unmask_initial[8] <- TRUE

generation_command <- paste(
  c(
    "oracle_tmp=$(mktemp -d /tmp/tensorqtl-susier-ash-oracle.XXXXXX)",
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
      "Rscript tests/oracles/susier_ash_state_reference.R",
      "tests/fixtures/susier_ash_state_reference.json"
    )
  ),
  collapse = "\n"
)

old_skip <- getOption("susie.skip_mrash", FALSE)
on.exit(options(susie.skip_mrash = old_skip), add = TRUE)
options(susie.skip_mrash = TRUE)

# End-to-end individual-data oracle. The design is standardized before calling
# susie because the pinned individual ash helper passes the stored matrix
# directly to mr.ash even though sparse SER operations honor scale attributes.
# Standardizing here isolates the implemented model from that upstream
# raw-design inconsistency.
n_e2e <- 36L
p_e2e <- 7L
index <- seq_len(n_e2e)
X_e2e_raw <- vapply(
  seq_len(p_e2e),
  function(j) {
    sin(index * (j + 0.5) / 4.7) +
      cos((index + 2 * j) / (3.1 + j / 4)) +
      ((index * j) %% 11) / 13
  },
  numeric(n_e2e)
)
X_e2e_raw[, 2] <- 0.72 * X_e2e_raw[, 1] + 0.28 * X_e2e_raw[, 2]
X_e2e_raw[, 5] <- -0.55 * X_e2e_raw[, 3] + 0.45 * X_e2e_raw[, 5]
beta_e2e <- c(1.15, -0.45, 0, 0.8, 0, -0.6, 0.25)
y_e2e <- drop(
  X_e2e_raw %*% beta_e2e +
    0.17 * sin(index / 2.3) -
    0.09 * cos(index / 4.1)
)
X_e2e <- unclass(scale(X_e2e_raw))
attributes(X_e2e) <- list(dim = dim(X_e2e))
y_e2e <- y_e2e - mean(y_e2e)

options(susie.skip_mrash = FALSE)
fit_e2e <- suppressWarnings(suppressMessages(susie(
  X_e2e,
  y_e2e,
  L = 5L,
  scaled_prior_variance = 0.25,
  unmappable_effects = "ash",
  coverage = NULL,
  max_iter = 50L,
  tol = 1e-3,
  verbose = FALSE
)))
options(susie.skip_mrash = TRUE)

weighted_alpha_e2e <- sweep(fit_e2e$alpha, 1, fit_e2e$c_hat, `*`)
sparse_e2e <- colSums(weighted_alpha_e2e * fit_e2e$mu)
fitted_identity_e2e <- drop(
  X_e2e %*% (sparse_e2e + fit_e2e$theta)
)
if (max(abs(fit_e2e$fitted - fitted_identity_e2e)) > 1e-8) {
  stop("Pinned susieR end-to-end ash fitted identity failed")
}

fixture <- list(
  schema_version = 1L,
  provenance = list(
    upstream = "stephenslab/susieR",
    source_repository = "/mnt/ssd/lalli/tmp_research/susieR",
    source_commit = source_commit,
    package_version = as.character(packageVersion("susieR")),
    R_version = R.version.string,
    generation_command = generation_command,
    upstream_function = "update_ash_variance_components",
    skip_mrash = TRUE
  ),
  input = list(
    Xcorr = unname(Xcorr),
    mu = unname(mu),
    c_hat = unname(c_hat)
  ),
  end_to_end = list(
    input = list(
      X = unname(X_e2e),
      y = unname(y_e2e)
    ),
    args = list(
      L = 5L,
      scaled_prior_variance = 0.25,
      estimate_prior_variance = TRUE,
      estimate_prior_method = "optim",
      estimate_residual_variance = TRUE,
      coverage = NULL,
      max_iter = 50L,
      tol = 1e-3,
      standardize = TRUE,
      intercept = TRUE
    ),
    expected = list(
      niter = fit_e2e$niter,
      converged = isTRUE(fit_e2e$converged),
      sigma2 = as.numeric(fit_e2e$sigma2),
      tau2 = as.numeric(fit_e2e$tau2),
      alpha = unname(fit_e2e$alpha),
      mu = unname(fit_e2e$mu),
      mu2 = unname(fit_e2e$mu2),
      V = as.numeric(fit_e2e$V),
      c_hat = as.numeric(fit_e2e$c_hat),
      C_hat = as.numeric(fit_e2e$C_hat),
      theta = as.numeric(fit_e2e$theta),
      sparse_effect_standardized = as.numeric(sparse_e2e),
      total_effect_standardized = as.numeric(
        sparse_e2e + fit_e2e$theta
      ),
      fitted = as.numeric(fit_e2e$fitted),
      pip = as.numeric(fit_e2e$pip),
      masked = as.logical(fit_e2e$masked),
      ever_diffuse = as.integer(fit_e2e$ever_diffuse),
      second_chance_used = as.logical(fit_e2e$second_chance_used)
    ),
    pinned_differences = list(
      residual_variance = paste(
        "run_final_ash_pass uses the returned Mr.ASH sigma2 to compute tau2",
        "but does not copy that sigma2 into the fitted model"
      ),
      raw_design = paste(
        "compute_ash_from_individual_data passes raw X to Mr.ASH while sparse",
        "SER operations honor X scaling attributes"
      ),
      final_sparse_subtraction = paste(
        "run_final_ash_pass subtracts colSums(alpha * mu) without c_hat"
      )
    )
  ),
  cases = list(
    wait_expose_second_chance = list(
      alpha = lapply(wait_expose_second_chance, unname),
      expected = run_sequence(wait_expose_second_chance)
    ),
    oscillation_reversal = list(
      alpha = lapply(oscillation, unname),
      expected = run_sequence(oscillation)
    ),
    collision = list(
      alpha = lapply(collision, unname),
      expected = run_sequence(collision)
    ),
    all_three_cases = list(
      alpha = lapply(all_cases, unname),
      expected = run_sequence(all_cases)
    ),
    delayed_unmask = list(
      initial_masked = as.logical(delayed_unmask_initial),
      alpha = lapply(delayed_unmask, unname),
      expected = run_sequence(
        delayed_unmask,
        initial_masked = delayed_unmask_initial
      )
    )
  )
)

write_json(
  fixture, output_path,
  auto_unbox = TRUE, pretty = TRUE, digits = 17, null = "null"
)
