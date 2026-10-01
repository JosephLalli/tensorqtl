# asSeq::trecase (0.99.501) on replicates [start, end) of one mirror-benchmark condition; called by
# external_benchmark_mirror.py, which writes the inputs and reads the output.
# Usage: Rscript external_benchmark_mirror_trecase.R <cond_dir> <start> <end> <out_tsv>
# Environment: R_LD_LIBRARY_PATH=/usr/lib/R/lib:/usr/lib/x86_64-linux-gnu LD_LIBRARY_PATH=/usr/local/cuda/lib64
# (CLAUDE.md, "R's BLAS crash is an environment clash").
# Inputs (float64, [reps x N] C order = R's N x reps): <cond_dir>/Y.bin (total counts), Y1.bin / Y2.bin
# (haplotype L / R allele-specific counts, integers), Z.bin (3 xL + xR: 0 ref|ref, 1 ref|alt, 3 alt|ref,
# 4 alt|alt; haplotype 1 = L, as benchmark/simulated_effects/05_run_trecase.py), offset.bin (log library size).
# X = log library size, the one covariate. asSeq cannot fit TReC without one: glmFit's intercept-only
# branch (glm.c, M == 0) never sets convg and glmFit returns irls && convg (glm.c:161), so every
# baseline TReC fit fails. Given the offset its true coefficient is 0; it costs one degree of freedom.
# asSeq defaults otherwise, except p.cut (above the 995 sentinel, so every test is written).
# One trecase call per replicate (the offset is per replicate); gene and marker share one position.
# Output: one row per replicate, written atomically to <out_tsv> after the whole chunk.

.libPaths(c("/mnt/ssd/lalli/usr/local/lib/R/library", .libPaths(), .Library.site))
suppressMessages(library(asSeq))

P_CUT <- 1000            # run_trecase.R: above asSeq's 995 sentinel for an unfitted model
LOCAL_DISTANCE <- 1e6    # one gene, one marker at the same position
TRACE <- 1               # prints why a joint fit failed

args <- commandArgs(trailingOnly = TRUE)
stopifnot("usage: cond_dir start end out_tsv" = length(args) == 4)
cond <- args[1]; start <- as.integer(args[2]); end <- as.integer(args[3]); out <- args[4]
stopifnot("BLAS: crossprod() is wrong; set R_LD_LIBRARY_PATH / LD_LIBRARY_PATH as above" =
            identical(crossprod(matrix(c(1, 2, 3, 4), 2)), matrix(c(5, 11, 11, 25), 2)))

offset <- readBin(file.path(cond, "offset.bin"), "double", n = file.size(file.path(cond, "offset.bin")) / 8)
N <- scan(file.path(cond, "N.txt"), quiet = TRUE)
mat <- function(name) matrix(readBin(file.path(cond, name), "double", n = length(offset)), nrow = N)
Y <- mat("Y.bin"); Y1 <- mat("Y1.bin"); Y2 <- mat("Y2.bin"); Z <- mat("Z.bin"); OFF <- matrix(offset, nrow = N)
stopifnot("Y1 and Y2 must be integer counts" = all(Y1 == round(Y1)) && all(Y2 == round(Y2)))
stopifnot("range outside the replicates" = start >= 0 && end <= ncol(Y) && start < end)

work <- paste0(out, ".work")
dir.create(work, showWarnings = FALSE)
keep <- c("NBod", "BBod", "TReC_b", "TReC_Pvalue", "ASE_b", "ASE_Pvalue", "Joint_b", "Joint_Chisq",
          "Joint_Pvalue", "trans_Pvalue", "final_Pvalue", "n_ASE", "n_ASE_Het")
rows <- vector("list", end - start)
for (r in start:(end - 1)) {
  k <- r + 1
  tag <- file.path(work, sprintf("rep%03d", r))
  t0 <- proc.time()[["elapsed"]]
  res <- tryCatch(
    trecase(Y[, k, drop = FALSE], Y1[, k, drop = FALSE], Y2[, k, drop = FALSE], OFF[, k, drop = FALSE],
            Z[, k, drop = FALSE], output.tag = tag, p.cut = P_CUT, offset = OFF[, k], local.only = TRUE,
            local.distance = LOCAL_DISTANCE, eChr = 1, ePos = 1, mChr = 1, mPos = 1, trace = TRACE),
    error = function(e) { message(sprintf("rep %d: trecase error: %s", r, conditionMessage(e))); NULL })
  secs <- proc.time()[["elapsed"]] - t0
  row <- data.frame(rep = r, error = is.null(res), succeed = if (is.null(res)) NA else res$succeed,
                    yFailBaselineModel = if (is.null(res)) NA else res$yFailBaselineModel, seconds = secs)
  f <- paste0(tag, "_eqtl.txt")
  vals <- setNames(as.list(rep(NA_real_, length(keep))), keep)
  if (!is.null(res) && file.exists(f)) {
    e <- read.delim(f)
    stopifnot("expected one test per replicate" = nrow(e) <= 1)
    if (nrow(e) == 1) vals <- as.list(e[1, keep])
  }
  rows[[r - start + 1]] <- cbind(row, as.data.frame(vals))
  unlink(paste0(tag, c("_eqtl.txt", "_freq.txt")))
}
tab <- do.call(rbind, rows)
tmp <- paste0(out, ".tmp")
write.table(tab, tmp, sep = "\t", quote = FALSE, row.names = FALSE)
stopifnot(file.rename(tmp, out))
unlink(work, recursive = TRUE)
