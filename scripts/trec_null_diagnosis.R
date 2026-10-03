# Why asSeq's total-count (TReC) test rejects null genes above nominal on the simulated-effects benchmark's all-null
# dataset (beta0.0 rep000). asSeq's own unmodified trec() (it reproduces trecase's TReC p exactly) on one seeded draw of
# tested variants per gene, three ways:
#   real      the dataset's totals with the benchmark's 17 covariates (what the benchmark scored)
#   sim       N_SIM totals simulated from asSeq's own null fit of the gene (glmNB, the fit trec's H0 uses: fitted means
#             and dispersion; Poisson where the dispersion is 0), same covariates: the test where its model holds
#   fewcov    the real totals with the 3 genotype PCs as the only covariates
# Inputs are 05_run_trecase.py's files for that dataset (Y, X, offset, genes.tsv) and its genotype tables (Z codes
# 0/1/3/4, converted to ALT dosage for trec). Real totals are fractional and simulated ones integers; rounding the totals
# changed no TReC result (input_diagnosis_20260928/trecase_integer).
# Output OUT/<gene>.tsv (part, rep, MarkerRowID, Chisq, Pvalue; a gene whose file exists is skipped) and
# OUT/<gene>.meta.tsv (dispersion, scale, residual df, median total, variants).
# Usage: Rscript trec_null_diagnosis.R <trecase_work dir> <out dir> <cores>
# Environment: R_LD_LIBRARY_PATH=/usr/lib/R/lib:/usr/lib/x86_64-linux-gnu LD_LIBRARY_PATH=/usr/local/cuda/lib64

.libPaths(c("/mnt/ssd/lalli/usr/local/lib/R/library", .libPaths(), .Library.site))
suppressMessages({library(asSeq); library(parallel)})

SEED <- 42            # one master seed; one L'Ecuyer stream per gene, in genes.tsv order
N_VARIANTS <- 300     # tested variants drawn per gene, the same draw for all three parts
N_SIM <- 20           # simulated totals per gene
LOCAL_DISTANCE <- 1e6 # run_trecase.R
N_GENO_PC <- 3        # the genotype PCs are X's last columns (05_run_trecase.py: RNA-tied covariates, then genotype PCs)

args <- commandArgs(trailingOnly = TRUE)
stopifnot("usage: trec_null_diagnosis.R work_dir out_dir cores" = length(args) == 3)
ds <- file.path(args[1], "beta0.0", "rep000"); gd <- file.path(args[1], "genotypes"); out <- args[2]
cores <- as.integer(args[3])
stopifnot("BLAS: crossprod() is wrong; set R_LD_LIBRARY_PATH / LD_LIBRARY_PATH as above" =
            identical(crossprod(matrix(c(1, 2, 3, 4), 2)), matrix(c(5, 11, 11, 25), 2)))
dir.create(out, showWarnings = FALSE, recursive = TRUE)

rd <- function(p, nrow) { x <- readBin(p, "double", n = file.size(p) / 8); matrix(x, nrow = nrow) }
offset <- readBin(file.path(ds, "offset.bin"), "double", n = file.size(file.path(ds, "offset.bin")) / 8)
N <- length(offset)
genes <- read.delim(file.path(ds, "genes.tsv"), colClasses = c("character", "integer", "integer"))
Y <- rd(file.path(ds, "Y.bin"), N); X <- rd(file.path(ds, "X.bin"), N)
stopifnot("Y has one column per gene" = ncol(Y) == nrow(genes), "X has the genotype PCs" = ncol(X) > N_GENO_PC)
cat(sprintf("%s: %d genes, %d donors, %d covariates\n", ds, nrow(genes), N, ncol(X)))

RNGkind("L'Ecuyer-CMRG"); set.seed(SEED)
streams <- vector("list", nrow(genes)); s <- .Random.seed
for (k in seq_len(nrow(genes))) { s <- nextRNGStream(s); streams[[k]] <- s }

write_tsv <- function(df, path) {
  tmp <- paste0(path, ".tmp"); write.table(df, tmp, sep = "\t", quote = FALSE, row.names = FALSE)
  stopifnot(file.rename(tmp, path))
}

one_gene <- function(k) {
  g <- genes$gene[k]; f <- file.path(out, paste0(g, ".tsv"))
  if (file.exists(f)) return(sprintf("%s skipped (file exists)", g))
  assign(".Random.seed", streams[[k]], envir = .GlobalEnv)
  chr <- genes$chr[k]
  markers <- read.delim(file.path(gd, sprintf("chr%d.markers.tsv", chr)), colClasses = c("character", "integer", "integer"))
  Z <- rd(file.path(gd, sprintf("chr%d.Z.bin", chr)), N)
  cis <- which(markers$chr == chr & abs(markers$pos - genes$pos[k]) <= LOCAL_DISTANCE)
  pick <- sort(if (length(cis) > N_VARIANTS) sample(cis, N_VARIANTS) else cis)
  Zs <- (Z[, pick, drop = FALSE] == 1 | Z[, pick, drop = FALSE] == 3) + 2 * (Z[, pick, drop = FALSE] == 4)
  run <- function(Ym, Xm, part) {
    tag <- file.path(tempdir(), paste0(g, "_", part))
    r <- trec(Ym, Xm, Zs, output.tag = tag, p.cut = 1000, offset = offset, local.only = TRUE,
              local.distance = LOCAL_DISTANCE, eChr = rep(chr, ncol(Ym)), ePos = rep(genes$pos[k], ncol(Ym)),
              mChr = markers$chr[pick], mPos = markers$pos[pick], trace = 0)
    stopifnot("trec did not report success" = r$succeed == 1)
    e <- read.delim(paste0(tag, "_eqtl.txt"))
    data.frame(part = rep(part, nrow(e)), rep = e$GeneRowID, MarkerRowID = pick[e$MarkerRowID], Chisq = e$Chisq,
               Pvalue = e$Pvalue)
  }
  real <- run(Y[, k, drop = FALSE], X, "real")
  if (nrow(real) == 0) {   # asSeq's null (baseline) model fails, as in the benchmark's run: the gene has no TReC p there
    writeLines("trec's baseline model failed on the real totals", file.path(out, paste0(g, ".baseline_failed")))
    write_tsv(real, f)
    return(sprintf("%s: baseline model failed, no TReC tests (as in the benchmark)", g))
  }
  h0 <- glmNB(Y[, k], X, offset = offset, trace = 0)
  sim <- sapply(seq_len(N_SIM), function(i)
    if (h0$phi > 0) rnbinom(N, size = 1 / h0$phi, mu = h0$fitted) else rpois(N, h0$fitted))
  res <- rbind(real, run(sim, X, "sim"),
               run(Y[, k, drop = FALSE], X[, (ncol(X) - N_GENO_PC + 1):ncol(X), drop = FALSE], "fewcov"))
  write_tsv(data.frame(gene = g, phi = h0$phi, scale = h0$scale, df_resid = h0$dfResid, median_total = median(Y[, k]),
                       cis_variants = length(cis), drawn = length(pick)), file.path(out, paste0(g, ".meta.tsv")))
  write_tsv(res, f)
  sprintf("%s: %d variants drawn of %d, rows real %d sim %d fewcov %d", g, length(pick), length(cis),
          sum(res$part == "real"), sum(res$part == "sim"), sum(res$part == "fewcov"))
}

msgs <- mclapply(seq_len(nrow(genes)), one_gene, mc.cores = cores, mc.preschedule = FALSE)
bad <- vapply(msgs, inherits, logical(1), "try-error")
cat(unlist(msgs[!bad]), sep = "\n")
if (any(bad)) stop(sprintf("%d genes failed: %s", sum(bad),
                           paste(genes$gene[bad], vapply(msgs[bad], as.character, ""), sep = ": ", collapse = "; ")))
cat("done\n")
