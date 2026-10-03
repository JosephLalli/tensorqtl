# Where asSeq's scale (the Pearson-type dispersion of its null fit, glm.c:525) sits when the model holds: per gene, its
# value on the real totals of the all-null dataset and its mean over N_SIM totals simulated from asSeq's own null fit
# (as in trec_null_diagnosis.R). Output OUT/scale.tsv. Usage: Rscript trec_null_scale_check.R <trecase_work dir> <out dir>
# Environment: R_LD_LIBRARY_PATH=/usr/lib/R/lib:/usr/lib/x86_64-linux-gnu LD_LIBRARY_PATH=/usr/local/cuda/lib64
.libPaths(c("/mnt/ssd/lalli/usr/local/lib/R/library", .libPaths(), .Library.site))
suppressMessages(library(asSeq))
SEED <- 42; N_SIM <- 5
args <- commandArgs(trailingOnly = TRUE)
stopifnot("usage: trec_null_scale_check.R work_dir out_dir" = length(args) == 2)
ds <- file.path(args[1], "beta0.0", "rep000"); out <- args[2]
rd <- function(p, nrow) { x <- readBin(p, "double", n = file.size(p) / 8); matrix(x, nrow = nrow) }
offset <- readBin(file.path(ds, "offset.bin"), "double", n = file.size(file.path(ds, "offset.bin")) / 8)
N <- length(offset); Y <- rd(file.path(ds, "Y.bin"), N); X <- rd(file.path(ds, "X.bin"), N)
genes <- read.delim(file.path(ds, "genes.tsv"), colClasses = c("character", "integer", "integer"))
set.seed(SEED)
res <- do.call(rbind, lapply(seq_len(nrow(genes)), function(k) {
  h0 <- glmNB(Y[, k], X, offset = offset, trace = 0)
  if (!(h0$phi > 0)) return(NULL)   # Poisson fit or failed baseline: scale is fixed at 1 there
  s <- sapply(seq_len(N_SIM), function(i) glmNB(rnbinom(N, size = 1 / h0$phi, mu = h0$fitted), X, offset = offset, trace = 0)$scale)
  data.frame(gene = genes$gene[k], scale_real = h0$scale, scale_sim = mean(s))
}))
tmp <- file.path(out, "scale.tsv.tmp"); write.table(res, tmp, sep = "\t", quote = FALSE, row.names = FALSE)
stopifnot(file.rename(tmp, file.path(out, "scale.tsv")))
cat(sprintf("%d genes: median scale real %.3f, simulated %.3f\n", nrow(res), median(res$scale_real), median(res$scale_sim)))
