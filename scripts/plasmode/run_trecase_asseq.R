# TReCASE (asSeq::trecase) on one gene of one plasmode dataset. Called by
# run_trecase_asseq.py, which writes the inputs and converts the output.
#
# Usage: Rscript run_trecase_asseq.R <dataset_dir> <genotype_dir> <gene> <out_tag>
# Run with R_LD_LIBRARY_PATH=/usr/lib/R/lib:/usr/lib/x86_64-linux-gnu
# LD_LIBRARY_PATH=/usr/local/cuda/lib64 (CLAUDE.md, "R's BLAS crash is an
# environment clash"); the crossprod() below stops at once if that is missing.
#
# Inputs (float64, R column-major): <dataset_dir>/Y.bin, Y1.bin, Y2.bin [N x genes],
# X.bin [N x covariates], offset.bin [N], genes.tsv (gene, chr, pos, in the Y
# column order); <genotype_dir>/chr<k>.Z.bin [N x markers] and chr<k>.markers.tsv
# (variant_id, chr, pos). asSeq's own defaults are kept: min.AS.reads = 5,
# min.AS.sample = 5, min.n.het = 5, converge 5e-5, convergeGLM 1e-8,
# scoreTestP 0.05, transTestP 0.05, maxit 100 (R/trecase.R signature).
#
# Y1 and Y2 must be integers: run_trecase_asseq.py rounds them, and sets both to 0
# on records outside make_datasets.allelic_kept (see its docstring). asSeq's
# beta-binomial adds lchoose(n, nA) (src/ase.c:43, :142), and R's lchoose rounds a
# non-integer nA to an integer with a warning while n stays fractional. H0 takes
# nA = Y1 (trecase.c:620) and H1 takes nA = Y2 for records with Zh 0, 1 or 4
# (:792-799), so the two likelihoods carry different constants, and asSeq stops at
# "likelihood decreases for ASE model" (trecase.c:1182-1185). On the 2026-09-26
# smoke all three genes stopped this way with fractional counts. Y (TReC) stays
# fractional: its likelihood uses only lgammafn / digamma (glm.c:867-918).
#
# Every warning raised inside trecase() is written to the log and counted; asSeq's
# trace-1 lines (TRACE) go to the same log.
#
# Output: <out_tag>_eqtl.txt and <out_tag>_freq.txt (asSeq), then
# <out_tag>_status.tsv (gene, succeed, yFailBaselineModel, seconds, n_markers_chr,
# n_warnings), written last and atomically, so its presence marks a finished gene.

.libPaths(c("/mnt/ssd/lalli/usr/local/lib/R/library", .libPaths(), .Library.site))
library(asSeq)

LOCAL_DISTANCE <- 1e6   # compare_mixqtl_replication.WIN; asSeq keeps |ePos - mPos| <= it (trecase.c:695-697)
P_CUT <- 1000           # above asSeq's 995 sentinel for an unfitted model (trecase.c:1149, 1192, 1253): with the
                        # strict test at trecase.c:1278 every cis test is written, including all-failed ones
TRACE <- 1              # trecase.c prints why a joint fit failed at trace >= 1 (:925, :1004-1009, :1064);
                        # the log is the record run_trecase_asseq.py counts. Maxit (:1111) prints only at > 1

args <- commandArgs(trailingOnly = TRUE)
stopifnot("usage: run_trecase_asseq.R dataset_dir genotype_dir gene out_tag" = length(args) == 4)
ds_dir <- args[1]
geno_dir <- args[2]
gene <- args[3]
tag <- args[4]
stopifnot("BLAS: crossprod() is wrong; set R_LD_LIBRARY_PATH / LD_LIBRARY_PATH as above" =
            identical(crossprod(matrix(c(1, 2, 3, 4), 2)), matrix(c(5, 11, 11, 25), 2)))

read_matrix <- function(path, nrow) {
  x <- readBin(path, "double", n = file.size(path) / 8)
  stopifnot("file length is not a whole number of rows" = length(x) %% nrow == 0)
  matrix(x, nrow = nrow)
}

offset <- readBin(file.path(ds_dir, "offset.bin"), "double", n = file.size(file.path(ds_dir, "offset.bin")) / 8)
N <- length(offset)
genes <- read.delim(file.path(ds_dir, "genes.tsv"), colClasses = c("character", "integer", "integer"))
k <- match(gene, genes$gene)
stopifnot("gene not in genes.tsv" = !is.na(k))
Y <- read_matrix(file.path(ds_dir, "Y.bin"), N)
Y1 <- read_matrix(file.path(ds_dir, "Y1.bin"), N)
Y2 <- read_matrix(file.path(ds_dir, "Y2.bin"), N)
stopifnot("Y1 and Y2 must be integer counts (see the header)" = all(Y1 == round(Y1)) && all(Y2 == round(Y2)))
stopifnot("Y, Y1, Y2 do not have one column per gene" = ncol(Y) == nrow(genes) && ncol(Y1) == nrow(genes) &&
            ncol(Y2) == nrow(genes))
X <- read_matrix(file.path(ds_dir, "X.bin"), N)
chr <- genes$chr[k]
markers <- read.delim(file.path(geno_dir, sprintf("chr%d.markers.tsv", chr)),
                      colClasses = c("character", "integer", "integer"))
Z <- read_matrix(file.path(geno_dir, sprintf("chr%d.Z.bin", chr)), N)
stopifnot("Z columns differ from the marker table" = ncol(Z) == nrow(markers))

n_warn <- 0
t0 <- proc.time()[["elapsed"]]
r <- withCallingHandlers(
  trecase(Y[, k, drop = FALSE], Y1[, k, drop = FALSE], Y2[, k, drop = FALSE], X, Z,
          output.tag = tag, p.cut = P_CUT, offset = offset, local.only = TRUE,
          local.distance = LOCAL_DISTANCE, eChr = chr, ePos = genes$pos[k],
          mChr = markers$chr, mPos = markers$pos, trace = TRACE),
  warning = function(w) {   # logged one per line and counted, instead of R's "50 or more warnings"
    message("trecase warning: ", conditionMessage(w))
    n_warn <<- n_warn + 1
    invokeRestart("muffleWarning")
  })
secs <- proc.time()[["elapsed"]] - t0
stopifnot("trecase did not report success" = r$succeed == 1)

status <- data.frame(gene = gene, succeed = r$succeed, yFailBaselineModel = r$yFailBaselineModel,
                     seconds = secs, n_markers_chr = ncol(Z), n_warnings = n_warn)
tmp <- paste0(tag, "_status.tsv.tmp")
write.table(status, tmp, sep = "\t", quote = FALSE, row.names = FALSE)
stopifnot(file.rename(tmp, paste0(tag, "_status.tsv")))
