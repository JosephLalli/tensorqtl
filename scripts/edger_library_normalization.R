#!/usr/bin/env Rscript
# edgeR library normalization for hapmixQTL (user rule, 2026-09-25): CPM is
# computed as edgeR computes it, from the effective library size
# lib.size x norm.factors, not from raw column sums.
#
# The standard edgeR sequence, on Salmon POINT-estimate gene counts:
#   1. DGEList of the count matrix for EVERY gene (lib.size = column sums)
#   2. filterByExpr with no design (edgeR:::filterByExpr.default: CPM cutoff
#      10 / median lib size in millions in >= 10 + 0.7 (n - 10) samples, and a
#      total count >= 15)
#   3. restrict to the calibration gene set (the caller's --restrict list; in
#      the calibration phase, protein-coding autosomal genes)
#   4. subset with keep.lib.sizes = FALSE, so lib.size is recomputed from the
#      kept genes
#   5. calcNormFactors (TMM)
# Effective library size = lib.size x norm.factors, the value cpm() divides by.
#
# Only filterByExpr, calcNormFactors and DGEList are used; none fits a linear
# model, so the host's mixed-BLAS crash (lmFit, lm, %*%) does not apply.
#
# usage: edger_library_normalization.R counts.tsv.gz restrict.txt out_dir
suppressMessages(library(edgeR))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("usage: edger_library_normalization.R counts.tsv.gz restrict.txt out_dir")
counts <- as.matrix(read.delim(args[1], row.names = 1, check.names = FALSE))
restrict <- readLines(args[2])
dir.create(args[3], showWarnings = FALSE, recursive = TRUE)
d <- DGEList(counts)
keep_expr <- suppressWarnings(filterByExpr(d))
keep <- keep_expr & rownames(d) %in% restrict
d <- d[keep, , keep.lib.sizes = FALSE]
d <- calcNormFactors(d, method = "TMM")
s <- data.frame(sample = colnames(d), lib_size = d$samples$lib.size,
                norm_factor = d$samples$norm.factors,
                eff_lib_size = d$samples$lib.size * d$samples$norm.factors)
write.table(s, file.path(args[3], "edger_samples.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
writeLines(rownames(d), file.path(args[3], "calibration_genes.txt"))
writeLines(rownames(counts)[keep_expr], file.path(args[3], "filter_by_expr_all_biotypes.txt"))
cat(sprintf("edgeR %s: %d genes in, %d pass filterByExpr, %d kept after the restriction; TMM factors %.3f-%.3f\n",
            as.character(packageVersion("edgeR")), nrow(counts), sum(keep_expr), sum(keep),
            min(s$norm_factor), max(s$norm_factor)))
