# hapmixQTL input and permutation contract

1. Salmon quant.sf NumReads point estimates define all expression values.
   Fractional counts are retained. Gibbs draws define allelic measurement
   variance; they do not replace point estimates.
2. Total counts include every mapped transcript of the gene. They are not
   inferred by adding counts of haplotype-paired transcripts.
3. Total expression and expression PCs use the same half-read transform,
   log2((count + 0.5)/(effective_library_size + 1) * 1e6).
4. edgeR filters the point-estimate total matrix, applies any explicit gene
   restriction, recomputes kept-gene library sizes and applies TMM.
   The effective library size is lib.size times norm.factor. The same filtered
   genes and effective sizes must be used for expression PCs and mapping.
5. Allelic working variance is the Gibbs variance of the log2 haplotype ratio
   plus the default counting term. It supplies relative variance shape; the
   residual scale is fitted. Total working variance is one.
6. A donor-gene pair has no allelic contribution if its variance is at most
   1e-12, it has no paired counts, or exactly one haplotype has fewer than
   0.5 reads. Excluded entries have finite phenotype values and Va = 0.
7. Sample IDs and orders must agree across expression, genotypes, phase and
   covariates. Haplotype suffix order must agree with phased GT allele order.
8. RNA-tied covariates travel with the donor's RNA record under permutation.
   Genotype PCs stay with genotypes. The allelic channel is through the origin;
   the total channel includes an intercept and covariates.
9. Gene-level detection uses permutation p-values or their Beta approximation.
   A selected lead's nominal p-value is not a gene-level p-value.
10. Published mixQTL replication reads point estimates without Gibbs draws and
    preserves its natural-log response and published filters.

The Salmon runner validates covariate provenance and refuses mismatched units,
gene sets or effective library sizes. The portable preparation command produces
the matched files; do not bypass that check to compensate for an incomplete
preparation. See [hapmixqtl_inputs.md](hapmixqtl_inputs.md).
