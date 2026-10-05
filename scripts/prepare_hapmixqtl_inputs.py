#!/usr/bin/env python3
"""Prepare point-estimate normalization and covariates for hapmixQTL.

The manifest is a two-column TSV (sample_id, Salmon directory).  ``tx2gene``
is a two-column TSV (base transcript ID, gene ID).  The optional sample
covariates file is a samples-by-numeric-covariates TSV with sample IDs in its
first column.  All three sample sources must describe exactly the manifest
samples.  This tool deliberately prepares no Gibbs draws: the mapper reads
those separately for its measurement-variance inputs.
"""
import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_covariates as C  # noqa: E402
import run_hapmixqtl_from_salmon as H  # noqa: E402


def _open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def _manifest(path):
    rows = []
    for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 2 or not fields[0].strip() or not fields[1].strip():
            raise SystemExit(f"{path}:{line_no}: expected sample_id and Salmon directory")
        rows.append((fields[0].strip(), fields[1].strip()))
    samples = [sample for sample, _ in rows]
    if not samples:
        raise SystemExit(f"{path}: no manifest rows")
    duplicate = sorted({sample for sample in samples if samples.count(sample) > 1})
    if duplicate:
        raise SystemExit(f"{path}: duplicate sample IDs, e.g. {duplicate[:3]}")
    return samples


def _tx2gene(path):
    mapping = {}
    for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 2 or not fields[0].strip() or not fields[1].strip():
            raise SystemExit(f"{path}:{line_no}: expected transcript_id and gene_id")
        tx, gene = fields[0].strip(), fields[1].strip()
        if tx in mapping and mapping[tx] != gene:
            raise SystemExit(f"{path}:{line_no}: transcript {tx!r} maps to multiple genes")
        mapping[tx] = gene
    if not mapping:
        raise SystemExit(f"{path}: no transcript-to-gene mappings")
    return mapping


def _vcf_samples(path):
    with _open(path) as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
                if len(set(samples)) != len(samples):
                    raise SystemExit(f"{path}: duplicate VCF sample IDs")
                return samples
    raise SystemExit(f"{path}: missing #CHROM VCF header")


def _sample_covariates(path, samples):
    if path is None:
        return pd.DataFrame(index=samples)
    header = Path(path).read_text().splitlines()[0].split("\t")
    if len(header) < 2:
        raise SystemExit(f"{path}: expected a sample-ID column and at least one covariate")
    if len(set(header)) != len(header):
        raise SystemExit(f"{path}: duplicate column names")
    reserved = [name for name in header[1:] if name.startswith(("geno_pc", "expr_pc"))]
    if reserved:
        raise SystemExit(f"{path}: reserved covariate column names, e.g. {reserved[:3]}")
    cov = pd.read_csv(path, sep="\t", index_col=0)
    cov.index = cov.index.astype(str)
    if cov.index.has_duplicates:
        raise SystemExit(f"{path}: duplicate sample IDs")
    if set(cov.index) != set(samples):
        missing, extra = sorted(set(samples) - set(cov.index)), sorted(set(cov.index) - set(samples))
        raise SystemExit(f"{path}: sample IDs must equal manifest IDs; missing={missing[:3]}, extra={extra[:3]}")
    if cov.shape[1] == 0:
        return pd.DataFrame(index=samples)
    cov = cov.apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(cov.to_numpy(float)).all():
        raise SystemExit(f"{path}: covariates must be finite numeric values")
    return cov.loc[samples]


def _full_rank(frame, label):
    design = np.column_stack([np.ones(len(frame)), frame.to_numpy(float)])
    rank = np.linalg.matrix_rank(design)
    if rank != design.shape[1]:
        raise SystemExit(f"{label} are not full rank with an intercept ({rank}/{design.shape[1]})")
    return rank


def prepare(manifest, tx2gene, vcf, out, sample_covariates=None,
            hap_suffix=("_hapA", "_hapB"), gene_restrict=None,
            n_expr_pc=10, n_geno_pc=3):
    samples = _manifest(manifest)
    vcf_samples = _vcf_samples(vcf)
    missing = [sample for sample in samples if sample not in vcf_samples]
    if missing:
        raise SystemExit(f"{vcf}: manifest samples absent from VCF, e.g. {missing[:3]}")
    if n_geno_pc < 0 or n_expr_pc < 0:
        raise SystemExit("PC counts must be non-negative")
    if n_geno_pc > len(samples) - 1:
        raise SystemExit(f"--n-geno-pc {n_geno_pc} exceeds available sample rank {len(samples)-1}")

    mapping = _tx2gene(tx2gene)
    genes = sorted(set(mapping.values()))
    out = Path(out)
    point_dir = out / "point_estimates"
    edger_dir = point_dir / "edger"
    point_dir.mkdir(parents=True, exist_ok=True)
    _, _, _, totals_all = H.load_point_estimates(manifest, tx2gene, tuple(hap_suffix), genes, samples)
    if totals_all.index.has_duplicates or totals_all.columns.duplicated().any():
        raise SystemExit("point-estimate aggregation produced duplicate genes or samples")
    totals_all.to_csv(point_dir / "totals_all.tsv.gz", sep="\t")
    restrict = Path(gene_restrict).read_text().split() if gene_restrict else list(totals_all.index)
    if not set(restrict) & set(totals_all.index):
        raise SystemExit("--gene-restrict has no genes in --tx2gene")
    eff_lib, eqtl_genes = H.edger_normalize(totals_all, restrict, edger_dir)
    if not eqtl_genes:
        raise SystemExit("edgeR retained no calibration genes")

    supplied = _sample_covariates(sample_covariates, samples)
    genotype = (C.genotype_pcs(vcf, samples, n_pc=n_geno_pc).loc[samples]
                if n_geno_pc else pd.DataFrame(index=samples))
    base = pd.concat([supplied, genotype], axis=1)
    base_rank = _full_rank(base, "sample and genotype covariates")
    available = len(samples) - base_rank
    if n_expr_pc > available:
        raise SystemExit(f"--n-expr-pc {n_expr_pc} exceeds residual sample rank {available}")
    if n_expr_pc > len(eqtl_genes):
        raise SystemExit(f"--n-expr-pc {n_expr_pc} exceeds {len(eqtl_genes)} edgeR calibration genes")
    pcs = C.expression_pcs_point(totals_all.loc[eqtl_genes, samples], eff_lib,
                                  eqtl_genes, base.to_numpy(float), n_pc=n_expr_pc)
    expr = pd.DataFrame(pcs, index=samples,
                        columns=[f"expr_pc{i + 1}" for i in range(n_expr_pc)])
    covariates = pd.concat([base, expr], axis=1)
    _full_rank(covariates, "final covariates")
    out.mkdir(parents=True, exist_ok=True)
    cov_path = out / "covariates.tsv"
    covariates.to_csv(cov_path, sep="\t")
    genotype_cols = list(genotype.columns)
    (out / "genotype_covariates.txt").write_text("\n".join(genotype_cols) + ("\n" if genotype_cols else ""))
    point_ref = os.path.relpath(point_dir, Path.cwd())
    (out / "covariate_build.json").write_text(json.dumps({
        "columns": list(covariates.columns), "genotype_tied": genotype_cols,
        "rna_tied": [c for c in covariates.columns if c not in genotype_cols],
        "expression_pc_unit": "half_read_log_cpm", "point_estimates": point_ref,
        "expression_pcs": "half-read log-CPM of Salmon point estimates, edgeR effective library sizes, edgeR calibration genes, residualized on supplied and genotype covariates",
    }, indent=2) + "\n")
    return cov_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tx2gene", required=True)
    ap.add_argument("--vcf", required=True)
    ap.add_argument("--sample-covariates")
    ap.add_argument("--hap-suffix", default="_hapA,_hapB")
    ap.add_argument("--gene-restrict")
    ap.add_argument("--out", default="prepared")
    ap.add_argument("--n-expr-pc", type=int, default=10)
    ap.add_argument("--n-geno-pc", type=int, default=3)
    args = ap.parse_args(argv)
    suffixes = tuple(args.hap_suffix.split(","))
    if len(suffixes) != 2 or not all(suffixes) or suffixes[0] == suffixes[1]:
        raise SystemExit("--hap-suffix needs two distinct comma-separated suffixes")
    prepare(args.manifest, args.tx2gene, args.vcf, args.out, args.sample_covariates,
            suffixes, args.gene_restrict, args.n_expr_pc, args.n_geno_pc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
