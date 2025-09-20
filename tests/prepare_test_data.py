#!/usr/bin/env python3
"""
Prepare test data for tensorQTL test suite.

This script creates smaller test datasets from the GEUVADIS example data
and generates synthetic data for edge case testing.
"""

import os
import sys
import pandas as pd
import numpy as np
import gzip
import shutil
from pathlib import Path

# Add tensorqtl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tensorqtl
    from tensorqtl import pgen
except ImportError:
    print("Warning: Could not import tensorqtl modules. Some functionality may be limited.")

def create_test_genotypes(source_prefix, output_prefix, n_variants=200, n_samples=50, random_seed=42):
    """
    Create test genotype files by subsetting the GEUVADIS data.

    Parameters:
    -----------
    source_prefix : str
        Path prefix to source pgen/pvar/psam files
    output_prefix : str
        Path prefix for output test files
    n_variants : int
        Number of variants to keep
    n_samples : int
        Number of samples to keep
    random_seed : int
        Random seed for reproducibility
    """
    np.random.seed(random_seed)

    print(f"Creating test genotype data from {source_prefix}")

    # Read pvar to get variant info
    pvar_file = f"{source_prefix}.pvar"
    if not os.path.exists(pvar_file):
        print(f"Warning: {pvar_file} not found, skipping genotype preparation")
        return

    pvar_df = pd.read_csv(pvar_file, sep='\t', comment='#',
                         names=['chrom', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info'],
                         dtype={'chrom': str, 'pos': np.int32})

    # Filter variants with reasonable MAF
    print(f"Total variants in source: {len(pvar_df)}")

    # Select random subset of variants
    if len(pvar_df) > n_variants:
        selected_variants = np.random.choice(len(pvar_df), n_variants, replace=False)
        selected_variants.sort()
    else:
        selected_variants = np.arange(len(pvar_df))

    pvar_subset = pvar_df.iloc[selected_variants].copy()

    # Read psam to get sample info
    psam_file = f"{source_prefix}.psam"
    psam_df = pd.read_csv(psam_file, sep='\t', index_col=0)

    print(f"Total samples in source: {len(psam_df)}")

    # Select random subset of samples
    if len(psam_df) > n_samples:
        selected_samples = np.random.choice(len(psam_df), n_samples, replace=False)
        selected_samples.sort()
    else:
        selected_samples = np.arange(len(psam_df))

    psam_subset = psam_df.iloc[selected_samples].copy()

    # Write subset files
    pvar_subset.to_csv(f"{output_prefix}.pvar", sep='\t', index=False, header=False)
    psam_subset.to_csv(f"{output_prefix}.psam", sep='\t')

    print(f"Created test data with {len(pvar_subset)} variants and {len(psam_subset)} samples")

    # Note: Creating subset pgen file would require pgenlib functionality
    # For now, we'll document this limitation
    print("Note: Subset pgen file creation requires additional pgenlib operations")
    print("Tests should use the original pgen with variant/sample indexing")

def create_test_phenotypes(source_bed, output_bed, sample_ids, n_genes=10, random_seed=42):
    """
    Create test phenotype file by subsetting the GEUVADIS expression data.
    """
    np.random.seed(random_seed)

    print(f"Creating test phenotype data from {source_bed}")

    if not os.path.exists(source_bed):
        print(f"Warning: {source_bed} not found, skipping phenotype preparation")
        return

    # Read phenotype data
    if source_bed.endswith('.gz'):
        opener = gzip.open
    else:
        opener = open

    with opener(source_bed, 'rt') as f:
        header = f.readline().strip().split('\t')

    # Get available samples in the phenotype file
    pheno_samples = header[4:]  # Skip chr, start, end, gene_id

    # Find intersection with our test samples
    common_samples = [s for s in sample_ids if s in pheno_samples]
    if len(common_samples) < len(sample_ids) * 0.8:
        print(f"Warning: Only {len(common_samples)}/{len(sample_ids)} samples found in phenotype data")

    # Read full data to select genes
    phenotype_df = pd.read_csv(source_bed, sep='\t', index_col=3)

    # Filter to chr18 genes and select subset
    chr18_genes = phenotype_df[phenotype_df['#chr'] == 'chr18']
    if len(chr18_genes) > n_genes:
        selected_genes = chr18_genes.sample(n=n_genes, random_state=random_seed)
    else:
        selected_genes = chr18_genes

    # Keep only common samples
    sample_cols = ['#chr', 'start', 'end'] + common_samples
    subset_df = selected_genes[sample_cols]

    # Write subset
    with gzip.open(output_bed, 'wt') as f:
        f.write('#chr\tstart\tend\tphenotype_id\t' + '\t'.join(common_samples) + '\n')
        for gene_id, row in subset_df.iterrows():
            f.write(f"{row['#chr']}\t{row['start']}\t{row['end']}\t{gene_id}")
            for sample in common_samples:
                f.write(f"\t{row[sample]}")
            f.write('\n')

    print(f"Created test phenotype data with {len(subset_df)} genes and {len(common_samples)} samples")
    return common_samples

def create_test_covariates(source_cov, output_cov, sample_ids, n_pcs=5):
    """
    Create test covariate file by subsetting the GEUVADIS covariates.
    """
    print(f"Creating test covariate data from {source_cov}")

    if not os.path.exists(source_cov):
        print(f"Warning: {source_cov} not found, skipping covariate preparation")
        return

    # Read covariate data
    cov_df = pd.read_csv(source_cov, sep='\t', index_col=0)

    # Get intersection of samples
    common_samples = [s for s in sample_ids if s in cov_df.columns]

    # Select first n_pcs principal components and subset samples
    pc_rows = [f'PC{i}' for i in range(1, n_pcs + 1) if f'PC{i}' in cov_df.index]
    subset_df = cov_df.loc[pc_rows, common_samples]

    # Write subset
    subset_df.to_csv(output_cov, sep='\t')

    print(f"Created test covariate data with {len(subset_df)} covariates and {len(common_samples)} samples")

def create_synthetic_data(output_dir, n_samples=20, n_variants=50, n_genes=5):
    """
    Create synthetic test data for edge cases.
    """
    np.random.seed(123)

    print("Creating synthetic test data")

    # Create synthetic genotypes (0, 1, 2)
    genotypes = np.random.choice([0, 1, 2], size=(n_variants, n_samples),
                                p=[0.64, 0.32, 0.04])  # MAF ~ 0.2

    # Create some edge cases
    genotypes[0, :] = 0  # Monomorphic variant
    genotypes[1, 0] = -9  # Missing genotype

    # Create synthetic phenotypes
    phenotypes = np.random.normal(0, 1, size=(n_genes, n_samples))

    # Add some true associations
    phenotypes[0, :] += 0.5 * genotypes[10, :]  # Strong effect
    phenotypes[1, :] += 0.2 * genotypes[20, :]  # Weak effect

    # Create sample and variant IDs
    sample_ids = [f"S{i:03d}" for i in range(n_samples)]
    variant_ids = [f"chr1_{i*1000+10000}_A_G" for i in range(n_variants)]
    gene_ids = [f"ENSG{i:08d}.1" for i in range(n_genes)]

    # Write synthetic genotype data as simple text (for easy loading in tests)
    geno_df = pd.DataFrame(genotypes, index=variant_ids, columns=sample_ids)
    geno_df.to_csv(os.path.join(output_dir, "synthetic_genotypes.txt"), sep='\t')

    # Write synthetic phenotypes in BED format
    with open(os.path.join(output_dir, "synthetic_phenotypes.bed"), 'w') as f:
        f.write('#chr\tstart\tend\tgene_id\t' + '\t'.join(sample_ids) + '\n')
        for i, gene_id in enumerate(gene_ids):
            f.write(f"chr1\t{i*10000}\t{i*10000+1}\t{gene_id}")
            for val in phenotypes[i, :]:
                f.write(f"\t{val:.6f}")
            f.write('\n')

    # Write synthetic covariates
    covariates = np.random.normal(0, 1, size=(3, n_samples))  # 3 PCs
    cov_df = pd.DataFrame(covariates, index=['PC1', 'PC2', 'PC3'], columns=sample_ids)
    cov_df.to_csv(os.path.join(output_dir, "synthetic_covariates.txt"), sep='\t')

    print(f"Created synthetic data with {n_variants} variants, {n_genes} genes, {n_samples} samples")

def main():
    """Main function to prepare all test data."""

    # Define paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    example_dir = base_dir.parent / "example" / "data"

    # GEUVADIS data paths
    geuvadis_prefix = example_dir / "GEUVADIS.445_samples.GRCh38.20170504.maf01.filtered.nodup.chr18"
    geuvadis_bed = example_dir / "GEUVADIS.445_samples.expression.bed.gz"
    geuvadis_cov = example_dir / "GEUVADIS.445_samples.covariates.txt"

    # Test data output paths
    test_prefix = data_dir / "test_geuvadis"
    test_bed = data_dir / "test_expression.bed.gz"
    test_cov = data_dir / "test_covariates.txt"

    print("Preparing tensorQTL test data...")
    print(f"Source data directory: {example_dir}")
    print(f"Output data directory: {data_dir}")

    # Get actual sample IDs from the phenotype file
    if geuvadis_bed.exists():
        with gzip.open(str(geuvadis_bed), 'rt') as f:
            header = f.readline().strip().split('\t')
        all_samples = header[4:]  # Skip chr, start, end, gene_id
        # Select subset of actual samples
        np.random.seed(42)
        test_samples = sorted(np.random.choice(all_samples, size=min(50, len(all_samples)), replace=False))
    else:
        test_samples = [f"HG00{100+i:03d}" for i in range(50)]  # Fallback

    # Create test datasets from GEUVADIS data
    if geuvadis_prefix.with_suffix('.pvar').exists():
        create_test_genotypes(str(geuvadis_prefix), str(test_prefix))

    if geuvadis_bed.exists():
        actual_samples = create_test_phenotypes(str(geuvadis_bed), str(test_bed), test_samples)
        if actual_samples:
            test_samples = actual_samples  # Use actual samples found

    if geuvadis_cov.exists():
        create_test_covariates(str(geuvadis_cov), str(test_cov), test_samples)

    # Create synthetic test data
    create_synthetic_data(str(data_dir))

    # Create a README for the test data
    readme_content = """# Test Data for tensorQTL

This directory contains test data for the tensorQTL test suite:

## GEUVADIS Subset Data
- `test_geuvadis.pvar/psam`: Subset of GEUVADIS genotype data (chr18, ~200 variants, ~50 samples)
- `test_expression.bed.gz`: Subset of expression data (~10 genes from chr18)
- `test_covariates.txt`: Matching covariate data (first 5 PCs)

## Synthetic Data
- `synthetic_genotypes.txt`: Small synthetic genotype matrix for unit tests
- `synthetic_phenotypes.bed`: Synthetic phenotype data with known associations
- `synthetic_covariates.txt`: Synthetic covariate data

## Usage
This data is automatically generated by `prepare_test_data.py` and is used
by the pytest test suite to validate tensorQTL functionality.

To regenerate test data:
```bash
cd tests
python prepare_test_data.py
```
"""

    with open(data_dir / "README.md", 'w') as f:
        f.write(readme_content)

    print("\nTest data preparation complete!")
    print(f"Generated files in: {data_dir}")

if __name__ == "__main__":
    main()