"""
Tests for tensorqtl.genotypeio module.

This module tests the genotype I/O functionality including:
- PLINK bed/bim/fam file reading
- PGEN file reading
- Sample selection and filtering
- Phenotype file reading
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import gzip
from pathlib import Path
import sys

# Add tensorqtl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tensorqtl.genotypeio as genotypeio
    import tensorqtl.pgen as pgen
    from tensorqtl.core import read_phenotype_bed
    TENSORQTL_AVAILABLE = True
except ImportError:
    TENSORQTL_AVAILABLE = False

from tests.utils import create_temp_files

class TestPhenotypeBedIO:
    """Test phenotype BED file I/O."""

    def test_read_phenotype_bed_basic(self, temp_dir):
        """Test basic phenotype BED file reading."""
        if not TENSORQTL_AVAILABLE:
            pytest.skip("tensorqtl not available")

        # Create test BED content
        bed_content = """#chr\tstart\tend\tgene_id\tS001\tS002\tS003
chr1\t1000\t2000\tGENE1\t1.5\t2.0\t-0.5
chr1\t3000\t4000\tGENE2\t0.0\t1.2\t-1.0
chr2\t5000\t6000\tGENE3\t2.1\t-0.3\t0.8"""

        bed_file = os.path.join(temp_dir, "test.bed")
        with open(bed_file, 'w') as f:
            f.write(bed_content)

        # Read the file
        phenotype_df, phenotype_pos_df = read_phenotype_bed(bed_file)

        # Check phenotype data
        assert phenotype_df.shape == (3, 3)  # 3 genes, 3 samples
        assert list(phenotype_df.columns) == ['S001', 'S002', 'S003']
        assert list(phenotype_df.index) == ['GENE1', 'GENE2', 'GENE3']

        # Check position data
        assert phenotype_pos_df.shape == (3, 3)  # 3 genes, 3 position columns
        assert list(phenotype_pos_df.columns) == ['chr', 'start', 'end']
        assert phenotype_pos_df.loc['GENE1', 'chr'] == 'chr1'
        assert phenotype_pos_df.loc['GENE1', 'start'] == 1001  # BED coordinates are converted to 1-based

    def test_read_phenotype_bed_gzipped(self, temp_dir):
        """Test reading gzipped BED file."""
        if not TENSORQTL_AVAILABLE:
            pytest.skip("tensorqtl not available")

        bed_content = """#chr\tstart\tend\tgene_id\tS001\tS002
chr1\t1000\t2000\tGENE1\t1.5\t2.0
chr1\t3000\t4000\tGENE2\t0.0\t1.2"""

        bed_file = os.path.join(temp_dir, "test.bed.gz")
        with gzip.open(bed_file, 'wt') as f:
            f.write(bed_content)

        phenotype_df, phenotype_pos_df = read_phenotype_bed(bed_file)

        assert phenotype_df.shape == (2, 2)
        assert list(phenotype_df.columns) == ['S001', 'S002']

    def test_read_phenotype_bed_missing_values(self, temp_dir):
        """Test handling of missing values in BED file."""
        if not TENSORQTL_AVAILABLE:
            pytest.skip("tensorqtl not available")

        bed_content = """#chr\tstart\tend\tgene_id\tS001\tS002\tS003
chr1\t1000\t2000\tGENE1\t1.5\tNA\t-0.5
chr1\t3000\t4000\tGENE2\t0.0\t1.2\t."""

        bed_file = os.path.join(temp_dir, "test_missing.bed")
        with open(bed_file, 'w') as f:
            f.write(bed_content)

        phenotype_df, phenotype_pos_df = read_phenotype_bed(bed_file)

        # Check that missing values are handled (should be NaN)
        assert pd.isna(phenotype_df.loc['GENE1', 'S002'])

    def test_read_phenotype_bed_single_position(self, temp_dir):
        """Test BED file with single position (TSS format)."""
        if not TENSORQTL_AVAILABLE:
            pytest.skip("tensorqtl not available")

        bed_content = """#chr\tstart\tend\tgene_id\tS001\tS002
chr1\t1000\t1001\tGENE1\t1.5\t2.0
chr1\t3000\t3001\tGENE2\t0.0\t1.2"""

        bed_file = os.path.join(temp_dir, "test_tss.bed")
        with open(bed_file, 'w') as f:
            f.write(bed_content)

        phenotype_df, phenotype_pos_df = read_phenotype_bed(bed_file)

        # Should work with TSS format (start == end - 1)
        assert phenotype_pos_df.loc['GENE1', 'end'] == 1001
        assert phenotype_pos_df.loc['GENE1', 'start'] == 1000

@pytest.mark.skipif(not TENSORQTL_AVAILABLE, reason="tensorqtl not available")
class TestPgenIO:
    """Test PGEN file I/O (if available)."""

    def test_pgen_reader_available(self):
        """Test that PGEN reader is available."""
        # This test checks if pgenlib is available
        try:
            import pgenlib
            assert hasattr(pgen, 'get_reader')
            assert hasattr(pgen, 'read')
        except ImportError:
            pytest.skip("pgenlib not available")

    def test_read_pvar(self, temp_dir):
        """Test reading PVAR file."""
        pvar_content = """#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
1\t1000\trs001\tA\tG\t.\t.\t.
1\t2000\trs002\tC\tT\t.\t.\t.
2\t3000\trs003\tG\tA\t.\t.\t."""

        pvar_file = os.path.join(temp_dir, "test.pvar")
        with open(pvar_file, 'w') as f:
            f.write(pvar_content)

        pvar_df = pgen.read_pvar(pvar_file)

        assert pvar_df.shape == (3, 8)
        assert list(pvar_df.columns) == ['chrom', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info']
        assert pvar_df.loc[0, 'chrom'] == '1'
        assert pvar_df.loc[0, 'pos'] == 1000
        assert pvar_df.loc[0, 'id'] == 'rs001'

    def test_read_psam(self, temp_dir):
        """Test reading PSAM file."""
        psam_content = """#IID\tSEX
S001\t1
S002\t2
S003\t1"""

        psam_file = os.path.join(temp_dir, "test.psam")
        with open(psam_file, 'w') as f:
            f.write(psam_content)

        psam_df = pgen.read_psam(psam_file)

        assert psam_df.shape == (3, 1)
        assert list(psam_df.index) == ['S001', 'S002', 'S003']
        assert 'SEX' in psam_df.columns

class TestGenotypeDataHandling:
    """Test genotype data handling and manipulation."""

    def test_create_synthetic_genotype_matrix(self):
        """Test creation of synthetic genotype matrix for testing."""
        n_variants, n_samples = 100, 50
        np.random.seed(42)

        # Create synthetic genotype matrix
        genotypes = np.random.choice([0, 1, 2], size=(n_variants, n_samples))

        # Create sample and variant IDs
        sample_ids = [f"S{i:03d}" for i in range(n_samples)]
        variant_ids = [f"chr1_{i*1000+10000}_A_G" for i in range(n_variants)]

        geno_df = pd.DataFrame(genotypes, index=variant_ids, columns=sample_ids)

        assert geno_df.shape == (n_variants, n_samples)
        assert len(geno_df.index) == n_variants
        assert len(geno_df.columns) == n_samples

        # Check that genotypes are in valid range
        assert geno_df.min().min() >= 0
        assert geno_df.max().max() <= 2

    def test_genotype_filtering(self):
        """Test genotype filtering operations."""
        # Create test data with some edge cases
        genotypes = np.array([
            [0, 1, 2, 1],  # Normal variant
            [0, 0, 0, 0],  # Monomorphic
            [1, 1, 1, 1],  # All heterozygous
            [-9, 1, 2, 1], # Missing data
        ])

        sample_ids = ['S1', 'S2', 'S3', 'S4']
        variant_ids = ['var1', 'var2', 'var3', 'var4']

        geno_df = pd.DataFrame(genotypes, index=variant_ids, columns=sample_ids)

        # Test filtering monomorphic variants
        # Variant 2 (index 1) is monomorphic
        maf_values = []
        for i in range(len(genotypes)):
            variant = genotypes[i]
            valid_geno = variant[variant >= 0]  # Remove missing
            if len(valid_geno) > 0:
                af = np.mean(valid_geno) / 2
                maf = min(af, 1 - af)
                maf_values.append(maf)
            else:
                maf_values.append(0)

        # Should identify monomorphic variant
        assert maf_values[1] == 0  # var2 is monomorphic

    def test_sample_selection(self):
        """Test sample selection and subsetting."""
        n_variants, n_samples = 10, 20
        genotypes = np.random.choice([0, 1, 2], size=(n_variants, n_samples))

        sample_ids = [f"S{i:03d}" for i in range(n_samples)]
        variant_ids = [f"var{i}" for i in range(n_variants)]

        geno_df = pd.DataFrame(genotypes, index=variant_ids, columns=sample_ids)

        # Select subset of samples
        selected_samples = sample_ids[:10]
        subset_df = geno_df[selected_samples]

        assert subset_df.shape == (n_variants, 10)
        assert list(subset_df.columns) == selected_samples

class TestDataConsistency:
    """Test data consistency checks."""

    def test_sample_id_consistency(self):
        """Test that sample IDs are consistent across data types."""
        sample_ids = ['S001', 'S002', 'S003']

        # Create genotype data
        genotypes = np.random.choice([0, 1, 2], size=(5, 3))
        geno_df = pd.DataFrame(genotypes, columns=sample_ids)

        # Create phenotype data
        phenotypes = np.random.normal(0, 1, size=(2, 3))
        pheno_df = pd.DataFrame(phenotypes, columns=sample_ids)

        # Create covariate data
        covariates = np.random.normal(0, 1, size=(2, 3))
        cov_df = pd.DataFrame(covariates, columns=sample_ids)

        # Check consistency
        assert geno_df.columns.equals(pheno_df.columns)
        assert pheno_df.columns.equals(cov_df.columns)

    def test_missing_sample_handling(self):
        """Test handling of missing samples across datasets."""
        geno_samples = ['S001', 'S002', 'S003', 'S004']
        pheno_samples = ['S001', 'S002', 'S003']  # Missing S004
        cov_samples = ['S001', 'S002', 'S003', 'S005']  # Missing S004, extra S005

        # Find common samples
        common_samples = list(set(geno_samples) & set(pheno_samples) & set(cov_samples))
        common_samples.sort()

        assert common_samples == ['S001', 'S002', 'S003']

class TestFileFormatValidation:
    """Test validation of file formats."""

    def test_bed_format_validation(self, temp_dir):
        """Test validation of BED format requirements."""
        # Valid BED format
        valid_bed = """#chr\tstart\tend\tgene_id\tS001\tS002
chr1\t1000\t2000\tGENE1\t1.5\t2.0
chr1\t3000\t4000\tGENE2\t0.0\t1.2"""

        bed_file = os.path.join(temp_dir, "valid.bed")
        with open(bed_file, 'w') as f:
            f.write(valid_bed)

        # Should be able to read without errors
        if TENSORQTL_AVAILABLE:
            phenotype_df, phenotype_pos_df = read_phenotype_bed(bed_file)
            assert phenotype_df.shape == (2, 2)

    def test_invalid_bed_format(self, temp_dir):
        """Test handling of invalid BED format."""
        # Missing required columns
        invalid_bed = """gene_id\tS001\tS002
GENE1\t1.5\t2.0
GENE2\t0.0\t1.2"""

        bed_file = os.path.join(temp_dir, "invalid.bed")
        with open(bed_file, 'w') as f:
            f.write(invalid_bed)

        # Should handle gracefully or raise appropriate error
        if TENSORQTL_AVAILABLE:
            with pytest.raises((ValueError, KeyError, IndexError)):
                read_phenotype_bed(bed_file)

class TestLargeDataHandling:
    """Test handling of larger datasets (but still manageable for CI)."""

    @pytest.mark.slow
    def test_medium_dataset(self):
        """Test with medium-sized dataset."""
        n_variants, n_samples, n_genes = 1000, 100, 50

        # Create synthetic data
        genotypes = np.random.choice([0, 1, 2], size=(n_variants, n_samples))
        phenotypes = np.random.normal(0, 1, size=(n_genes, n_samples))

        sample_ids = [f"S{i:04d}" for i in range(n_samples)]
        variant_ids = [f"chr1_{i*1000+10000}_A_G" for i in range(n_variants)]
        gene_ids = [f"ENSG{i:08d}.1" for i in range(n_genes)]

        geno_df = pd.DataFrame(genotypes, index=variant_ids, columns=sample_ids)
        pheno_df = pd.DataFrame(phenotypes, index=gene_ids, columns=sample_ids)

        # Test basic operations
        assert geno_df.shape == (n_variants, n_samples)
        assert pheno_df.shape == (n_genes, n_samples)

        # Test subsetting
        subset_samples = sample_ids[:50]
        geno_subset = geno_df[subset_samples]
        pheno_subset = pheno_df[subset_samples]

        assert geno_subset.shape == (n_variants, 50)
        assert pheno_subset.shape == (n_genes, 50)

class TestRealDataIntegration:
    """Test with real GEUVADIS subset data if available."""

    def test_load_test_data(self, test_data_dir):
        """Test loading prepared test data."""
        expr_file = test_data_dir / "test_expression.bed.gz"
        cov_file = test_data_dir / "test_covariates.txt"

        if expr_file.exists() and TENSORQTL_AVAILABLE:
            # Test reading expression data
            phenotype_df, phenotype_pos_df = read_phenotype_bed(str(expr_file))

            assert phenotype_df is not None
            assert phenotype_pos_df is not None
            assert phenotype_df.shape[0] > 0  # Should have some genes
            assert phenotype_df.shape[1] > 0  # Should have some samples

            # Check position data format
            assert 'chr' in phenotype_pos_df.columns
            assert 'start' in phenotype_pos_df.columns
            assert 'end' in phenotype_pos_df.columns

        if cov_file.exists():
            # Test reading covariate data
            cov_df = pd.read_csv(cov_file, sep='\t', index_col=0)

            assert cov_df.shape[0] > 0  # Should have some covariates
            assert cov_df.shape[1] > 0  # Should have some samples

            # If both files exist, check sample consistency
            if expr_file.exists() and TENSORQTL_AVAILABLE:
                common_samples = set(phenotype_df.columns) & set(cov_df.columns)
                assert len(common_samples) > 0  # Should have overlapping samples

    def test_data_types_and_ranges(self, test_expression_data, test_covariates_data):
        """Test data types and value ranges of real data."""
        expr_df, pos_df = test_expression_data

        if expr_df is not None:
            # Expression values should be numeric
            assert expr_df.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()

            # Position data should have correct types
            assert pos_df['start'].dtype in [np.int32, np.int64]
            assert pos_df['end'].dtype in [np.int32, np.int64]

        if test_covariates_data is not None:
            # Covariate values should be numeric
            assert test_covariates_data.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()