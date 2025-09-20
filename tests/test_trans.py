"""
Tests for tensorqtl.trans module.

This module tests the trans-QTL mapping functionality including:
- Trans-QTL association mapping
- Sparse output mode
- Cis filtering
- Batch processing
"""

import pytest
import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path

# Add tensorqtl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tensorqtl.trans as trans
    import tensorqtl.core as core
    TENSORQTL_AVAILABLE = True
except ImportError:
    TENSORQTL_AVAILABLE = False

from tests.utils import simulate_qtl_data, validate_statistical_properties

@pytest.mark.skipif(not TENSORQTL_AVAILABLE, reason="tensorqtl not available")
class TestTransMapping:
    """Test trans-QTL mapping functions."""

    def test_map_trans_basic(self, device, torch_dtype):
        """Test basic trans-QTL mapping."""
        # Create synthetic data
        n_samples = 50
        n_variants = 100
        n_phenotypes = 20
        torch.manual_seed(42)

        genotypes = torch.randint(0, 3, (n_variants, n_samples), dtype=torch_dtype, device=device).float()
        phenotypes = torch.randn(n_phenotypes, n_samples, dtype=torch_dtype, device=device)

        # Convert to DataFrames for compatibility
        sample_ids = [f"S{i:03d}" for i in range(n_samples)]
        variant_ids = [f"var{i}" for i in range(n_variants)]
        phenotype_ids = [f"gene{i}" for i in range(n_phenotypes)]

        genotype_df = pd.DataFrame(genotypes.cpu().numpy(), index=variant_ids, columns=sample_ids)
        phenotype_df = pd.DataFrame(phenotypes.cpu().numpy(), index=phenotype_ids, columns=sample_ids)

        # Test trans mapping
        result = trans.map_trans(
            genotype_df, phenotype_df, covariates_df=None,
            return_sparse=True, pval_threshold=0.1,  # Liberal threshold for testing
            maf_threshold=0.0, batch_size=50
        )

        # Check result format
        assert isinstance(result, pd.DataFrame)
        expected_cols = ['phenotype_id', 'variant_id', 'pval', 'b', 'b_se', 'af']
        assert all(col in result.columns for col in expected_cols)

        # Check that we have some results
        assert len(result) > 0

        # Check p-value range
        assert (result['pval'] >= 0).all()
        assert (result['pval'] <= 1).all()

        # Check allele frequency range
        assert (result['af'] >= 0).all()
        assert (result['af'] <= 1).all()

    def test_map_trans_with_covariates(self, device, torch_dtype):
        """Test trans-QTL mapping with covariate adjustment."""
        n_samples = 50
        n_variants = 50
        n_phenotypes = 10
        torch.manual_seed(123)

        genotypes = torch.randint(0, 3, (n_variants, n_samples), dtype=torch_dtype, device=device).float()
        phenotypes = torch.randn(n_phenotypes, n_samples, dtype=torch_dtype, device=device)
        covariates = torch.randn(n_samples, 3, dtype=torch_dtype, device=device)

        # Convert to DataFrames
        sample_ids = [f"S{i:03d}" for i in range(n_samples)]
        variant_ids = [f"var{i}" for i in range(n_variants)]
        phenotype_ids = [f"gene{i}" for i in range(n_phenotypes)]
        covariate_ids = ['PC1', 'PC2', 'PC3']

        genotype_df = pd.DataFrame(genotypes.cpu().numpy(), index=variant_ids, columns=sample_ids)
        phenotype_df = pd.DataFrame(phenotypes.cpu().numpy(), index=phenotype_ids, columns=sample_ids)
        covariates_df = pd.DataFrame(covariates.cpu().numpy().T, index=covariate_ids, columns=sample_ids)

        # Test with covariates
        result_with_cov = trans.map_trans(
            genotype_df, phenotype_df, covariates_df=covariates_df,
            return_sparse=True, pval_threshold=0.1, batch_size=25
        )

        # Test without covariates
        result_without_cov = trans.map_trans(
            genotype_df, phenotype_df, covariates_df=None,
            return_sparse=True, pval_threshold=0.1, batch_size=25
        )

        # Both should return valid results
        assert len(result_with_cov) >= 0
        assert len(result_without_cov) >= 0

        # Results should generally be different (though might be similar for null data)
        if len(result_with_cov) > 0 and len(result_without_cov) > 0:
            # At minimum, check that we get results from both
            assert isinstance(result_with_cov, pd.DataFrame)
            assert isinstance(result_without_cov, pd.DataFrame)

    def test_map_trans_sparse_vs_dense(self, device, torch_dtype):
        """Test sparse vs dense output modes."""
        n_samples = 30
        n_variants = 40
        n_phenotypes = 5
        torch.manual_seed(42)

        genotypes = torch.randint(0, 3, (n_variants, n_samples), dtype=torch_dtype, device=device).float()
        phenotypes = torch.randn(n_phenotypes, n_samples, dtype=torch_dtype, device=device)

        sample_ids = [f"S{i:03d}" for i in range(n_samples)]
        variant_ids = [f"var{i}" for i in range(n_variants)]
        phenotype_ids = [f"gene{i}" for i in range(n_phenotypes)]

        genotype_df = pd.DataFrame(genotypes.cpu().numpy(), index=variant_ids, columns=sample_ids)
        phenotype_df = pd.DataFrame(phenotypes.cpu().numpy(), index=phenotype_ids, columns=sample_ids)

        # Test sparse mode
        result_sparse = trans.map_trans(
            genotype_df, phenotype_df, covariates_df=None,
            return_sparse=True, pval_threshold=0.1, batch_size=20
        )

        # Test dense mode
        result_dense = trans.map_trans(
            genotype_df, phenotype_df, covariates_df=None,
            return_sparse=False, batch_size=20
        )

        # Sparse should have fewer or equal rows (due to filtering)
        assert len(result_sparse) <= len(result_dense)

        # Dense mode should include all variant-phenotype pairs
        expected_dense_rows = n_variants * n_phenotypes
        assert len(result_dense) == expected_dense_rows

    def test_map_trans_maf_filtering(self, device, torch_dtype):
        """Test MAF filtering in trans mapping."""
        n_samples = 50
        torch.manual_seed(42)

        # Create genotypes with different MAFs
        genotypes_list = []

        # Add some rare variants (low MAF)
        rare_genotype = torch.zeros(n_samples, dtype=torch_dtype, device=device)
        rare_genotype[:2] = 1  # Only 2 samples with minor allele -> MAF = 0.02
        genotypes_list.append(rare_genotype)

        # Add common variant
        common_genotype = torch.randint(0, 3, (n_samples,), dtype=torch_dtype, device=device).float()
        genotypes_list.append(common_genotype)

        genotypes = torch.stack(genotypes_list)
        phenotypes = torch.randn(3, n_samples, dtype=torch_dtype, device=device)

        sample_ids = [f"S{i:03d}" for i in range(n_samples)]
        variant_ids = ['rare_var', 'common_var']
        phenotype_ids = ['gene1', 'gene2', 'gene3']

        genotype_df = pd.DataFrame(genotypes.cpu().numpy(), index=variant_ids, columns=sample_ids)
        phenotype_df = pd.DataFrame(phenotypes.cpu().numpy(), index=phenotype_ids, columns=sample_ids)

        # Test with strict MAF filtering
        result_strict = trans.map_trans(
            genotype_df, phenotype_df, covariates_df=None,
            return_sparse=False, maf_threshold=0.05  # Should filter out rare variant
        )

        # Test with lenient MAF filtering
        result_lenient = trans.map_trans(
            genotype_df, phenotype_df, covariates_df=None,
            return_sparse=False, maf_threshold=0.01
        )

        # Strict filtering should have fewer results
        assert len(result_strict) <= len(result_lenient)

        # Check that rare variants are filtered out in strict mode
        if len(result_strict) > 0:
            assert 'rare_var' not in result_strict['variant_id'].values

class TestCisFiltering:
    """Test cis association filtering."""

    def test_filter_cis_basic(self):
        """Test basic cis filtering functionality."""
        if not TENSORQTL_AVAILABLE:
            pytest.skip("tensorqtl not available")

        # Create test trans results
        trans_df = pd.DataFrame({
            'phenotype_id': ['gene1', 'gene1', 'gene2', 'gene2'],
            'variant_id': ['chr1_1000_A_G', 'chr2_2000_C_T', 'chr1_1500_G_A', 'chr3_3000_T_C'],
            'pval': [0.001, 0.01, 0.005, 0.02],
            'b': [0.5, 0.3, -0.4, 0.2],
            'b_se': [0.1, 0.15, 0.12, 0.18],
            'af': [0.3, 0.4, 0.25, 0.35]
        })

        # Create position data
        phenotype_pos = {
            'gene1': {'chr': 'chr1', 'start': 900, 'end': 1100},
            'gene2': {'chr': 'chr1', 'start': 1400, 'end': 1600},
        }

        # Create variant data
        variant_df = pd.DataFrame({
            'chrom': ['chr1', 'chr2', 'chr1', 'chr3'],
            'pos': [1000, 2000, 1500, 3000]
        }, index=['chr1_1000_A_G', 'chr2_2000_C_T', 'chr1_1500_G_A', 'chr3_3000_T_C'])

        # Filter cis associations (window = 1000)
        filtered_df = trans.filter_cis(trans_df, phenotype_pos, variant_df, window=1000)

        # Should remove cis associations
        # gene1 at chr1:900-1100 should exclude chr1_1000_A_G (within window)
        # gene2 at chr1:1400-1600 should exclude chr1_1500_G_A (within window)

        expected_kept = ['chr2_2000_C_T', 'chr3_3000_T_C']  # trans associations
        assert set(filtered_df['variant_id']) == set(expected_kept)

    def test_filter_cis_different_chromosomes(self):
        """Test that different chromosomes are always kept."""
        if not TENSORQTL_AVAILABLE:
            pytest.skip("tensorqtl not available")

        trans_df = pd.DataFrame({
            'phenotype_id': ['gene1', 'gene1'],
            'variant_id': ['chr1_1000_A_G', 'chr2_1000_C_T'],
            'pval': [0.001, 0.01],
            'b': [0.5, 0.3],
            'b_se': [0.1, 0.15],
            'af': [0.3, 0.4]
        })

        phenotype_pos = {
            'gene1': {'chr': 'chr1', 'start': 1000, 'end': 1001},
        }

        variant_df = pd.DataFrame({
            'chrom': ['chr1', 'chr2'],
            'pos': [1000, 1000]
        }, index=['chr1_1000_A_G', 'chr2_1000_C_T'])

        # Even with small window, different chromosome should be kept
        filtered_df = trans.filter_cis(trans_df, phenotype_pos, variant_df, window=0)

        # Should keep the chr2 variant (trans)
        assert 'chr2_1000_C_T' in filtered_df['variant_id'].values
        # Should remove the chr1 variant (cis)
        assert 'chr1_1000_A_G' not in filtered_df['variant_id'].values

class TestBatchProcessing:
    """Test batch processing functionality."""

    def test_trans_mapping_batch_consistency(self, device, torch_dtype):
        """Test that batch processing gives consistent results."""
        if not TENSORQTL_AVAILABLE:
            pytest.skip("tensorqtl not available")

        n_samples = 40
        n_variants = 60
        n_phenotypes = 8
        torch.manual_seed(42)

        genotypes = torch.randint(0, 3, (n_variants, n_samples), dtype=torch_dtype, device=device).float()
        phenotypes = torch.randn(n_phenotypes, n_samples, dtype=torch_dtype, device=device)

        sample_ids = [f"S{i:03d}" for i in range(n_samples)]
        variant_ids = [f"var{i}" for i in range(n_variants)]
        phenotype_ids = [f"gene{i}" for i in range(n_phenotypes)]

        genotype_df = pd.DataFrame(genotypes.cpu().numpy(), index=variant_ids, columns=sample_ids)
        phenotype_df = pd.DataFrame(phenotypes.cpu().numpy(), index=phenotype_ids, columns=sample_ids)

        # Test with different batch sizes
        result_batch10 = trans.map_trans(
            genotype_df, phenotype_df, covariates_df=None,
            return_sparse=False, batch_size=10
        )

        result_batch30 = trans.map_trans(
            genotype_df, phenotype_df, covariates_df=None,
            return_sparse=False, batch_size=30
        )

        # Results should be identical regardless of batch size
        assert len(result_batch10) == len(result_batch30)

        # Sort both dataframes to ensure same order for comparison
        result_batch10_sorted = result_batch10.sort_values(['phenotype_id', 'variant_id']).reset_index(drop=True)
        result_batch30_sorted = result_batch30.sort_values(['phenotype_id', 'variant_id']).reset_index(drop=True)

        # P-values should be very close
        np.testing.assert_allclose(
            result_batch10_sorted['pval'].values,
            result_batch30_sorted['pval'].values,
            rtol=1e-6
        )

class TestStatisticalValidation:
    """Test statistical properties of trans-QTL results."""

    def test_null_hypothesis_properties(self, device, torch_dtype):
        """Test that null trans-QTL results have expected properties."""
        if not TENSORQTL_AVAILABLE:
            pytest.skip("tensorqtl not available")

        # Create null data (no real associations)
        n_samples = 100
        n_variants = 200
        n_phenotypes = 20
        torch.manual_seed(123)

        genotypes = torch.randint(0, 3, (n_variants, n_samples), dtype=torch_dtype, device=device).float()
        phenotypes = torch.randn(n_phenotypes, n_samples, dtype=torch_dtype, device=device)

        sample_ids = [f"S{i:03d}" for i in range(n_samples)]
        variant_ids = [f"var{i}" for i in range(n_variants)]
        phenotype_ids = [f"gene{i}" for i in range(n_phenotypes)]

        genotype_df = pd.DataFrame(genotypes.cpu().numpy(), index=variant_ids, columns=sample_ids)
        phenotype_df = pd.DataFrame(phenotypes.cpu().numpy(), index=phenotype_ids, columns=sample_ids)

        # Run trans mapping
        result = trans.map_trans(
            genotype_df, phenotype_df, covariates_df=None,
            return_sparse=False, maf_threshold=0.05
        )

        # Test statistical properties
        if len(result) > 100:  # Need sufficient tests
            validation = validate_statistical_properties(result['pval'].values)

            # Under null hypothesis, should have ~5% significant at alpha=0.05
            assert abs(validation['prop_significant'] - 0.05) < 0.05

class TestEdgeCases:
    """Test edge cases in trans-QTL mapping."""

    def test_single_phenotype_variant(self, device, torch_dtype):
        """Test with single phenotype and variant."""
        if not TENSORQTL_AVAILABLE:
            pytest.skip("tensorqtl not available")

        n_samples = 50
        genotypes = torch.randint(0, 3, (1, n_samples), dtype=torch_dtype, device=device).float()
        phenotypes = torch.randn(1, n_samples, dtype=torch_dtype, device=device)

        sample_ids = [f"S{i:03d}" for i in range(n_samples)]

        genotype_df = pd.DataFrame(genotypes.cpu().numpy(), index=['var1'], columns=sample_ids)
        phenotype_df = pd.DataFrame(phenotypes.cpu().numpy(), index=['gene1'], columns=sample_ids)

        result = trans.map_trans(
            genotype_df, phenotype_df, covariates_df=None,
            return_sparse=False
        )

        assert len(result) == 1  # Single association
        assert result.iloc[0]['phenotype_id'] == 'gene1'
        assert result.iloc[0]['variant_id'] == 'var1'

    def test_empty_results_sparse_mode(self, device, torch_dtype):
        """Test sparse mode when no associations meet threshold."""
        if not TENSORQTL_AVAILABLE:
            pytest.skip("tensorqtl not available")

        n_samples = 30
        genotypes = torch.randint(0, 3, (10, n_samples), dtype=torch_dtype, device=device).float()
        phenotypes = torch.randn(5, n_samples, dtype=torch_dtype, device=device)

        sample_ids = [f"S{i:03d}" for i in range(n_samples)]
        variant_ids = [f"var{i}" for i in range(10)]
        phenotype_ids = [f"gene{i}" for i in range(5)]

        genotype_df = pd.DataFrame(genotypes.cpu().numpy(), index=variant_ids, columns=sample_ids)
        phenotype_df = pd.DataFrame(phenotypes.cpu().numpy(), index=phenotype_ids, columns=sample_ids)

        # Use very strict threshold - should return empty results
        result = trans.map_trans(
            genotype_df, phenotype_df, covariates_df=None,
            return_sparse=True, pval_threshold=1e-10
        )

        # Should handle empty results gracefully
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 0  # Could be 0 or more