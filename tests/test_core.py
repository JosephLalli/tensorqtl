"""
Tests for tensorqtl.core module.

This module tests the core functionality including:
- Residualizer class
- Statistical functions (MAF calculation, correlation, etc.)
- Data type handling and conversions
- Device management
"""

import pytest
import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path

# Add tensorqtl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorqtl.core as core
from tests.utils import assert_allclose_with_dtype, generate_synthetic_genotypes

class TestResidualizer:
    """Test the Residualizer class for covariate adjustment."""

    def test_residualizer_init(self, synthetic_covariates, device, torch_dtype):
        """Test Residualizer initialization."""
        cov_tensor = torch.tensor(synthetic_covariates.values.T, dtype=torch_dtype, device=device)
        residualizer = core.Residualizer(cov_tensor)

        assert residualizer.Q_t.shape[0] == cov_tensor.shape[0]  # n_samples
        assert residualizer.Q_t.shape[1] <= cov_tensor.shape[1]  # n_covariates (after centering)
        assert residualizer.dof == cov_tensor.shape[0] - 2 - cov_tensor.shape[1]

    def test_residualizer_transform_center_true(self, residualizer, device, torch_dtype):
        """Test Residualizer transform with centering."""
        n_features, n_samples = 10, residualizer.Q_t.shape[0]
        M_t = torch.randn(n_features, n_samples, dtype=torch_dtype, device=device)

        M_res = residualizer.transform(M_t, center=True)

        assert M_res.shape == M_t.shape
        assert M_res.dtype == M_t.dtype
        assert M_res.device == M_t.device

        # Check that the result is orthogonal to covariates
        # After residualization, correlation with covariates should be near zero
        Q_expanded = residualizer.Q_t.T.unsqueeze(0).expand(n_features, -1, -1)
        correlations = torch.bmm(M_res.unsqueeze(1), Q_expanded).squeeze(1)
        assert torch.allclose(correlations, torch.zeros_like(correlations), atol=1e-5)

    def test_residualizer_transform_center_false(self, residualizer, device, torch_dtype):
        """Test Residualizer transform without centering."""
        n_features, n_samples = 10, residualizer.Q_t.shape[0]
        M_t = torch.randn(n_features, n_samples, dtype=torch_dtype, device=device)

        M_res = residualizer.transform(M_t, center=False)

        assert M_res.shape == M_t.shape
        assert M_res.dtype == M_t.dtype
        assert M_res.device == M_t.device

    def test_residualizer_reproducible(self, synthetic_covariates, device, torch_dtype):
        """Test that Residualizer gives reproducible results."""
        cov_tensor = torch.tensor(synthetic_covariates.values.T, dtype=torch_dtype, device=device)

        residualizer1 = core.Residualizer(cov_tensor)
        residualizer2 = core.Residualizer(cov_tensor)

        M_t = torch.randn(5, cov_tensor.shape[0], dtype=torch_dtype, device=device)

        res1 = residualizer1.transform(M_t)
        res2 = residualizer2.transform(M_t)

        torch.testing.assert_close(res1, res2)

class TestStatisticalFunctions:
    """Test statistical functions in core module."""

    def test_calculate_maf_basic(self, device, torch_dtype):
        """Test basic MAF calculation."""
        # Test case: all heterozygotes (AF = 0.5, MAF = 0.5)
        genotypes = torch.tensor([[1, 1, 1, 1]], dtype=torch_dtype, device=device)
        maf = core.calculate_maf(genotypes)

        expected_maf = torch.tensor([0.5], dtype=torch_dtype, device=device)
        torch.testing.assert_close(maf, expected_maf)

    def test_calculate_maf_homozygous(self, device, torch_dtype):
        """Test MAF calculation with homozygous genotypes."""
        # All homozygous reference (MAF = 0)
        genotypes = torch.tensor([[0, 0, 0, 0]], dtype=torch_dtype, device=device)
        maf = core.calculate_maf(genotypes)
        assert torch.allclose(maf, torch.zeros_like(maf))

        # All homozygous alternate (MAF = 0)
        genotypes = torch.tensor([[2, 2, 2, 2]], dtype=torch_dtype, device=device)
        maf = core.calculate_maf(genotypes)
        assert torch.allclose(maf, torch.zeros_like(maf))

    def test_calculate_maf_mixed(self, device, torch_dtype):
        """Test MAF calculation with mixed genotypes."""
        # 1 het, 3 hom ref -> Sum=1, AF = 1/8 = 0.125, MAF = 0.125
        genotypes = torch.tensor([[0, 0, 0, 1]], dtype=torch_dtype, device=device)
        maf = core.calculate_maf(genotypes)

        expected_maf = torch.tensor([0.125], dtype=torch_dtype, device=device)
        torch.testing.assert_close(maf, expected_maf, atol=1e-6)

    def test_get_allele_stats(self, device, torch_dtype):
        """Test allele statistics calculation."""
        # Test with known genotypes
        genotypes = torch.tensor([
            [0, 1, 2, 1],  # AF = 0.5, MAF = 0.5
            [0, 0, 0, 1],  # AF = 0.125, MAF = 0.125
        ], dtype=torch_dtype, device=device)

        af_t, ma_samples_t, ma_count_t = core.get_allele_stats(genotypes)

        # Check allele frequencies
        expected_af = torch.tensor([0.5, 0.125], dtype=torch_dtype, device=device)
        torch.testing.assert_close(af_t, expected_af, atol=1e-6)

        # Check minor allele sample counts
        expected_ma_samples = torch.tensor([3, 1], dtype=torch.int32, device=device)
        torch.testing.assert_close(ma_samples_t, expected_ma_samples)

        # Check minor allele counts
        expected_ma_count = torch.tensor([4, 1], dtype=torch.int32, device=device)
        torch.testing.assert_close(ma_count_t, expected_ma_count)

    def test_filter_maf(self, device, torch_dtype):
        """Test MAF filtering."""
        genotypes = torch.tensor([
            [0, 1, 2, 1],  # MAF = 0.5 (keep)
            [0, 0, 0, 0],  # MAF = 0 (filter)
            [0, 0, 0, 1],  # MAF = 0.125 (filter if threshold > 0.125)
        ], dtype=torch_dtype, device=device)

        variant_ids = np.array(['var1', 'var2', 'var3'])

        # Test with threshold 0.1
        filtered_geno, filtered_ids, filtered_af = core.filter_maf(
            genotypes, variant_ids, maf_threshold=0.1
        )

        assert filtered_geno.shape[0] == 2  # Should keep var1 and var3
        assert len(filtered_ids) == 2
        np.testing.assert_array_equal(filtered_ids, ['var1', 'var3'])

    def test_impute_mean(self, device, torch_dtype):
        """Test mean imputation of missing genotypes."""
        genotypes = torch.tensor([
            [0, 1, -9, 1],  # Missing value at position 2
            [0, 0, 0, 1],   # No missing values
        ], dtype=torch_dtype, device=device)

        original = genotypes.clone()
        core.impute_mean(genotypes, missing=-9)

        # Check that non-missing values are unchanged
        assert torch.equal(genotypes[1], original[1])

        # Check that missing value was imputed to mean
        # Mean of [0, 1, 1] = 2/3
        expected_imputed = 2.0/3.0
        assert torch.allclose(genotypes[0, 2], torch.tensor(expected_imputed, dtype=torch_dtype, device=device))

    def test_center_normalize(self, device, torch_dtype):
        """Test center and normalize function."""
        M_t = torch.tensor([
            [1, 2, 3, 4],
            [10, 20, 30, 40]
        ], dtype=torch_dtype, device=device)

        # Test along dimension 0 (across features)
        M_norm = core.center_normalize(M_t, dim=0)

        # Check that each column (sample) has mean 0 and norm 1
        assert torch.allclose(M_norm.mean(dim=0), torch.zeros(4, dtype=torch_dtype, device=device), atol=1e-6)
        assert torch.allclose(torch.norm(M_norm, dim=0), torch.ones(4, dtype=torch_dtype, device=device), atol=1e-6)

        # Test along dimension 1 (across samples)
        M_norm = core.center_normalize(M_t, dim=1)

        # Check that each row (feature) has mean 0 and norm 1
        assert torch.allclose(M_norm.mean(dim=1), torch.zeros(2, dtype=torch_dtype, device=device), atol=1e-6)
        assert torch.allclose(torch.norm(M_norm, dim=1), torch.ones(2, dtype=torch_dtype, device=device), atol=1e-6)

    def test_calculate_corr_basic(self, device, torch_dtype):
        """Test basic correlation calculation."""
        # Create perfectly correlated data
        genotype_t = torch.tensor([[1, 2, 3, 4]], dtype=torch_dtype, device=device)
        phenotype_t = torch.tensor([[2, 4, 6, 8]], dtype=torch_dtype, device=device)  # 2 * genotype

        corr = core.calculate_corr(genotype_t, phenotype_t)

        # Should be perfect correlation
        expected_corr = torch.ones(1, 1, dtype=torch_dtype, device=device)
        torch.testing.assert_close(corr, expected_corr, atol=1e-5)

    def test_calculate_corr_with_residualizer(self, device, torch_dtype):
        """Test correlation calculation with residualization."""
        n_samples = 100
        torch.manual_seed(42)

        # Create genotypes and phenotypes
        genotypes = torch.randn(5, n_samples, dtype=torch_dtype, device=device)
        phenotypes = torch.randn(3, n_samples, dtype=torch_dtype, device=device)

        # Create covariates
        covariates = torch.randn(n_samples, 2, dtype=torch_dtype, device=device)
        residualizer = core.Residualizer(covariates)

        # Calculate correlation with residualizer
        corr_res = core.calculate_corr(genotypes, phenotypes, residualizer=residualizer)

        # Calculate correlation without residualizer
        corr_raw = core.calculate_corr(genotypes, phenotypes, residualizer=None)

        assert corr_res.shape == (5, 3)
        assert corr_raw.shape == (5, 3)

        # Results should be different
        assert not torch.allclose(corr_res, corr_raw, atol=1e-3)

    def test_calculate_corr_return_var(self, device, torch_dtype):
        """Test correlation calculation with variance return."""
        genotypes = torch.randn(3, 50, dtype=torch_dtype, device=device)
        phenotypes = torch.randn(2, 50, dtype=torch_dtype, device=device)

        corr, geno_var, pheno_var = core.calculate_corr(
            genotypes, phenotypes, return_var=True
        )

        assert corr.shape == (3, 2)
        assert geno_var.shape == (3,)
        assert pheno_var.shape == (2,)

        # Variances should be positive
        assert torch.all(geno_var >= 0)
        assert torch.all(pheno_var >= 0)

class TestDataTypes:
    """Test data type handling."""

    def test_output_dtype_dict(self):
        """Test that output dtype dictionary is properly defined."""
        assert 'pval_nominal' in core.output_dtype_dict
        assert 'slope' in core.output_dtype_dict
        assert 'af' in core.output_dtype_dict

        # Check that dtypes are valid numpy types
        for col, dtype in core.output_dtype_dict.items():
            assert hasattr(np, dtype.__name__) or dtype in [str]

class TestLinearRegression:
    """Test linear regression functions."""

    def test_linreg_basic(self, device, torch_dtype):
        """Test basic linear regression."""
        # Create simple linear relationship: y = 2*x + 1 + noise
        n_samples = 100
        torch.manual_seed(42)

        X = torch.randn(n_samples, 2, dtype=torch_dtype, device=device)
        X[:, 0] = 1  # Intercept column

        true_beta = torch.tensor([1.0, 2.0], dtype=torch_dtype, device=device)
        y = torch.mv(X, true_beta) + 0.1 * torch.randn(n_samples, dtype=torch_dtype, device=device)

        # Run regression
        beta_hat, se, tstat, pval, dof = core.linreg(X.T, y.unsqueeze(0), dtype=torch_dtype)

        # Check shapes
        assert beta_hat.shape == (1, 2)
        assert se.shape == (1, 2)
        assert tstat.shape == (1, 2)
        assert pval.shape == (1, 2)

        # Check that estimated coefficients are close to true values
        torch.testing.assert_close(beta_hat[0], true_beta, atol=0.2)

        # Check that degrees of freedom is correct
        assert dof == n_samples - 2

    def test_linreg_perfect_fit(self, device, torch_dtype):
        """Test linear regression with perfect fit (no noise)."""
        n_samples = 50
        X = torch.randn(n_samples, 2, dtype=torch_dtype, device=device)
        X[:, 0] = 1  # Intercept

        true_beta = torch.tensor([0.5, 1.5], dtype=torch_dtype, device=device)
        y = torch.mv(X, true_beta)  # No noise

        beta_hat, se, tstat, pval, dof = core.linreg(X.T, y.unsqueeze(0), dtype=torch_dtype)

        # Should recover true coefficients exactly (within numerical precision)
        torch.testing.assert_close(beta_hat[0], true_beta, atol=1e-5)

        # P-values should be very small for perfect fit
        assert torch.all(pval < 1e-10)

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_monomorphic_variants(self, device, torch_dtype):
        """Test handling of monomorphic variants."""
        # All zeros
        genotypes = torch.zeros(1, 10, dtype=torch_dtype, device=device)
        maf = core.calculate_maf(genotypes)
        assert torch.allclose(maf, torch.zeros_like(maf))

        af_t, ma_samples_t, ma_count_t = core.get_allele_stats(genotypes)
        assert torch.allclose(af_t, torch.zeros_like(af_t))

    def test_single_sample(self, device, torch_dtype):
        """Test functions with single sample."""
        genotypes = torch.tensor([[1]], dtype=torch_dtype, device=device)

        maf = core.calculate_maf(genotypes)
        assert maf.shape == (1,)

        af_t, ma_samples_t, ma_count_t = core.get_allele_stats(genotypes)
        assert af_t.shape == (1,)

    def test_empty_tensors(self, device, torch_dtype):
        """Test handling of empty tensors."""
        empty_geno = torch.empty(0, 10, dtype=torch_dtype, device=device)
        maf = core.calculate_maf(empty_geno)
        assert maf.shape == (0,)

    @pytest.mark.parametrize("missing_val", [-9, np.nan])
    def test_missing_data_handling(self, device, torch_dtype, missing_val):
        """Test handling of different missing data encodings."""
        if missing_val is np.nan and torch_dtype == torch.int32:
            pytest.skip("NaN not supported for integer tensors")

        genotypes = torch.tensor([
            [0, 1, missing_val, 1],
            [0, 0, 0, missing_val]
        ], dtype=torch_dtype, device=device)

        # Test imputation
        if missing_val == -9:
            original = genotypes.clone()
            core.impute_mean(genotypes, missing=missing_val)
            # Check that non-missing values are unchanged
            mask = original != missing_val
            assert torch.equal(genotypes[mask], original[mask])

    def test_dtype_consistency(self, device):
        """Test that functions maintain dtype consistency."""
        for dtype in [torch.float32, torch.float64]:
            genotypes = torch.tensor([[0, 1, 2, 1]], dtype=dtype, device=device)

            maf = core.calculate_maf(genotypes)
            assert maf.dtype == dtype

            af_t, _, _ = core.get_allele_stats(genotypes)
            assert af_t.dtype == dtype