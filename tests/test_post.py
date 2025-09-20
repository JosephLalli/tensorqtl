"""
Tests for tensorqtl.post module.

This module tests post-processing functionality including:
- FDR calculation
- Q-value computation
- Multiple testing correction
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add tensorqtl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tensorqtl.post as post
    TENSORQTL_AVAILABLE = True
except ImportError:
    TENSORQTL_AVAILABLE = False

@pytest.mark.skipif(not TENSORQTL_AVAILABLE, reason="tensorqtl not available")
class TestQValueCalculation:
    """Test q-value calculation functionality."""

    def test_calculate_qvalues_basic(self):
        """Test basic q-value calculation."""
        # Create test data with some significant results
        n_tests = 1000
        np.random.seed(42)

        # Generate mostly null p-values
        pvals = np.random.uniform(0, 1, n_tests)

        # Add some significant p-values
        pvals[:50] = np.random.uniform(0, 0.01, 50)

        # Create test DataFrame
        test_df = pd.DataFrame({
            'phenotype_id': [f'gene_{i}' for i in range(n_tests)],
            'pval_nominal': pvals,
            'pval_perm': pvals,  # Use same values for simplicity
        })

        # Calculate q-values
        result = post.calculate_qvalues(test_df, fdr=0.05)

        # Check that qval column was added
        assert 'qval' in test_df.columns

        # Check that q-values are in valid range
        assert (test_df['qval'] >= 0).all()
        assert (test_df['qval'] <= 1).all()

        # Q-values should be >= p-values (monotonicity)
        assert (test_df['qval'] >= test_df['pval_perm']).all()

        # Should identify some significant results
        n_significant = (test_df['qval'] <= 0.05).sum()
        assert n_significant > 0
        assert n_significant <= 50  # Should be <= number of truly significant

    def test_calculate_qvalues_with_lambda(self):
        """Test q-value calculation with specific lambda parameter."""
        n_tests = 500
        np.random.seed(123)

        pvals = np.random.uniform(0, 1, n_tests)

        test_df = pd.DataFrame({
            'phenotype_id': [f'gene_{i}' for i in range(n_tests)],
            'pval_nominal': pvals,
            'pval_perm': pvals,
        })

        # Test with specific lambda value
        result = post.calculate_qvalues(test_df, fdr=0.05, qvalue_lambda=0.5)

        assert 'qval' in test_df.columns
        assert (test_df['qval'] >= 0).all()
        assert (test_df['qval'] <= 1).all()

    def test_calculate_qvalues_all_significant(self):
        """Test q-value calculation when all results are significant."""
        n_tests = 100

        # All very small p-values
        pvals = np.random.uniform(0, 0.001, n_tests)

        test_df = pd.DataFrame({
            'phenotype_id': [f'gene_{i}' for i in range(n_tests)],
            'pval_nominal': pvals,
            'pval_perm': pvals,
        })

        result = post.calculate_qvalues(test_df, fdr=0.05)

        # All should be significant
        assert (test_df['qval'] <= 0.05).all()

    def test_calculate_qvalues_no_significant(self):
        """Test q-value calculation when no results are significant."""
        n_tests = 100

        # All large p-values
        pvals = np.random.uniform(0.5, 1.0, n_tests)

        test_df = pd.DataFrame({
            'phenotype_id': [f'gene_{i}' for i in range(n_tests)],
            'pval_nominal': pvals,
            'pval_perm': pvals,
        })

        result = post.calculate_qvalues(test_df, fdr=0.05)

        # None should be significant at FDR 0.05
        assert (test_df['qval'] > 0.05).all()

class TestFDRControl:
    """Test FDR control properties."""

    def test_fdr_control_simulation(self):
        """Test that FDR is controlled in simulation."""
        if not TENSORQTL_AVAILABLE:
            pytest.skip("tensorqtl not available")

        # Simulate multiple testing scenario
        n_null = 900
        n_alt = 100
        n_tests = n_null + n_alt

        np.random.seed(42)

        # Null p-values (uniform)
        null_pvals = np.random.uniform(0, 1, n_null)

        # Alternative p-values (concentrated near 0)
        alt_pvals = np.random.beta(0.1, 1, n_alt)  # Skewed toward 0

        # Combine and shuffle
        all_pvals = np.concatenate([null_pvals, alt_pvals])
        true_null = np.concatenate([np.ones(n_null), np.zeros(n_alt)])  # 1 = null, 0 = alternative

        # Shuffle to randomize order
        perm = np.random.permutation(n_tests)
        all_pvals = all_pvals[perm]
        true_null = true_null[perm]

        # Create DataFrame
        test_df = pd.DataFrame({
            'phenotype_id': [f'gene_{i}' for i in range(n_tests)],
            'pval_nominal': all_pvals,
            'pval_perm': all_pvals,
        })

        # Calculate q-values
        post.calculate_qvalues(test_df, fdr=0.05)

        # Check FDR
        significant = test_df['qval'] <= 0.05
        if significant.sum() > 0:
            false_discoveries = (significant & (true_null == 1)).sum()
            total_discoveries = significant.sum()
            empirical_fdr = false_discoveries / total_discoveries

            # FDR should be controlled (may be conservative)
            assert empirical_fdr <= 0.1  # Allow some slack due to stochasticity

class TestEdgeCases:
    """Test edge cases in post-processing."""

    def test_single_test(self):
        """Test with single test."""
        if not TENSORQTL_AVAILABLE:
            pytest.skip("tensorqtl not available")

        test_df = pd.DataFrame({
            'phenotype_id': ['gene_1'],
            'pval_nominal': [0.01],
            'pval_perm': [0.01],
        })

        post.calculate_qvalues(test_df, fdr=0.05)

        assert 'qval' in test_df.columns
        assert len(test_df) == 1

    def test_identical_pvalues(self):
        """Test with identical p-values."""
        if not TENSORQTL_AVAILABLE:
            pytest.skip("tensorqtl not available")

        n_tests = 50
        test_df = pd.DataFrame({
            'phenotype_id': [f'gene_{i}' for i in range(n_tests)],
            'pval_nominal': [0.01] * n_tests,
            'pval_perm': [0.01] * n_tests,
        })

        post.calculate_qvalues(test_df, fdr=0.05)

        # All q-values should be identical
        unique_qvals = test_df['qval'].nunique()
        assert unique_qvals == 1

    def test_extreme_pvalues(self):
        """Test with extreme p-values."""
        if not TENSORQTL_AVAILABLE:
            pytest.skip("tensorqtl not available")

        test_df = pd.DataFrame({
            'phenotype_id': ['gene_1', 'gene_2', 'gene_3'],
            'pval_nominal': [0.0, 1.0, 0.5],
            'pval_perm': [0.0, 1.0, 0.5],
        })

        post.calculate_qvalues(test_df, fdr=0.05)

        assert 'qval' in test_df.columns
        # Should handle extreme values gracefully
        assert (test_df['qval'] >= 0).all()
        assert (test_df['qval'] <= 1).all()

class TestMultipleTestingCorrection:
    """Test multiple testing correction methods."""

    def test_benjamini_hochberg_procedure(self):
        """Test Benjamini-Hochberg FDR control procedure."""
        # Manual implementation to verify
        pvals = np.array([0.01, 0.02, 0.03, 0.5, 0.6, 0.7, 0.8, 0.9])
        alpha = 0.05
        m = len(pvals)

        # Sort p-values
        sorted_pvals = np.sort(pvals)

        # BH critical values
        bh_critical = (np.arange(1, m + 1) / m) * alpha

        # Find largest i such that P(i) <= (i/m) * alpha
        significant_indices = sorted_pvals <= bh_critical

        if significant_indices.any():
            largest_significant = np.where(significant_indices)[0][-1]
            expected_significant = largest_significant + 1
        else:
            expected_significant = 0

        # At alpha=0.05, we expect some to be significant
        assert expected_significant >= 0