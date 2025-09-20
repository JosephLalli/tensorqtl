"""
Tests for tensorqtl.cis module.

This module tests the cis-QTL mapping functionality including:
- Nominal associations
- Permutation testing
- Independent QTL discovery
- Interaction testing
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
    import tensorqtl.cis as cis
    import tensorqtl.core as core
    TENSORQTL_AVAILABLE = True
except ImportError:
    TENSORQTL_AVAILABLE = False

from tests.utils import simulate_qtl_data, assert_allclose_with_dtype, validate_statistical_properties

@pytest.mark.skipif(not TENSORQTL_AVAILABLE, reason="tensorqtl not available")
class TestCisNominal:
    """Test nominal cis-QTL mapping functions."""

    def test_calculate_cis_nominal_basic(self, simple_association_data, device, torch_dtype):
        """Test basic nominal cis-QTL calculation."""
        data = simple_association_data
        genotypes = data['genotypes'][:5]  # Use subset
        phenotypes = data['phenotypes'][:1]  # Single phenotype

        # Test without covariates
        result = cis.calculate_cis_nominal(genotypes, phenotypes[0], residualizer=None)

        tstat, slope, slope_se, af, ma_samples, ma_count = result

        # Check output shapes
        assert tstat.shape == (5,)
        assert slope.shape == (5,)
        assert slope_se.shape == (5,)
        assert af.shape == (5,)
        assert ma_samples.shape == (5,)
        assert ma_count.shape == (5,)

        # Check that first association is strongest (we added known effect)
        assert abs(tstat[0]) > abs(tstat[2])  # Should be stronger than null

        # Check that effect direction matches expected (positive effect added)
        assert slope[0] > 0

    def test_calculate_cis_nominal_with_residualizer(self, simple_association_data, device, torch_dtype):
        """Test nominal cis-QTL calculation with covariate adjustment."""
        data = simple_association_data
        genotypes = data['genotypes'][:3]
        phenotypes = data['phenotypes'][:1]
        covariates = data['covariates']

        # Create residualizer
        residualizer = core.Residualizer(covariates)

        # Test with covariates
        result_with_cov = cis.calculate_cis_nominal(genotypes, phenotypes[0], residualizer=residualizer)
        result_without_cov = cis.calculate_cis_nominal(genotypes, phenotypes[0], residualizer=None)

        tstat_with, slope_with = result_with_cov[:2]
        tstat_without, slope_without = result_without_cov[:2]

        # Results should be different
        assert not torch.allclose(tstat_with, tstat_without, atol=1e-3)

        # But shapes should be the same
        assert tstat_with.shape == tstat_without.shape

    def test_calculate_cis_nominal_return_af(self, simple_association_data, device, torch_dtype):
        """Test that allele frequency calculations are correct."""
        data = simple_association_data
        genotypes = data['genotypes'][:3]
        phenotypes = data['phenotypes'][:1]

        # Test with return_af=True (default)
        result_with_af = cis.calculate_cis_nominal(genotypes, phenotypes[0], return_af=True)
        result_without_af = cis.calculate_cis_nominal(genotypes, phenotypes[0], return_af=False)

        assert len(result_with_af) == 6  # tstat, slope, slope_se, af, ma_samples, ma_count
        assert len(result_without_af) == 3  # tstat, slope, slope_se

        # Check allele frequency values are reasonable
        af = result_with_af[3]
        assert torch.all(af >= 0)
        assert torch.all(af <= 1)

class TestCisPermutations:
    """Test permutation-based cis-QTL mapping."""

    def test_calculate_cis_permutations_basic(self, simple_association_data, device, torch_dtype):
        """Test basic permutation calculation."""
        data = simple_association_data
        genotypes = data['genotypes'][:5]
        phenotypes = data['phenotypes'][0]  # Single phenotype vector

        # Create permutation indices
        n_samples = phenotypes.shape[0]
        n_perms = 100
        torch.manual_seed(42)
        permutation_ix = torch.stack([torch.randperm(n_samples) for _ in range(n_perms)])

        # Run permutation test
        r_nominal, std_ratio, best_ix, r2_perm, best_genotype = cis.calculate_cis_permutations(
            genotypes, phenotypes, permutation_ix
        )

        # Check outputs
        assert isinstance(r_nominal, torch.Tensor)
        assert isinstance(std_ratio, torch.Tensor)
        assert isinstance(best_ix, torch.Tensor)
        assert r2_perm.shape == (n_perms,)
        assert best_genotype.shape == (n_samples,)

        # Nominal correlation should be stronger than most permutations
        r2_nominal = r_nominal.pow(2)
        prop_stronger = (r2_perm >= r2_nominal).float().mean()
        assert prop_stronger < 0.5  # Should be better than random

    def test_calculate_cis_permutations_with_residualizer(self, simple_association_data, device, torch_dtype):
        """Test permutation calculation with covariate adjustment."""
        data = simple_association_data
        genotypes = data['genotypes'][:3]
        phenotypes = data['phenotypes'][0]
        covariates = data['covariates']

        residualizer = core.Residualizer(covariates)

        n_samples = phenotypes.shape[0]
        n_perms = 50
        torch.manual_seed(42)
        permutation_ix = torch.stack([torch.randperm(n_samples) for _ in range(n_perms)])

        # Run with and without residualizer
        result_with = cis.calculate_cis_permutations(
            genotypes, phenotypes, permutation_ix, residualizer=residualizer
        )
        result_without = cis.calculate_cis_permutations(
            genotypes, phenotypes, permutation_ix, residualizer=None
        )

        # Results should be different
        assert not torch.allclose(result_with[0], result_without[0], atol=1e-3)

class TestCisMapping:
    """Test high-level cis-QTL mapping functions."""

    def test_map_cis_synthetic_data(self, device, torch_dtype):
        """Test complete cis-QTL mapping pipeline with synthetic data."""
        if not TENSORQTL_AVAILABLE:
            pytest.skip("tensorqtl not available")

        # Create synthetic QTL data
        qtl_data = simulate_qtl_data(
            n_samples=50, n_variants=100, n_phenotypes=10, n_qtls=3,
            seed=42
        )

        # Convert to tensors
        genotype_df = qtl_data['genotypes']
        phenotype_df = qtl_data['phenotypes']
        phenotype_pos_df = qtl_data['positions']
        covariates_df = qtl_data['covariates']

        # Convert to appropriate device/dtype
        genotype_tensor = torch.tensor(genotype_df.values, dtype=torch_dtype, device=device)
        phenotype_tensor = torch.tensor(phenotype_df.values, dtype=torch_dtype, device=device)

        # Test cis nominal mapping (just check it runs without error)
        # Note: This tests the core calculation functions, not the full map_nominal pipeline
        n_genes = phenotype_tensor.shape[0]
        n_variants = genotype_tensor.shape[0]

        # Test first gene
        if n_genes > 0:
            result = cis.calculate_cis_nominal(genotype_tensor, phenotype_tensor[0])
            tstat, slope, slope_se, af, ma_samples, ma_count = result

            assert tstat.shape == (n_variants,)
            assert not torch.isnan(tstat).any()
            assert not torch.isinf(tstat).any()

    @pytest.mark.slow
    def test_map_cis_with_known_qtl(self, device, torch_dtype):
        """Test cis-QTL mapping with known true QTL."""
        if not TENSORQTL_AVAILABLE:
            pytest.skip("tensorqtl not available")

        # Create data with strong known association
        n_samples = 100
        torch.manual_seed(123)

        # Create single variant and phenotype with strong association
        genotype = torch.randint(0, 3, (1, n_samples), dtype=torch_dtype, device=device)

        # Create phenotype with strong effect from genotype
        effect_size = 1.0
        noise = torch.randn(n_samples, dtype=torch_dtype, device=device) * 0.5
        phenotype = effect_size * genotype.squeeze() + noise

        # Test association
        result = cis.calculate_cis_nominal(genotype, phenotype)
        tstat, slope, slope_se = result[:3]

        # Should detect strong association
        assert abs(tstat[0]) > 3.0  # Should be highly significant
        assert abs(slope[0] - effect_size) < 0.3  # Should recover effect size

class TestCisInteractions:
    """Test cis-QTL interaction mapping."""

    def test_interaction_data_preparation(self, device, torch_dtype):
        """Test preparation of interaction data."""
        n_samples = 50

        # Create interaction vector
        interaction = torch.randn(n_samples, dtype=torch_dtype, device=device)
        interaction_t = interaction.reshape(1, -1)

        assert interaction_t.shape == (1, n_samples)

class TestStatisticalValidation:
    """Test statistical properties of cis-QTL results."""

    def test_null_distribution_properties(self, device, torch_dtype):
        """Test that null statistics have expected properties."""
        if not TENSORQTL_AVAILABLE:
            pytest.skip("tensorqtl not available")

        # Create null data (no associations)
        n_samples = 100
        n_variants = 200
        torch.manual_seed(42)

        genotypes = torch.randint(0, 3, (n_variants, n_samples), dtype=torch_dtype, device=device)
        phenotype = torch.randn(n_samples, dtype=torch_dtype, device=device)

        # Calculate associations
        result = cis.calculate_cis_nominal(genotypes, phenotype)
        tstat = result[0]

        # Convert to p-values
        dof = n_samples - 2  # No covariates
        pvals = core.get_t_pval(tstat.cpu().numpy(), dof)

        # Test statistical properties
        validation = validate_statistical_properties(pvals, alpha=0.05, tolerance=0.15)

        assert validation['n_tests'] == n_variants
        # Under null, should have ~5% significant results
        assert abs(validation['prop_significant'] - 0.05) < 0.15

    def test_effect_size_estimation(self, device, torch_dtype):
        """Test that effect sizes are estimated correctly."""
        if not TENSORQTL_AVAILABLE:
            pytest.skip("tensorqtl not available")

        n_samples = 200
        true_effect = 0.5
        torch.manual_seed(123)

        # Create genotype
        genotype = torch.randint(0, 3, (1, n_samples), dtype=torch_dtype, device=device).float()

        # Create phenotype with known effect
        noise = torch.randn(n_samples, dtype=torch_dtype, device=device) * 0.3
        phenotype = true_effect * genotype.squeeze() + noise

        # Estimate effect
        result = cis.calculate_cis_nominal(genotype, phenotype)
        estimated_effect = result[1][0]  # slope

        # Should be close to true effect
        assert abs(estimated_effect - true_effect) < 0.1

class TestEdgeCases:
    """Test edge cases in cis-QTL mapping."""

    def test_monomorphic_variants(self, device, torch_dtype):
        """Test handling of monomorphic variants."""
        if not TENSORQTL_AVAILABLE:
            pytest.skip("tensorqtl not available")

        n_samples = 50

        # Create monomorphic genotype (all zeros)
        genotype = torch.zeros(1, n_samples, dtype=torch_dtype, device=device)
        phenotype = torch.randn(n_samples, dtype=torch_dtype, device=device)

        # Should handle gracefully (may return NaN or zero)
        result = cis.calculate_cis_nominal(genotype, phenotype)
        tstat = result[0]

        # Should not crash, result may be NaN
        assert tstat.shape == (1,)

    def test_constant_phenotype(self, device, torch_dtype):
        """Test handling of constant phenotype."""
        if not TENSORQTL_AVAILABLE:
            pytest.skip("tensorqtl not available")

        n_samples = 50

        genotype = torch.randint(0, 3, (1, n_samples), dtype=torch_dtype, device=device).float()
        phenotype = torch.ones(n_samples, dtype=torch_dtype, device=device)  # Constant

        # Should handle gracefully
        result = cis.calculate_cis_nominal(genotype, phenotype)
        tstat = result[0]

        assert tstat.shape == (1,)
        # Result should be NaN or zero for constant phenotype
        assert torch.isnan(tstat[0]) or abs(tstat[0]) < 1e-6

    def test_single_sample(self, device, torch_dtype):
        """Test behavior with single sample (should fail gracefully)."""
        if not TENSORQTL_AVAILABLE:
            pytest.skip("tensorqtl not available")

        genotype = torch.tensor([[1]], dtype=torch_dtype, device=device)
        phenotype = torch.tensor([1.0], dtype=torch_dtype, device=device)

        # Should handle single sample case
        with pytest.raises((RuntimeError, ValueError, AssertionError)):
            cis.calculate_cis_nominal(genotype, phenotype)