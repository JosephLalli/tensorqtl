"""
Integration tests for tensorQTL.

This module tests complete workflows and end-to-end functionality,
using the prepared test data when available.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import sys
from pathlib import Path

# Add tensorqtl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tensorqtl
    from tensorqtl import cis, trans, post
    from tensorqtl.core import read_phenotype_bed, Residualizer
    TENSORQTL_AVAILABLE = True
except ImportError:
    TENSORQTL_AVAILABLE = False

from tests.utils import simulate_qtl_data, create_temp_files

@pytest.mark.integration
@pytest.mark.skipif(not TENSORQTL_AVAILABLE, reason="tensorqtl not available")
class TestCisWorkflow:
    """Test complete cis-QTL analysis workflows."""

    def test_cis_nominal_workflow(self, synthetic_genotypes, synthetic_phenotypes, synthetic_covariates):
        """Test complete cis nominal workflow."""
        genotype_df = synthetic_genotypes
        expr_df, pos_df = synthetic_phenotypes
        covariates_df = synthetic_covariates

        # Ensure sample alignment
        common_samples = list(set(genotype_df.columns) & set(expr_df.columns) & set(covariates_df.columns))
        assert len(common_samples) >= 10  # Need sufficient samples

        genotype_df = genotype_df[common_samples]
        expr_df = expr_df[common_samples]
        covariates_df = covariates_df[common_samples]

        # Test basic workflow components
        # Note: map_nominal is complex and requires proper file I/O setup
        # Here we test the core calculation functions that power it

        import torch
        device = torch.device("cpu")  # Use CPU for reproducibility

        # Convert to tensors
        genotype_tensor = torch.tensor(genotype_df.values, dtype=torch.float32, device=device)
        phenotype_tensor = torch.tensor(expr_df.values, dtype=torch.float32, device=device)
        covariate_tensor = torch.tensor(covariates_df.values.T, dtype=torch.float32, device=device)

        # Create residualizer
        residualizer = Residualizer(covariate_tensor)

        # Test association for first phenotype
        result = cis.calculate_cis_nominal(genotype_tensor, phenotype_tensor[0], residualizer=residualizer)
        tstat, slope, slope_se, af, ma_samples, ma_count = result

        # Verify results structure
        n_variants = genotype_tensor.shape[0]
        assert tstat.shape == (n_variants,)
        assert slope.shape == (n_variants,)
        assert not torch.isnan(tstat).any()

    @pytest.mark.slow
    def test_cis_permutation_workflow(self, synthetic_genotypes, synthetic_phenotypes, synthetic_covariates):
        """Test cis-QTL permutation workflow."""
        genotype_df = synthetic_genotypes.iloc[:20]  # Subset for speed
        expr_df, pos_df = synthetic_phenotypes
        expr_df = expr_df.iloc[:3]  # Subset for speed
        covariates_df = synthetic_covariates

        # Align samples
        common_samples = list(set(genotype_df.columns) & set(expr_df.columns) & set(covariates_df.columns))
        genotype_df = genotype_df[common_samples]
        expr_df = expr_df[common_samples]
        covariates_df = covariates_df[common_samples]

        import torch
        device = torch.device("cpu")

        # Convert to tensors
        genotype_tensor = torch.tensor(genotype_df.values, dtype=torch.float32, device=device)
        phenotype_tensor = torch.tensor(expr_df.values, dtype=torch.float32, device=device)
        covariate_tensor = torch.tensor(covariates_df.values.T, dtype=torch.float32, device=device)

        residualizer = Residualizer(covariate_tensor)

        # Create permutation indices
        n_samples = phenotype_tensor.shape[1]
        n_perms = 10  # Small number for testing
        torch.manual_seed(42)
        permutation_ix = torch.stack([torch.randperm(n_samples) for _ in range(n_perms)])

        # Test permutation calculation
        result = cis.calculate_cis_permutations(
            genotype_tensor, phenotype_tensor[0], permutation_ix, residualizer=residualizer
        )

        r_nominal, std_ratio, best_ix, r2_perm, best_genotype = result

        # Verify results
        assert isinstance(r_nominal, torch.Tensor)
        assert r2_perm.shape == (n_perms,)
        assert best_genotype.shape == (n_samples,)

@pytest.mark.integration
@pytest.mark.skipif(not TENSORQTL_AVAILABLE, reason="tensorqtl not available")
class TestTransWorkflow:
    """Test complete trans-QTL analysis workflows."""

    def test_trans_mapping_workflow(self, synthetic_genotypes, synthetic_phenotypes, synthetic_covariates):
        """Test complete trans-QTL mapping workflow."""
        genotype_df = synthetic_genotypes.iloc[:30]  # Subset for speed
        expr_df, pos_df = synthetic_phenotypes
        expr_df = expr_df.iloc[:5]  # Subset for speed
        covariates_df = synthetic_covariates

        # Align samples
        common_samples = list(set(genotype_df.columns) & set(expr_df.columns) & set(covariates_df.columns))
        genotype_df = genotype_df[common_samples]
        expr_df = expr_df[common_samples]
        covariates_df = covariates_df[common_samples]

        # Run trans mapping
        result = trans.map_trans(
            genotype_df, expr_df, covariates_df=covariates_df,
            return_sparse=True, pval_threshold=0.1,  # Liberal threshold
            maf_threshold=0.0, batch_size=10
        )

        # Verify result format
        assert isinstance(result, pd.DataFrame)
        expected_cols = ['phenotype_id', 'variant_id', 'pval', 'b', 'b_se', 'af']
        assert all(col in result.columns for col in expected_cols)

        # Basic validation
        if len(result) > 0:
            assert (result['pval'] >= 0).all()
            assert (result['pval'] <= 1).all()
            assert (result['af'] >= 0).all()
            assert (result['af'] <= 1).all()

    def test_trans_cis_filtering_workflow(self, synthetic_genotypes, synthetic_phenotypes):
        """Test trans-QTL with cis filtering."""
        genotype_df = synthetic_genotypes.iloc[:20]
        expr_df, pos_df = synthetic_phenotypes
        expr_df = expr_df.iloc[:3]

        # Create mock trans results
        trans_results = []
        for _, gene_id in enumerate(expr_df.index):
            for _, variant_id in enumerate(genotype_df.index[:5]):  # Subset variants
                trans_results.append({
                    'phenotype_id': gene_id,
                    'variant_id': variant_id,
                    'pval': np.random.uniform(0.01, 0.1),
                    'b': np.random.normal(0, 0.5),
                    'b_se': np.random.uniform(0.1, 0.3),
                    'af': np.random.uniform(0.1, 0.5)
                })

        trans_df = pd.DataFrame(trans_results)

        # Create position dictionaries
        phenotype_pos = {}
        for gene_id, row in pos_df.iterrows():
            phenotype_pos[gene_id] = {
                'chr': row['chr'],
                'start': row['start'],
                'end': row['end']
            }

        # Create variant position data
        variant_df = pd.DataFrame({
            'chrom': ['chr1'] * len(genotype_df.index),
            'pos': [i * 1000 + 10000 for i in range(len(genotype_df.index))]
        }, index=genotype_df.index)

        # Filter cis associations
        filtered_df = trans.filter_cis(trans_df, phenotype_pos, variant_df, window=100000)

        # Should have some filtering (though exact amount depends on positions)
        assert len(filtered_df) <= len(trans_df)
        assert isinstance(filtered_df, pd.DataFrame)

@pytest.mark.integration
@pytest.mark.skipif(not TENSORQTL_AVAILABLE, reason="tensorqtl not available")
class TestPostProcessingWorkflow:
    """Test post-processing workflows."""

    def test_qvalue_calculation_workflow(self):
        """Test complete q-value calculation workflow."""
        # Simulate QTL mapping results
        n_tests = 500
        np.random.seed(42)

        # Create realistic p-value distribution
        # Most are null, some are significant
        null_pvals = np.random.uniform(0.1, 1.0, int(0.9 * n_tests))
        sig_pvals = np.random.uniform(0.0, 0.05, int(0.1 * n_tests))
        all_pvals = np.concatenate([null_pvals, sig_pvals])
        np.random.shuffle(all_pvals)

        # Create results DataFrame
        results_df = pd.DataFrame({
            'phenotype_id': [f'gene_{i}' for i in range(n_tests)],
            'variant_id': [f'variant_{i}' for i in range(n_tests)],
            'pval_nominal': all_pvals,
            'pval_perm': all_pvals,  # Use same for simplicity
            'slope': np.random.normal(0, 0.5, n_tests),
        })

        # Calculate q-values
        n_significant_before = (results_df['pval_perm'] <= 0.05).sum()
        post.calculate_qvalues(results_df, fdr=0.05)

        # Verify q-values were added
        assert 'qval' in results_df.columns

        # Check properties
        assert (results_df['qval'] >= 0).all()
        assert (results_df['qval'] <= 1).all()
        assert (results_df['qval'] >= results_df['pval_perm']).all()

        # Should have some significant results at FDR 0.05
        n_significant_after = (results_df['qval'] <= 0.05).sum()
        assert n_significant_after > 0
        assert n_significant_after <= n_significant_before

@pytest.mark.integration
class TestRealDataIntegration:
    """Test integration with real GEUVADIS data if available."""

    def test_load_real_test_data(self, test_expression_data, test_covariates_data):
        """Test loading and basic validation of real test data."""
        expr_df, pos_df = test_expression_data
        cov_df = test_covariates_data

        if expr_df is not None and cov_df is not None:
            # Basic format validation
            assert isinstance(expr_df, pd.DataFrame)
            assert isinstance(pos_df, pd.DataFrame)
            assert isinstance(cov_df, pd.DataFrame)

            # Check dimensions
            assert expr_df.shape[0] > 0  # Should have genes
            assert expr_df.shape[1] > 0  # Should have samples
            assert pos_df.shape[0] == expr_df.shape[0]  # Same number of genes

            # Check sample overlap
            common_samples = set(expr_df.columns) & set(cov_df.columns)
            assert len(common_samples) > 0

            # Check data types
            assert expr_df.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()
            assert cov_df.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()

    @pytest.mark.slow
    def test_real_data_cis_calculation(self, test_expression_data, test_covariates_data):
        """Test cis calculation with real data if available."""
        if not TENSORQTL_AVAILABLE:
            pytest.skip("tensorqtl not available")

        expr_df, pos_df = test_expression_data
        cov_df = test_covariates_data

        if expr_df is not None and cov_df is not None and len(expr_df) > 0:
            # Create mock genotype data for samples
            common_samples = list(set(expr_df.columns) & set(cov_df.columns))
            if len(common_samples) >= 10:
                # Create synthetic genotypes for real samples
                n_variants = 50
                np.random.seed(42)
                genotypes = np.random.choice([0, 1, 2], size=(n_variants, len(common_samples)))

                variant_ids = [f"test_var_{i}" for i in range(n_variants)]
                genotype_df = pd.DataFrame(genotypes, index=variant_ids, columns=common_samples)

                # Subset data to common samples
                expr_subset = expr_df[common_samples]
                cov_subset = cov_df[common_samples]

                # Test association calculation
                import torch
                device = torch.device("cpu")

                genotype_tensor = torch.tensor(genotype_df.values, dtype=torch.float32, device=device)
                phenotype_tensor = torch.tensor(expr_subset.values, dtype=torch.float32, device=device)
                covariate_tensor = torch.tensor(cov_subset.values.T, dtype=torch.float32, device=device)

                residualizer = Residualizer(covariate_tensor)

                # Test first gene
                result = cis.calculate_cis_nominal(genotype_tensor, phenotype_tensor[0], residualizer=residualizer)

                # Should work without errors
                assert len(result) == 6  # Expected number of return values
                assert result[0].shape == (n_variants,)  # t-statistics

@pytest.mark.integration
class TestWorkflowConsistency:
    """Test consistency across different workflow components."""

    def test_data_flow_consistency(self):
        """Test that data flows consistently between workflow steps."""
        # Create comprehensive test dataset
        qtl_data = simulate_qtl_data(
            n_samples=30, n_variants=50, n_phenotypes=5, n_qtls=2, seed=123
        )

        genotype_df = qtl_data['genotypes']
        phenotype_df = qtl_data['phenotypes']
        pos_df = qtl_data['positions']
        covariates_df = qtl_data['covariates']

        # Verify sample consistency
        geno_samples = set(genotype_df.columns)
        pheno_samples = set(phenotype_df.columns)
        cov_samples = set(covariates_df.columns)

        assert geno_samples == pheno_samples == cov_samples

        # Verify index consistency
        assert len(set(genotype_df.index)) == len(genotype_df.index)  # No duplicate variants
        assert len(set(phenotype_df.index)) == len(phenotype_df.index)  # No duplicate genes
        assert phenotype_df.index.equals(pos_df.index)  # Matching gene positions

    def test_numerical_consistency(self):
        """Test numerical consistency across repeated runs."""
        if not TENSORQTL_AVAILABLE:
            pytest.skip("tensorqtl not available")

        # Fixed seed for reproducibility
        np.random.seed(42)
        import torch
        torch.manual_seed(42)

        n_samples = 30
        genotypes = torch.randint(0, 3, (10, n_samples), dtype=torch.float32)
        phenotype = torch.randn(n_samples, dtype=torch.float32)

        # Run calculation twice
        result1 = cis.calculate_cis_nominal(genotypes, phenotype)
        result2 = cis.calculate_cis_nominal(genotypes, phenotype)

        # Results should be identical
        for r1, r2 in zip(result1, result2):
            torch.testing.assert_close(r1, r2)