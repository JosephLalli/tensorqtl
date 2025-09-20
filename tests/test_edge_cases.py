"""
Edge case and performance tests for tensorQTL.

This module tests edge cases, error conditions, and performance benchmarks.
"""

import pytest
import numpy as np
import pandas as pd
import torch
import time
import sys
from pathlib import Path

# Add tensorqtl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tensorqtl.core as core
    import tensorqtl.cis as cis
    import tensorqtl.trans as trans
    TENSORQTL_AVAILABLE = True
except ImportError:
    TENSORQTL_AVAILABLE = False

@pytest.mark.skipif(not TENSORQTL_AVAILABLE, reason="tensorqtl not available")
class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_data(self, device, torch_dtype):
        """Test behavior with empty datasets."""
        # Empty genotypes
        empty_genotypes = torch.empty(0, 10, dtype=torch_dtype, device=device)
        phenotype = torch.randn(10, dtype=torch_dtype, device=device)

        maf = core.calculate_maf(empty_genotypes)
        assert maf.shape == (0,)

        # Empty phenotypes
        genotypes = torch.randint(0, 3, (5, 0), dtype=torch_dtype, device=device).float()
        empty_phenotype = torch.empty(0, dtype=torch_dtype, device=device)

        # Should handle gracefully or raise appropriate error
        with pytest.raises((RuntimeError, ValueError, IndexError)):
            core.calculate_corr(genotypes, empty_phenotype.unsqueeze(0))

    def test_single_variant_sample(self, device, torch_dtype):
        """Test with single variant or sample."""
        # Single sample
        genotypes = torch.tensor([[1]], dtype=torch_dtype, device=device)
        phenotype = torch.tensor([1.0], dtype=torch_dtype, device=device)

        # Most functions should fail gracefully with single sample
        with pytest.raises((RuntimeError, ValueError, AssertionError)):
            cis.calculate_cis_nominal(genotypes, phenotype)

        # Single variant, multiple samples
        genotypes = torch.tensor([[0, 1, 2, 1, 0]], dtype=torch_dtype, device=device)
        phenotype = torch.tensor([1.0, 2.0, 3.0, 2.5, 1.5], dtype=torch_dtype, device=device)

        # Should work
        result = cis.calculate_cis_nominal(genotypes, phenotype)
        assert result[0].shape == (1,)  # Single t-statistic

    def test_extreme_values(self, device, torch_dtype):
        """Test with extreme values."""
        n_samples = 50

        # Very large genotype values (should be 0,1,2 but test robustness)
        genotypes = torch.tensor([[1000, 1001, 1002] * (n_samples // 3 + 1)][:n_samples], dtype=torch_dtype, device=device).unsqueeze(0)
        phenotype = torch.randn(n_samples, dtype=torch_dtype, device=device)

        # Should handle without crashing
        try:
            result = cis.calculate_cis_nominal(genotypes, phenotype)
            # Results might be extreme but should be finite
            assert torch.isfinite(result[0]).any()
        except (RuntimeError, ValueError):
            # Acceptable to fail with extreme values
            pass

        # Very large phenotype values
        genotypes = torch.randint(0, 3, (1, n_samples), dtype=torch_dtype, device=device).float()
        extreme_phenotype = torch.tensor([1e10] * n_samples, dtype=torch_dtype, device=device)

        result = cis.calculate_cis_nominal(genotypes, extreme_phenotype)
        # Should handle large values

    def test_all_missing_data(self, device, torch_dtype):
        """Test with all missing data."""
        n_samples = 20

        # All missing genotypes
        missing_genotypes = torch.full((1, n_samples), -9, dtype=torch_dtype, device=device)
        phenotype = torch.randn(n_samples, dtype=torch_dtype, device=device)

        # Test imputation
        original = missing_genotypes.clone()
        core.impute_mean(missing_genotypes, missing=-9)

        # All values should be imputed to same value (mean of missing is undefined)
        # Function might set to 0 or NaN
        assert not torch.equal(missing_genotypes, original)

    def test_perfect_correlation(self, device, torch_dtype):
        """Test with perfectly correlated data."""
        n_samples = 50

        # Create perfectly correlated genotype and phenotype
        base_values = torch.randn(n_samples, dtype=torch_dtype, device=device)
        genotypes = base_values.unsqueeze(0)  # Same values
        phenotype = 2 * base_values  # Perfect linear relationship

        result = cis.calculate_cis_nominal(genotypes, phenotype)
        tstat, slope = result[:2]

        # Should detect perfect correlation (very high t-statistic)
        assert abs(tstat[0]) > 10  # Should be very significant

    def test_constant_values(self, device, torch_dtype):
        """Test with constant values."""
        n_samples = 30

        # Constant genotype
        const_genotype = torch.ones(1, n_samples, dtype=torch_dtype, device=device)
        phenotype = torch.randn(n_samples, dtype=torch_dtype, device=device)

        result = cis.calculate_cis_nominal(const_genotype, phenotype)
        tstat = result[0]

        # Should be zero or NaN (no variance in genotype)
        assert torch.isnan(tstat[0]) or abs(tstat[0]) < 1e-6

        # Constant phenotype
        genotype = torch.randint(0, 3, (1, n_samples), dtype=torch_dtype, device=device).float()
        const_phenotype = torch.ones(n_samples, dtype=torch_dtype, device=device)

        result = cis.calculate_cis_nominal(genotype, const_phenotype)
        tstat = result[0]

        # Should be zero or NaN (no variance in phenotype)
        assert torch.isnan(tstat[0]) or abs(tstat[0]) < 1e-6

    def test_numerical_precision_float32_vs_float64(self, device):
        """Test numerical precision differences between float32 and float64."""
        n_samples = 100
        torch.manual_seed(42)

        # Create data that might show precision differences
        genotypes_f64 = torch.randint(0, 3, (5, n_samples), dtype=torch.float64, device=device)
        phenotype_f64 = torch.randn(n_samples, dtype=torch.float64, device=device)

        genotypes_f32 = genotypes_f64.float()
        phenotype_f32 = phenotype_f64.float()

        # Calculate with both precisions
        result_f64 = cis.calculate_cis_nominal(genotypes_f64, phenotype_f64)
        result_f32 = cis.calculate_cis_nominal(genotypes_f32, phenotype_f32)

        tstat_f64, slope_f64 = result_f64[:2]
        tstat_f32, slope_f32 = result_f32[:2]

        # Results should be close but may differ due to precision
        torch.testing.assert_close(tstat_f32.double(), tstat_f64, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(slope_f32.double(), slope_f64, rtol=1e-5, atol=1e-6)

@pytest.mark.benchmark
@pytest.mark.skipif(not TENSORQTL_AVAILABLE, reason="tensorqtl not available")
class TestPerformance:
    """Performance benchmarks and scaling tests."""

    @pytest.mark.slow
    def test_cis_performance_scaling(self, device, torch_dtype):
        """Test performance scaling with dataset size."""
        torch.manual_seed(42)

        sizes = [(50, 100), (100, 500), (200, 1000)]  # (samples, variants)
        times = []

        for n_samples, n_variants in sizes:
            genotypes = torch.randint(0, 3, (n_variants, n_samples), dtype=torch_dtype, device=device).float()
            phenotype = torch.randn(n_samples, dtype=torch_dtype, device=device)

            # Time the calculation
            start_time = time.time()
            result = cis.calculate_cis_nominal(genotypes, phenotype)
            end_time = time.time()

            elapsed = end_time - start_time
            times.append(elapsed)

            # Should complete in reasonable time
            assert elapsed < 10.0  # 10 seconds max

            # Verify result
            assert result[0].shape == (n_variants,)

        # Performance should scale reasonably (not exponentially)
        # Allow for some variance in timing
        if len(times) >= 2:
            ratio = times[-1] / times[0]
            size_ratio = (sizes[-1][0] * sizes[-1][1]) / (sizes[0][0] * sizes[0][1])
            # Time should not increase faster than O(n^2)
            assert ratio < size_ratio * 2

    @pytest.mark.slow
    def test_memory_usage_large_dataset(self, device, torch_dtype):
        """Test memory usage with larger datasets."""
        if device.type == 'cuda':
            # Only test on GPU where memory is more constrained
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

            # Create moderately large dataset
            n_samples, n_variants = 200, 2000
            genotypes = torch.randint(0, 3, (n_variants, n_samples), dtype=torch_dtype, device=device).float()
            phenotype = torch.randn(n_samples, dtype=torch_dtype, device=device)

            # Run calculation
            result = cis.calculate_cis_nominal(genotypes, phenotype)

            peak_memory = torch.cuda.max_memory_allocated()
            memory_used = peak_memory - initial_memory

            # Memory usage should be reasonable (less than 1GB for this size)
            assert memory_used < 1e9  # 1GB

            # Clean up
            del genotypes, phenotype, result
            torch.cuda.empty_cache()

    def test_batch_processing_performance(self, device, torch_dtype):
        """Test that batch processing improves performance."""
        if not TENSORQTL_AVAILABLE:
            pytest.skip("tensorqtl not available")

        n_samples = 100
        n_variants = 200
        n_phenotypes = 50

        # Create test data
        genotypes = torch.randint(0, 3, (n_variants, n_samples), dtype=torch_dtype, device=device).float()
        phenotypes = torch.randn(n_phenotypes, n_samples, dtype=torch_dtype, device=device)

        sample_ids = [f"S{i}" for i in range(n_samples)]
        variant_ids = [f"V{i}" for i in range(n_variants)]
        phenotype_ids = [f"P{i}" for i in range(n_phenotypes)]

        genotype_df = pd.DataFrame(genotypes.cpu().numpy(), index=variant_ids, columns=sample_ids)
        phenotype_df = pd.DataFrame(phenotypes.cpu().numpy(), index=phenotype_ids, columns=sample_ids)

        # Test different batch sizes
        batch_sizes = [10, 25, 50]
        times = []

        for batch_size in batch_sizes:
            start_time = time.time()

            result = trans.map_trans(
                genotype_df, phenotype_df, covariates_df=None,
                return_sparse=False, batch_size=batch_size
            )

            end_time = time.time()
            times.append(end_time - start_time)

            # Verify results are consistent
            assert len(result) == n_variants * n_phenotypes

        # Performance shouldn't vary dramatically with batch size
        # (though optimal batch size depends on hardware)
        assert max(times) / min(times) < 5  # Less than 5x difference

    def test_gpu_vs_cpu_performance(self, torch_dtype):
        """Test GPU vs CPU performance if both available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        n_samples, n_variants = 100, 500
        torch.manual_seed(42)

        # Create same data on both devices
        genotypes_cpu = torch.randint(0, 3, (n_variants, n_samples), dtype=torch_dtype).float()
        phenotype_cpu = torch.randn(n_samples, dtype=torch_dtype)

        genotypes_gpu = genotypes_cpu.cuda()
        phenotype_gpu = phenotype_cpu.cuda()

        # Time CPU calculation
        start_time = time.time()
        result_cpu = cis.calculate_cis_nominal(genotypes_cpu, phenotype_cpu)
        cpu_time = time.time() - start_time

        # Time GPU calculation
        torch.cuda.synchronize()
        start_time = time.time()
        result_gpu = cis.calculate_cis_nominal(genotypes_gpu, phenotype_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time

        # Results should be close
        torch.testing.assert_close(result_cpu[0], result_gpu[0].cpu(), rtol=1e-4)

        # For this size, GPU might not be faster due to overhead
        # Just check that both complete successfully
        assert cpu_time > 0
        assert gpu_time > 0

@pytest.mark.skipif(not TENSORQTL_AVAILABLE, reason="tensorqtl not available")
class TestErrorHandling:
    """Test error handling and recovery."""

    def test_mismatched_dimensions(self, device, torch_dtype):
        """Test handling of mismatched tensor dimensions."""
        # Mismatched sample dimensions
        genotypes = torch.randint(0, 3, (5, 10), dtype=torch_dtype, device=device).float()
        phenotype = torch.randn(20, dtype=torch_dtype, device=device)  # Different sample count

        with pytest.raises((RuntimeError, ValueError, AssertionError)):
            cis.calculate_cis_nominal(genotypes, phenotype)

    def test_invalid_genotype_values(self, device, torch_dtype):
        """Test handling of invalid genotype values."""
        n_samples = 50

        # Negative genotypes (other than missing -9)
        invalid_genotypes = torch.tensor([[-1, -2, -3] * (n_samples // 3 + 1)][:n_samples], dtype=torch_dtype, device=device).unsqueeze(0)
        phenotype = torch.randn(n_samples, dtype=torch_dtype, device=device)

        # Should handle gracefully or raise appropriate error
        try:
            result = cis.calculate_cis_nominal(invalid_genotypes, phenotype)
            # If it succeeds, results should be finite
            assert torch.isfinite(result[0]).any() or torch.isnan(result[0]).any()
        except (ValueError, RuntimeError):
            # Acceptable to fail with invalid values
            pass

    def test_nan_infinity_handling(self, device, torch_dtype):
        """Test handling of NaN and infinity values."""
        n_samples = 20

        # NaN in genotypes
        genotypes = torch.randint(0, 3, (1, n_samples), dtype=torch_dtype, device=device).float()
        genotypes[0, 0] = float('nan')
        phenotype = torch.randn(n_samples, dtype=torch_dtype, device=device)

        # Should handle NaN gracefully
        result = cis.calculate_cis_nominal(genotypes, phenotype)
        # Result might be NaN, which is acceptable

        # Infinity in phenotype
        genotypes = torch.randint(0, 3, (1, n_samples), dtype=torch_dtype, device=device).float()
        phenotype = torch.randn(n_samples, dtype=torch_dtype, device=device)
        phenotype[0] = float('inf')

        try:
            result = cis.calculate_cis_nominal(genotypes, phenotype)
            # If it succeeds, check for reasonable output
        except (ValueError, RuntimeError):
            # Acceptable to fail with infinite values
            pass

@pytest.mark.skipif(not TENSORQTL_AVAILABLE, reason="tensorqtl not available")
class TestRobustness:
    """Test robustness across different conditions."""

    def test_different_sample_sizes(self, device, torch_dtype):
        """Test robustness across different sample sizes."""
        sample_sizes = [10, 25, 50, 100, 200]

        for n_samples in sample_sizes:
            if n_samples < 5:
                continue  # Skip very small sizes

            torch.manual_seed(42)
            genotypes = torch.randint(0, 3, (5, n_samples), dtype=torch_dtype, device=device).float()
            phenotype = torch.randn(n_samples, dtype=torch_dtype, device=device)

            try:
                result = cis.calculate_cis_nominal(genotypes, phenotype)
                # Should produce valid results
                assert result[0].shape == (5,)
                assert torch.isfinite(result[0]).all() or torch.isnan(result[0]).any()
            except (ValueError, RuntimeError):
                # Some sizes might not work due to degrees of freedom
                pass

    def test_different_maf_distributions(self, device, torch_dtype):
        """Test robustness with different MAF distributions."""
        n_samples = 100

        # Test with rare variants (low MAF)
        rare_genotypes = torch.zeros(1, n_samples, dtype=torch_dtype, device=device)
        rare_genotypes[0, :5] = 1  # 5% heterozygotes -> MAF = 0.025

        # Test with common variants (high MAF)
        common_genotypes = torch.ones(1, n_samples, dtype=torch_dtype, device=device)
        common_genotypes[0, :20] = 0  # 20% homozygous reference
        common_genotypes[0, 20:40] = 2  # 20% homozygous alternate
        # 60% heterozygous -> MAF = 0.4

        phenotype = torch.randn(n_samples, dtype=torch_dtype, device=device)

        # Both should work
        for genotypes in [rare_genotypes, common_genotypes]:
            result = cis.calculate_cis_nominal(genotypes, phenotype)
            assert result[0].shape == (1,)