"""
Shared pytest fixtures for tensorQTL tests.

This module provides common fixtures that can be used across all test modules,
including test data loading, device configuration, and temporary directories.
"""

import pytest
import numpy as np
import pandas as pd
import torch
import tempfile
import shutil
from pathlib import Path
import gzip
import os

# Add tensorqtl to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tensorqtl
    from tensorqtl import genotypeio, pgen
    from tensorqtl.core import Residualizer
except ImportError:
    tensorqtl = None

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"

@pytest.fixture(scope="session")
def test_data_dir():
    """Path to the test data directory."""
    return TEST_DATA_DIR

@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available for testing."""
    return torch.cuda.is_available()

@pytest.fixture(scope="session")
def device(gpu_available):
    """Get the appropriate device for testing."""
    return torch.device("cuda" if gpu_available else "cpu")

@pytest.fixture(scope="session")
def torch_dtype():
    """Default torch dtype for tests."""
    return torch.float32

@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(scope="session")
def synthetic_genotypes(test_data_dir):
    """Load synthetic genotype data for unit tests."""
    geno_file = test_data_dir / "synthetic_genotypes.txt"
    if geno_file.exists():
        return pd.read_csv(geno_file, sep='\t', index_col=0)
    else:
        # Fallback synthetic data
        np.random.seed(123)
        n_variants, n_samples = 50, 20
        genotypes = np.random.choice([0, 1, 2], size=(n_variants, n_samples))
        sample_ids = [f"S{i:03d}" for i in range(n_samples)]
        variant_ids = [f"chr1_{i*1000+10000}_A_G" for i in range(n_variants)]
        return pd.DataFrame(genotypes, index=variant_ids, columns=sample_ids)

@pytest.fixture(scope="session")
def synthetic_phenotypes(test_data_dir):
    """Load synthetic phenotype data for unit tests."""
    pheno_file = test_data_dir / "synthetic_phenotypes.bed"
    if pheno_file.exists():
        df = pd.read_csv(pheno_file, sep='\t', index_col=3)
        return df.iloc[:, 3:], df.iloc[:, :3]  # expression data, position data
    else:
        # Fallback synthetic data
        np.random.seed(123)
        n_genes, n_samples = 5, 20
        expression = np.random.normal(0, 1, size=(n_genes, n_samples))
        sample_ids = [f"S{i:03d}" for i in range(n_samples)]
        gene_ids = [f"ENSG{i:08d}.1" for i in range(n_genes)]

        expr_df = pd.DataFrame(expression, index=gene_ids, columns=sample_ids)
        pos_df = pd.DataFrame({
            'chr': ['chr1'] * n_genes,
            'start': [i*10000 for i in range(n_genes)],
            'end': [i*10000+1 for i in range(n_genes)]
        }, index=gene_ids)

        return expr_df, pos_df

@pytest.fixture(scope="session")
def synthetic_covariates(test_data_dir):
    """Load synthetic covariate data for unit tests."""
    cov_file = test_data_dir / "synthetic_covariates.txt"
    if cov_file.exists():
        return pd.read_csv(cov_file, sep='\t', index_col=0)
    else:
        # Fallback synthetic data
        np.random.seed(123)
        n_samples = 20
        covariates = np.random.normal(0, 1, size=(3, n_samples))
        sample_ids = [f"S{i:03d}" for i in range(n_samples)]
        return pd.DataFrame(covariates, index=['PC1', 'PC2', 'PC3'], columns=sample_ids)

@pytest.fixture(scope="session")
def test_expression_data(test_data_dir):
    """Load GEUVADIS subset expression data if available."""
    expr_file = test_data_dir / "test_expression.bed.gz"
    if expr_file.exists():
        # Read the BED file
        with gzip.open(expr_file, 'rt') as f:
            header = f.readline().strip().split('\t')

        df = pd.read_csv(expr_file, sep='\t', index_col=3)
        sample_cols = header[4:]  # Skip chr, start, end, gene_id

        expr_df = df[sample_cols]
        pos_df = df[['#chr', 'start', 'end']].rename(columns={'#chr': 'chr'})

        return expr_df, pos_df
    else:
        return None, None

@pytest.fixture(scope="session")
def test_covariates_data(test_data_dir):
    """Load GEUVADIS subset covariate data if available."""
    cov_file = test_data_dir / "test_covariates.txt"
    if cov_file.exists():
        return pd.read_csv(cov_file, sep='\t', index_col=0)
    else:
        return None

@pytest.fixture
def residualizer(synthetic_covariates, device, torch_dtype):
    """Create a Residualizer object for testing."""
    if tensorqtl is None:
        pytest.skip("tensorqtl not available")

    cov_tensor = torch.tensor(synthetic_covariates.values.T, dtype=torch_dtype, device=device)
    return Residualizer(cov_tensor)

@pytest.fixture
def genotype_tensor(synthetic_genotypes, device, torch_dtype):
    """Convert genotype DataFrame to PyTorch tensor."""
    return torch.tensor(synthetic_genotypes.values, dtype=torch_dtype, device=device)

@pytest.fixture
def phenotype_tensor(synthetic_phenotypes, device, torch_dtype):
    """Convert phenotype DataFrame to PyTorch tensor."""
    expr_df, _ = synthetic_phenotypes
    return torch.tensor(expr_df.values, dtype=torch_dtype, device=device)

@pytest.fixture
def simple_association_data(device, torch_dtype):
    """Create simple test data with known associations for validation."""
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = 100
    n_variants = 10
    n_phenotypes = 5

    # Create genotypes
    genotypes = torch.randint(0, 3, (n_variants, n_samples), dtype=torch_dtype, device=device)

    # Create phenotypes with some true associations
    phenotypes = torch.randn(n_phenotypes, n_samples, dtype=torch_dtype, device=device)

    # Add known associations
    phenotypes[0] += 0.5 * genotypes[0]  # Strong association
    phenotypes[1] += 0.2 * genotypes[1]  # Weak association

    # Create covariates
    covariates = torch.randn(n_samples, 3, dtype=torch_dtype, device=device)

    return {
        'genotypes': genotypes,
        'phenotypes': phenotypes,
        'covariates': covariates,
        'true_effects': [0.5, 0.2, 0.0, 0.0, 0.0]  # Expected effect sizes
    }

@pytest.fixture
def edge_case_data(device, torch_dtype):
    """Create data with edge cases for robust testing."""
    n_samples = 50

    # Monomorphic variant (all zeros)
    mono_variant = torch.zeros(n_samples, dtype=torch_dtype, device=device)

    # Variant with missing data (-9)
    missing_variant = torch.ones(n_samples, dtype=torch_dtype, device=device)
    missing_variant[0] = -9

    # High correlation phenotypes
    base_pheno = torch.randn(n_samples, dtype=torch_dtype, device=device)
    corr_pheno = base_pheno + 0.01 * torch.randn(n_samples, dtype=torch_dtype, device=device)

    # Constant phenotype
    const_pheno = torch.ones(n_samples, dtype=torch_dtype, device=device)

    return {
        'monomorphic_variant': mono_variant,
        'missing_variant': missing_variant,
        'correlated_phenotypes': torch.stack([base_pheno, corr_pheno]),
        'constant_phenotype': const_pheno
    }

# Parametrized fixtures for testing different data types and devices
@pytest.fixture(params=[torch.float32, torch.float64])
def dtype_param(request):
    """Parametrized fixture for testing different data types."""
    return request.param

@pytest.fixture(params=["cpu", "cuda"])
def device_param(request, gpu_available):
    """Parametrized fixture for testing different devices."""
    if request.param == "cuda" and not gpu_available:
        pytest.skip("CUDA not available")
    return torch.device(request.param)

# Markers for organizing tests
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu_required: marks tests requiring GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks performance benchmark tests"
    )