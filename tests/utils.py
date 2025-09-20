"""
Utility functions for tensorQTL tests.

This module provides helper functions for generating test data,
making comparisons, and other common testing operations.
"""

import numpy as np
import pandas as pd
import torch
import tempfile
import os
from pathlib import Path

def assert_allclose_with_dtype(actual, expected, rtol=1e-5, atol=1e-8, dtype=None):
    """
    Assert arrays are close with appropriate tolerances for data type.

    Parameters:
    -----------
    actual : array-like
        Actual values
    expected : array-like
        Expected values
    rtol : float
        Relative tolerance (adjusted based on dtype)
    atol : float
        Absolute tolerance (adjusted based on dtype)
    dtype : torch.dtype or np.dtype
        Data type for tolerance adjustment
    """
    # Adjust tolerances based on data type
    if dtype == torch.float32 or dtype == np.float32:
        rtol = max(rtol, 1e-4)
        atol = max(atol, 1e-6)
    elif dtype == torch.float64 or dtype == np.float64:
        rtol = max(rtol, 1e-12)
        atol = max(atol, 1e-14)

    if torch.is_tensor(actual):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    else:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

def generate_synthetic_genotypes(n_variants, n_samples, maf=0.2, missing_rate=0.0, seed=None):
    """
    Generate synthetic genotype data.

    Parameters:
    -----------
    n_variants : int
        Number of variants
    n_samples : int
        Number of samples
    maf : float
        Minor allele frequency
    missing_rate : float
        Proportion of missing genotypes
    seed : int
        Random seed

    Returns:
    --------
    pd.DataFrame
        Genotype matrix (variants x samples)
    """
    if seed is not None:
        np.random.seed(seed)

    # Calculate allele frequencies based on MAF
    p = 1 - maf  # Frequency of major allele
    q = maf      # Frequency of minor allele

    # Generate genotypes following Hardy-Weinberg equilibrium
    probs = [p**2, 2*p*q, q**2]  # P(0), P(1), P(2)
    genotypes = np.random.choice([0, 1, 2], size=(n_variants, n_samples), p=probs)

    # Add missing data
    if missing_rate > 0:
        missing_mask = np.random.random((n_variants, n_samples)) < missing_rate
        genotypes[missing_mask] = -9

    # Create DataFrame
    sample_ids = [f"S{i:04d}" for i in range(n_samples)]
    variant_ids = [f"chr1_{i*1000+10000}_A_G" for i in range(n_variants)]

    return pd.DataFrame(genotypes, index=variant_ids, columns=sample_ids)

def generate_synthetic_phenotypes(n_phenotypes, n_samples, noise_std=1.0, seed=None):
    """
    Generate synthetic phenotype data.

    Parameters:
    -----------
    n_phenotypes : int
        Number of phenotypes
    n_samples : int
        Number of samples
    noise_std : float
        Standard deviation of noise
    seed : int
        Random seed

    Returns:
    --------
    tuple of pd.DataFrame
        (expression_df, position_df)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate expression data
    expression = np.random.normal(0, noise_std, size=(n_phenotypes, n_samples))

    # Create DataFrames
    sample_ids = [f"S{i:04d}" for i in range(n_samples)]
    gene_ids = [f"ENSG{i:08d}.1" for i in range(n_phenotypes)]

    expr_df = pd.DataFrame(expression, index=gene_ids, columns=sample_ids)

    # Create position data
    pos_df = pd.DataFrame({
        'chr': ['chr1'] * n_phenotypes,
        'start': [i*50000 for i in range(n_phenotypes)],
        'end': [i*50000+1 for i in range(n_phenotypes)]
    }, index=gene_ids)

    return expr_df, pos_df

def generate_synthetic_covariates(n_samples, n_covariates=5, seed=None):
    """
    Generate synthetic covariate data.

    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_covariates : int
        Number of covariates
    seed : int
        Random seed

    Returns:
    --------
    pd.DataFrame
        Covariate matrix (covariates x samples)
    """
    if seed is not None:
        np.random.seed(seed)

    covariates = np.random.normal(0, 1, size=(n_covariates, n_samples))

    sample_ids = [f"S{i:04d}" for i in range(n_samples)]
    covariate_ids = [f"PC{i+1}" for i in range(n_covariates)]

    return pd.DataFrame(covariates, index=covariate_ids, columns=sample_ids)

def add_known_associations(genotypes_df, phenotypes_df, effect_sizes, variant_indices=None, phenotype_indices=None):
    """
    Add known associations between genotypes and phenotypes.

    Parameters:
    -----------
    genotypes_df : pd.DataFrame
        Genotype matrix
    phenotypes_df : pd.DataFrame
        Phenotype matrix
    effect_sizes : list
        Effect sizes for associations
    variant_indices : list
        Indices of variants to use (default: first N variants)
    phenotype_indices : list
        Indices of phenotypes to use (default: first N phenotypes)

    Returns:
    --------
    pd.DataFrame
        Modified phenotype matrix with added associations
    """
    modified_phenotypes = phenotypes_df.copy()

    n_associations = len(effect_sizes)
    if variant_indices is None:
        variant_indices = list(range(min(n_associations, len(genotypes_df))))
    if phenotype_indices is None:
        phenotype_indices = list(range(min(n_associations, len(phenotypes_df))))

    for i, effect_size in enumerate(effect_sizes):
        if i < len(variant_indices) and i < len(phenotype_indices):
            variant_idx = variant_indices[i]
            phenotype_idx = phenotype_indices[i]

            genotype_values = genotypes_df.iloc[variant_idx].values
            modified_phenotypes.iloc[phenotype_idx] += effect_size * genotype_values

    return modified_phenotypes

def create_temp_files(file_dict, temp_dir=None):
    """
    Create temporary files from dictionary of filename -> content.

    Parameters:
    -----------
    file_dict : dict
        Dictionary mapping filenames to content
    temp_dir : str
        Temporary directory (created if None)

    Returns:
    --------
    str
        Path to temporary directory
    """
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()

    for filename, content in file_dict.items():
        filepath = os.path.join(temp_dir, filename)

        if isinstance(content, pd.DataFrame):
            content.to_csv(filepath, sep='\t')
        elif isinstance(content, str):
            with open(filepath, 'w') as f:
                f.write(content)
        else:
            raise ValueError(f"Unsupported content type for {filename}: {type(content)}")

    return temp_dir

def simulate_qtl_data(n_samples=100, n_variants=1000, n_phenotypes=50, n_qtls=10,
                     effect_size_range=(0.1, 0.5), maf_range=(0.05, 0.5), seed=None):
    """
    Simulate comprehensive QTL dataset with known true associations.

    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_variants : int
        Number of variants
    n_phenotypes : int
        Number of phenotypes
    n_qtls : int
        Number of true QTL associations
    effect_size_range : tuple
        Range of effect sizes for true QTLs
    maf_range : tuple
        Range of minor allele frequencies
    seed : int
        Random seed

    Returns:
    --------
    dict
        Dictionary with 'genotypes', 'phenotypes', 'positions', 'covariates', 'true_qtls'
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate genotypes with varying MAF
    mafs = np.random.uniform(maf_range[0], maf_range[1], n_variants)
    genotypes_list = []

    for i, maf in enumerate(mafs):
        p = 1 - maf
        q = maf
        probs = [p**2, 2*p*q, q**2]
        variant_geno = np.random.choice([0, 1, 2], size=n_samples, p=probs)
        genotypes_list.append(variant_geno)

    genotypes = np.array(genotypes_list)

    # Generate baseline phenotypes
    phenotypes = np.random.normal(0, 1, size=(n_phenotypes, n_samples))

    # Add true QTL effects
    true_qtls = []
    for i in range(n_qtls):
        variant_idx = np.random.randint(0, n_variants)
        phenotype_idx = np.random.randint(0, n_phenotypes)
        effect_size = np.random.uniform(effect_size_range[0], effect_size_range[1])

        # Add effect
        phenotypes[phenotype_idx] += effect_size * genotypes[variant_idx]

        true_qtls.append({
            'variant_idx': variant_idx,
            'phenotype_idx': phenotype_idx,
            'effect_size': effect_size,
            'maf': mafs[variant_idx]
        })

    # Create DataFrames
    sample_ids = [f"S{i:04d}" for i in range(n_samples)]
    variant_ids = [f"chr1_{i*1000+10000}_A_G" for i in range(n_variants)]
    gene_ids = [f"ENSG{i:08d}.1" for i in range(n_phenotypes)]

    genotypes_df = pd.DataFrame(genotypes, index=variant_ids, columns=sample_ids)
    phenotypes_df = pd.DataFrame(phenotypes, index=gene_ids, columns=sample_ids)

    # Create position data
    positions_df = pd.DataFrame({
        'chr': ['chr1'] * n_phenotypes,
        'start': [i*50000 for i in range(n_phenotypes)],
        'end': [i*50000+1 for i in range(n_phenotypes)]
    }, index=gene_ids)

    # Create covariates
    covariates_df = generate_synthetic_covariates(n_samples, n_covariates=5, seed=seed)

    return {
        'genotypes': genotypes_df,
        'phenotypes': phenotypes_df,
        'positions': positions_df,
        'covariates': covariates_df,
        'true_qtls': true_qtls
    }

def check_output_format(df, required_columns, optional_columns=None):
    """
    Check if output DataFrame has required format.

    Parameters:
    -----------
    df : pd.DataFrame
        Output DataFrame to check
    required_columns : list
        List of required column names
    optional_columns : list
        List of optional column names

    Returns:
    --------
    bool
        True if format is correct
    """
    if optional_columns is None:
        optional_columns = []

    # Check required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise AssertionError(f"Missing required columns: {missing_cols}")

    # Check data types for common columns
    if 'pval_nominal' in df.columns:
        assert df['pval_nominal'].dtype in [np.float32, np.float64], "pval_nominal should be float"
        assert (df['pval_nominal'] >= 0).all(), "p-values should be non-negative"
        assert (df['pval_nominal'] <= 1).all(), "p-values should be <= 1"

    if 'slope' in df.columns:
        assert df['slope'].dtype in [np.float32, np.float64], "slope should be float"

    if 'af' in df.columns:
        assert df['af'].dtype in [np.float32, np.float64], "af should be float"
        assert (df['af'] >= 0).all(), "allele frequencies should be non-negative"
        assert (df['af'] <= 1).all(), "allele frequencies should be <= 1"

    return True

def validate_statistical_properties(pvals, alpha=0.05, tolerance=0.1):
    """
    Validate statistical properties of p-values under null hypothesis.

    Parameters:
    -----------
    pvals : array-like
        P-values to test
    alpha : float
        Significance level
    tolerance : float
        Tolerance for uniform distribution test

    Returns:
    --------
    dict
        Dictionary with validation results
    """
    pvals = np.array(pvals)

    # Remove NaN values
    valid_pvals = pvals[~np.isnan(pvals)]

    results = {
        'n_tests': len(valid_pvals),
        'prop_significant': np.mean(valid_pvals < alpha),
        'expected_prop_significant': alpha,
        'uniform_ks_pval': None,
        'is_uniform': False
    }

    if len(valid_pvals) > 0:
        # Test for uniform distribution (under null hypothesis)
        from scipy import stats
        _, ks_pval = stats.kstest(valid_pvals, 'uniform')
        results['uniform_ks_pval'] = ks_pval
        results['is_uniform'] = ks_pval > 0.05  # Not significantly different from uniform

        # Check if proportion of significant results is reasonable
        prop_diff = abs(results['prop_significant'] - alpha)
        results['prop_reasonable'] = prop_diff < tolerance

    return results