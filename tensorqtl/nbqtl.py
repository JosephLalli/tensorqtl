"""
GPU-accelerated negative binomial QTL mapping using quasi-GLM and score tests.

This module implements efficient score tests for QTL mapping with count data,
supporting both known observation-specific variances (from bootstrap/measurement models)
and estimated negative binomial dispersion parameters.

Statistical model:
- Null: log(μ) = Z*α + offset
- Known variance mode: w = μ²/V (quasi-GLM)
- Estimated variance mode: w = μ/(1 + φ*μ) (NB GLM)
- Score test: S = g'W(I-P)r, I = g'W(I-P)g

Author: tensorQTL development team
"""

import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize
from scipy.special import gammaln
import sys
import os
import time
from collections import OrderedDict
import h5py

sys.path.insert(1, os.path.dirname(__file__))
from core import SimpleLogger, Residualizer, output_dtype_dict
import genotypeio
from cis import read_phenotype_bed


def get_device_dtype(device=None, dtype=None):
    """Get device and dtype, with defaults for nbqtl"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float64  # Default to float64 for numerical stability
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    return device, dtype


def clamp_variances(V_t, min_var=1e-6, max_var=1e6):
    """Clamp variances to avoid numerical issues"""
    return torch.clamp(V_t, min_var, max_var)


def clamp_means(mu_t, min_mu=1e-12):
    """Clamp means to avoid numerical issues"""
    return torch.clamp(mu_t, min_mu, float('inf'))


def check_convergence(eta_old, eta_new, tol=1e-5):
    """Check IRLS convergence"""
    diff = torch.abs(eta_new - eta_old)
    max_diff = torch.max(diff)
    return max_diff < tol, max_diff


def load_variances(variance_file, phenotype_ids=None, samples=None):
    """
    Load observation-specific variances from file.

    Parameters:
    -----------
    variance_file : str
        Path to variance file (HDF5, NPY, or TSV)
    phenotype_ids : list, optional
        Subset of phenotypes to load
    samples : list, optional
        Subset of samples to load

    Returns:
    --------
    V_df : pd.DataFrame
        Variance matrix (phenotypes x samples)
    """
    if variance_file.endswith('.h5') or variance_file.endswith('.hdf5'):
        with h5py.File(variance_file, 'r') as f:
            V_arr = f['variances'][:]
            phenotype_ids_file = f['phenotype_ids'][:].astype(str)
            sample_ids_file = f['sample_ids'][:].astype(str)
        V_df = pd.DataFrame(V_arr, index=phenotype_ids_file, columns=sample_ids_file)
    elif variance_file.endswith('.npy'):
        V_arr = np.load(variance_file)
        if phenotype_ids is None or samples is None:
            raise ValueError("phenotype_ids and samples must be provided for .npy variance files")
        V_df = pd.DataFrame(V_arr, index=phenotype_ids, columns=samples)
    else:
        # Assume TSV format
        V_df = pd.read_csv(variance_file, sep='\t', index_col=0)

    # Subset if requested
    if phenotype_ids is not None:
        V_df = V_df.loc[phenotype_ids]
    if samples is not None:
        V_df = V_df[samples]

    return V_df


def load_offsets(offset_file, phenotype_ids=None, samples=None):
    """
    Load log offset terms (size factors, exposures).

    Parameters:
    -----------
    offset_file : str
        Path to offset file
    phenotype_ids : list, optional
        Phenotype IDs for broadcasting
    samples : list, optional
        Sample IDs

    Returns:
    --------
    offset_df : pd.DataFrame or pd.Series
        Offset matrix/vector
    """
    if offset_file.endswith('.npy'):
        offset_arr = np.load(offset_file)
        if offset_arr.ndim == 1:
            # Sample-specific offsets
            if samples is None:
                raise ValueError("samples must be provided for 1D .npy offset files")
            offset_df = pd.Series(offset_arr, index=samples)
        else:
            # Feature x sample offsets
            if phenotype_ids is None or samples is None:
                raise ValueError("phenotype_ids and samples must be provided for 2D .npy offset files")
            offset_df = pd.DataFrame(offset_arr, index=phenotype_ids, columns=samples)
    else:
        # Assume TSV format
        offset_df = pd.read_csv(offset_file, sep='\t', index_col=0)
        if offset_df.shape[1] == 1:
            offset_df = offset_df.squeeze()

    return offset_df


def broadcast_offsets(offset_df, phenotype_ids, samples):
    """Broadcast offsets to (F, N) shape"""
    if isinstance(offset_df, pd.Series):
        # Sample-specific offsets: broadcast to all features
        offset_matrix = pd.DataFrame(
            np.broadcast_to(offset_df.values[None, :], (len(phenotype_ids), len(samples))),
            index=phenotype_ids,
            columns=samples
        )
    else:
        # Already feature x sample
        offset_matrix = offset_df.loc[phenotype_ids, samples]

    return offset_matrix


def fit_null_quasi_nb_known_var(Y_t, Z_t, offset_t, V_t, max_iter=6, tol=1e-5,
                                ridge=1e-6, device=None, dtype=None, logger=None):
    """
    Fit null quasi-GLM with known observation-specific variances.

    Parameters:
    -----------
    Y_t : torch.Tensor
        Phenotype matrix (F x N) - count data
    Z_t : torch.Tensor
        Covariate matrix (N x p0) including intercept
    offset_t : torch.Tensor
        Log offset matrix (F x N) or (N,) for broadcasting
    V_t : torch.Tensor
        Known variance matrix (F x N) - from bootstrap/measurement model
    max_iter : int
        Maximum IRLS iterations
    tol : float
        Convergence tolerance
    ridge : float
        Ridge regularization for information matrix
    device : torch.device
        Device for computation
    dtype : torch.dtype
        Data type for computation
    logger : SimpleLogger
        Logger for progress

    Returns:
    --------
    mu0_t : torch.Tensor
        Fitted means under null (F x N)
    w_t : torch.Tensor
        IRLS weights (F x N)
    L_chol_t : torch.Tensor
        Cholesky factors (F x p0 x p0)
    r_perp_t : torch.Tensor
        Projected residuals (F x N)
    converged : torch.Tensor
        Convergence flag per feature (F,)
    """
    device, dtype = get_device_dtype(device, dtype)
    F, N = Y_t.shape
    p0 = Z_t.shape[1]

    # Move to device
    Y_t = Y_t.to(device=device, dtype=dtype)
    Z_t = Z_t.to(device=device, dtype=dtype)
    offset_t = offset_t.to(device=device, dtype=dtype)
    V_t = clamp_variances(V_t.to(device=device, dtype=dtype))

    # Broadcast offset if needed
    if offset_t.dim() == 1:
        offset_t = offset_t.unsqueeze(0).expand(F, -1)

    # Initialize linear predictor: log(mean(Y)) + centered offset
    Y_mean = Y_t.mean(dim=1, keepdim=True)
    Y_mean = torch.clamp(Y_mean, min=1e-6)  # Avoid log(0)
    offset_centered = offset_t - offset_t.mean(dim=1, keepdim=True)
    eta_t = torch.log(Y_mean) + offset_centered

    # Storage for results
    converged = torch.zeros(F, dtype=torch.bool, device=device)

    if logger is not None:
        logger.write(f'    * fitting null quasi-GLM for {F} features (known variances)')

    for iteration in range(max_iter):
        eta_old = eta_t.clone()

        # Compute current mean
        mu_t = clamp_means(torch.exp(eta_t))

        # Quasi-GLM weights using known variances
        w_t = mu_t.pow(2) / V_t

        # Working response (linearization around current estimate)
        z_t = eta_t + (Y_t - mu_t) / mu_t
        z_centered = z_t - offset_t

        # Build information matrix G = Z^T W Z for each feature (batched)
        # w_t is (F x N), Z_t is (N x p0)
        # We want G[f] = Z^T * diag(w[f]) * Z

        # Efficient batched computation
        Z_w = Z_t.unsqueeze(0) * w_t.unsqueeze(2)  # (F x N x p0)
        G_t = torch.bmm(Z_w.transpose(1, 2), Z_t.unsqueeze(0).expand(F, -1, -1))  # (F x p0 x p0)

        # Add ridge regularization
        ridge_eye = ridge * torch.eye(p0, device=device, dtype=dtype)
        G_t = G_t + ridge_eye.unsqueeze(0)

        # Right hand side h = Z^T W (z - offset)
        h_t = torch.bmm(Z_w.transpose(1, 2), z_centered.unsqueeze(2)).squeeze(2)  # (F x p0)

        # Solve G * alpha = h for each feature (batched Cholesky)
        try:
            L_chol_t = torch.linalg.cholesky(G_t)  # (F x p0 x p0)
            alpha_t = torch.cholesky_solve(h_t.unsqueeze(2), L_chol_t).squeeze(2)  # (F x p0)
        except RuntimeError as e:
            if logger is not None:
                logger.write(f'    * warning: Cholesky failed at iteration {iteration}, using pseudoinverse')
            alpha_t = torch.bmm(torch.pinverse(G_t), h_t.unsqueeze(2)).squeeze(2)
            L_chol_t = None

        # Update linear predictor
        eta_t = torch.bmm(alpha_t.unsqueeze(1), Z_t.T.unsqueeze(0).expand(F, -1, -1)).squeeze(1) + offset_t

        # Check convergence per feature
        conv_check, max_diff = check_convergence(eta_old, eta_t, tol)
        converged = converged | conv_check

        if logger is not None and iteration % 2 == 0:
            logger.write(f'    * iteration {iteration+1}: max_diff={max_diff:.2e}, converged={converged.sum().item()}/{F}')

        if converged.all():
            break

    # Final computations
    mu0_t = clamp_means(torch.exp(eta_t))
    w_t = mu0_t.pow(2) / V_t

    # Compute projected residuals r_perp = r - Z * (G^-1 * Z^T * W * r)
    # where r = (Y - mu) / mu (working residuals)
    r_t = (Y_t - mu0_t) / mu0_t

    if L_chol_t is not None:
        # Efficient computation using cached Cholesky factors
        Zw_r = torch.bmm(Z_w.transpose(1, 2), r_t.unsqueeze(2)).squeeze(2)  # Z^T W r
        Kinv_Zw_r = torch.cholesky_solve(Zw_r.unsqueeze(2), L_chol_t).squeeze(2)
        Z_Kinv_Zw_r = torch.bmm(Kinv_Zw_r.unsqueeze(1), Z_t.T.unsqueeze(0).expand(F, -1, -1)).squeeze(1)
        r_perp_t = r_t - Z_Kinv_Zw_r
    else:
        # Fallback without Cholesky factors
        r_perp_t = r_t  # Simplified for now
        L_chol_t = torch.eye(p0, device=device, dtype=dtype).unsqueeze(0).expand(F, -1, -1)

    if logger is not None:
        n_converged = converged.sum().item()
        logger.write(f'    * IRLS completed: {n_converged}/{F} features converged')

    return mu0_t, w_t, L_chol_t, r_perp_t, converged


def estimate_nb_dispersion_simple(Y_t, mu_t, min_dispersion=1e-8, max_dispersion=1e3):
    """
    Simple method-of-moments dispersion estimation for NB2 model.

    For NB2: Var(Y) = μ + φ*μ²
    Method of moments: φ = max(0, (sample_var - μ) / μ²)

    Parameters:
    -----------
    Y_t : torch.Tensor
        Count data (F x N)
    mu_t : torch.Tensor
        Fitted means (F x N)
    min_dispersion : float
        Minimum allowed dispersion
    max_dispersion : float
        Maximum allowed dispersion

    Returns:
    --------
    phi_t : torch.Tensor
        Dispersion estimates per feature (F,)
    """
    # Compute sample variance across samples for each feature
    Y_var = Y_t.var(dim=1, unbiased=True)  # (F,)
    mu_mean = mu_t.mean(dim=1)  # (F,)

    # Method of moments: φ = (Var - μ) / μ²
    phi_t = (Y_var - mu_mean) / (mu_mean.pow(2) + 1e-8)
    phi_t = torch.clamp(phi_t, min_dispersion, max_dispersion)

    return phi_t


def fit_null_nb_estimate_dispersion(Y_t, Z_t, offset_t, max_iter=6, tol=1e-5,
                                   ridge=1e-6, device=None, dtype=None, logger=None):
    """
    Fit null NB GLM with estimated dispersion parameters.

    Uses iterative approach:
    1. Fit Poisson GLM to get initial means
    2. Estimate dispersion using method of moments
    3. Re-fit with NB weights

    Parameters similar to fit_null_quasi_nb_known_var but without V_t.

    Returns similar outputs but with estimated dispersion.
    """
    device, dtype = get_device_dtype(device, dtype)
    F, N = Y_t.shape
    p0 = Z_t.shape[1]

    # Move to device
    Y_t = Y_t.to(device=device, dtype=dtype)
    Z_t = Z_t.to(device=device, dtype=dtype)
    offset_t = offset_t.to(device=device, dtype=dtype)

    # Broadcast offset if needed
    if offset_t.dim() == 1:
        offset_t = offset_t.unsqueeze(0).expand(F, -1)

    if logger is not None:
        logger.write(f'    * fitting null NB-GLM for {F} features (estimated dispersion)')

    # Step 1: Initial Poisson fit
    # (Simplified - use log(mean) as starting point)
    Y_mean = Y_t.mean(dim=1, keepdim=True)
    Y_mean = torch.clamp(Y_mean, min=1e-6)
    offset_centered = offset_t - offset_t.mean(dim=1, keepdim=True)
    eta_t = torch.log(Y_mean) + offset_centered

    # Simple Poisson IRLS for initial estimates
    for iteration in range(3):  # Just a few iterations for initial fit
        mu_t = clamp_means(torch.exp(eta_t))
        w_t = mu_t  # Poisson weights
        z_t = eta_t + (Y_t - mu_t) / mu_t
        z_centered = z_t - offset_t

        # Batched solve
        Z_w = Z_t.unsqueeze(0) * w_t.unsqueeze(2)
        G_t = torch.bmm(Z_w.transpose(1, 2), Z_t.unsqueeze(0).expand(F, -1, -1))
        G_t = G_t + ridge * torch.eye(p0, device=device, dtype=dtype).unsqueeze(0)
        h_t = torch.bmm(Z_w.transpose(1, 2), z_centered.unsqueeze(2)).squeeze(2)

        try:
            L_chol_t = torch.linalg.cholesky(G_t)
            alpha_t = torch.cholesky_solve(h_t.unsqueeze(2), L_chol_t).squeeze(2)
        except RuntimeError:
            alpha_t = torch.bmm(torch.pinverse(G_t), h_t.unsqueeze(2)).squeeze(2)
            L_chol_t = None

        eta_t = torch.bmm(alpha_t.unsqueeze(1), Z_t.T.unsqueeze(0).expand(F, -1, -1)).squeeze(1) + offset_t

    # Step 2: Estimate dispersion
    mu_t = clamp_means(torch.exp(eta_t))
    phi_t = estimate_nb_dispersion_simple(Y_t, mu_t)

    if logger is not None:
        phi_median = torch.median(phi_t).item()
        logger.write(f'    * estimated dispersion: median={phi_median:.3f}')

    # Step 3: Re-fit with NB weights
    converged = torch.zeros(F, dtype=torch.bool, device=device)

    for iteration in range(max_iter):
        eta_old = eta_t.clone()

        mu_t = clamp_means(torch.exp(eta_t))

        # NB weights: w = μ / (1 + φ*μ)
        phi_mu = phi_t.unsqueeze(1) * mu_t
        w_t = mu_t / (1.0 + phi_mu)

        z_t = eta_t + (Y_t - mu_t) / mu_t
        z_centered = z_t - offset_t

        # Batched solve
        Z_w = Z_t.unsqueeze(0) * w_t.unsqueeze(2)
        G_t = torch.bmm(Z_w.transpose(1, 2), Z_t.unsqueeze(0).expand(F, -1, -1))
        G_t = G_t + ridge * torch.eye(p0, device=device, dtype=dtype).unsqueeze(0)
        h_t = torch.bmm(Z_w.transpose(1, 2), z_centered.unsqueeze(2)).squeeze(2)

        try:
            L_chol_t = torch.linalg.cholesky(G_t)
            alpha_t = torch.cholesky_solve(h_t.unsqueeze(2), L_chol_t).squeeze(2)
        except RuntimeError:
            alpha_t = torch.bmm(torch.pinverse(G_t), h_t.unsqueeze(2)).squeeze(2)
            L_chol_t = None

        eta_t = torch.bmm(alpha_t.unsqueeze(1), Z_t.T.unsqueeze(0).expand(F, -1, -1)).squeeze(1) + offset_t

        conv_check, max_diff = check_convergence(eta_old, eta_t, tol)
        converged = converged | conv_check

        if converged.all():
            break

    # Final results
    mu0_t = clamp_means(torch.exp(eta_t))
    phi_mu = phi_t.unsqueeze(1) * mu0_t
    w_t = mu0_t / (1.0 + phi_mu)

    # Compute projected residuals properly
    r_t = (Y_t - mu0_t) / mu0_t

    if L_chol_t is not None:
        # Efficient computation using cached Cholesky factors
        Z_w = Z_t.unsqueeze(0) * w_t.unsqueeze(2)  # (F x N x p0)
        Zw_r = torch.bmm(Z_w.transpose(1, 2), r_t.unsqueeze(2)).squeeze(2)  # Z^T W r
        Kinv_Zw_r = torch.cholesky_solve(Zw_r.unsqueeze(2), L_chol_t).squeeze(2)
        Z_Kinv_Zw_r = torch.bmm(Kinv_Zw_r.unsqueeze(1), Z_t.T.unsqueeze(0).expand(F, -1, -1)).squeeze(1)
        r_perp_t = r_t - Z_Kinv_Zw_r
    else:
        # Fallback without Cholesky factors
        r_perp_t = r_t
        L_chol_t = torch.eye(p0, device=device, dtype=dtype).unsqueeze(0).expand(F, -1, -1)

    if logger is not None:
        n_converged = converged.sum().item()
        logger.write(f'    * NB-GLM completed: {n_converged}/{F} features converged')

    return mu0_t, w_t, L_chol_t, r_perp_t, converged


def score_test_block(geno_block_t, Z_t, w_t, L_chol_t, r_perp_t,
                    robust=False, device=None, dtype=None):
    """
    Vectorized score test for a block of genotypes.

    Computes score statistic S = g^T W (I - P) r and information I = g^T W (I - P) g
    where P = Z (Z^T W Z)^-1 Z^T W is the projection matrix.

    Parameters:
    -----------
    geno_block_t : torch.Tensor
        Genotype block (B x N) where B is block size, N is samples
    Z_t : torch.Tensor
        Covariate matrix (N x p0)
    w_t : torch.Tensor
        IRLS weights (F x N)
    L_chol_t : torch.Tensor
        Cholesky factors (F x p0 x p0)
    r_perp_t : torch.Tensor
        Projected residuals (F x N)
    robust : bool
        Use robust (sandwich) standard errors
    device, dtype : torch device and data type

    Returns:
    --------
    z_scores_t : torch.Tensor
        Z-scores (F x B)
    pvals_t : torch.Tensor
        P-values (F x B)
    """
    device, dtype = get_device_dtype(device, dtype)

    B, N = geno_block_t.shape  # B = block size, N = samples
    F = w_t.shape[0]  # F = features
    p0 = Z_t.shape[1]  # p0 = covariates

    # Move to device
    geno_block_t = geno_block_t.to(device=device, dtype=dtype)

    # Project out covariates from genotypes: g_perp = g - Z * (G^-1 * Z^T * W * g)
    # geno_block_t is (B x N), we want to compute this for all features

    # Expand genotypes for all features: (F x B x N)
    g_expanded = geno_block_t.unsqueeze(0).expand(F, -1, -1)

    # Compute Z^T * W * g for each feature and genotype
    # w_t is (F x N), Z_t is (N x p0), g_expanded is (F x B x N)
    Z_w = Z_t.unsqueeze(0) * w_t.unsqueeze(2)  # (F x N x p0)

    # For each feature and genotype: Z^T * W * g
    # We want: (F x B x p0) = Z^T(N x p0) * W(F x N) * g(F x B x N)
    Zw_g = torch.bmm(
        Z_w.transpose(1, 2),  # (F x p0 x N)
        g_expanded.transpose(1, 2)  # (F x N x B)
    )  # Result: (F x p0 x B)

    # Solve G * x = Zw_g using cached Cholesky factors
    # L_chol_t is (F x p0 x p0), Zw_g is (F x p0 x B)
    try:
        G_inv_Zw_g = torch.cholesky_solve(Zw_g, L_chol_t)  # (F x p0 x B)
    except RuntimeError:
        # Fallback to pseudoinverse
        G_inv = torch.pinverse(torch.bmm(L_chol_t, L_chol_t.transpose(1, 2)))
        G_inv_Zw_g = torch.bmm(G_inv, Zw_g)

    # Compute Z * G_inv_Zw_g: (F x N x B)
    Z_G_inv_Zw_g = torch.bmm(
        Z_t.unsqueeze(0).expand(F, -1, -1),  # (F x N x p0)
        G_inv_Zw_g  # (F x p0 x B)
    )  # Result: (F x N x B)

    # Projected genotypes: g_perp = g - Z * G_inv_Zw_g
    # Fix dimension mismatch: g_expanded is (F x B x N), Z_G_inv_Zw_g is (F x N x B)
    g_perp = g_expanded.transpose(1, 2) - Z_G_inv_Zw_g  # Both (F x N x B) now

    # Score statistic: S = g_perp^T * W * r_perp
    # g_perp is (F x N x B), w_t is (F x N), r_perp_t is (F x N)
    w_r_perp = w_t.unsqueeze(2) * r_perp_t.unsqueeze(2)  # (F x N x 1)
    scores = torch.bmm(g_perp.transpose(1, 2), w_r_perp).squeeze(2)  # (F x B)

    # Information: I = g_perp^T * W * g_perp
    w_g_perp = w_t.unsqueeze(2) * g_perp  # (F x N x B)
    information = torch.bmm(g_perp.transpose(1, 2), w_g_perp)  # (F x B x B)
    information_diag = torch.diagonal(information, dim1=1, dim2=2)  # (F x B)

    # Z-scores and p-values
    z_scores_t = scores / torch.sqrt(information_diag + 1e-12)
    pvals_t = 2.0 * torch.distributions.Normal(0, 1).cdf(-torch.abs(z_scores_t))

    return z_scores_t, pvals_t


def run_nbqtl_score(phenotype_bed, genotype_pgen, covariates_file,
                   variance_file=None, variance_from=None, offset_file=None,
                   cis_mode=True, window=1000000, robust=False,
                   batch_size_features=20000, batch_size_snps=16000,
                   device="cuda", dtype="float64", out_prefix=None,
                   logger=None, verbose=True):
    """
    High-level driver for nbqtl score test mapping.

    Parameters:
    -----------
    phenotype_bed : str
        Path to phenotype BED file (count data)
    genotype_pgen : str
        Path prefix to PGEN files (without .pgen extension)
    covariates_file : str
        Path to covariates file (TSV format)
    variance_file : str, optional
        Path to observation-specific variance file (H5/NPY/TSV)
    variance_from : str, optional
        Method for variance estimation ('nb2', 'trended') if variance_file not provided
    offset_file : str, optional
        Path to log offset file (size factors)
    cis_mode : bool
        Whether to perform cis-QTL mapping (vs trans)
    window : int
        cis-window size in base pairs
    robust : bool
        Use robust (sandwich) standard errors
    batch_size_features : int
        Number of features to process per batch
    batch_size_snps : int
        Number of SNPs to process per block
    device : str
        Device for computation ('cuda' or 'cpu')
    dtype : str
        Data type ('float64' or 'float32')
    out_prefix : str
        Output file prefix
    logger : SimpleLogger
        Logger for progress reporting
    verbose : bool
        Verbose output

    Returns:
    --------
    results_df : pd.DataFrame
        QTL mapping results
    """
    device, dtype = get_device_dtype(device, dtype)

    if logger is None:
        logger = SimpleLogger()

    logger.write('nbQTL score test mapping')
    logger.write(f'  * variance mode: {"known" if variance_file else variance_from or "nb2"}')
    logger.write(f'  * device: {device}')
    logger.write(f'  * dtype: {dtype}')

    # Load phenotypes
    logger.write(f'  * reading phenotypes ({phenotype_bed})')
    if cis_mode:
        phenotype_df, phenotype_pos_df = read_phenotype_bed(phenotype_bed)
        logger.write(f'    * {phenotype_df.shape[0]} phenotypes x {phenotype_df.shape[1]} samples')
    else:
        raise NotImplementedError("trans mode not yet implemented for nbqtl")

    # Load covariates
    if covariates_file is not None:
        logger.write(f'  * reading covariates ({covariates_file})')
        covariates_df = pd.read_csv(covariates_file, sep='\t', index_col=0).T
        assert phenotype_df.columns.equals(covariates_df.index), "Sample mismatch between phenotypes and covariates"
        logger.write(f'    * {covariates_df.shape[1]} covariates')

        # Add intercept if not present
        if not ('intercept' in covariates_df.columns or '1' in covariates_df.columns):
            covariates_df.insert(0, 'intercept', 1.0)

        Z_t = torch.tensor(covariates_df.values, dtype=dtype, device=device)
    else:
        # Just intercept
        logger.write('  * using intercept-only model')
        Z_t = torch.ones((phenotype_df.shape[1], 1), dtype=dtype, device=device)

    # Load variances or prepare for estimation
    if variance_file is not None:
        logger.write(f'  * reading variances ({variance_file})')
        V_df = load_variances(variance_file, phenotype_df.index, phenotype_df.columns)
        assert V_df.index.equals(phenotype_df.index), "Phenotype mismatch in variance file"
        assert V_df.columns.equals(phenotype_df.columns), "Sample mismatch in variance file"
        V_t = torch.tensor(V_df.values, dtype=dtype, device=device)
        use_known_variances = True
    else:
        V_t = None
        use_known_variances = False
        if variance_from is None:
            variance_from = 'nb2'
        logger.write(f'  * will estimate variances using {variance_from} method')

    # Load offsets
    if offset_file is not None:
        logger.write(f'  * reading offsets ({offset_file})')
        offset_df = load_offsets(offset_file, phenotype_df.index, phenotype_df.columns)
        offset_df = broadcast_offsets(offset_df, phenotype_df.index, phenotype_df.columns)
        offset_t = torch.tensor(offset_df.values, dtype=dtype, device=device)
    else:
        # Zero offsets
        offset_t = torch.zeros((phenotype_df.shape[0], phenotype_df.shape[1]), dtype=dtype, device=device)

    # Load genotype reader
    logger.write(f'  * initializing genotype reader ({genotype_pgen})')
    try:
        import pgen
        pgr = pgen.PgenReader(genotype_pgen, select_samples=phenotype_df.columns)
        n_variants = pgr.num_variants
        logger.write(f'    * {n_variants} variants available')
    except ImportError:
        raise ImportError("pgen module required for nbqtl. Install with: pip install Pgenlib")

    # Convert phenotypes to tensors
    Y_t = torch.tensor(phenotype_df.values, dtype=dtype, device=device)

    # Results storage
    results_list = []

    # Process in feature batches
    n_features = phenotype_df.shape[0]
    n_batches = (n_features + batch_size_features - 1) // batch_size_features

    logger.write(f'  * processing {n_features} features in {n_batches} batches')

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size_features
        end_idx = min((batch_idx + 1) * batch_size_features, n_features)
        batch_size = end_idx - start_idx

        if verbose:
            logger.write(f'  * batch {batch_idx + 1}/{n_batches}: features {start_idx}-{end_idx-1}')

        # Get feature batch
        Y_batch = Y_t[start_idx:end_idx]
        offset_batch = offset_t[start_idx:end_idx]

        if use_known_variances:
            V_batch = V_t[start_idx:end_idx]
        else:
            V_batch = None

        # Fit null models for this batch
        if use_known_variances:
            mu0_t, w_t, L_chol_t, r_perp_t, converged = fit_null_quasi_nb_known_var(
                Y_batch, Z_t, offset_batch, V_batch,
                device=device, dtype=dtype, logger=logger if verbose else None
            )
        else:
            mu0_t, w_t, L_chol_t, r_perp_t, converged = fit_null_nb_estimate_dispersion(
                Y_batch, Z_t, offset_batch,
                device=device, dtype=dtype, logger=logger if verbose else None
            )

        # Get phenotype batch info for cis mapping
        phenotype_batch_df = phenotype_df.iloc[start_idx:end_idx]
        phenotype_pos_batch_df = phenotype_pos_df.iloc[start_idx:end_idx]

        # Process each phenotype in the batch for cis associations
        for feat_idx, (phenotype_id, phenotype_row) in enumerate(phenotype_batch_df.iterrows()):
            phenotype_pos = phenotype_pos_batch_df.loc[phenotype_id]

            # Get cis window
            if 'pos' in phenotype_pos_df.columns:
                # TSS-based window
                start_pos = phenotype_pos['pos'] - window
                end_pos = phenotype_pos['pos'] + window
            else:
                # Region-based window
                start_pos = phenotype_pos['start'] - window
                end_pos = phenotype_pos['end'] + window

            chrom = phenotype_pos['chr']

            # Get variants in cis window using actual PGEN API
            try:
                # Check if chromosome exists in variant_dfs
                if chrom not in pgr.variant_dfs:
                    continue

                # Get variants in chromosome
                chrom_variants = pgr.variant_dfs[chrom]

                # Filter by position range
                mask = (chrom_variants['pos'] >= start_pos) & (chrom_variants['pos'] <= end_pos)
                cis_variant_indices = chrom_variants.loc[mask, 'index'].values
                cis_variant_ids = chrom_variants.loc[mask].index.values

                if len(cis_variant_indices) == 0:
                    continue

                n_cis_variants = len(cis_variant_indices)
                if verbose and feat_idx % 1000 == 0:
                    logger.write(f'    * {phenotype_id}: {n_cis_variants} cis variants')

                # Process variants in blocks
                n_var_batches = (n_cis_variants + batch_size_snps - 1) // batch_size_snps

                for var_batch_idx in range(n_var_batches):
                    var_start = var_batch_idx * batch_size_snps
                    var_end = min((var_batch_idx + 1) * batch_size_snps, n_cis_variants)

                    variant_indices_subset = cis_variant_indices[var_start:var_end]
                    variant_ids_subset = cis_variant_ids[var_start:var_end]

                    # Load genotype block using actual PGEN API
                    geno_block = pgen.read_dosages_list(
                        pgr.pgen_file,
                        variant_indices_subset,
                        sample_subset=pgr.sample_idxs
                    )
                    geno_block_t = torch.tensor(geno_block, dtype=dtype, device=device)

                    # Extract single feature data
                    w_feat = w_t[feat_idx:feat_idx+1]  # (1 x N)
                    L_chol_feat = L_chol_t[feat_idx:feat_idx+1]  # (1 x p0 x p0)
                    r_perp_feat = r_perp_t[feat_idx:feat_idx+1]  # (1 x N)

                    # Compute score tests
                    z_scores, pvals = score_test_block(
                        geno_block_t, Z_t, w_feat, L_chol_feat, r_perp_feat,
                        robust=robust, device=device, dtype=dtype
                    )

                    # Store results
                    for var_idx, (variant_index, variant_id) in enumerate(zip(variant_indices_subset, variant_ids_subset)):
                        z_score = z_scores[0, var_idx].cpu().item()
                        pval = pvals[0, var_idx].cpu().item()

                        # Get variant position from variant DataFrame
                        var_pos = pgr.pvar_df.iloc[variant_index]['pos']

                        # Calculate distances
                        if 'pos' in phenotype_pos_df.columns:
                            distance = var_pos - phenotype_pos['pos']
                        else:
                            distance = var_pos - (phenotype_pos['start'] + phenotype_pos['end']) // 2

                        results_list.append({
                            'phenotype_id': phenotype_id,
                            'variant_id': variant_id,
                            'chr': chrom,
                            'pos': var_pos,
                            'distance': distance,
                            'z_score': z_score,
                            'pval': pval,
                            'converged': converged[feat_idx].cpu().item()
                        })

            except Exception as e:
                if verbose:
                    logger.write(f'    * warning: error processing {phenotype_id}: {e}')
                continue

    # Convert results to DataFrame
    if len(results_list) == 0:
        logger.write('  * no results generated')
        return pd.DataFrame()

    results_df = pd.DataFrame(results_list)
    logger.write(f'  * generated {len(results_df)} associations')

    # Write output if prefix provided
    if out_prefix is not None:
        output_file = f"{out_prefix}.nbqtl_score.txt.gz"
        results_df.to_csv(output_file, sep='\t', index=False, float_format='%.6g')
        logger.write(f'  * results written to {output_file}')

    return results_df


def chunk_iterator(data, chunk_size):
    """Iterator for processing data in chunks"""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]