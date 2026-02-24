import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch

sys.path.insert(1, os.path.dirname(__file__))
from core import *

EPSILON = 1e-12
MISSING_GENOTYPE = -9


def _read_summary_matrix(path):
    if path.lower().endswith(('.bed', '.bed.gz', '.bed.parquet')):
        return read_phenotype_bed(path)
    raise ValueError(f'Unsupported matrix format: {path}')


def read_hapmix_inputs(A_path, T_path, Va_path, Vt_path, tauL_path, tauR_path, covLR_path=None):
    A_df, pos_df = _read_summary_matrix(A_path)
    T_df, T_pos_df = _read_summary_matrix(T_path)
    Va_df, Va_pos_df = _read_summary_matrix(Va_path)
    Vt_df, Vt_pos_df = _read_summary_matrix(Vt_path)
    tauL_df, tauL_pos_df = _read_summary_matrix(tauL_path)
    tauR_df, tauR_pos_df = _read_summary_matrix(tauR_path)
    if covLR_path is not None:
        covLR_df, covLR_pos_df = _read_summary_matrix(covLR_path)
    else:
        covLR_df = pd.DataFrame(np.zeros_like(A_df.values, dtype=np.float32), index=A_df.index, columns=A_df.columns)
        covLR_pos_df = pos_df

    for name, df in [('T', T_df), ('Va', Va_df), ('Vt', Vt_df), ('tauL', tauL_df), ('tauR', tauR_df), ('covLR', covLR_df)]:
        if not A_df.columns.equals(df.columns):
            raise ValueError(f'Sample order mismatch between A and {name} matrices.')
        if not A_df.index.equals(df.index):
            raise ValueError(f'Feature order mismatch between A and {name} matrices.')
    for name, x in [('T', T_pos_df), ('Va', Va_pos_df), ('Vt', Vt_pos_df), ('tauL', tauL_pos_df), ('tauR', tauR_pos_df), ('covLR', covLR_pos_df)]:
        if not pos_df.equals(x):
            raise ValueError(f'Feature position mismatch between A and {name} matrices.')
    return A_df, T_df, Va_df, Vt_df, tauL_df, tauR_df, covLR_df, pos_df


def _adjust_ase_variance_with_covariance(Va_t, covLR_t, yL_mean_t, yR_mean_t, kappa):
    """
    Apply delta-method covariance correction to ASE variance.

    For A = log(yL+k) - log(yR+k):
      dA/dyL = 1/(yL+k), dA/dyR = -1/(yR+k),
      covariance contribution = 2*(dA/dyL)*(dA/dyR)*Cov(yL,yR)
                             = -2*Cov(yL,yR)/((yL+k)(yR+k)).
    Therefore negative Cov(yL,yR) increases Var(A), while positive Cov reduces it.
    Retained for diagnostic use. Not called in the production weight path
    because Va from Gibbs draws is treated as already containing full
    inferential uncertainty (including L/R covariance).
    """
    denL = torch.clamp(yL_mean_t + kappa, min=EPSILON)
    denR = torch.clamp(yR_mean_t + kappa, min=EPSILON)
    covLR_t = torch.nan_to_num(covLR_t, nan=0.0)
    cov_term_t = -2.0 * covLR_t / torch.clamp(denL * denR, min=EPSILON)
    Va_adj_t = torch.clamp(Va_t + cov_term_t, min=EPSILON)
    return Va_adj_t


def summarize_nonstandard_haplotype_inputs(phenotype_df, mapping_overdispersion_df, kappa=1.0):
    """
    Convert non-standard haplotype matrices (samples x 2*features) into
    hapmix summary matrices (features x samples).

    The expected input layout is:
      phenotype_df.iloc[i, 2*f + h], h in {0,1}
      mapping_overdispersion_df has the same shape/layout as phenotype_df.

    Negative expression values are clipped to 0 before log transforms.

    Parameters
    ----------
    phenotype_df : pd.DataFrame
        Matrix with samples in rows and paired haplotype feature columns
        in L/R order.
    mapping_overdispersion_df : pd.DataFrame
        Tau matrix with the same shape/order as phenotype_df.
    kappa : float
        Pseudocount used in log transforms.

    Returns
    -------
    tuple
        (A_df, T_df, Va_df, Vt_df, tauL_df, tauR_df), each features x samples.

    Warning
    -------
    This fallback sets Va=Vt=1 for all feature-sample entries (equal weighting).
    It is valid when inferential variances are unavailable, but typically less
    powerful than Gibbs-derived Va/Vt because sample-specific uncertainty is not
    reflected in the weights.
    """
    if not phenotype_df.index.equals(mapping_overdispersion_df.index):
        raise ValueError('phenotype_df and mapping_overdispersion_df sample indexes must match.')
    if not phenotype_df.columns.equals(mapping_overdispersion_df.columns):
        raise ValueError('phenotype_df and mapping_overdispersion_df columns must match.')
    if phenotype_df.shape[1] % 2 != 0:
        raise ValueError('Non-standard haplotype inputs must have 2 columns per feature.')

    A_rows = []
    T_rows = []
    Va_rows = []
    Vt_rows = []
    tauL_rows = []
    tauR_rows = []
    feature_ids = []
    sample_ids = phenotype_df.index

    for f in range(phenotype_df.shape[1] // 2):
        cL = phenotype_df.columns[2 * f]
        cR = phenotype_df.columns[2 * f + 1]
        yL = phenotype_df.iloc[:, 2 * f].astype(np.float32).values
        yR = phenotype_df.iloc[:, 2 * f + 1].astype(np.float32).values
        tauL = mapping_overdispersion_df.iloc[:, 2 * f].astype(np.float32).values
        tauR = mapping_overdispersion_df.iloc[:, 2 * f + 1].astype(np.float32).values

        if str(cL).endswith('_L') and str(cR) == str(cL)[:-2] + '_R':
            feature_id = str(cL)[:-2]
        else:
            feature_id = f'feature_{f}'
        feature_ids.append(feature_id)

        # Guard log transforms from invalid negative values in upstream inputs.
        yL = np.clip(yL, a_min=0.0, a_max=None)
        yR = np.clip(yR, a_min=0.0, a_max=None)
        A_rows.append(np.log(yL + kappa) - np.log(yR + kappa))
        T_rows.append(np.log((yL + yR) / 2.0 + kappa))
        # Non-standard input does not include Gibbs-draw inferential variances.
        # Use a neutral baseline variance of 1.0 per sample; tau still controls
        # relative weighting via variance inflation in downstream WLS.
        Va_rows.append(np.ones_like(yL, dtype=np.float32))
        Vt_rows.append(np.ones_like(yL, dtype=np.float32))
        tauL_rows.append(tauL)
        tauR_rows.append(tauR)

    A_df = pd.DataFrame(np.vstack(A_rows), index=feature_ids, columns=sample_ids)
    T_df = pd.DataFrame(np.vstack(T_rows), index=feature_ids, columns=sample_ids)
    Va_df = pd.DataFrame(np.vstack(Va_rows), index=feature_ids, columns=sample_ids)
    Vt_df = pd.DataFrame(np.vstack(Vt_rows), index=feature_ids, columns=sample_ids)
    tauL_df = pd.DataFrame(np.vstack(tauL_rows), index=feature_ids, columns=sample_ids)
    tauR_df = pd.DataFrame(np.vstack(tauR_rows), index=feature_ids, columns=sample_ids)
    return A_df, T_df, Va_df, Vt_df, tauL_df, tauR_df


def _check_haplotype_sample_alignment(genotype_columns, samples):
    if len(genotype_columns) != 2 * len(samples):
        raise ValueError('Haplotype genotype input must contain exactly 2 columns per sample (L/R).')
    for i, sample in enumerate(samples):
        c0 = str(genotype_columns[2 * i])
        c1 = str(genotype_columns[2 * i + 1])
        if not (c0 == sample or c0.startswith(sample)):
            raise ValueError(f'Genotype haplotype column {c0} is not aligned to sample {sample}.')
        if not (c1 == sample or c1.startswith(sample)):
            raise ValueError(f'Genotype haplotype column {c1} is not aligned to sample {sample}.')


def _weighted_channel_ols_vectorized(X_t, y_t, w_t, covariates_t=None):
    m_t = w_t > 0
    n_eff = int(m_t.sum().item())
    n_cov = 0 if covariates_t is None else covariates_t.shape[1]
    if n_eff <= (n_cov + 2):
        z = torch.full([X_t.shape[0]], np.nan, device=X_t.device)
        return z, z.clone(), z.clone(), n_eff, np.nan

    # Standard weighted least-squares identity:
    #   min sum_i w_i (y_i - x_i b)^2
    # is equivalent to OLS on transformed variables:
    #   y*_i = sqrt(w_i) y_i, x*_i = sqrt(w_i) x_i.
    # This lets us reuse tensorQTL's fast OLS/residualization machinery.
    sw_t = torch.sqrt(w_t[m_t])
    y_star_t = y_t[m_t] * sw_t
    X_star_t = X_t[:, m_t] * sw_t
    if covariates_t is not None:
        C_star_t = covariates_t[m_t] * sw_t.unsqueeze(1)
        residualizer = Residualizer(C_star_t)
        y_r_t = residualizer.transform(y_star_t.reshape(1, -1)).squeeze(0)
        X_r_t = residualizer.transform(X_star_t)
    else:
        y_r_t = y_star_t - y_star_t.mean()
        X_r_t = X_star_t - X_star_t.mean(1, keepdim=True)

    r_t, gx_var_t, y_var_t = calculate_corr(X_r_t, y_r_t.reshape(1, -1), residualizer=None, return_var=True)
    r_t = r_t.squeeze(1)
    std_ratio_t = torch.sqrt(y_var_t.reshape(1, -1) / gx_var_t.reshape(-1, 1)).squeeze(1)
    slope_t = r_t * std_ratio_t

    dof = n_eff - 2 - n_cov
    tstat_t = r_t * torch.sqrt(dof / (1 - r_t.double().pow(2)))
    slope_se_t = (slope_t.double() / tstat_t).float()
    return slope_t, slope_se_t, tstat_t, n_eff, dof


def _weighted_channel_ols_single(x_t, y_t, w_t, covariates_t=None):
    m_t = w_t > 0
    n_eff = int(m_t.sum().item())
    n_cov = 0 if covariates_t is None else covariates_t.shape[1]
    if n_eff <= (n_cov + 2):
        return np.nan, np.nan, np.nan, n_eff, np.nan

    # Same sqrt(weight) WLS transform as above, for single-variant ASE fits.
    sw_t = torch.sqrt(w_t[m_t])
    y_star_t = y_t[m_t] * sw_t
    x_star_t = x_t[m_t] * sw_t
    if covariates_t is not None:
        C_star_t = covariates_t[m_t] * sw_t.unsqueeze(1)
        residualizer = Residualizer(C_star_t)
        y_r_t = residualizer.transform(y_star_t.reshape(1, -1)).squeeze(0)
        x_r_t = residualizer.transform(x_star_t.reshape(1, -1)).squeeze(0)
    else:
        y_r_t = y_star_t - y_star_t.mean()
        x_r_t = x_star_t - x_star_t.mean()

    x_var = x_r_t.var()
    y_var = y_r_t.var()
    r = torch.dot(
        x_r_t / torch.sqrt((x_r_t * x_r_t).sum()),
        y_r_t / torch.sqrt((y_r_t * y_r_t).sum())
    )
    slope = r * torch.sqrt(y_var / x_var)
    dof = n_eff - 2 - n_cov
    tstat = r * torch.sqrt(dof / (1 - r.double().pow(2)))
    slope_se = (slope.double() / tstat).float()
    return float(slope.item()), float(slope_se.item()), float(tstat.item()), n_eff, dof


def _combine_channels(beta_a, se_a, beta_t, se_t):
    if np.isfinite(beta_a) and np.isfinite(se_a) and np.isfinite(beta_t) and np.isfinite(se_t):
        w_a = 1.0 / (se_a * se_a)
        w_t = 1.0 / (se_t * se_t)
        beta = (beta_a * w_a + beta_t * w_t) / (w_a + w_t)
        se = np.sqrt(1.0 / (w_a + w_t))
        pval = 2 * stats.norm.sf(np.abs(beta / se))
    elif np.isfinite(beta_t) and np.isfinite(se_t):
        beta, se = beta_t, se_t
        pval = 2 * stats.norm.sf(np.abs(beta / se))
    elif np.isfinite(beta_a) and np.isfinite(se_a):
        beta, se = beta_a, se_a
        pval = 2 * stats.norm.sf(np.abs(beta / se))
    else:
        beta = np.nan
        se = np.nan
        pval = np.nan
    return beta, se, pval


def map_nominal(genotype_df, variant_df, A_df, T_df, Va_df, Vt_df, tauL_df, tauR_df, phenotype_pos_df, prefix,
                covLR_df=None,
                covariates_df=None, maf_threshold=0, window=1000000, min_hets_ase=5, kappa=1.0,
                output_dir='.', logger=None, verbose=True):
    """
    Nominal cis mapping for haplotype-aware two-channel QTL model (Method A).

    Channel models:
      ASE channel:   a_i ~ beta * s_i + covariates,   s_i = xL_i - xR_i
      Total channel: t_i ~ beta * (g_i / 2) + covariates, g_i = xL_i + xR_i

    The total-channel regressor uses g/2 (not g) so both channels estimate the
    same log-aFC beta scale. Channel estimates are combined by inverse-variance
    meta-analysis.
    """
    import genotypeio
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if logger is None:
        logger = SimpleLogger()

    if covariates_df is not None and not A_df.columns.equals(covariates_df.index):
        raise ValueError('Covariates must be aligned to hapmix input samples.')
    if covLR_df is not None and not (A_df.index.equals(covLR_df.index) and A_df.columns.equals(covLR_df.columns)):
        raise ValueError('covLR_df must be aligned to A_df.')
    _check_haplotype_sample_alignment(genotype_df.columns, A_df.columns)

    logger.write('hapmixQTL nominal mapping')
    logger.write(f'  * {A_df.shape[1]} samples')
    logger.write(f'  * {A_df.shape[0]} features')
    logger.write(f'  * {variant_df.shape[0]} variants')
    logger.write(f'  * cis-window: ±{window:,}')

    if covariates_df is not None:
        covariates_t = torch.tensor(covariates_df.values, dtype=torch.float32).to(device)
    else:
        covariates_t = None

    igc = genotypeio.InputGeneratorCis(genotype_df, variant_df, A_df, phenotype_pos_df, window=window)
    idx = {pid: i for i, pid in enumerate(A_df.index)}

    for chrom in igc.chrs:
        logger.write(f'    Mapping chromosome {chrom}')
        n = 0
        for phenotype_id in igc.phenotype_pos_df[igc.phenotype_pos_df['chr'] == chrom].index:
            r = igc.cis_ranges[phenotype_id]
            n += r[1] - r[0] + 1

        chr_res = OrderedDict()
        chr_res['phenotype_id'] = []
        chr_res['variant_id'] = []
        chr_res['start_distance'] = np.empty(n, dtype=np.int32)
        if 'pos' not in phenotype_pos_df:
            chr_res['end_distance'] = np.empty(n, dtype=np.int32)
        chr_res['af'] = np.empty(n, dtype=np.float32)
        chr_res['ma_samples'] = np.empty(n, dtype=np.int32)
        chr_res['ma_count'] = np.empty(n, dtype=np.int32)
        chr_res['tauL_median'] = np.empty(n, dtype=np.float32)
        chr_res['tauR_median'] = np.empty(n, dtype=np.float32)
        for c in ['beta', 'beta_se', 'pval', 'beta_a', 'beta_a_se', 'pval_a', 'beta_t', 'beta_t_se', 'pval_t']:
            chr_res[c] = np.empty(n, dtype=np.float64 if c.startswith('pval') else np.float32)

        start = 0
        for _, genotypes, genotype_range, phenotype_id in igc.generate_data(chrom=chrom, verbose=verbose):
            i = idx[phenotype_id]
            A_t = torch.tensor(A_df.values[i], dtype=torch.float32).to(device)
            T_t = torch.tensor(T_df.values[i], dtype=torch.float32).to(device)
            Va_t = torch.tensor(Va_df.values[i], dtype=torch.float32).to(device)
            Vt_t = torch.tensor(Vt_df.values[i], dtype=torch.float32).to(device)
            tauL_t = torch.tensor(tauL_df.values[i], dtype=torch.float32).to(device)
            tauR_t = torch.tensor(tauR_df.values[i], dtype=torch.float32).to(device)
            if covLR_df is not None:
                covLR_t = torch.tensor(covLR_df.values[i], dtype=torch.float32).to(device)
            else:
                covLR_t = torch.zeros_like(Va_t)
            # covLR_t is retained for diagnostics/backward compatibility only;
            # production weight computation uses Va_t/Vt_t directly.

            # Production weights use Gibbs-draw summary variances directly.
            w_a_base_t = 1.0 / torch.clamp(Va_t, min=EPSILON)
            w_t_t = 1.0 / torch.clamp(Vt_t, min=EPSILON)

            # Tau tensors are retained for QC outputs only.
            tauL_median = float(torch.nanmedian(tauL_t).item())
            tauR_median = float(torch.nanmedian(tauR_t).item())

            g_hap_t = torch.tensor(genotypes, dtype=torch.float32).to(device)
            xL_t = g_hap_t[:, 0::2]
            xR_t = g_hap_t[:, 1::2]
            phase_known_t = torch.isfinite(xL_t) & torch.isfinite(xR_t) & (xL_t >= 0) & (xR_t >= 0)
            s_t = xL_t - xR_t
            g_t = xL_t + xR_t
            g_t = torch.where(phase_known_t, g_t, torch.full_like(g_t, MISSING_GENOTYPE))
            impute_mean(g_t)
            s_t = torch.where(phase_known_t, s_t, torch.zeros_like(s_t))

            if maf_threshold > 0:
                variant_ids = genotype_df.index[genotype_range]
                g_t, mask_t = filter_maf(g_t, variant_ids, maf_threshold)
                s_t = s_t[mask_t]
                phase_known_t = phase_known_t[mask_t]
                genotype_range = genotype_range[mask_t.cpu().numpy()]

            beta_t_t, beta_t_se_t, tstat_t_t, n_eff_t, dof_t = _weighted_channel_ols_vectorized(g_t / 2, T_t, w_t_t, covariates_t)
            pval_t = get_t_pval(tstat_t_t.cpu().numpy(), dof_t) if np.isfinite(dof_t) else np.full(beta_t_t.shape[0], np.nan)

            nvar = g_t.shape[0]
            beta_a = np.empty(nvar, dtype=np.float32)
            beta_a_se = np.empty(nvar, dtype=np.float32)
            pval_a = np.empty(nvar, dtype=np.float64)
            beta = np.empty(nvar, dtype=np.float32)
            beta_se = np.empty(nvar, dtype=np.float32)
            pval = np.empty(nvar, dtype=np.float64)

            for j in range(nvar):
                # ASE is only informative in phased heterozygotes. Homozygotes
                # (s=0) and unphased samples are assigned zero ASE weight.
                ase_mask_t = (s_t[j] != 0) & phase_known_t[j]
                w_a_t = torch.where(ase_mask_t, w_a_base_t, torch.zeros_like(w_a_base_t))
                if int(ase_mask_t.sum().item()) < min_hets_ase:
                    ba, bse, pa = np.nan, np.nan, np.nan
                else:
                    ba, bse, t_a, _, dof_a = _weighted_channel_ols_single(s_t[j], A_t, w_a_t, covariates_t)
                    pa = get_t_pval(t_a, dof_a) if np.isfinite(dof_a) else np.nan
                bt = float(beta_t_t[j].item()) if torch.isfinite(beta_t_t[j]) else np.nan
                btse = float(beta_t_se_t[j].item()) if torch.isfinite(beta_t_se_t[j]) else np.nan
                b, bse_comb, p_comb = _combine_channels(ba, bse, bt, btse)

                beta_a[j] = ba
                beta_a_se[j] = bse
                pval_a[j] = pa
                beta[j] = b
                beta_se[j] = bse_comb
                pval[j] = p_comb

            af_t, ma_samples_t, ma_count_t = get_allele_stats(g_t)
            end = start + nvar
            chr_res['phenotype_id'].extend([phenotype_id] * nvar)
            chr_res['variant_id'].extend(genotype_df.index[genotype_range].tolist())
            pos = phenotype_pos_df.loc[phenotype_id]
            chr_res['start_distance'][start:end] = variant_df['pos'].values[genotype_range] - pos.iloc[1]
            if 'pos' not in phenotype_pos_df:
                chr_res['end_distance'][start:end] = variant_df['pos'].values[genotype_range] - pos['end']
            chr_res['af'][start:end] = af_t.cpu().numpy()
            chr_res['ma_samples'][start:end] = ma_samples_t.cpu().numpy()
            chr_res['ma_count'][start:end] = ma_count_t.cpu().numpy()
            chr_res['tauL_median'][start:end] = tauL_median
            chr_res['tauR_median'][start:end] = tauR_median
            chr_res['beta_t'][start:end] = beta_t_t.cpu().numpy()
            chr_res['beta_t_se'][start:end] = beta_t_se_t.cpu().numpy()
            chr_res['pval_t'][start:end] = pval_t
            chr_res['beta_a'][start:end] = beta_a
            chr_res['beta_a_se'][start:end] = beta_a_se
            chr_res['pval_a'][start:end] = pval_a
            chr_res['beta'][start:end] = beta
            chr_res['beta_se'][start:end] = beta_se
            chr_res['pval'][start:end] = pval
            start = end

        out_file = os.path.join(output_dir, f'{prefix}.hapmixqtl_pairs.{chrom}.parquet')
        pd.DataFrame(chr_res).to_parquet(out_file, index=False)


def map_nominal_nonstandard(genotype_df, variant_df, phenotype_df, mapping_overdispersion_df, phenotype_pos_df, prefix,
                            covariates_df=None, maf_threshold=0, window=1000000, min_hets_ase=5, kappa=1.0,
                            output_dir='.', logger=None, verbose=True):
    """
    Wrapper for non-standard haplotype input layout:
      phenotype_df: samples x (2*features) with paired L/R columns.
      mapping_overdispersion_df: same shape/order as phenotype_df with tau values.

    Returns
    -------
    None
        Writes nominal hapmixqtl parquet outputs to output_dir.
    """
    A_df, T_df, Va_df, Vt_df, tauL_df, tauR_df = summarize_nonstandard_haplotype_inputs(
        phenotype_df, mapping_overdispersion_df, kappa=kappa
    )
    if not A_df.index.equals(phenotype_pos_df.index):
        raise ValueError('phenotype_pos_df index must match derived feature IDs from non-standard phenotype_df columns.')
    return map_nominal(
        genotype_df, variant_df, A_df, T_df, Va_df, Vt_df, tauL_df, tauR_df, phenotype_pos_df, prefix,
        covLR_df=None, covariates_df=covariates_df, maf_threshold=maf_threshold, window=window,
        min_hets_ase=min_hets_ase, kappa=kappa, output_dir=output_dir, logger=logger, verbose=verbose
    )
