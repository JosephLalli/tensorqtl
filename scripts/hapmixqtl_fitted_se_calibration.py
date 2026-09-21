"""hapmixQTL under Var(eps_i) = sigma^2 v_i, measured on its OWN statistic.

Until se_mode='fitted' existed, hapmixQTL could not express this model at
all: se_mode was 'model' (known-variance 1/sqrt(xx)) or 'robust' (HC1
sandwich), neither of which is sigma_hat/sqrt(xx). The calibration figure
quoted for it in earlier reports (1.033) came from the weighting ablation,
which computes that SE with mixQTL's regression function on hapmixQTL's
ALLELIC channel alone. This measures hapmixQTL's own combined statistic.

Calibration ratio = var(beta_hat) across null genotype permutations divided
by mean(se^2). One means the reported uncertainty matches the realized
error; above one means the arm understates its own error.

Configurations, all on identical data, donors and variants:

  shipped        tau_mode='estimate', se_mode='model'   w = 1/(v+tau), absolute
  fitted_tau     tau_mode='estimate', se_mode='fitted'  w = 1/(v+tau), shape only
  sigma2_v       tau_mode='zero',     se_mode='fitted'  w = 1/v,       shape only
  sigma2_v_known tau_mode='zero',     se_mode='model'   w = 1/v,       absolute

The last is the configuration the module warns about, included as the
reference point the warning is about.

NOTE the combined statistic is an inverse-variance meta-analysis of the two
channels on their SEs, so unlike a single channel its POINT ESTIMATE moves
with the SE form: under the known-variance form 1/se^2 = xx and it collapses
to the pooled score, under a fitted sigma each channel carries its own scale
and the two get reweighted. That is mechanism 4 of the disagreement plan,
and it is reported here as the beta correlation against the shipped arm.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
D = '/mnt/ssd/lalli/brainvar_hapmix_deploy'
OUT = f'{D}/mixqtl_replication_20260919'
SEED = 42
NPERM = int(os.environ.get('NP', '40'))

sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

CONFIGS = {
    'shipped':        dict(tau_mode='estimate', se_mode='model'),
    'fitted_tau':     dict(tau_mode='estimate', se_mode='fitted'),
    'sigma2_v':       dict(tau_mode='zero',     se_mode='fitted'),
    'sigma2_v_known': dict(tau_mode='zero',     se_mode='model'),
}


def main():
    import tensorqtl.hapmixqtl as HM
    from compare_mixqtl_replication import load_inputs, gene_variant_index

    I = load_inputs()
    genes, keep = I['genes'], I['keep']
    A, T, Va, Vt, _ = HM.compute_summaries_from_gibbs(
        I['YL'], I['YR'], yT=I['YT'], count_noise=True)
    A, T, Va, Vt = A[:, keep], T[:, keep], Va[:, keep], Vt[:, keep]
    dos = I['dos'][I['idx']][:, keep]
    sgn = (I['xL'] - I['xR'])[I['idx']][:, keep]
    dev = torch.device('cpu')
    Tt = lambda x: torch.tensor(np.asarray(x), dtype=torch.float64, device=dev)
    cov_t = Tt(I['cov_df'].values)
    n = len(I['order'])

    rows = []
    for j, g in enumerate(genes):
        vsel = gene_variant_index(I, g)
        if vsel.size == 0:
            continue
        gt = np.nan_to_num(dos[vsel].astype(float), nan=1.0)
        sg = sgn[vsel].astype(float)
        varying = gt.var(axis=1) > 0
        if not varying.any():
            continue
        gt, sg = gt[varying], sg[varying]
        a_t, t_t = Tt(A[j]), Tt(T[j])
        va_t, vt_t = Tt(Va[j]), Tt(Vt[j])

        for name, cfg in CONFIGS.items():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                wa, wt, ra, rt = HM._prepare_channels(
                    a_t, t_t, va_t, vt_t, cov_t, cfg['tau_mode'], dev,
                    ase_covariates_t=None)
            fitted = cfg['se_mode'] == 'fitted'
            bet = np.empty((NPERM, gt.shape[0]))
            se2 = np.empty((NPERM, gt.shape[0]))
            for pi in range(NPERM):
                prm = np.random.RandomState(SEED + 10007 + pi).permutation(n)
                out = HM.calculate_hapmixqtl_nominal(
                    Tt(gt[:, prm]), Tt(sg[:, prm]), a_t, t_t, wa, wt, ra, rt,
                    fitted=fitted)
                bet[pi] = out[1].cpu().numpy()
                se2[pi] = out[2].cpu().numpy() ** 2
            vb = np.nanvar(bet, axis=0, ddof=1)
            ms = np.nanmean(se2, axis=0)
            ok = np.isfinite(vb) & np.isfinite(ms) & (ms > 0)
            if ok.sum() < 20:
                continue
            # observed-pass beta, for the mechanism-4 comparison
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                obs = HM.calculate_hapmixqtl_nominal(
                    Tt(gt), Tt(sg), a_t, t_t, wa, wt, ra, rt, fitted=fitted)
            rows.append(dict(gene=g, config=name, n_var=int(ok.sum()),
                             calibration=float(np.median(vb[ok] / ms[ok])),
                             median_var_beta=float(np.median(vb[ok])),
                             obs_beta=obs[1].cpu().numpy()[ok],
                             obs_se=obs[2].cpu().numpy()[ok]))
        print(f'  {g}', flush=True)

    df = pd.DataFrame([{k: v for k, v in r.items()
                        if k not in ('obs_beta', 'obs_se')} for r in rows])
    df.to_csv(f'{OUT}/hapmixqtl_fitted_se_calibration.tsv', sep='\t', index=False)

    print('\nCALIBRATION of the COMBINED statistic, median over genes')
    print('  (1 = reported uncertainty matches realized error)')
    res = {}
    piv = df.pivot(index='gene', columns='config', values='calibration')
    vbp = df.pivot(index='gene', columns='config', values='median_var_beta')
    for name in CONFIGS:
        if name not in piv:
            continue
        res[name] = dict(
            calibration=float(piv[name].median()),
            calibration_iqr=[float(piv[name].quantile(.25)),
                             float(piv[name].quantile(.75))],
            var_beta_vs_shipped=float((vbp[name] / vbp['shipped']).median()),
        )
        print(f'  {name:16s} calibration {res[name]["calibration"]:7.3f}'
              f'   IQR {res[name]["calibration_iqr"][0]:.3f}-'
              f'{res[name]["calibration_iqr"][1]:.3f}'
              f'   var(beta) vs shipped {res[name]["var_beta_vs_shipped"]:.3f}')

    # mechanism 4: does a per-channel fitted sigma move the combined beta?
    by = {}
    for r in rows:
        by.setdefault(r['gene'], {})[r['config']] = r
    from scipy.stats import pearsonr
    for name in ('fitted_tau', 'sigma2_v'):
        rr, mag = [], []
        for g, d in by.items():
            if 'shipped' in d and name in d and len(d['shipped']['obs_beta']) == len(d[name]['obs_beta']):
                x, y = d['shipped']['obs_beta'], d[name]['obs_beta']
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() > 20:
                    rr.append(pearsonr(x[ok], y[ok])[0])
                    mag.append(np.median(np.abs(y[ok])) / np.median(np.abs(x[ok])))
        res.setdefault(name, {})['beta_r_vs_shipped'] = float(np.median(rr))
        res[name]['abs_beta_ratio_vs_shipped'] = float(np.median(mag))
        print(f'  mechanism 4: {name:14s} beta r vs shipped '
              f'{np.median(rr):.4f}   |beta| ratio {np.median(mag):.4f}')

    json.dump(dict(nperm=NPERM, seed=SEED, configs=CONFIGS, results=res),
              open(f'{OUT}/hapmixqtl_fitted_se_calibration.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
