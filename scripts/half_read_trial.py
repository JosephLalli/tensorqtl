"""Bounded total half-read trial; original ASE, unit-total weights, shipped GPU scan.

Record-null variance is conditional on observed records. Independent NB sampling
is a total-only sanity model, not a model of fresh Salmon assignment uncertainty.
Run in the existing benchmark environment with SIMULATED_EFFECTS_GENE_SET set per stratum.
"""
import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from half_read_io import atomic_path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark' / 'simulated_effects'))
import common as C
import tensorqtl.hapmixqtl as HM


def half_read(count, library):
    count, library = np.asarray(count), np.asarray(library)
    if not np.isfinite(count).all() or (count < 0).any():
        raise ValueError('counts must be finite and nonnegative')
    if not np.isfinite(library).all() or (library <= 0).any():
        raise ValueError('effective libraries must be finite and positive')
    return np.log2((count + 0.5) / (library + 1.0) * 1e6)


def selected_variant(I, candidates):
    dosage = I['dos'][candidates]
    candidates = candidates[(dosage != dosage[:, [0]]).any(1)]
    if not len(candidates):
        raise ValueError('no variable-dosage tested variant')
    sign = I['xL'][candidates].astype(float) - I['xR'][candidates].astype(float)
    return candidates[np.argmax((sign != 0).sum(1))]


def summarize(beta, se, p, truth=0.0):
    good = np.isfinite(beta) & np.isfinite(se) & (se > 0) & np.isfinite(p)
    b, e, pv = beta[good], se[good], p[good]
    return dict(n=int(good.sum()), mean=float(b.mean()) if len(b) else np.nan,
                variance=float(b.var(ddof=1)) if len(b) > 1 else np.nan,
                mse=float(np.mean((b-truth)**2)) if len(b) else np.nan,
                mean_se2=float(np.mean(e**2)) if len(b) else np.nan,
                **{f'rate_{a:g}': float(np.mean(pv < a)) if len(b) else np.nan for a in C.ALPHAS})


def check_mapper(ref, gene, vid, channel, b, se, p, checks):
    suffix = {'allelic': '_a', 'total': '_t', 'combined': ''}[channel]
    row = ref.loc[(gene, vid)]
    for name, value in [('slope'+suffix, b), ('slope'+suffix+'_se', se)]:
        expected = float(row[name])
        if np.isfinite(value) and np.isfinite(expected):
            delta = abs(value-expected)/max(float(row['slope'+suffix+'_se']), 1e-8)
            if delta > 1e-3:
                raise AssertionError((gene, channel, name, value, expected, delta))
            checks.append(delta)
        elif not (value == expected or np.isnan(value) and np.isnan(expected)):
            raise AssertionError((gene, channel, name, value, expected))
    pcol = {'allelic': 'pval_a', 'total': 'pval_t', 'combined': 'pval_nominal'}[channel]
    if not np.isfinite(p) == np.isfinite(row[pcol]):
        raise AssertionError((gene, channel, p, row[pcol]))
    if np.isfinite(p) and abs(p-row[pcol]) > 1e-4:
        raise AssertionError((gene, channel, p, row[pcol]))


def independent_counts(I, R, tested, output, reps, tensor, S, anchor):
    """Paired NB draws; exact same residualized design and draws for both transforms."""
    rng = np.random.default_rng(20260930)
    design = np.column_stack([np.ones(len(R['eff_lib'])), I['cov_df'], I['geno_cov_df']])
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise AssertionError('rank-deficient NB nuisance design')
    q = tensor(np.linalg.qr(design, mode='reduced')[0])
    dof = design.shape[0]-design.shape[1]-1
    lib = R['eff_lib']
    rows, check_values, check_counts = [], {}, []
    for k, gene in enumerate(I['genes']):
        j = selected_variant(I, tested[k])
        g = I['dos'][j].astype(float)/2
        gt = tensor(g)
        gr = gt - q @ (q.T @ gt)
        xx = float(gr @ gr)
        mu0 = lib * np.mean(R['pT'][k]/lib)
        if not (mu0 > 0).all() or xx <= 0:
            raise AssertionError((gene, 'invalid count-sampling design'))
        for phi in (0.05, 0.2):
            for beta in (0.0, -0.8, -0.4, 0.4, 0.8):
                mu = mu0 * 2**(beta*g)
                y = rng.negative_binomial(1/phi, (1/phi)/(1/phi+mu), size=(reps, len(lib)))
                if phi == 0.2 and beta == 0.4:
                    check_counts.append(y[0])
                for arm in ('original', 'half_read'):
                    transform = (lambda v: np.log2(v/lib*1e6+1)) if arm == 'original' else (lambda v: half_read(v, lib))
                    yt = tensor(transform(y))
                    residual = yt - (yt @ q) @ q.T
                    b = (residual @ gr)/xx
                    error = residual - b[:, None]*gr
                    se = torch.sqrt((error*error).sum(1)/dof/xx)
                    b, se = b.cpu().numpy(), se.cpu().numpy()
                    pv = 2*stats.t.sf(np.abs(b/se), dof)
                    if beta:
                        shift = transform(mu)-transform(mu0)
                        response = float(gr @ tensor(shift))/xx/beta
                    else:
                        prior = lib/1e6 if arm == 'original' else 0.5
                        derivative = g*mu0/(mu0+prior)
                        response = float(gr @ tensor(derivative))/xx
                    summary = summarize(b, se, pv, beta)
                    if phi == 0.2 and beta == 0.4:
                        check_values[(gene, arm)] = (str(I['vdf'].index[j]), b[0], se[0], pv[0])
                    rows.append(dict(gene=gene, variant_id=str(I['vdf'].index[j]), phi=phi,
                        beta=beta, arm=arm, dof=dof, baseline_mean_count=float(mu0.mean()),
                        response=response, normalized_variance=summary['variance']/response**2,
                        coverage95=float(np.mean(np.abs((b-beta)/se) <= stats.t.ppf(.975, dof))),
                        target_coverage95=float(np.mean(np.abs((b-response*beta)/se) <= stats.t.ppf(.975, dof))),
                        **summary))
        if (k+1) % 20 == 0:
            print(f'NB counts: {k+1}/{len(I["genes"])} genes', flush=True)
    with atomic_path(output/'independent_nb.parquet') as temporary:
        pd.DataFrame(rows).to_parquet(temporary, index=False)
    counts = np.asarray(check_counts)
    checks = []
    # A total-only scan with no ASE support verifies the independent-count instrument.
    zeros = np.zeros_like(counts, dtype=float)
    ds = anchor | dict(A=zeros, Va=zeros, pL=zeros, pR=zeros, pT=counts, eff_lib=lib,
                       perm=np.arange(len(lib)), swap=np.ones(len(lib)))
    for arm in ('original', 'half_read'):
        total = np.log2(counts/lib*1e6+1) if arm == 'original' else half_read(counts, lib)
        ref = C.run_nominal(S, ds | dict(T=total), 'split', output/'scratch')[0]
        ref = ref.set_index(['phenotype_id', 'variant_id'])
        for gene in I['genes']:
            vid, b, se, p = check_values[(gene, arm)]
            check_mapper(ref, gene, vid, 'total', b, se, p, checks)
            if ref.loc[(gene, vid), 'dof_t'] != dof:
                raise AssertionError((gene, 'NB dof mismatch'))
    return dict(seed=20260930, replicates=reps, phi=[0.05, 0.2], beta=[0, -.8, -.4, .4, .8],
        mean='mu0_i = Leff_i * mean(pT/Leff); mu_i = mu0_i * 2^(beta*dosage_i/2)',
        variance='mu + phi*mu^2', covariates='fixed real covariates; constant baseline CPM',
        weights='unit total only', precision='variance / squared noise-free finite-effect response; derivative at beta=0',
        max_mapper_discrepancy_se_units=max(checks), mapper_check='first NB draw at phi=.2 beta=.4; both transforms',
        limitation='No Salmon ambiguity, ASE channel, or resampled Gibbs weights')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--nperm', type=int, default=2000)
    ap.add_argument('--count-reps', type=int, default=1000)
    args = ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        ap.error('output must be absent or empty')
    if min(args.nperm, args.count_reps) < 2:
        ap.error('replicate counts must be at least two')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda')
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float64, device=device)
    I, R, tested = C.load()
    S = C.setup(I)
    A, T, Va, Vt, _ = HM.summaries_from_point_estimates(
        R['pL'], R['pR'], R['pT'], R['eff_lib'], R['YL'], R['YR'], R['YT'])
    kept = C.allelic_kept(R['pL'], R['pR'], Va)
    variance = np.where(kept, Va, 0.)
    t0 = time.perf_counter()
    for _ in range(1000):
        Th = half_read(R['pT'], R['eff_lib'])
    preprocessing_seconds = (time.perf_counter()-t0)/1000
    G, N = A.shape
    rng = np.random.default_rng(20260929)
    perm = np.stack([rng.permutation(N) for _ in range(args.nperm+2)])
    flip = rng.choice([-1., 1.], size=(args.nperm+2, N))
    anchor = C.load_dataset(C.DATASETS, 'beta0.0', 0)
    perm[0], flip[0] = anchor['perm'], anchor['swap'].astype(float)
    perm[1], flip[1] = np.arange(N), np.ones(N)
    for arr, name in ((A[:, perm[0]]*flip[0], 'A'), (T[:, perm[0]], 'T'), (Va[:, perm[0]], 'Va')):
        np.testing.assert_allclose(arr, anchor[name], rtol=0, atol=1e-12)
    index = ['phenotype_id', 'variant_id']
    stored = C.read_results(C.RESULTS/'beta0.0/split/nominal_rep000.parquet', C.COLS+C.DOF_COLS)
    stored = stored.set_index(index).sort_index()
    reference, timings = {}, []
    # Matched scans validate both instruments and give a small paired timing check.
    for m in range(3):
        pp, ff = perm[m], flip[m]
        ll, rr = R['pL'][:, pp], R['pR'][:, pp]
        moved = anchor | dict(A=A[:, pp]*ff, Va=Va[:, pp], Vt=Vt[:, pp],
            pL=np.where(ff < 0, rr, ll), pR=np.where(ff < 0, ll, rr),
            pT=R['pT'][:, pp], eff_lib=R['eff_lib'][pp], perm=pp, swap=ff)
        order = ('original', 'half_read') if m % 2 == 0 else ('half_read', 'original')
        for arm in order:
            ds = moved | dict(T=(T if arm == 'original' else Th)[:, pp])
            torch.cuda.synchronize()
            start = time.perf_counter()
            df = C.run_nominal(S, ds, 'split', args.output/'scratch')[0]
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            reference[(m, arm)] = df.set_index(index).sort_index()
            timings.append(dict(arrangement=m, arm=arm, seconds=elapsed, pairs=len(df)))
            print(f'mapper arrangement {m} {arm}: {len(df):,} pairs, {elapsed:.3f}s', flush=True)
        if m == 0:
            pd.testing.assert_frame_equal(reference[(0, 'original')], stored, check_exact=True)
        cols = ['slope_a', 'slope_a_se', 'pval_a', 'dof_a', 'allelic_admitted']
        pd.testing.assert_frame_equal(reference[(m, 'original')][cols],
                                      reference[(m, 'half_read')][cols], check_exact=True)
        for col in C.COLS[2:]:
            np.testing.assert_array_equal(np.isfinite(reference[(m, 'original')][col]),
                                          np.isfinite(reference[(m, 'half_read')][col]))
    with atomic_path(args.output/'mapper_timing.tsv') as temporary:
        pd.DataFrame(timings).to_csv(temporary, sep='\t', index=False)
    p_t, f_t = torch.as_tensor(perm, device=device), tensor(flip)
    cov = tensor(np.column_stack([I['cov_df'], I['geno_cov_df']]))
    ones = tensor(np.ones(N))
    rows, checks, mc = [], [], {}
    input_hash = hashlib.sha256()
    for arr in (A, T, Th, Va, Vt, R['pL'], R['pR'], R['pT'], R['eff_lib'], cov.cpu().numpy()):
        input_hash.update(np.ascontiguousarray(arr).tobytes())
    start = time.perf_counter()
    for k, gene in enumerate(I['genes']):
        j = selected_variant(I, tested[k])
        vid = str(I['vdf'].index[j])
        s = tensor((I['xL'][j].astype(float)-I['xR'][j].astype(float))[None, :])
        g = tensor(I['dos'][j].astype(float)[None, :]/2)
        input_hash.update(gene.encode()+vid.encode()+s.cpu().numpy().tobytes()+g.cpu().numpy().tobytes())
        a = tensor(A[k])
        swa, _, ra, rt = HM._prepare_channels(a, tensor(T[k]), tensor(variance[k]), ones,
            cov, 'zero', device, ase_covariates_t=None, fitted_scale=True)
        rt.n_fixed_cov = I['geno_cov_df'].shape[1]
        da, dt = HM._channel_dof(ra), HM._channel_dof(rt)
        admitted = HM._allelic_admitted(ra, rt, HM.MIN_ALLELIC_DONORS, True)
        xa, xxa, yya = HM._record_permutation_channel(s, a, swa, ra, p_t, flip_t=f_t)
        ba = torch.where(xxa > 0, xa/xxa, torch.zeros_like(xa))
        sea2 = torch.where(xxa > 0, ((yya-xa**2/xxa).clamp(min=0)/da)/xxa,
                           torch.full_like(xa, float('inf')))
        for arm, total in (('original', T), ('half_read', Th)):
            xt, xxt, yyt = HM._record_permutation_channel(g, tensor(total[k]), ones, rt, p_t)
            bt = xt/xxt
            set2 = ((yyt-xt**2/xxt).clamp(min=0)/dt)/xxt
            _, bc, dc = HM._combined_tstat2(xa, xxa, yya, xt, xxt, yyt, min(da, dt),
                fitted=True, dof_a=da, dof_t=dt, allelic=admitted, return_slope=True, return_dof=True)
            ia = torch.where(torch.isfinite(sea2) & (sea2 > 0), 1/sea2, torch.zeros_like(sea2))
            if not admitted:
                ia = torch.zeros_like(ia)
            it = 1/set2
            sec = torch.sqrt(HM._meier_factor(ia, it, da, dt)/(ia+it))
            arrays = [('allelic', ba, torch.sqrt(sea2), da), ('total', bt, torch.sqrt(set2), dt),
                      ('combined', bc, sec, dc.cpu().numpy().ravel())]
            for channel, b, se, df in arrays:
                b, se = b.cpu().numpy().ravel(), se.cpu().numpy().ravel()
                pv = 2*stats.t.sf(np.abs(b/se), df)
                if channel == 'allelic' and not np.isfinite(HM._reported_dof(ra)):
                    pv[:] = np.nan
                for m in range(3):
                    check_mapper(reference[(m, arm)], gene, vid, channel, b[m], se[m], pv[m], checks)
                rows.append(dict(gene=gene, variant_id=vid, arm=arm, channel=channel, admitted=admitted,
                    dof_a=da, dof_t=dt, **summarize(b[2:], se[2:], pv[2:])))
                good = np.isfinite(b[2:]) & np.isfinite(se[2:]) & (se[2:] > 0) & np.isfinite(pv[2:])
                for scope in (['all', 'admitted'] if admitted else ['all']):
                    counts = mc.setdefault((arm, channel, scope), np.zeros((1+len(C.ALPHAS), args.nperm)))
                    counts[0] += good
                    for z, alpha in enumerate(C.ALPHAS):
                        counts[z+1] += good & (pv[2:] < alpha)
        if (k+1) % 20 == 0:
            print(f'record null: {k+1}/{G} genes; {time.perf_counter()-start:.1f}s', flush=True)
    with atomic_path(args.output/'per_gene_null.parquet') as temporary:
        pd.DataFrame(rows).to_parquet(temporary, index=False)
    mc_rows = []
    for (arm, channel, scope), counts in mc.items():
        for z, alpha in enumerate(C.ALPHAS):
            rate = counts[z+1]/counts[0]
            rec = dict(arm=arm, channel=channel, scope=scope, alpha=alpha,
                rate=float(rate.mean()), mc_se=float(rate.std(ddof=1)/np.sqrt(args.nperm)))
            if arm == 'half_read':
                old = mc[('original', channel, scope)]
                delta = rate-old[z+1]/old[0]
                rec.update(paired_delta=float(delta.mean()), paired_mc_se=float(delta.std(ddof=1)/np.sqrt(args.nperm)))
            mc_rows.append(rec)
    with atomic_path(args.output/'null_monte_carlo.tsv') as temporary:
        pd.DataFrame(mc_rows).to_csv(temporary, sep='\t', index=False)
    nb = independent_counts(I, R, tested, args.output, args.count_reps, tensor, S, anchor)
    C.write_json(args.output/'manifest.json', dict(gene_set=C.GENE_SET, genes=G, donors=N,
        random_replicates=args.nperm, seed=20260929, additional_anchor_arrangements=2,
        original_saved_scan_exact=True, allelic_outputs_exact=True, finite_patterns_identical=True,
        mapper_checks='both arms at three arrangements; float64 instrument versus float32 mapper',
        max_mapper_discrepancy_se_units=max(checks), preprocessing_seconds=preprocessing_seconds,
        input_sha256=input_hash.hexdigest(), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        core_sha256=hashlib.sha256(Path(HM.__file__).read_bytes()).hexdigest(), independent_nb=nb,
        torch_version=torch.__version__, gpu=torch.cuda.get_device_name(), null_and_nb_seconds=time.perf_counter()-start))
    shutil.rmtree(args.output/'scratch')
    print(f'Complete: {args.output}', flush=True)


if __name__ == '__main__':
    main()
