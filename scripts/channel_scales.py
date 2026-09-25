"""Do the two channels share a residual scale? Stacking assumes they do.

The stacked fit pools both channels under ONE sigma^2. If their whitened
residual scales differ, that single scale is a compromise: it inflates the
standard error on one channel and deflates it on the other, and the combined
statistic inherits whichever way the deflation falls. The inverse-variance
meta-analysis does not make this assumption -- it fits a scale per channel.

Measured on the observed data, per gene: s2a and s2t are the fitted residual
scales of the allelic and total channels at the null design.
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import compare_mixqtl_replication as CM   # noqa: E402

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
RUN = D / 'rasqual_default_mode_20260923'
EPS = 1e-12

null46 = {l.strip() for l in open(D / 'genes_null46_20260923.txt') if l.strip()}
I = CM.load_inputs(gene_list=str(D / 'genes_59_stratified_20260923.txt'),
                   regions=str(RUN / 'regions.bed'))
keep = I['keep']
import run_hapmixqtl_from_salmon as H  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    A, T, Va, Vt, _ = H.compute_summaries_from_gibbs(
        I['YL'][:, keep], I['YR'][:, keep], yT=I['YT'][:, keep])
C = I['cov_df'].values
N = C.shape[0]
Z = np.column_stack([np.ones(N), C])

rows = []
for i, g in enumerate(I['genes']):
    if g not in null46:
        continue
    ka = np.isfinite(A[i]) & np.isfinite(Va[i]) & (Va[i] > EPS)
    kt = np.isfinite(T[i]) & np.isfinite(Vt[i]) & (Vt[i] > EPS)
    if ka.sum() < 10 or kt.sum() < 20:
        continue
    wa = 1.0 / Va[i][ka]
    ya = A[i][ka] * np.sqrt(wa)                   # through origin, no nuisance
    s2a = float(ya @ ya) / max(ka.sum() - 1, 1)
    wt = 1.0 / Vt[i][kt]
    sw = np.sqrt(wt)[:, None]
    yw, Zw = T[i][kt] * sw.ravel(), Z[kt] * sw
    q, _ = np.linalg.qr(Zw)
    et = yw - q @ (q.T @ yw)
    s2t = float(et @ et) / max(kt.sum() - 1 - Z.shape[1], 1)
    rows.append(dict(gene=g, s2a=s2a, s2t=s2t, ratio=s2a / s2t))

t = pd.DataFrame(rows)
print(f'{len(t)} genes\n')
print('fitted residual scale, allelic vs total channel (observed data):')
print(f'  allelic s2a  median {t.s2a.median():.3f}   IQR {t.s2a.quantile(.25):.3f}-{t.s2a.quantile(.75):.3f}')
print(f'  total   s2t  median {t.s2t.median():.3f}   IQR {t.s2t.quantile(.25):.3f}-{t.s2t.quantile(.75):.3f}')
print(f'\n  ratio s2a/s2t  median {t.ratio.median():.2f}   '
      f'IQR {t.ratio.quantile(.25):.2f}-{t.ratio.quantile(.75):.2f}   '
      f'range {t.ratio.min():.2f}-{t.ratio.max():.2f}')
print(f'  ratio within 2-fold of 1 in {int(((t.ratio > 0.5) & (t.ratio < 2)).sum())}/{len(t)} genes')
t.to_csv(D / 'se_fixes_20260924/channel_scales.tsv', sep='\t', index=False)
print('\nwrote channel_scales.tsv')
