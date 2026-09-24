"""How accurate is each method's standard error, per gene?

The earlier comparison put the median reported se beside the median realized
null spread -- two separately summarised distributions, which says nothing about
whether the se is right for any particular gene. This pairs them.

Both sides come from the SAME null draws, so nothing depends on comparing an
observed fit to a null one:

    claimed_g   = mean over draws of the se the method reported
    realized_g  = sd over draws of beta_hat

Under the null the true slope is zero, so realized_g IS the estimator's error.
A method whose se means what it says has claimed_g / realized_g = 1 for every
gene. Standard errors are recovered from each arm's own reported quantities in
the right form -- mixQTL's stat is a signed Z so its se is |beta|/|stat|;
hapmixQTL's is T^2 and RASQUAL's is a chi-square, so both take the square root.

TWO THINGS MATTER AND THE MEDIAN SHOWS ONLY ONE. A se that is 15% low on average
but swings between half and double per gene is far worse than one uniformly 15%
low, because the second is a fixable bias and the first is unusable per gene. So
the spread of the ratio is reported alongside its centre.

With 30 draws, sd(beta) carries about 1/sqrt(2*29) = 13% relative uncertainty of
its own, which inflates the observed spread. The variance of log-ratio is
decomposed against that floor rather than quoted raw.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
LONG = D / 'realized_variance_20260924/null_long.tsv'

lg = pd.read_csv(LONG, sep='\t')
n_draw = int(lg.perm.max()) + 1
strata = pd.read_csv(D / 'genes_59_strata_20260923.tsv', sep='\t').set_index('gene')


def se_from(arm, beta, stat):
    b, s = np.abs(np.asarray(beta, float)), np.asarray(stat, float)
    with np.errstate(invalid='ignore', divide='ignore'):
        # mixQTL reports a SIGNED Z; the other two report a chi-square-like
        # statistic, so only those take a square root
        return b / np.abs(s) if arm == 'mixQTL' else b / np.sqrt(s)


rows = []
for (arm, g), sub in lg.groupby(['arm', 'gene']):
    b = sub.beta.values
    se = se_from(arm, sub.beta.values, sub.stat.values)
    ok = np.isfinite(se) & (se > 0)
    if ok.sum() < 10 or len(b) < 10:
        continue
    realized = float(np.std(b, ddof=1))
    claimed = float(np.mean(se[ok]))
    if realized <= 0:
        continue
    rows.append(dict(arm=arm, gene=g,
                     stratum=strata.loc[g, 'stratum'] if g in strata.index else '?',
                     n=int(ok.sum()), claimed=claimed, realized=realized,
                     ratio=claimed / realized))
t = pd.DataFrame(rows)
out = D / 'se_accuracy_20260924'
out.mkdir(exist_ok=True)
t.to_csv(out / 'per_gene.tsv', sep='\t', index=False)

# noise floor: with n draws, log sd(beta) has sd ~ 1/sqrt(2(n-1))
floor = 1.0 / np.sqrt(2 * (n_draw - 1))
print(f'{n_draw} draws per gene; log-ratio noise floor from estimating sd alone: '
      f'{floor:.3f}\n')
print('claimed se / realized null sd, one value per gene')
print('1.00 = the standard error means what it says\n')
res = {'n_draw': n_draw, 'log_noise_floor': float(floor)}
for arm in ('hapmixQTL', 'mixQTL', 'RASQUAL'):
    s = t[t.arm == arm]
    if not len(s):
        continue
    r = s.ratio.values
    lr = np.log(r)
    obs_sd = float(np.std(lr, ddof=1))
    true_sd = float(np.sqrt(max(obs_sd ** 2 - floor ** 2, 0.0)))
    within10 = float(np.mean((r > 1 / 1.1) & (r < 1.1)))
    within25 = float(np.mean((r > 1 / 1.25) & (r < 1.25)))
    print(f'  {arm:10s} n={len(r):3d}  median {np.median(r):5.2f}   '
          f'IQR {np.quantile(r, .25):.2f}-{np.quantile(r, .75):.2f}   '
          f'range {r.min():.2f}-{r.max():.2f}')
    print(f'             within +/-10% {within10:.0%}   within +/-25% {within25:.0%}   '
          f'gene-to-gene sd of log ratio {obs_sd:.3f} observed, '
          f'{true_sd:.3f} after removing the {floor:.3f} floor')
    res[arm] = dict(n=int(len(r)), median=float(np.median(r)),
                    q25=float(np.quantile(r, .25)), q75=float(np.quantile(r, .75)),
                    min=float(r.min()), max=float(r.max()),
                    within_10pct=within10, within_25pct=within25,
                    log_sd_observed=obs_sd, log_sd_after_floor=true_sd)

print('\nby stratum (median claimed/realized):')
print(t.pivot_table(index='stratum', columns='arm', values='ratio',
                    aggfunc='median').round(2).to_string())
(out / 'summary.json').write_text(json.dumps(res, indent=2, default=float))
print(f'\nwrote {out}')
