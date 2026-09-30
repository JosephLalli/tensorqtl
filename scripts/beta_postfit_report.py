#!/usr/bin/env python3
"""The post-fit adjustment page: what each adjustment of beta_postfit_adjustment.py does to effect-size recovery and
mean squared error on the half-read default. Reads OUT/postfit_<set>.json and embeds it into beta_postfit_template.html.

  python3 scripts/beta_postfit_report.py      # writes OUT/beta_postfit.html
"""
import json
import os
from pathlib import Path

OUT = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy/beta_shortfall_20260929')
SETS = {'deep': 'corrected_null_store_20260925', 'lowcov': 'stratum30_100'}
TEMPLATE = Path(__file__).with_name('beta_postfit_template.html')


def main():
    data = {k: json.loads((OUT / f'postfit_{gs}.json').read_text())['result'] for k, gs in SETS.items()}
    page = TEMPLATE.read_text().replace('/*DATA*/null', json.dumps(data, separators=(',', ':'), allow_nan=False))
    tmp = OUT / 'beta_postfit.tmp.html'
    tmp.write_text(page)
    os.replace(tmp, OUT / 'beta_postfit.html')
    print(f'wrote {OUT / "beta_postfit.html"} ({len(page):,} bytes)')


if __name__ == '__main__':
    main()
