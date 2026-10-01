#!/usr/bin/env python3
"""Build the Gibbs-draw cache that build_point_estimate_cache.py, build_covariates.py and mixQTL mode's driver read.

Writes <cache-dir>/gibbs_<key>/ with YL.npy, YR.npy, YT.npy ([genes, samples, draws], memory-mappable), genes.txt and
samples.txt. The arrays come from run_hapmixqtl_from_salmon.load_counts, the runner's own reader: YL and YR sum the
haplotype-paired transcripts of each gene, YT every transcript. key is the first 16 hex digits of the sha256 of the
manifest text, the tx2gene text and the comma-joined suffixes, so a different cohort, annotation or suffix pair gets a
different directory; the BrainVar cohort's cache, cache/gibbs_56b63c3b37ed5df8, is cohort/salmon.tsv, annot/tx2gene.tsv
and --hap-suffix _L,_R. This is the code compare_pipelines.py --cache-dir ran before that driver was retired
(2026-10-01). Loading the 92-donor cohort takes about three quarters of an hour.

  python3 scripts/build_gibbs_cache.py --salmon cohort/salmon.tsv --tx2gene annot/tx2gene.tsv \\
      --hap-suffix _L,_R --cache-dir cache
"""
import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_hapmixqtl_from_salmon as H   # noqa: E402


def cache_key(salmon, tx2gene, suffixes):
    text = Path(salmon).read_text() + Path(tx2gene).read_text() + ','.join(suffixes)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--salmon', required=True, help='manifest: sample <TAB> Salmon output directory')
    ap.add_argument('--tx2gene', required=True, help='transcript <TAB> gene')
    ap.add_argument('--hap-suffix', default='_hapA,_hapB', help="haplotype suffixes, as the runner's --hap-suffix")
    ap.add_argument('--cache-dir', required=True)
    args = ap.parse_args()
    sufs = tuple(args.hap_suffix.split(','))
    cache = Path(args.cache_dir) / f'gibbs_{cache_key(args.salmon, args.tx2gene, sufs)}'
    if (cache / 'genes.txt').exists():
        print(f'{cache} already exists; nothing to do')
        return
    genes, samples, YL, YR, YT = H.load_counts(args.salmon, args.tx2gene, sufs, None)
    print(f'{len(genes)} genes x {len(samples)} samples x {YL.shape[2]} draws')
    tmp = cache.with_name(cache.name + '.tmp')
    shutil.rmtree(tmp, ignore_errors=True)   # a previous interrupted build of this same key
    tmp.mkdir(parents=True)
    for k, v in (('YL', YL), ('YR', YR), ('YT', YT)):
        np.save(tmp / f'{k}.npy', v)
    (tmp / 'genes.txt').write_text('\n'.join(map(str, genes)))
    (tmp / 'samples.txt').write_text('\n'.join(map(str, samples)))
    os.replace(tmp, cache)
    print(f'wrote {cache}')


if __name__ == '__main__':
    main()
