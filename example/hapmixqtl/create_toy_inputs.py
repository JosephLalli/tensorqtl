#!/usr/bin/env python3
"""Create deterministic fabricated inputs for the public hapmixQTL walkthrough.

This is a file-format example only.  It does not model a cohort or validate
scientific performance.
"""
import argparse
import gzip
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def create(out):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rng = np.random.RandomState(20261004)
    samples = [f"S{i:03d}" for i in range(1, 33)]
    genes = [f"GENE{i:03d}" for i in range(1, 17)]
    tx = [f"TX{i:03d}" for i in range(1, 17)]
    names = [f"{t}{suffix}" for t in tx for suffix in ("_hapA", "_hapB")]
    (out / "tx2gene.tsv").write_text("\n".join(f"{t}\t{g}" for t, g in zip(tx, genes)) + "\n")
    (out / "gene_pos.tsv").write_text("\n".join(
        f"{g}\tchr1\t{1_000_000 + 2_000_000*i}" for i, g in enumerate(genes)) + "\n")

    manifest = []
    for sample_i, sample in enumerate(samples):
        sample_dir = out / "salmon" / sample
        boot_dir = sample_dir / "aux_info" / "bootstrap"
        boot_dir.mkdir(parents=True)
        values = rng.poisson(180 + 8 * np.arange(len(names)) + sample_i, len(names)).astype(float) + 20
        pd.DataFrame({"Name": names, "Length": 1000, "EffectiveLength": 800,
                      "TPM": values / values.sum() * 1e6, "NumReads": values}).to_csv(
                          sample_dir / "quant.sf", sep="\t", index=False)
        draws = rng.poisson(values[None, :], size=(12, len(names))).astype(np.float64)
        with gzip.open(boot_dir / "names.tsv.gz", "wt") as handle:
            handle.write("\t".join(names))
        with gzip.open(boot_dir / "bootstraps.gz", "wb") as handle:
            handle.write(draws.tobytes(order="C"))
        (sample_dir / "aux_info" / "meta_info.json").write_text(json.dumps({"num_bootstraps": 12, "samp_type": "gibbs"}) + "\n")
        manifest.append(f"{sample}\t{os.path.relpath(sample_dir, Path.cwd())}")
    (out / "manifest.tsv").write_text("\n".join(manifest) + "\n")
    pd.DataFrame({"age": np.linspace(20.0, 51.0, len(samples))}, index=samples).to_csv(
        out / "sample_covariates.tsv", sep="\t")

    with open(out / "phased.vcf", "w") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples) + "\n")
        for variant in range(240):
            p = 0.12 + 0.03 * (variant % 20)
            left = rng.binomial(1, p, len(samples))
            right = rng.binomial(1, p, len(samples))
            if (left + right).sum() == 0:
                left[0] = 1
            if (left + right).sum() == 2 * len(samples):
                right[0] = 0
            gt = [f"{a}|{b}" for a, b in zip(left, right)]
            handle.write(f"chr1\t{500_000 + variant * 120_000}\tv{variant + 1}\tA\tG\t.\tPASS\t.\tGT\t" + "\t".join(gt) + "\n")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="toy")
    args = ap.parse_args(argv)
    create(args.out)
    print(f"wrote deterministic fabricated input files to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
