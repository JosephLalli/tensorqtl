import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_hapmixqtl_inputs", ROOT / "scripts/prepare_hapmixqtl_inputs.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def _inputs(tmp_path):
    (tmp_path / "manifest.tsv").write_text("s1\tx\ns2\ty\ns3\tz\ns4\tw\n")
    (tmp_path / "tx2gene.tsv").write_text("tx1\tg1\ntx2\tg2\ntx3\tg3\n")
    (tmp_path / "x.vcf").write_text("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\ts4\n")
    pd.DataFrame({"age": [1., 2., 3., 4.]}, index=["s1", "s2", "s3", "s4"]).to_csv(tmp_path / "cov.tsv", sep="\t")


def test_prepare_alignment_and_provenance(tmp_path, monkeypatch):
    _inputs(tmp_path)
    samples = ["s1", "s2", "s3", "s4"]
    totals = pd.DataFrame(np.arange(12, dtype=float).reshape(3, 4) + 10,
                          index=["g1", "g2", "g3"], columns=samples)
    monkeypatch.setattr(P.H, "load_point_estimates", lambda *a: (None, None, None, totals))
    def edger(t, r, out):
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"sample": samples, "eff_lib_size": [10., 11., 12., 13.]}).to_csv(out / "edger_samples.tsv", sep="\t", index=False)
        (out / "calibration_genes.txt").write_text("g1\ng2\ng3\n")
        return np.array([10., 11., 12., 13.]), ["g1", "g2", "g3"]
    monkeypatch.setattr(P.H, "edger_normalize", edger)
    monkeypatch.setattr(P.C, "genotype_pcs", lambda v, s, n_pc: pd.DataFrame({"geno_pc1": [-1., 1., -1., 1.]}, index=s) if n_pc else pd.DataFrame(index=s))
    monkeypatch.setattr(P.C, "expression_pcs_point", lambda *a, **k: np.array([[1.], [-1.], [-1.], [1.]])[:, :k["n_pc"]])
    cov = P.prepare(tmp_path / "manifest.tsv", tmp_path / "tx2gene.tsv", tmp_path / "x.vcf", tmp_path / "prepared", tmp_path / "cov.tsv", n_expr_pc=1, n_geno_pc=1)
    meta = json.loads((tmp_path / "prepared" / "covariate_build.json").read_text())
    assert pd.read_csv(cov, sep="\t", index_col=0).index.tolist() == samples
    assert meta["expression_pc_unit"] == "half_read_log_cpm"
    P.H.check_covariate_provenance(cov, ["g1", "g2", "g3"], np.array([10., 11., 12., 13.]), samples)
    monkeypatch.setattr(P.C, "genotype_pcs", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run for zero PCs")))
    P.prepare(tmp_path / "manifest.tsv", tmp_path / "tx2gene.tsv", tmp_path / "x.vcf", tmp_path / "prepared_zero", tmp_path / "cov.tsv", n_expr_pc=0, n_geno_pc=0)


def test_rejects_duplicate_manifest_and_non_numeric_covariates(tmp_path):
    _inputs(tmp_path)
    (tmp_path / "manifest.tsv").write_text("s1\tx\ns1\ty\n")
    with pytest.raises(SystemExit, match="duplicate sample"):
        P._manifest(tmp_path / "manifest.tsv")
    pd.DataFrame({"age": ["bad", "2", "3", "4"]}, index=["s1", "s2", "s3", "s4"]).to_csv(tmp_path / "bad.tsv", sep="\t")
    with pytest.raises(SystemExit, match="finite numeric"):
        P._sample_covariates(tmp_path / "bad.tsv", ["s1", "s2", "s3", "s4"])
