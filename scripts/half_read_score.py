"""Score the existing half-read (voom) refits without rerunning nominal scans.

The half-read total phenotype is already represented by the ``voom`` and
``target_voom`` configurations in beta_shortfall_refits.py.  This script only
joins those completed results to their planted dataset truths and reports the
paired observed-versus-half-read comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REFIT_ROOT = Path("/mnt/ssd/lalli/brainvar_hapmix_deploy/beta_shortfall_20260929")
SETS = {
    "corrected_null_store_20260925": Path("/mnt/ssd/lalli/brainvar_hapmix_deploy/plasmode_meier_20260927"),
    "stratum30_100": Path("/mnt/ssd/lalli/brainvar_hapmix_deploy/plasmode_lowcov_meier_20260927"),
}
KEY = ["gene", "variant_id", "scenario", "rep", "arm"]
CONFIGS = ("observed", "voom", "target", "target_voom")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def exact(a, b):
    """Exact numeric equality, treating a NaN pair as equal."""
    a, b = np.asarray(a), np.asarray(b)
    return np.array_equal(a, b, equal_nan=True)


def load_truth(root, scenario, rep, meta_genes):
    path = root / "datasets" / scenario / f"rep{rep:03d}.npz"
    with np.load(path, allow_pickle=False) as d:
        required = ("causal_variant", "is_null", "allelic_truth", "total_truth")
        missing = set(required).difference(d.files)
        if missing:
            raise AssertionError(f"{path}: missing {sorted(missing)}")
        if len(d["causal_variant"]) != len(meta_genes):
            raise AssertionError(f"{path}: gene order does not match meta.json")
        keep = ~d["is_null"].astype(bool)
        return pd.DataFrame({
            "gene": np.asarray(meta_genes, dtype=str)[keep],
            "variant_id": d["causal_variant"].astype(str)[keep],
            "allelic_truth": d["allelic_truth"].astype(float)[keep],
            "total_truth": d["total_truth"].astype(float)[keep],
        }), path


def target_combined(target, weight_source):
    """Noise-free combined slope at a run's channel inverse-SE-squared weights."""
    ase = weight_source.slope_a_se.to_numpy(float)
    wa = np.zeros(len(ase))
    aok = weight_source.allelic_admitted.to_numpy(bool) & np.isfinite(ase) & (ase > 0)
    wa[aok] = 1.0 / ase[aok] ** 2
    tse = weight_source.slope_t_se.to_numpy(float)
    wt = np.zeros(len(tse))
    tok = np.isfinite(tse) & (tse > 0)
    wt[tok] = 1.0 / tse[tok] ** 2
    den = wa + wt
    out = np.full(len(den), np.nan)
    good = den > 0
    # Assign active terms separately so an inactive 0 * NaN does not propagate.
    num = np.zeros(len(den))
    ta, tt = target.slope_a.to_numpy(float), target.slope_t.to_numpy(float)
    num[aok] = wa[aok] * ta[aok]
    num[tok] += wt[tok] * tt[tok]
    out[good] = num[good] / den[good]
    return out


def bootstrap_gene(frame, value, kind, rng, n_boot=2000):
    """Gene-cluster bootstrap: resample genes, retaining all their replicate units."""
    ids, genes = pd.factorize(frame.gene, sort=False)
    value = np.asarray(value, float)
    value2 = value[:, None] if value.ndim == 1 else value
    sums = np.zeros((len(genes), value2.shape[1]))
    np.add.at(sums, ids, value2)
    sizes = np.bincount(ids, minlength=len(genes))
    draws = rng.multinomial(len(genes), np.full(len(genes), 1 / len(genes)), size=n_boot)
    totals = draws @ sums
    denom = draws @ sizes
    if kind == "mean":
        samples = totals[:, 0] / denom
    elif kind == "mse_ratio":
        samples = totals[:, 1] / totals[:, 0]
    else:
        raise ValueError(kind)
    return {"lo": float(np.quantile(samples, 0.025)), "hi": float(np.quantile(samples, 0.975)),
            "n_gene_clusters": int(len(genes)), "n_units": int(len(frame)), "n_boot": n_boot}


def describe_pair(frame, channel, rng):
    truth_col = "total_truth" if channel == "total" else "allelic_truth"
    obs = frame[f"observed_{channel}"].to_numpy(float)
    half = frame[f"voom_{channel}"].to_numpy(float)
    truth = frame[truth_col].to_numpy(float)
    se_obs = frame[f"observed_{channel}_se"].to_numpy(float)
    se_half = frame[f"voom_{channel}_se"].to_numpy(float)
    finite = np.isfinite(obs) & np.isfinite(half) & np.isfinite(truth) & (truth != 0)
    if channel == "allelic":
        finite &= frame.observed_allelic_admitted.to_numpy(bool)
    finite_se = finite & np.isfinite(se_obs) & np.isfinite(se_half) & (se_obs > 0) & (se_half > 0)
    x = frame.loc[finite].copy()
    oo, hh, tt = obs[finite], half[finite], truth[finite]
    recovery_o, recovery_h = oo / tt, hh / tt
    bias_o, bias_h = oo - tt, hh - tt
    mse_o, mse_h = (oo - tt) ** 2, (hh - tt) ** 2
    out = {
        "channel": channel,
        "scope": "admitted_ASE_only" if channel == "allelic" else "all_finite_combined_or_total",
        "n_finite": int(finite.sum()),
        "mean_bias": {"observed": float(bias_o.mean()), "half_read": float(bias_h.mean()),
                      "half_over_observed": float(bias_h.mean() / bias_o.mean()) if bias_o.mean() != 0 else None},
        "mean_recovery": {"observed": float(recovery_o.mean()), "half_read": float(recovery_h.mean()),
                          "half_over_observed": float(recovery_h.mean() / recovery_o.mean())},
        "mse": {"observed": float(mse_o.mean()), "half_read": float(mse_h.mean()),
                "half_over_observed": float(mse_h.mean() / mse_o.mean()) if mse_o.mean() != 0 else None},
    }
    out["mean_recovery"].update(bootstrap_gene(x, recovery_h, "mean", rng))
    out["mse"].update(bootstrap_gene(x, np.column_stack((mse_o, mse_h)),
                                      "mse_ratio", rng))
    # The recovery CI is for the half-read mean; the ratio is intentionally separate.
    out["mean_recovery"].update({"half_read_lo": out["mean_recovery"].pop("lo"),
                                 "half_read_hi": out["mean_recovery"].pop("hi")})
    out["mse"].update({"ratio_lo": out["mse"].pop("lo"), "ratio_hi": out["mse"].pop("hi")})
    if finite_se.any():
        z_o = np.sign(truth[finite_se]) * obs[finite_se] / se_obs[finite_se]
        z_h = np.sign(truth[finite_se]) * half[finite_se] / se_half[finite_se]
        out["raw_se"] = {"n_finite": int(finite_se.sum()), "observed": float(se_obs[finite_se].mean()),
                         "half_read": float(se_half[finite_se].mean()),
                         "half_over_observed": float(se_half[finite_se].mean() / se_obs[finite_se].mean())}
        out["signed_z_paired_difference"] = {"n_finite": int(finite_se.sum()),
                                               "half_minus_observed_mean": float((z_h - z_o).mean())}
    return out


def oracle_se_diagnostic(frame):
    """Effect-normalized SE display; it is not repeated-sample precision."""
    truth = frame.total_truth.to_numpy(float)
    response_orig = frame.target_total.to_numpy(float) / truth
    response_half = frame.target_voom_total.to_numpy(float) / truth
    se_orig, se_half = frame.observed_total_se.to_numpy(float), frame.voom_total_se.to_numpy(float)
    support = (np.isfinite(truth) & np.isfinite(response_orig) & np.isfinite(response_half) &
               np.isfinite(se_orig) & np.isfinite(se_half) & (response_orig > 0.1) &
               (response_half > 0.1))
    orig, half = se_orig[support] / response_orig[support], se_half[support] / response_half[support]
    return {"label": "oracle estimated-SE response normalization; not empirical repeated precision",
            "support": "shared finite target/truth and target_voom/truth responses > 0.1, with both finite SEs",
            "n": int(support.sum()), "observed_mean": float(orig.mean()) if len(orig) else None,
            "half_read_mean": float(half.mean()) if len(half) else None,
            "half_over_observed": float(half.mean() / orig.mean()) if len(orig) and orig.mean() else None}


def score_set(name, root):
    refit = REFIT_ROOT / f"refits_{name}.parquet"
    meta_path = root / "datasets" / "meta.json"
    meta = json.loads(meta_path.read_text())
    f = pd.read_parquet(refit)
    if set(CONFIGS).difference(f.config.unique()):
        raise AssertionError(f"{refit}: required refit configuration is absent")
    f = f[f.config.isin(CONFIGS)].copy()
    if f.duplicated(KEY + ["config"]).any():
        raise AssertionError(f"{refit}: duplicate causal config keys")
    pieces, truth_paths = [], []
    for (scenario, rep), g in f.groupby(["scenario", "rep"], sort=True):
        t, p = load_truth(root, scenario, int(rep), meta["genes"])
        truth_paths.append(p)
        for arm, a in g.groupby("arm", sort=True):
            w = {c: a[a.config == c].set_index(KEY).sort_index() for c in CONFIGS}
            index = w["observed"].index
            for c in CONFIGS[1:]:
                if not index.equals(w[c].index):
                    raise AssertionError(f"{name} {scenario} rep {rep} {arm}: {c} keys misalign observed")
            for col in ("slope_a", "slope_a_se", "allelic_admitted"):
                if not exact(w["observed"][col], w["voom"][col]):
                    raise AssertionError(f"{name} {scenario} rep {rep} {arm}: observed/voom {col} changed")
            base = w["observed"].reset_index()[KEY].merge(t, on=["gene", "variant_id"], how="left",
                                                            validate="1:1", indicator="_truth_match")
            if (base._truth_match != "both").any():
                raise AssertionError(f"{name} {scenario} rep {rep} {arm}: truth key mismatch")
            base = base.drop(columns="_truth_match")
            target_weights = {"target": "observed", "target_voom": "voom"}
            for c, q in w.items():
                q = q.reset_index()
                if c in target_weights:
                    source = w[target_weights[c]].reset_index()
                    combined = target_combined(q, source)
                    base[f"{c}_combined"] = combined
                else:
                    base[f"{c}_combined"] = q.slope.to_numpy(float)
                base[f"{c}_combined_se"] = q.slope_se.to_numpy(float)
                base[f"{c}_total"] = q.slope_t.to_numpy(float)
                base[f"{c}_total_se"] = q.slope_t_se.to_numpy(float)
                base[f"{c}_allelic"] = q.slope_a.to_numpy(float)
                base[f"{c}_allelic_se"] = q.slope_a_se.to_numpy(float)
                base[f"{c}_allelic_admitted"] = q.allelic_admitted.to_numpy(bool)
            pieces.append(base)
    return pd.concat(pieces, ignore_index=True), [refit, meta_path, *truth_paths]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true", help="replace this script's two output files")
    args = ap.parse_args()
    targets = [args.output / "causal_comparison.parquet", args.output / "causal_summary.json"]
    if any(p.exists() for p in targets) and not args.overwrite:
        raise SystemExit("refusing to overwrite existing scorer output; pass --overwrite")
    args.output.mkdir(parents=True, exist_ok=True)
    parts, inputs = [], []
    for name, root in SETS.items():
        part, used = score_set(name, root)
        part.insert(0, "dataset_set", name)
        parts.append(part)
        inputs.extend(used)
    comparison = pd.concat(parts, ignore_index=True)
    summary = {"method": "completed voom refits: total log2((counts + 0.5)/(library + 1)*1e6)",
               "comparison": "observed versus half-read total pseudocount; no new nominal scans",
               "primary_arm": "split", "sensitivity_arm": "unit", "bootstrap": "2000 gene-cluster resamples",
               "target_combined": "target uses observed and target_voom uses voom channel inverse-SE-squared weights",
               "ase_scope": "ASE summaries require allelic_admitted; combined and total include all finite rows",
               "oracle_estimated_se_diagnostic": "effect-normalized only; not empirical repeated precision",
               "input_sha256": {str(p): digest(p) for p in sorted(set(inputs))}, "sets": {}}
    for (name, arm, scenario), g in comparison.groupby(["dataset_set", "arm", "scenario"], sort=True):
        d = summary["sets"].setdefault(name, {}).setdefault(arm, {})
        per_rep = g.groupby("rep").size()
        if not (per_rep == 50).all():
            raise AssertionError(f"{name} {arm} {scenario}: expected 50 non-null genes per replicate, got {per_rep.to_dict()}")
        d[scenario] = {"combined": describe_pair(g, "combined", np.random.default_rng(179 + len(d))),
                       "total": describe_pair(g, "total", np.random.default_rng(719 + len(d))),
                       "allelic_admitted": describe_pair(g, "allelic", np.random.default_rng(991 + len(d))),
                       "oracle_total_estimated_se": oracle_se_diagnostic(g),
                       "n_ase_admitted": int(g.observed_allelic_admitted.sum()),
                       "n_nonnull_genes_per_replicate": 50, "n_units_across_three_replicates": int(len(g)),
                       "n_unique_gene_clusters": int(g.gene.nunique())}
    comparison.to_parquet(targets[0], index=False)
    targets[1].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {targets[0]} ({len(comparison)} rows)")
    print(f"wrote {targets[1]}")


if __name__ == "__main__":
    main()
