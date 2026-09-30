"""Aggregate completed half-read trial artifacts; never launches scans."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from half_read_io import DEPLOY, RESULTS, atomic_path

ROOT = RESULTS / 'half_read_trial_20260929'
ARCHIVE = DEPLOY / 'beta_balance_trial_20260929'
STRAINS = {"deep": "corrected_null_store_20260925", "low": "stratum30_100"}
ARMS = ("original", "half_read")
NBOOT = 2000


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def boot_pair(gene, original, half, kind, seed):
    """Paired gene bootstrap, retaining the paired value for every drawn gene."""
    ids, genes = pd.factorize(gene, sort=False)
    if len(genes) != 100:
        raise AssertionError(f"expected 100 genes, got {len(genes)}")
    vals = np.column_stack([original, half])
    sums = np.zeros((len(genes), 2))
    np.add.at(sums, ids, vals)
    draws = np.random.default_rng(seed).multinomial(len(genes), np.full(len(genes), 1 / len(genes)), size=NBOOT)
    totals = draws @ sums
    if kind == "mean_ratio":
        samples = totals[:, 1] / totals[:, 0]
    elif kind == "geom_ratio":
        logs = np.zeros((len(genes), 1))
        np.add.at(logs[:, 0], ids, np.log(half / original))
        samples = np.exp((draws @ logs)[:, 0] / len(genes))
    else:
        raise ValueError(kind)
    point = float(np.mean(half) / np.mean(original)) if kind == "mean_ratio" else float(np.exp(np.mean(np.log(half / original))))
    return {"point": point,
            "lo": float(np.quantile(samples, .025)), "hi": float(np.quantile(samples, .975)), "n_boot": NBOOT,
            "n_genes": int(len(genes))}


def nb_summary(path, stratum):
    n = pd.read_parquet(path)
    required = {"gene", "variant_id", "phi", "beta", "arm", "response", "normalized_variance", "mse", "mean",
                "coverage95", "target_coverage95", "rate_0.05", "n"}
    if missing := required - set(n):
        raise AssertionError(f"{path}: missing {sorted(missing)}")
    out = {}
    for (phi, beta), x in n.groupby(["phi", "beta"], sort=True):
        w = {a: x[x.arm == a].sort_values(["gene", "variant_id"]).reset_index(drop=True) for a in ARMS}
        if any(len(w[a]) != 100 for a in ARMS) or not w["original"][["gene", "variant_id"]].equals(w["half_read"][["gene", "variant_id"]]):
            raise AssertionError(f"{stratum} phi={phi} beta={beta}: paired 100-gene rows/keys failed")
        if not (w["original"].n == 1000).all() or not (w["half_read"].n == 1000).all():
            raise AssertionError(f"{stratum} phi={phi} beta={beta}: expected 1,000 count replicates")
        if not (np.isfinite(w["original"].response).all() and np.isfinite(w["half_read"].response).all() and
                (w["original"].response > 0).all() and (w["half_read"].response > 0).all()):
            raise AssertionError(f"{stratum} phi={phi} beta={beta}: nonpositive/nonfinite response would be an exclusion")
        for col in ("normalized_variance", "mse", "mean", "coverage95", "target_coverage95"):
            if not (np.isfinite(w["original"][col]).all() and np.isfinite(w["half_read"][col]).all()):
                raise AssertionError(f"{stratum} phi={phi} beta={beta}: nonfinite {col}")
        o, h = w["original"], w["half_read"]
        mean_v = boot_pair(o.gene, o.normalized_variance.to_numpy(), h.normalized_variance.to_numpy(), "mean_ratio", 1000 + len(out))
        geo_v = boot_pair(o.gene, o.normalized_variance.to_numpy(), h.normalized_variance.to_numpy(), "geom_ratio", 2000 + len(out))
        mse = boot_pair(o.gene, o.mse.to_numpy(), h.mse.to_numpy(), "mean_ratio", 3000 + len(out))
        bias_o, bias_h = float((o["mean"] - beta).mean()), float((h["mean"] - beta).mean())
        cov = {a: {"truth": float(w[a].coverage95.mean()), "target": float(w[a].target_coverage95.mean())} for a in ARMS}
        out[f"phi{phi:g}_beta{beta:g}"] = {
            "phi": float(phi), "beta": float(beta), "n_genes": 100, "count_replicates_per_gene": 1000,
            "no_exclusions": True, "positive_response": True,
            "normalized_variance_mean_ratio": mean_v, "normalized_variance_geomean_per_gene_ratio": geo_v,
            "raw_mse_ratio": mse, "mean_raw_bias": {"original": bias_o, "half_read": bias_h}, "coverage95": cov,
            "null_rate_0.05": {"original": float(o["rate_0.05"].mean()), "half_read": float(h["rate_0.05"].mean())} if beta == 0 else None,
            "mean_recovery": None if beta == 0 else {"original": float((o["mean"] / beta).mean()),
                                                       "half_read": float((h["mean"] / beta).mean())},
            "screen": {"variance_upper_lt_1_10": bool(mean_v["hi"] < 1.10 and geo_v["hi"] < 1.10),
                       "mse_point_le_1_and_upper_lt_1_10": bool(mse["point"] <= 1 and mse["hi"] < 1.10),
                       "mean_abs_bias_no_worse": bool(abs(bias_h) <= abs(bias_o)),
                       "coverage_93_to_97_candidate_truth_target": bool(all(.93 <= v <= .97 for v in cov["half_read"].values())),
                       "null_rate_0_04_to_0_06_candidate": None if beta != 0 else bool(.04 <= h["rate_0.05"].mean() <= .06)}}
    return out


def null_summary(path, stratum):
    n = pd.read_parquet(path)
    old = pd.read_parquet(ARCHIVE / f"null_{stratum}" / "per_gene.parquet")
    cur = n[n.arm == "original"].sort_values(["gene", "variant_id", "channel"]).reset_index(drop=True)
    old = old[old.arm == "split"].sort_values(["gene", "variant_id", "channel"]).reset_index(drop=True)
    keys = ["gene", "variant_id", "channel"]
    if not cur[keys].equals(old[keys]):
        raise AssertionError(f"{stratum}: archived original null keys differ")
    numeric = [c for c in cur.columns if c in old and pd.api.types.is_numeric_dtype(cur[c])]
    for c in numeric:
        if not np.array_equal(np.isfinite(cur[c]), np.isfinite(old[c])):
            raise AssertionError(f"{stratum}: archived original null finite pattern differs in {c}")
    max_abs = max(float(np.max(np.abs(cur[c].to_numpy(float)[np.isfinite(cur[c].to_numpy(float))] - old[c].to_numpy(float)[np.isfinite(old[c].to_numpy(float))]))) for c in numeric)
    if max_abs > 1e-10:
        raise AssertionError(f"{stratum}: archived original null reproduction max abs {max_abs}")
    result = {"archived_original_reproduced": True, "max_abs_difference": max_abs, "variance_ratio_descriptive": {}}
    for channel, g in n.groupby("channel"):
        scopes = [("all", g)]
        if channel == "allelic":
            scopes.append(("admitted", g[g.admitted]))
        for scope, h in scopes:
            a = h[h.arm == "original"].set_index(keys[:2]).variance
            b = h[h.arm == "half_read"].set_index(keys[:2]).variance
            common = a.index.intersection(b.index)
            if len(common): result["variance_ratio_descriptive"][f"{channel}:{scope}"] = float(b.loc[common].mean() / a.loc[common].mean())
    mc = pd.read_csv(path.parent / "null_monte_carlo.tsv", sep="\t")
    result["nominal_rates"] = []
    for _, r in mc[mc.arm == "half_read"].iterrows():
        result["nominal_rates"].append({"channel": r.channel, "scope": r.scope, "alpha": float(r.alpha),
            "original_rate": float(mc[(mc.arm == "original") & (mc.channel == r.channel) & (mc.scope == r.scope) & (mc.alpha == r.alpha)].rate.iloc[0]),
            "half_read_rate": float(r.rate), "paired_delta": float(r.paired_delta),
            "paired_mc_ci": [float(r.paired_delta - 1.96*r.paired_mc_se), float(r.paired_delta + 1.96*r.paired_mc_se)]})
    return result


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    ROOT = args.root
    required = [ROOT / s / f for s in STRAINS for f in ("per_gene_null.parquet", "null_monte_carlo.tsv", "independent_nb.parquet", "mapper_timing.tsv", "manifest.json")]
    missing = [str(p) for p in required if not p.exists()]
    if missing: raise SystemExit("not ready: " + ", ".join(missing))
    targets = [ROOT / "repeated_summary.json", ROOT / "REPORT.md"]
    if any(p.exists() for p in targets) and not args.overwrite: raise SystemExit("refusing to overwrite report outputs")
    causal = json.loads((ROOT / "causal_summary.json").read_text())
    summary = {"scope": "aggregation of completed trial artifacts; no scans launched", "nb": {}, "record_null": {},
               "causal_counterfactual": causal["sets"], "input_sha256": {str(p): sha(p) for p in required + [ROOT / "causal_summary.json"]}}
    for stratum in STRAINS:
        summary["nb"][stratum] = nb_summary(ROOT / stratum / "independent_nb.parquet", stratum)
        summary["record_null"][stratum] = null_summary(ROOT / stratum / "per_gene_null.parquet", stratum)
    comparison = pd.read_parquet(ROOT / "causal_comparison.parquet")
    unique_truth = comparison.drop_duplicates(["dataset_set", "scenario", "rep", "gene"])
    bad_total_truth = int((~np.isfinite(unique_truth.total_truth) | (unique_truth.total_truth == 0)).sum())
    # Required fact for the final interpretation: bias recovery did not uniformly satisfy MSE screening.
    low = causal["sets"]["stratum30_100"]["split"]["beta0.4"]["combined"]["mse"]
    if not (abs(low["half_over_observed"] - 1.2574) < .002 and abs(low["ratio_lo"] - 1.064) < .003 and abs(low["ratio_hi"] - 1.470) < .003):
        raise AssertionError("unexpected low beta0.4 combined MSE result")
    summary["causal_note"] = ("Low beta0.4 combined MSE ratio is 1.257 [1.064, 1.470]; bias improvement alone does not meet a uniform accuracy goal. "
                              f"The unique comparison units have {bad_total_truth} zero/nonfinite total-truth row before finite scoring.")
    with atomic_path(ROOT / "repeated_summary.json") as temporary:
        temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = ["# Half-read trial report", "", "This report aggregates completed outputs only. The original trial tested a uniform-precision criterion; the later 2026-09-29 decision adopted half-read split as an accuracy/precision tradeoff (implementation 6f8ad35, merge 86b947f). See ../half_read_default_adoption_20260929/verification.json and ../half_read_unit_power_pr_20260929/index.html. Numerical results below remain the original trial evidence.", "",
             "The independent NB results are a total-only oracle-response diagnostic. They contain no Salmon ambiguity, ASE channel, or resampled Gibbs weights. Record-null variance is conditional on the observed records; its transform-scale variance ratios are descriptive, not a common precision scale or evidence of gene independence.", "",
             "The causal counterfactual has 50 non-null genes per replicate and 150 units across three replicates (88 unique gene clusters). Deep total has 149 finite-scored units because one planted total truth is zero/nonfinite. Low beta=0.4 combined MSE is 1.257 [1.064, 1.470], so bias improvement alone does not meet the uniform-accuracy goal.", "",
             "## Causal counterfactual (split arm)", "", "| stratum | beta | combined recovery original -> half | combined MSE ratio [95% CI] |", "|---|---:|---:|---:|"]
    for s, setname in STRAINS.items():
        for beta in ("beta0.4", "beta0.8"):
            c = causal["sets"][setname]["split"][beta]["combined"]
            lines.append(f"| {s} | {beta[4:]} | {c['mean_recovery']['observed']:.3f} -> {c['mean_recovery']['half_read']:.3f} | {c['mse']['half_over_observed']:.3f} [{c['mse']['ratio_lo']:.3f}, {c['mse']['ratio_hi']:.3f}] |")
    lines += ["", "## Independent NB total-only diagnostic", "", "| stratum | phi | beta | normalized-variance mean ratio [CI] | MSE ratio [CI] | half coverage truth/target | half null p<0.05 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for s, values in summary["nb"].items():
        for v in values.values():
            q, m, cv = v["normalized_variance_mean_ratio"], v["raw_mse_ratio"], v["coverage95"]["half_read"]
            rate = "" if v["null_rate_0.05"] is None else f"{v['null_rate_0.05']['half_read']:.3f}"
            lines.append(f"| {s} | {v['phi']:.2f} | {v['beta']:.1f} | {q['point']:.3f} [{q['lo']:.3f}, {q['hi']:.3f}] | {m['point']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}] | {cv['truth']:.3f}/{cv['target']:.3f} | {rate} |")
    lines += ["", "## Record-null nominal rates (alpha 0.001)", "", "| stratum | channel/scope | original | half-read | paired delta [95% MC CI] |", "|---|---|---:|---:|---:|"]
    for s, v in summary["record_null"].items():
        for r in v["nominal_rates"]:
            if r["alpha"] == .001:
                lines.append(f"| {s} | {r['channel']}/{r['scope']} | {r['original_rate']:.4f} | {r['half_read_rate']:.4f} | {r['paired_delta']:.4f} [{r['paired_mc_ci'][0]:.4f}, {r['paired_mc_ci'][1]:.4f}] |")
    lines += ["", "## Matched mapper timing", "", "| stratum | original mean seconds | half-read mean seconds | pairs per scan |", "|---|---:|---:|---:|"]
    for s in STRAINS:
        timing = pd.read_csv(ROOT / s / "mapper_timing.tsv", sep="\t")
        o = timing[timing.arm == "original"].seconds.mean()
        h = timing[timing.arm == "half_read"].seconds.mean()
        lines.append(f"| {s} | {o:.3f} | {h:.3f} | {int(timing.pairs.iloc[0]):,} |")
    lines += ["", "Archived split-original null outputs reproduce numerically (finite patterns identical; maximum finite absolute difference at most 1e-10). Detailed screens and all rates are in `repeated_summary.json`."]
    with atomic_path(ROOT / "REPORT.md") as temporary:
        temporary.write_text("\n".join(lines) + "\n")
    print(f"wrote {targets[0]} and {targets[1]}")


if __name__ == "__main__": main()
