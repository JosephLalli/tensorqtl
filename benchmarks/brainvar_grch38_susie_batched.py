#!/usr/bin/env python3
"""Benchmark gene-batched ordinary SuSiE on the BrainVar GRCh38 eGene set.

This is deliberately a benchmark, not a replacement for the validated
SuSiE-ash analysis. It reuses that analysis's immutable input freeze and
preprocessing contract, writes only to a fresh output root, and checkpoints
results by chromosome.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch


CONTIGS = tuple(f"chr{i}" for i in range(1, 23))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensorqtl-source", type=Path, required=True)
    parser.add_argument("--adapter-source", type=Path, required=True)
    parser.add_argument("--genotype-parquet", type=Path, required=True)
    parser.add_argument("--variant-table", type=Path, required=True)
    parser.add_argument("--phenotype-bed", type=Path, required=True)
    parser.add_argument("--covariates", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--permutation-results", type=Path, required=True)
    parser.add_argument("--validated-summary", type=Path, required=True)
    parser.add_argument("--provenance-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contigs", nargs="+", default=list(CONTIGS))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--scalar-baseline-genes", type=int, default=128)
    parser.add_argument("--effects", type=int, default=10)
    parser.add_argument("--window", type=int, default=1_000_000)
    parser.add_argument("--maf-threshold", type=float, default=0.05)
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--minimum-absolute-correlation", type=float, default=0.5)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--max-genes", type=int)
    parser.add_argument("--allow-dirty-source", action="store_true")
    return parser.parse_args()


def atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_parquet(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    table.to_parquet(temporary, index=False)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def git_state(path: Path) -> dict[str, object]:
    commit = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain"],
        text=True,
    )
    return {"commit": commit, "clean": not bool(status.strip())}


def validate_frozen_inputs(
    arguments: argparse.Namespace,
    provenance: dict[str, object],
) -> None:
    declared = {
        "covariates": arguments.covariates,
        "genotype_parquet": arguments.genotype_parquet,
        "permutation_results": arguments.permutation_results,
        "phenotype_bed": arguments.phenotype_bed,
        "sample_manifest": arguments.sample_manifest,
        "variant_table": arguments.variant_table,
    }
    records = provenance["inputs"]
    for name, path in declared.items():
        record = records[name]
        resolved = path.resolve()
        if resolved != Path(record["path"]).resolve():
            raise ValueError(f"{name} path differs from the frozen manifest")
        stat = resolved.stat()
        if stat.st_size != int(record["size_bytes"]):
            raise ValueError(f"{name} size differs from the frozen manifest")
        if stat.st_mtime_ns != int(record["mtime_ns"]):
            raise ValueError(f"{name} mtime differs from the frozen manifest")


def load_modules(arguments: argparse.Namespace):
    tensorqtl_package = arguments.tensorqtl_source / "tensorqtl"
    adapter_package = arguments.adapter_source / "src"
    for path in (str(tensorqtl_package), str(adapter_package)):
        if path not in sys.path:
            sys.path.insert(0, path)
    susie = importlib.import_module("susie")
    core = importlib.import_module("core")
    finemapping = importlib.import_module("brainvar_eqtl.finemapping")
    genotypes = importlib.import_module("brainvar_eqtl.genotypes")
    runner = importlib.import_module("brainvar_eqtl.runner")
    if not hasattr(susie, "susie_batched"):
        raise RuntimeError("TensorQTL source does not provide susie_batched")
    return susie, core, finemapping, genotypes, runner


def map_genotype_samples(
    genotypes: pd.DataFrame,
    sample_manifest: Path,
    phenotype_samples: pd.Index,
) -> pd.DataFrame:
    manifest = pd.read_csv(sample_manifest, sep="\t")
    required = {"SubjectID", "matchingDNALibrary"}
    if required - set(manifest):
        raise ValueError("Sample manifest lacks fine-mapping identifiers")
    source = pd.Index(manifest["matchingDNALibrary"].astype(str))
    analysis = pd.Index(manifest["SubjectID"].astype(str))
    if source.has_duplicates or analysis.has_duplicates:
        raise ValueError("Sample manifest identifiers are duplicated")
    if set(genotypes.columns.astype(str)) != set(source):
        raise ValueError("Genotype samples differ from sample manifest")
    mapped = genotypes.loc[:, source].copy(deep=False)
    mapped.columns = analysis
    if not mapped.columns.equals(phenotype_samples):
        raise ValueError("Mapped genotype and phenotype sample orders differ")
    return mapped


def locus_bounds(position: pd.Series, window: int) -> tuple[int, int]:
    if "pos" in position:
        return max(1, int(position["pos"]) - window), int(position["pos"]) + window
    return max(1, int(position["start"]) - window), int(position["end"]) + window


def credible_set_signature(fit: dict[str, object]) -> str:
    sets = fit.get("sets")
    if not isinstance(sets, dict) or sets.get("cs") is None:
        return "[]"
    signature = [
        [str(name), sorted(np.asarray(members, dtype=int).tolist())]
        for name, members in sorted(
            sets["cs"].items(),
            key=lambda item: int(item[0].replace("L", "")),
        )
    ]
    return json.dumps(signature, separators=(",", ":"))


def select_scalar_baseline(
    validated_summary: pd.DataFrame,
    selected_ids: set[str],
    count: int,
) -> set[str]:
    if count <= 0:
        return set()
    candidates = validated_summary.loc[
        validated_summary["phenotype_id"].astype(str).isin(selected_ids)
    ].sort_values(
        ["variants_fine_mapped", "phenotype_id"],
        kind="stable",
    )
    count = min(count, len(candidates))
    positions = np.linspace(0, len(candidates) - 1, count, dtype=int)
    return set(candidates.iloc[np.unique(positions)]["phenotype_id"].astype(str))


def load_contig_variant_index(
    variant_table: Path,
    contigs: tuple[str, ...],
    genotypes_module,
) -> dict[str, pd.DataFrame] | None:
    """Load and validate the variant TSV once, indexed by contig.

    The immutable BrainVar adapter exposes the row-aware metadata loader used
    by its per-contig API.  Keeping the source-row column here lets its
    Parquet reader retain the same order and bounds validation for each
    contig, without rereading the large TSV for every chromosome.
    """
    if not all(
        hasattr(genotypes_module, name)
        for name in ("variant_rows_for_contigs", "read_genotype_row_subset")
    ):
        return None
    variants = genotypes_module.variant_rows_for_contigs(variant_table, contigs)
    return {
        contig: variants.loc[variants["chrom"].eq(contig)].copy()
        for contig in contigs
    }


def build_genotype_contig_slice_from_index(
    genotype_parquet: Path,
    contig: str,
    contig_variant_index: dict[str, pd.DataFrame],
    genotypes_module,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Use cached row-aware metadata with the adapter's Parquet loader."""
    variants = contig_variant_index[contig]
    if variants.empty:
        raise ValueError(f"No cached variants found on {contig}")
    genotypes = genotypes_module.read_genotype_row_subset(
        genotype_parquet,
        variants,
    )
    output_variants = variants.drop(columns="source_row")
    summary = {
        "contigs": [contig],
        "variants": int(len(output_variants)),
        "samples": int(genotypes.shape[1]),
        "source_first_row": int(variants["source_row"].iloc[0]),
        "source_last_row": int(variants["source_row"].iloc[-1]),
    }
    return genotypes, output_variants, summary


def prepare_contig(
    contig: str,
    egenes: pd.DataFrame,
    phenotypes: pd.DataFrame,
    positions: pd.DataFrame,
    arguments: argparse.Namespace,
    finemapping,
    genotypes_module,
    contig_variant_index: dict[str, pd.DataFrame] | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    load_started = time.perf_counter()
    if contig_variant_index is None:
        genotype_table, variants, genotype_summary = (
            genotypes_module.build_genotype_contig_slice(
                arguments.genotype_parquet,
                arguments.variant_table,
                contigs=(contig,),
            )
        )
    else:
        genotype_table, variants, genotype_summary = (
            build_genotype_contig_slice_from_index(
                arguments.genotype_parquet,
                contig,
                contig_variant_index,
                genotypes_module,
            )
        )
    genotype_table = genotypes_module.normalize_missing_dosages(genotype_table)
    genotype_table = map_genotype_samples(
        genotype_table,
        arguments.sample_manifest,
        phenotypes.columns,
    )
    load_seconds = time.perf_counter() - load_started

    prep_started = time.perf_counter()
    genotype_values, retained_ids, _ = (
        finemapping.impute_and_filter_genotypes(
            genotype_table,
            maf_threshold=arguments.maf_threshold,
        )
    )
    retained_index = pd.Index(retained_ids.astype(str))
    filtered_variants = variants.loc[retained_index]
    if not filtered_variants.index.equals(retained_index):
        raise ValueError(f"{contig} MAF filtering changed variant order")
    variant_positions = filtered_variants["pos"].to_numpy(dtype=np.int64)
    if np.any(np.diff(variant_positions) < 0):
        raise ValueError(f"{contig} variants are not position sorted")
    prepared: list[dict[str, object]] = []
    for egene in egenes.itertuples(index=False):
        phenotype_id = str(egene.phenotype_id)
        lower, upper = locus_bounds(
            positions.loc[phenotype_id],
            arguments.window,
        )
        left = int(np.searchsorted(variant_positions, lower, side="left"))
        right = int(np.searchsorted(variant_positions, upper, side="right"))
        if right <= left:
            raise ValueError(
                f"{phenotype_id} has no variants after MAF filtering"
            )
        prepared.append(
            {
                "phenotype_id": phenotype_id,
                # NumPy slicing produces a view into the one filtered contig
                # matrix. Overlapping cis windows therefore share storage
                # until they are packed into a GPU batch.
                "X_variant_major": genotype_values[left:right],
                "y": phenotypes.loc[phenotype_id].to_numpy(
                    dtype=np.float32,
                ),
                "variant_count": right - left,
            }
        )
    prep_seconds = time.perf_counter() - prep_started
    prepared.sort(key=lambda job: (int(job["variant_count"]), str(job["phenotype_id"])))
    del genotype_table, variants, filtered_variants
    gc.collect()
    return prepared, {
        "genotype_load_seconds": load_seconds,
        "genotype_preparation_seconds": prep_seconds,
        "genotype_slice": genotype_summary,
    }


def fit_contig(
    contig: str,
    jobs: list[dict[str, object]],
    scalar_ids: set[str],
    covariates: pd.DataFrame,
    arguments: argparse.Namespace,
    susie,
    core,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    fit_seconds = 0.0
    packing_seconds = 0.0
    residualization_seconds = 0.0
    scalar_seconds = 0.0
    peak_memory = 0
    batch_count = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    covariate_t = torch.as_tensor(
        covariates.to_numpy(dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    residualizer = core.Residualizer(covariate_t)
    fit_options = {
        "L": arguments.effects,
        "coverage": arguments.coverage,
        "min_abs_corr": arguments.minimum_absolute_correlation,
        "max_iter": arguments.max_iterations,
        "tol": arguments.tolerance,
        "estimate_residual_variance": True,
        "estimate_prior_variance": True,
        "estimate_prior_method": "EM",
        "standardize": True,
        "intercept": True,
    }
    for start in range(0, len(jobs), arguments.batch_size):
        batch = jobs[start : start + arguments.batch_size]
        variant_counts = [int(job["variant_count"]) for job in batch]
        p_max = max(variant_counts)
        sample_count = int(np.asarray(batch[0]["y"]).size)
        padding_efficiency = (
            sum(variant_counts)
            / (len(batch) * p_max)
        )
        use_pinned_staging = device.type == "cuda"
        packing_started = time.perf_counter()
        host_X = torch.zeros(
            (len(batch), p_max, sample_count),
            dtype=torch.float32,
            pin_memory=use_pinned_staging,
        )
        host_y = torch.empty(
            (len(batch), sample_count),
            dtype=torch.float32,
            pin_memory=use_pinned_staging,
        )
        for local_i, (job, p) in enumerate(zip(batch, variant_counts)):
            source_X = torch.from_numpy(
                np.asarray(job["X_variant_major"])
            )
            if source_X.shape != (p, sample_count):
                raise ValueError(
                    f"{job['phenotype_id']} has an invalid genotype view"
                )
            host_X[local_i, :p].copy_(source_X)
            host_y[local_i].copy_(
                torch.from_numpy(np.asarray(job["y"], dtype=np.float32))
            )
        packed_X = host_X.to(device=device, non_blocking=use_pinned_staging)
        packed_y = host_y.to(device=device, non_blocking=use_pinned_staging)
        synchronize()
        packing_seconds += time.perf_counter() - packing_started

        residualization_started = time.perf_counter()
        for local_i, p in enumerate(variant_counts):
            packed_X[local_i, :p].copy_(
                residualizer.transform(packed_X[local_i, :p])
            )
            outcome_t = packed_y[local_i].reshape(1, -1)
            packed_y[local_i].copy_(
                residualizer.transform(outcome_t).reshape(-1)
            )
        synchronize()
        residualization_seconds += (
            time.perf_counter() - residualization_started
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        synchronize()
        started = time.perf_counter()
        fits = susie.susie_batched_packed(
            packed_X,
            packed_y,
            variant_counts,
            **fit_options,
        )
        synchronize()
        batch_seconds = time.perf_counter() - started
        fit_seconds += batch_seconds
        batch_count += 1
        if torch.cuda.is_available():
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())

        for local_i, (job, fit, p) in enumerate(
                zip(batch, fits, variant_counts)):
            phenotype_id = str(job["phenotype_id"])
            rows.append(
                {
                    "phenotype_id": phenotype_id,
                    "chromosome": contig,
                    "variants_fine_mapped": int(job["variant_count"]),
                    "batch_index": batch_count,
                    "batch_genes": len(batch),
                    "batch_max_variants": p_max,
                    "batch_padding_efficiency": padding_efficiency,
                    "batch_seconds": batch_seconds,
                    "converged": bool(fit["converged"]),
                    "iterations": int(fit["niter"]),
                    "residual_variance": float(fit["sigma2"]),
                    "maximum_pip": float(np.max(fit["pip"])),
                    "credible_sets": (
                        0
                        if fit.get("sets", {}).get("cs") is None
                        else len(fit["sets"]["cs"])
                    ),
                    "credible_set_signature": credible_set_signature(fit),
                }
            )
            if phenotype_id not in scalar_ids:
                continue
            synchronize()
            scalar_started = time.perf_counter()
            scalar = susie.susie(
                packed_X[local_i, :p].transpose(0, 1).contiguous(),
                packed_y[local_i].reshape(-1, 1),
                **fit_options,
            )
            synchronize()
            elapsed = time.perf_counter() - scalar_started
            scalar_seconds += elapsed
            comparisons.append(
                {
                    "phenotype_id": phenotype_id,
                    "chromosome": contig,
                    "variants_fine_mapped": int(job["variant_count"]),
                    "scalar_seconds": elapsed,
                    "batched_converged": bool(fit["converged"]),
                    "scalar_converged": bool(scalar["converged"]),
                    "iteration_difference": int(fit["niter"]) - int(scalar["niter"]),
                    "maximum_absolute_pip_difference": float(
                        np.max(np.abs(fit["pip"] - scalar["pip"]))
                    ),
                    "maximum_absolute_fitted_difference": float(
                        torch.max(torch.abs(fit["fitted"] - scalar["fitted"]))
                    ),
                    "maximum_absolute_alpha_difference": float(
                        torch.max(torch.abs(fit["alpha"] - scalar["alpha"]))
                    ),
                    "residual_variance_difference": (
                        float(fit["sigma2"]) - float(scalar["sigma2"])
                    ),
                    "credible_sets_identical": (
                        credible_set_signature(fit)
                        == credible_set_signature(scalar)
                    ),
                }
            )
            del scalar
        del fits, packed_X, packed_y, host_X, host_y
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del residualizer, covariate_t
    return pd.DataFrame(rows), pd.DataFrame(comparisons), {
        "batches": batch_count,
        "fit_seconds": fit_seconds,
        "packing_seconds": packing_seconds,
        "residualization_seconds": residualization_seconds,
        "scalar_baseline_seconds": scalar_seconds,
        "peak_gpu_memory_bytes": peak_memory,
    }


def distribution(values: pd.Series) -> dict[str, float]:
    return {
        "minimum": float(values.min()),
        "p25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "p75": float(values.quantile(0.75)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
    }


def main() -> int:
    arguments = parse_arguments()
    if set(arguments.contigs) - set(CONTIGS):
        raise ValueError("Only chr1 through chr22 are supported")
    if arguments.batch_size < 1 or arguments.scalar_baseline_genes < 0:
        raise ValueError("Batch size and scalar baseline count are invalid")
    if (arguments.output / "FINAL_BENCHMARK.json").exists():
        raise FileExistsError(
            f"Benchmark is already complete: {arguments.output}"
        )
    arguments.output.mkdir(parents=True, exist_ok=True)

    source_state = git_state(arguments.tensorqtl_source)
    if not source_state["clean"] and not arguments.allow_dirty_source:
        raise RuntimeError("TensorQTL benchmark source must have a clean worktree")
    susie, core, finemapping, genotypes_module, runner = load_modules(arguments)

    provenance = json.loads(arguments.provenance_manifest.read_text())
    validate_frozen_inputs(arguments, provenance)
    permutation = pd.read_csv(arguments.permutation_results, sep="\t")
    egenes = finemapping.select_egenes(permutation, fdr=arguments.fdr)
    phenotypes, positions = runner._read_phenotype_bed(arguments.phenotype_bed)
    covariates = runner._read_covariates(arguments.covariates)
    if not phenotypes.columns.equals(covariates.index):
        raise ValueError("Phenotype and covariate sample orders differ")
    egenes = egenes.loc[
        egenes["phenotype_id"].astype(str).isin(positions.index)
        & positions.loc[egenes["phenotype_id"], "chr"].isin(arguments.contigs).to_numpy()
    ].copy()
    if arguments.max_genes is not None:
        egenes = egenes.sort_values(
            ["pval_beta", "phenotype_id"],
            kind="stable",
        ).head(arguments.max_genes)
    selected_ids = set(egenes["phenotype_id"].astype(str))
    validated = pd.read_parquet(arguments.validated_summary)
    validated_ids = set(validated["phenotype_id"].astype(str))
    if not selected_ids <= validated_ids:
        raise ValueError("Selected eGenes differ from the validated GRCh38 set")
    if (
        arguments.max_genes is None
        and set(arguments.contigs) == set(CONTIGS)
        and selected_ids != validated_ids
    ):
        raise ValueError("Full benchmark selection is not the 2,888 validated eGenes")
    scalar_ids = select_scalar_baseline(
        validated,
        selected_ids,
        arguments.scalar_baseline_genes,
    )
    active_contigs = tuple(
        contig
        for contig in arguments.contigs
        if (positions["chr"].eq(contig) & positions.index.isin(selected_ids)).any()
    )

    run_configuration = {
        "status": "running",
        "started_at_epoch_seconds": time.time(),
        "parameters": {
            "batch_size": arguments.batch_size,
            "coverage": arguments.coverage,
            "effects": arguments.effects,
            "fdr": arguments.fdr,
            "maf_threshold": arguments.maf_threshold,
            "max_iterations": arguments.max_iterations,
            "minimum_absolute_correlation": (
                arguments.minimum_absolute_correlation
            ),
            "scalar_baseline_genes": len(scalar_ids),
            "tolerance": arguments.tolerance,
            "window": arguments.window,
        },
        "selection": {
            "contigs": arguments.contigs,
            "genes": len(egenes),
        },
        "tensorqtl": source_state,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "susie_sha256": sha256(
                arguments.tensorqtl_source / "tensorqtl" / "susie.py"
            ),
            "benchmark_sha256": sha256(Path(__file__)),
        },
        "input_provenance": provenance["inputs"],
    }
    configuration_path = arguments.output / "RUN_CONFIGURATION.json"
    if configuration_path.exists():
        existing = json.loads(configuration_path.read_text())
        identity_keys = (
            "parameters",
            "selection",
            "tensorqtl",
            "software",
            "input_provenance",
        )
        if any(existing[key] != run_configuration[key] for key in identity_keys):
            raise ValueError("Resume configuration differs from the existing run")
        run_configuration = existing
    else:
        atomic_json(run_configuration, configuration_path)

    wall_started = time.perf_counter()
    contig_manifests = []
    contig_variant_index: dict[str, pd.DataFrame] | None = None
    variant_index_load_seconds = 0.0
    for contig in arguments.contigs:
        contig_ids = positions.index[
            positions["chr"].eq(contig)
            & positions.index.isin(selected_ids)
        ]
        contig_egenes = egenes.loc[
            egenes["phenotype_id"].astype(str).isin(contig_ids)
        ].copy()
        if contig_egenes.empty:
            continue
        contig_egenes = contig_egenes.sort_values(
            ["pval_beta", "phenotype_id"],
            kind="stable",
        )
        contig_output = arguments.output / "contigs" / contig
        contig_manifest_path = contig_output / "manifest.json"
        if contig_manifest_path.exists():
            contig_manifest = json.loads(contig_manifest_path.read_text())
            if (
                contig_manifest.get("status") != "complete"
                or int(contig_manifest["genes"]) != len(contig_egenes)
                or not (contig_output / "gene_results.parquet").exists()
                or not (contig_output / "scalar_comparisons.parquet").exists()
            ):
                raise ValueError(f"Incomplete or inconsistent checkpoint: {contig}")
            contig_manifests.append(contig_manifest)
            print(f"SKIP {contig} complete", flush=True)
            continue
        print(f"PREPARE {contig} genes={len(contig_egenes)}", flush=True)
        if contig_variant_index is None:
            index_started = time.perf_counter()
            contig_variant_index = load_contig_variant_index(
                arguments.variant_table,
                active_contigs,
                genotypes_module,
            )
            variant_index_load_seconds = time.perf_counter() - index_started
        prepared, prep_metrics = prepare_contig(
            contig,
            contig_egenes,
            phenotypes,
            positions,
            arguments,
            finemapping,
            genotypes_module,
            contig_variant_index,
        )
        prep_metrics["genotype_load_seconds"] += variant_index_load_seconds
        prep_metrics["variant_index_load_seconds"] = (
            variant_index_load_seconds
        )
        variant_index_load_seconds = 0.0
        print(f"FIT {contig} genes={len(prepared)}", flush=True)
        gene_table, comparison_table, fit_metrics = fit_contig(
            contig,
            prepared,
            scalar_ids,
            covariates,
            arguments,
            susie,
            core,
        )
        atomic_parquet(gene_table, contig_output / "gene_results.parquet")
        atomic_parquet(
            comparison_table,
            contig_output / "scalar_comparisons.parquet",
        )
        contig_manifest = {
            "status": "complete",
            "contig": contig,
            "genes": len(gene_table),
            "scalar_comparisons": len(comparison_table),
            **prep_metrics,
            **fit_metrics,
        }
        atomic_json(contig_manifest, contig_manifest_path)
        contig_manifests.append(contig_manifest)
        print(
            f"DONE {contig} fit_seconds={fit_metrics['fit_seconds']:.3f}",
            flush=True,
        )
        del prepared, gene_table, comparison_table
        gc.collect()

    gene_files = sorted(
        (arguments.output / "contigs").glob("chr*/gene_results.parquet")
    )
    comparison_files = sorted(
        (arguments.output / "contigs").glob("chr*/scalar_comparisons.parquet")
    )
    genes = pd.concat(
        [pd.read_parquet(path) for path in gene_files],
        ignore_index=True,
    )
    comparisons = pd.concat(
        [pd.read_parquet(path) for path in comparison_files],
        ignore_index=True,
    )
    atomic_parquet(genes, arguments.output / "gene_results.parquet")
    atomic_parquet(
        comparisons,
        arguments.output / "scalar_comparisons.parquet",
    )

    total_fit_seconds = sum(float(item["fit_seconds"]) for item in contig_manifests)
    total_scalar_seconds = sum(
        float(item["scalar_baseline_seconds"]) for item in contig_manifests
    )
    total_load_seconds = sum(
        float(item["genotype_load_seconds"]) for item in contig_manifests
    )
    total_prep_seconds = sum(
        float(item["residualization_seconds"]) for item in contig_manifests
    )
    total_packing_seconds = sum(
        float(item.get("packing_seconds", 0)) for item in contig_manifests
    )
    total_genotype_preparation_seconds = sum(
        float(item.get("genotype_preparation_seconds", 0))
        for item in contig_manifests
    )
    estimated_scalar_seconds = (
        total_scalar_seconds / len(comparisons) * len(genes)
        if len(comparisons)
        else None
    )
    final_manifest = {
        **run_configuration,
        "status": "complete",
        "completed_at_epoch_seconds": time.time(),
        "wall_seconds": time.perf_counter() - wall_started,
        "counts": {
            "genes": len(genes),
            "batches": sum(int(item["batches"]) for item in contig_manifests),
            "converged": int(genes["converged"].sum()),
            "credible_sets": int(genes["credible_sets"].sum()),
            "scalar_comparisons": len(comparisons),
        },
        "timing": {
            "genotype_load_seconds": total_load_seconds,
            "genotype_preparation_seconds": (
                total_genotype_preparation_seconds
            ),
            "packing_seconds": total_packing_seconds,
            "residualization_seconds": total_prep_seconds,
            "batched_fit_seconds": total_fit_seconds,
            "batched_genes_per_second": len(genes) / total_fit_seconds,
            "scalar_baseline_seconds": total_scalar_seconds,
            "estimated_full_scalar_seconds": estimated_scalar_seconds,
            "estimated_solver_speedup": (
                estimated_scalar_seconds / total_fit_seconds
                if estimated_scalar_seconds is not None
                else None
            ),
        },
        "distributions": {
            "variants": distribution(genes["variants_fine_mapped"]),
            "iterations": distribution(genes["iterations"]),
            "padding_efficiency": distribution(
                genes["batch_padding_efficiency"]
            ),
        },
        "parity": {
            "convergence_status_matches": int(
                (
                    comparisons["batched_converged"]
                    == comparisons["scalar_converged"]
                ).sum()
            ),
            "iteration_matches": int(
                comparisons["iteration_difference"].eq(0).sum()
            ),
            "credible_sets_identical": int(
                comparisons["credible_sets_identical"].sum()
            ),
            "maximum_absolute_pip_difference": float(
                comparisons["maximum_absolute_pip_difference"].max()
            ),
            "maximum_absolute_fitted_difference": float(
                comparisons["maximum_absolute_fitted_difference"].max()
            ),
            "maximum_absolute_alpha_difference": float(
                comparisons["maximum_absolute_alpha_difference"].max()
            ),
            "maximum_absolute_residual_variance_difference": float(
                comparisons["residual_variance_difference"].abs().max()
            ),
        },
        "peak_gpu_memory_bytes": max(
            int(item["peak_gpu_memory_bytes"]) for item in contig_manifests
        ),
        "contigs": contig_manifests,
    }
    atomic_json(final_manifest, arguments.output / "FINAL_BENCHMARK.json")
    print(
        json.dumps(
            {
                "genes": len(genes),
                "batched_fit_seconds": total_fit_seconds,
                "estimated_solver_speedup": final_manifest["timing"][
                    "estimated_solver_speedup"
                ],
                "output": str(arguments.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
