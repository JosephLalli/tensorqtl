import importlib.util
from pathlib import Path
import threading

import pandas as pd


BENCHMARK = (
    Path(__file__).parents[1] / "benchmarks" / "brainvar_grch38_susie_batched.py"
)
SPEC = importlib.util.spec_from_file_location("brainvar_batched_benchmark", BENCHMARK)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


class RowAwareGenotypes:
    @staticmethod
    def read_genotype_row_subset(path, variants):
        assert path == Path("genotypes.parquet")
        assert variants["source_row"].is_monotonic_increasing
        return pd.DataFrame(
            {"sample": variants["source_row"].to_numpy()},
            index=variants.index,
        )


def test_cached_variant_index_scans_tsv_once_and_keeps_source_rows(
        tmp_path, monkeypatch):
    variant_table = tmp_path / "variants.tsv"
    variant_table.write_text(
        "id\tchrom\tpos\tindex\n"
        "v1\tchr1\t10\t0\n"
        "v2\tchr2\t20\t1\n"
        "v3\tchr1\t30\t2\n",
        encoding="utf-8",
    )
    scans = 0
    original_read_csv = pd.read_csv

    def counted_read_csv(*args, **kwargs):
        nonlocal scans
        scans += 1
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", counted_read_csv)
    index = benchmark.load_contig_variant_index(
        variant_table,
        ("chr1", "chr2"),
        RowAwareGenotypes,
    )

    chr1_genotypes, chr1_variants, chr1_summary = (
        benchmark.build_genotype_contig_slice_from_index(
            Path("genotypes.parquet"), "chr1", index, RowAwareGenotypes
        )
    )
    chr2_genotypes, chr2_variants, chr2_summary = (
        benchmark.build_genotype_contig_slice_from_index(
            Path("genotypes.parquet"), "chr2", index, RowAwareGenotypes
        )
    )

    assert scans == 1
    assert chr1_genotypes.index.tolist() == ["v1", "v3"]
    assert chr1_variants.index.tolist() == ["v1", "v3"]
    assert "source_row" not in chr1_variants
    assert chr1_summary == {
        "contigs": ["chr1"],
        "variants": 2,
        "samples": 1,
        "source_first_row": 0,
        "source_last_row": 2,
    }
    assert chr2_genotypes.index.tolist() == ["v2"]
    assert chr2_variants.index.tolist() == ["v2"]
    assert chr2_summary["source_first_row"] == 1
    assert chr2_summary["source_last_row"] == 1


def test_bounded_prefetch_uses_workers_and_preserves_consumer_order():
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def loader(value):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        if value < 2:
            barrier.wait(timeout=5)
        result = value * 10
        with lock:
            active -= 1
        return result

    observed = list(
        benchmark.iter_prefetched(range(5), loader, max_workers=2)
    )

    assert [item for item, _, _ in observed] == list(range(5))
    assert [result for _, result, _ in observed] == [
        0, 10, 20, 30, 40,
    ]
    assert maximum_active == 2
    assert all(wait_seconds >= 0 for _, _, wait_seconds in observed)


def test_synchronous_loader_preserves_order():
    observed = list(
        benchmark.iter_synchronous(range(3), lambda value: value + 1)
    )

    assert [(item, result) for item, result, _ in observed] == [
        (0, 1), (1, 2), (2, 3),
    ]
