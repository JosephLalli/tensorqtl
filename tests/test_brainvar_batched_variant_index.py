import importlib.util
from pathlib import Path

import pandas as pd


BENCHMARK = (
    Path(__file__).parents[1] / "benchmarks" / "brainvar_grch38_susie_batched.py"
)
SPEC = importlib.util.spec_from_file_location("brainvar_batched_benchmark", BENCHMARK)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


class RowAwareGenotypes:
    scans = 0

    @classmethod
    def variant_rows_for_contigs(cls, path, contigs):
        cls.scans += 1
        assert path == Path("variants.tsv")
        assert contigs == ("chr1", "chr2")
        return pd.DataFrame(
            {
                "chrom": ["chr1", "chr2", "chr1"],
                "pos": [10, 20, 30],
                "source_row": [0, 1, 2],
            },
            index=pd.Index(["v1", "v2", "v3"], name="id"),
        )

    @staticmethod
    def read_genotype_row_subset(path, variants):
        assert path == Path("genotypes.parquet")
        assert variants["source_row"].is_monotonic_increasing
        return pd.DataFrame(
            {"sample": variants["source_row"].to_numpy()},
            index=variants.index,
        )


def test_cached_variant_index_scans_tsv_once_and_keeps_source_rows():
    RowAwareGenotypes.scans = 0
    index = benchmark.load_contig_variant_index(
        Path("variants.tsv"),
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

    assert RowAwareGenotypes.scans == 1
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
