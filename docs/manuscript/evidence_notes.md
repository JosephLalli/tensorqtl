# Manuscript evidence ledger notes

`evidence.tsv` records source paths and JSON keys for the current half-read simulated-effects configuration. Values are transcribed from saved result summaries or generated saved report tables, not conversational records. Each beta has 50 non-null genes per dataset across three datasets (150 units); replicate units share genes and should not be treated as independent genes. Null membership varies by dataset.

For causal-unit precision, `ratio_vs_unit` is not a common-target MSE against the generating beta: `channel_truths` constructs each arm's combined target with that arm's fitted channel weights. `ratio_vs_unit_count` substitutes common channel truths but still combines them using each arm's weights. The beta-zero null ratio is the exception recorded here: both arms have the common true target zero.

Do not include comparisons involving Gibbs weighting in both channels, `plus_one`, or deprecated arms. Native RASQUAL results are a 52-variant-per-gene truth-informed subset and must be described as retrospective ranking; returned-test null shares do not establish gene-level calibration. The historical `referee_replication_20260928` entry is retained only for its 92/135 partition and is not a current-configuration comparison.

The locally run TReCASE count-model study is a separate simulation, not independent external validation. Its matched-power values use each arm's empirical 95th-percentile null-statistic threshold.

## Raw source fingerprints

| source | SHA-256 |
| --- | --- |
| `simulated_effects_half_read_20261001/summary.json` | `0720cb6da3559c7968baa1fab0307c26b4693bb92baf67b73cb0c13473823ffc` |
| `simulated_effects_lowcov_half_read_20261001/summary.json` | `441a2dc57b45a815cb9b6c14607b80d3310ab389175b2b3e820824d6b2a139c5` |
| `stored_null_half_read_20261001/summary.json` | `531f8c56bb1bc1da4bee852d8fb92d2ed6e2898b097c38ce46ef59e88eaa6d6d` |
| `stored_null_lowcov_half_read_20261002/summary.json` | `fff398671e4f606fb9ce2023b845a1b6c1851f92865ef0ae6a0866644bba69d8` |
| `beta_recovery_current_20260930/recovery_corrected_null_store_20260925.json` | `a5bd42cc11a04aae123a0026d9281aaf3c941df3c3ad7f502718691134ca9f4b` |
| `beta_recovery_current_20260930/recovery_stratum30_100.json` | `064345ac9b1a94377a326ba7b23d0a5ce8c74473e62f8456e1cd05f424ae9f1f` |
| `combined_reference_exact_model_20260927/summary.json` | `16c568494d8d7966d92ac652d9466f9cbef996b8b264bd45e11644f2f039c864` |
| `external_benchmark_half_read_20261002/summary.json` | `9b8fec3284fc1730ca24d64b3d0e8aa5543f2df4a1e580312830019ce44daaae` |
| `referee_replication_20260928/facts.json` | `cce006d3f8a59052b9be700b543aa73447936dfb2808e83c94da53ccb85e0996` |
| `salmon_informative_reads_20260930/split_half/split_half_summary.tsv` | `738d5ba2844c59c1874fe901313d4df1e679ff77999217704f0079d0f23ad19a` |
| `simulated_effects_half_read_20261001/native/results_rasqual_subset/score.json` | `2bb718b714805573794970a530201954bad1a006525e17ab908c6aa126135ab3` |
| `simulated_effects_lowcov_half_read_20261001/native/results_rasqual_subset/score.json` | `45750be2845e6c42fb4d25604473975442cd9a3e399427d5217e620929163799` |
