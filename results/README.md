# Saved experimental evidence

This directory contains small aggregate extracts from the existing main experiment
`data/output/cleaned_full_model_notice_final_risk_overlay/` and a separately named
historical 20-seed `optimized_55` experiment. They use different configurations.
No experiment was rerun and no official result was replaced during packaging.

| File | Content |
|---|---|
| `main_results.csv` | Full-model recorded metrics in a two-column table |
| `reported_metrics.json` | The same metrics object extracted from the saved report |
| `ablation.csv` | Byte-identical copy of the eight-variant summary |
| `ablation_significance.csv` | Byte-identical copy of saved comparisons |
| `experiment_config.json` | Resolved configuration from the full-model report |
| `leakage_audit.json` | Saved audit including temporal split boundaries |
| `data_summary.json` | Saved preparation metadata/source notes and explicitly derived CSV counts/date ranges |
| `provenance.json` | Original relative paths, source hashes, transformation notes and release-file hashes |
| `optimized_55_20seed_aggregate.csv` | Original aggregate for seeds 2024–2043 under historical `optimized_55` |
| `optimized_55_20seed_summary.csv` | Original per-seed metrics for those 20 runs |
| `optimized_55_20seed_summary.json` | Original seed list, per-seed metrics and aggregate |
| `optimized_55_20seed_config.json` | Saved seed-2024 resolved config; other seeds differ only in seed/output path |
| `optimized_55_20seed_README.md` | Configuration mapping, comparisons, validation and interpretation limits |

Raw reports are not copied wholesale because they include machine-specific paths
and text-event examples. Original reports, predictions and prepared data remain
local and ignored. Aggregate counts in `data_summary.json` were derived during
packaging; they are diagnostics, not newly trained results. Date deduplication
keeps the first saved row for the count check. Original source files are not
distributed, so their hashes document provenance but do not independently prove
the historical experiment.

`experiment_config.json` preserves original Windows-relative output paths as
evidence, not as the recommended cross-platform runtime configuration. Use
`config/reproduction.yaml` for new runs. The original model report is a full
ablation run with external baselines disabled. Do not combine the table with
metrics from unrelated configurations.

For the main experiment, the full model predicts non-up on every evaluated date. Its direction accuracy
matches the observed majority-class fraction. The strategy-return comparison
uses asymmetric optimization settings between full and ablated variants. These
limitations are retained in the main README.

The 20-seed evidence is documented in
[its own experiment note](optimized_55_20seed_README.md). It does not evaluate the
final-risk-overlay configuration. Historical missing config fields must not be
silently filled from current defaults when interpreting these results.
