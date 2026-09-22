# Historical 20-seed experiment: optimized_55

## Configuration identity

These artifacts come from `data/output/optimized_55/300750/`, with individual
reports in `data/output/optimized_55/multiseed/seed_<seed>/300750/`.
They cover exactly **20 distinct seeds, 2024 through 2043 inclusive**. The separate
`optimized_experiment_20260507` directory contains only five saved seeds and is
not the source of this release.

The associated checked-in configuration is
[`config/optimized_55.yaml`](../config/optimized_55.yaml). All its explicitly
specified fields were compared with the saved configuration: they match except
for `train.seed`, `data.output_dir` and `backtest.run_baselines`, which the
multi-seed workflow overrides. The saved reports agree in every configuration
field across all 20 seeds except seed and per-seed output directory.

[`optimized_55_20seed_config.json`](optimized_55_20seed_config.json) extracts the
resolved configuration from the seed-2024 report without filling in missing fields.
It records seed 2024 and its historical relative output directory. For other runs,
the seed and corresponding `seed_<seed>` directory change. The existing current
dataclass defaults contain additional cleaning, gating and strategy fields not
recorded in these historical reports. **Absence is not evidence of a particular
historical default.** Neither the current YAML nor this snapshot supplies an
immutable historical source-code or environment version.

| Setting | Historical 20-seed experiment | README Main Results experiment |
|---|---|---|
| Associated YAML | `optimized_55.yaml` | `red_comment_revision.yaml` |
| Saved experiment directory | `optimized_55` | `cleaned_full_model_notice_final_risk_overlay` |
| Seeds represented here | 2024–2043 | 2026 |
| Lookback / horizon | 45 / 1 | 45 / 1 |
| Train / validation / test / step rows | 720 / 160 / 80 / 40 | 720 / 160 / 80 / 40 |
| Notice / macro / industry input weights | 0.25 / 0.40 / 0.40 | 0.04 / 0.20 / 0.40 |
| Strategy signal / optimization metric | `external` / `sharpe` | `hybrid` / `significance` |
| Separate baseline reporting | Disabled by multi-seed workflow | Disabled by ablation workflow |
| Position ensemble / final risk overlay | Fields absent from saved config | Enabled in saved config |

The selected historical model records hidden dimension 48, 4 attention heads,
dropout 0.15, learning rate 0.0005, batch size 128 and up to 60 epochs. Every seed
report records 560 evaluation dates. `run_baselines=false` describes the reporting
flag; it must not be interpreted as proof that the historical `external` strategy
is architecturally identical to the main experiment or contains no auxiliary
signals.

## Files and validation

- [Aggregate CSV](optimized_55_20seed_aggregate.csv): eight metrics with saved
  `mean`, `std`, `ci95_low`, `ci95_high` and `n` columns.
- [Per-seed CSV](optimized_55_20seed_summary.csv): one row for each of the 20 seeds.
- [Summary JSON](optimized_55_20seed_summary.json): original seed list, per-seed
  metrics and aggregate values.
- [Saved configuration](optimized_55_20seed_config.json): extraction described above.
- [Provenance](provenance.json): original relative paths and SHA-256 hashes,
  including the 20 source reports supporting the configuration comparison.

The three summary/aggregate files are **byte-for-byte copies**. No experiment
was rerun and no saved number was replaced. Every per-seed summary metric was
checked against its original report; CSV and JSON agree. Aggregate means and
population standard deviations (`ddof=0`) were independently checked as a
packaging validation, without rewriting the saved aggregate.

The current aggregation implementation describes a percentile bootstrap of the
mean across seed values: 2,000 resamples, 95% interval and bootstrap RNG seed 42.
The released endpoints are the saved values, not a newly estimated interval.
There is no frozen historical source revision to prove the exact generator used
at run time. These intervals concern seed variation for one historical setting;
they are not a time-series block bootstrap, prediction interval, cross-market
generalization interval or guarantee of trading profitability.

## Interpretation and reproduction boundary

Mean direction accuracy is 54.43%; mean simulated cumulative return is -2.89%,
with a saved mean interval spanning zero. These results do not establish
consistently positive strategy returns. They are **not robustness evidence for
the final-risk-overlay configuration** used in the main README table.

The existing CLI route for this configuration is
`python main.py --config config/optimized_55.yaml multiseed`, after preparing the
external inputs described in `data/README.md`. This command writes into the
historical output directory: run it only in a fresh clone, or first copy the YAML
and choose new cache/output roots to preserve existing artifacts. No new run was
performed for this release. Current code/defaults, changed source data and an
unrecorded model/environment revision prevent a claim of exact replication.
