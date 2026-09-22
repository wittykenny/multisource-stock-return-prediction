# Multi-Source Stock Return Prediction

A single-stock research pipeline combining financial time series and Chinese financial text to study next-day return, direction and prediction intervals.

## Overview

This undergraduate project studies whether heterogeneous information can support return forecasting for **CATL (300750.SZ)** over **2020–2025**. It integrates data acquisition and temporal alignment, frozen Chinese FinBERT representations, a TFT-style temporal model with text selection and gated fusion, and rolling evaluation.

**Recorded outcome:** the selected saved experiment has MAE **0.017050** and direction accuracy **55.18%** over **560** distinct evaluation dates. However, all evaluated direction labels are non-up, matching the non-up class frequency; **F1 and MCC are both zero**. These results do not establish useful directional discrimination. The positive simulated strategy return involves a separate validation-selected ensemble and risk controls, not just the forecasting network.

**Project contribution:** a code implementation integrating multi-source data preparation, numeric/text fusion, rolling backtests and modality ablations. FinBERT and the underlying learning libraries are third-party components; this repository does not claim a new pretrained language model or state-of-the-art performance. See [results](#main-results), [reproduction](#reproduction) and [limitations](#limitations).

## Research Objectives

- Model next-day stock returns, direction and quantile intervals using structured and textual inputs.
- Examine the contribution of news, announcements, sentiment, policy, financial, macroeconomic and industry inputs through existing ablations.
- Evaluate predictions and a simulated execution strategy under rolling temporal splits and transaction costs.

## Method

- Acquire price, financial, macroeconomic, industry and text inputs; apply relevance filtering, deduplication and source-quality controls.
- Align financial disclosures and text availability with configurable release lags. Apply rolling causal VMD/wavelet denoising to selected numerical features.
- Use a **local frozen Chinese FinBERT** for text embeddings and sentiment. The downstream text selector is trained, but raw-text FinBERT is not fine-tuned end to end.
- `TFTLite` combines variable selection, gated residual blocks, temporal convolutions, recurrent/attention components, known-future calendar inputs, differentiable text selection, cross-attention and modality gating.
- Train joint point, quantile and direction objectives; evaluate with rolling windows and train-window scaling.
- The selected trading configuration also trains shadow experts, selects/combines positions on validation data, and applies pruning, delayed execution, turnover limits and volatility/drawdown controls. Trading gains cannot be attributed solely to the multimodal network.

## Data

This is a **project-assembled single-stock dataset**, not a standard public benchmark. Inputs are obtained through AKShare and provider-specific acquisition code, including Eastmoney/Tencent market interfaces, financial and industry interfaces, CNInfo announcements, news search/RSS and official policy pages. Saved source notes report incomplete historical policy coverage and use of text-derived sentiment proxies.

- Requested period: 2020-01-01 to 2025-12-31.
- Saved prepared data: **1,454 rows**, 2020-01-02 to 2025-12-30.
- Selected experiment: **720 / 160 / 80 rows** for train/validation/test, advancing **40 rows**; **45-row** lookback and **one-day** horizon. These are observation counts, not calendar days.
- **1,040** overlapping prediction rows reduce to **560** distinct evaluation dates, 2023-08-18 to 2025-12-10.
- Raw corpora, market caches, dictionary and weights are excluded. See [data preparation and availability](data/README.md) and [saved data summary](results/data_summary.json).

## Repository Structure

```text
.
├── main.py                    # CLI entry point
├── stock_predictor/           # Data, text, features, model and backtest modules
├── config/                    # Original experiments + isolated reproduction paths
├── scripts/audit_release.py   # Pre-stage and index privacy/size checks
├── tests/                     # Existing strategy-selection regression tests
├── data/README.md             # External inputs and preparation instructions
├── results/                   # Small aggregate evidence and source hashes
├── docs/                      # Reproduction details and release report
├── requirements.txt
└── .gitignore
```

## Installation

Run commands from the repository root in a fresh clone. The release checks used **Python 3.13.3 on Windows** in an existing environment; a fresh dependency installation and full retraining have **not** been validated. Direct installed dependencies are versioned in `requirements.txt`; historical training-environment identity is unknown. See [environment details](docs/REPRODUCIBILITY.md).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux/macOS, activate with `source .venv/bin/activate`. GPU execution requires a PyTorch build compatible with the machine's driver; CPU fallback exists but full ablations are computationally expensive.

## Reproduction

First obtain the local FinBERT files and prepare external data access as described in [data/README.md](data/README.md). A clone alone does not contain these inputs. Use the isolated reproduction configuration so reruns do not overwrite the original experiment directories:

```bash
python main.py --help
python -m unittest discover -s tests -v
python main.py --config config/reproduction.yaml fetch
python main.py --config config/reproduction.yaml prepare
python main.py --config config/reproduction.yaml ablation
python main.py --config config/reproduction.yaml ablation-significance
```

`--config` precedes the subcommand. `ablation` runs the full model and the seven modality removals; it disables external baselines internally, matching the selected saved full-model report. Outputs go to `data/output/reproduction/`. For a standalone run with configured baselines, use `backtest`, but that is a different output scope. `audit`, `tune`, `multiseed` and `info-gain` are additional existing commands.

This is a **workflow reproduction recipe**, not a claim of exact numerical reproducibility. Provider revisions, missing historical snapshots, an unrecorded model revision and environment changes prevent an exact guarantee. No training was rerun for this repository cleanup.

## Main Results

The values below are copied from the saved `full` ablation in `cleaned_full_model_notice_final_risk_overlay`, associated with `config/red_comment_revision.yaml`. They are not newly computed training results. All metrics are available in [main_results.csv](results/main_results.csv); [provenance.json](results/provenance.json) records the original files, transformations and SHA-256 hashes.

| Metric | Recorded value |
|---|---:|
| Evaluation dates | 560 |
| MAE (decimal return) | 0.017050 |
| RMSE (decimal return) | 0.025939 |
| R² | -0.023795 |
| Direction accuracy | 55.18% |
| Up-class F1 / MCC | 0 / 0 |
| Fraction predicted up | 0.00% |
| IC / Rank IC | -0.071512 / -0.079552 |
| 10–90% interval empirical coverage | 89.11% |
| Simulated cumulative strategy return | 27.26% |
| Simulated annualized strategy return | 11.46% |
| Simulated Sharpe ratio | 1.063384 |
| Simulated maximum drawdown | -7.09% |

The saved direction counts are **309 non-up / 251 up**, with **560 non-up predictions**. The interval's nominal coverage is 80%; higher empirical coverage alone does not establish better calibrated intervals. Strategy metrics include 10 bps transaction cost and 10 bps slippage with a one-day execution delay. They describe a simulated composite strategy, not observed trading performance. The selected report contains no external-baseline results; none are invented here.

## Ablation / Robustness

- [Modality ablation results](results/ablation.csv) and [saved significance tests](results/ablation_significance.csv) are included. Removing industry inputs gives slightly lower MAE than the full variant, so the full model does not dominate all ablations.
- The strategy-return comparison flags do not establish a 5% significant advantage for the full variant. The full variant uses strategy optimization while ablated variants disable it in this experiment: economic-return differences are **not a controlled estimate of modality value**.
- The [leakage audit](results/leakage_audit.json) records split boundaries and train-only scaling checks. A feature-name audit is not proof against every form of temporal leakage or selection bias.
- A separate **20-seed historical `optimized_55` experiment (2024–2043)** is included: [aggregate](results/optimized_55_20seed_aggregate.csv), [per-seed summary](results/optimized_55_20seed_summary.csv), and [configuration/provenance notes](results/optimized_55_20seed_README.md). It is **not** a 20-seed evaluation of the `red_comment_revision` / final-risk-overlay configuration used in Main Results.

### Historical 20-seed experiment: `optimized_55`

The explicit settings in [config/optimized_55.yaml](config/optimized_55.yaml) match the saved run settings apart from the expected seed, per-seed output directory and disabled external-baseline reporting overrides. The [saved resolved configuration](results/optimized_55_20seed_config.json) is authoritative for the fields recorded at run time: current dataclass defaults contain additional fields absent from those historical reports, so the current YAML alone is not a complete historical specification.

| Metric | Mean across 20 seeds | Population SD | Saved 95% CI for the mean |
|---|---:|---:|---:|
| MAE | 0.017080 | 0.000053 | [0.017057, 0.017104] |
| RMSE | 0.025961 | 0.000045 | [0.025941, 0.025981] |
| Direction accuracy | 54.43% | 1.00 percentage points | [53.96%, 54.83%] |
| Simulated cumulative return | -2.89% | 15.97 percentage points | [-10.18%, 4.16%] |
| Simulated Sharpe ratio | -0.201731 | 0.686236 | [-0.506844, 0.100905] |

Values are rounded from the saved aggregate; original precision and all eight metrics remain in the linked files. Each run records 560 evaluation dates. Seeds repeat the same historical setting, not independent stocks or market periods. Mean cumulative return is negative and its interval spans zero; these results do not establish a consistently profitable strategy. They must not be combined with the main experiment's 27.26% single-run cumulative return as if they described the same configuration. See the [experiment note](results/optimized_55_20seed_README.md) for interval interpretation and reproduction limits.

## Limitations

- One stock and one historical period; no demonstrated cross-stock or out-of-market generalization.
- The reported direction accuracy is attained by an all-non-up classifier. Negative R² and IC further limit claims of predictive utility.
- Historical text retrieval and policy coverage are incomplete; some sentiment inputs are proxies. Publication-date lags are approximations, not a complete point-in-time vendor archive.
- Many configurations and strategy variants were explored. An untouched final holdout and independent replication are not established by the available artifacts.
- Modality ablation strategy settings are asymmetric; apparent economic gains cannot isolate architectural contributions.
- Exact corpus, pretrained-model revision, historical dependency lock and trained forecasting checkpoints are not provided. Frozen FinBERT embeddings are not end-to-end language-model adaptation.
- No repository license is assigned pending confirmation of code ownership and reuse permissions. Third-party materials retain their own terms.

## Reproducibility

The selected run records seed **2026**, and the code seeds Python, NumPy and PyTorch. Hardware/backend nondeterminism may remain. The original resolved configuration is preserved in [experiment_config.json](results/experiment_config.json); `config/reproduction.yaml` changes only cache/output locations relative to `config/red_comment_revision.yaml`. Other configurations remain available as historical experiments and should not be assumed to generate the same table.

Release validation covers syntax, configuration loading, CLI help, two existing tests, source hashes and selected-file audits. It does not cover a clean full data download, model retraining or recreation of the reported metrics. See [reproduction details](docs/REPRODUCIBILITY.md).

## Citation

This repository contains an undergraduate research/course project by Haoming Luo. No formal publication is claimed.

## Author

Haoming Luo

Zhongnan University of Economics and Law

B.Eng. in Artificial Intelligence & B.Mgt. in Accounting
