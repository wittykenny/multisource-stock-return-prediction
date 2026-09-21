from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    symbol: str = "300750"
    exchange: str = "sz"
    company_name: str = "\u5b81\u5fb7\u65f6\u4ee3"
    industry_name: str = "\u7535\u6c60"
    industry_code: str = "BK1033"
    secondary_industry_names: list[str] = field(default_factory=lambda: ["\u65b0\u80fd\u6e90\u8f66", "\u65b0\u80fd\u6c7d\u8f66"])
    secondary_industry_code: str = ""
    industry_pe_name: str = "\u7535\u6c14\u673a\u68b0\u548c\u5668\u6750\u5236\u9020\u4e1a"
    industry_pe_classification: str = "\u8bc1\u76d1\u4f1a\u884c\u4e1a\u5206\u7c7b"
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"
    cache_dir: str = "data/cache"
    output_dir: str = "data/output"
    refresh: bool = False
    financial_min_date: str = "2019-01-01"
    financial_report_lag_days: int = 90
    text_lag_business_days: int = 1
    news_pages: int = 50
    news_yearly_expansion: bool = True
    notice_page_size: int = 100
    gdelt_enabled: bool = True
    gdelt_company_query: str = "CATL battery"
    gdelt_industry_query: str = ""
    gdelt_max_records_per_window: int = 25
    gdelt_sleep_seconds: float = 5.2
    google_news_enabled: bool = True
    google_news_window_days: int = 31
    policy_enabled: bool = True
    policy_pages: int = 20
    policy_archive_pages: int = 8
    policy_official_queries: list[str] = field(
        default_factory=lambda: [
            "site:gov.cn \u52a8\u529b\u7535\u6c60 \u653f\u7b56",
            "site:gov.cn \u50a8\u80fd \u7535\u6c60 \u653f\u7b56",
            "site:miit.gov.cn \u52a8\u529b\u7535\u6c60",
            "site:nea.gov.cn \u50a8\u80fd \u653f\u7b56",
            "site:ndrc.gov.cn \u65b0\u80fd\u6e90\u6c7d\u8f66 \u7535\u6c60 \u653f\u7b56",
        ]
    )
    policy_official_seed_urls: list[dict[str, str]] = field(
        default_factory=lambda: [
            {"url": "https://www.gov.cn/zhengce/zhengceku/index.htm", "source": "gov_policy_archive"},
            {"url": "https://www.miit.gov.cn/zwgk/zcwj/wjfb/index.html", "source": "miit_policy_archive"},
            {"url": "https://www.nea.gov.cn/zwgk/zcfb/index.html", "source": "nea_policy_archive"},
            {"url": "https://www.ndrc.gov.cn/xxgk/zcfb/tz/index.html", "source": "ndrc_policy_archive"},
        ]
    )
    policy_official_domains: list[str] = field(
        default_factory=lambda: [
            "gov.cn",
            "miit.gov.cn",
            "nea.gov.cn",
            "ndrc.gov.cn",
            "gov.cn/zhengce",
        ]
    )
    policy_keywords: list[str] = field(
        default_factory=lambda: [
            "\u5de5\u4fe1\u90e8 \u52a8\u529b\u7535\u6c60",
            "\u56fd\u5bb6\u80fd\u6e90\u5c40 \u50a8\u80fd \u653f\u7b56",
            "\u9502\u79bb\u5b50\u7535\u6c60 \u884c\u4e1a\u89c4\u8303\u6761\u4ef6",
            "\u65b0\u80fd\u6e90\u6c7d\u8f66 \u52a8\u529b\u7535\u6c60 \u653f\u7b56",
        ]
    )
    sentiment_enabled: bool = True
    guba_pages: int = 300
    guba_min_rows_per_year: int = 30
    strict_official_policy: bool = True
    news_min_rows_per_year: int = 12
    news_quality_filter: bool = True
    news_min_relevance_score: float = 5.0
    news_industry_only_min_score: float = 8.0
    news_min_text_chars: int = 12
    news_max_garbled_ratio: float = 0.02
    news_near_duplicate_days: int = 7
    news_max_per_day: int = 24
    news_max_per_source_per_day: int = 4
    notice_quality_filter: bool = True
    notice_min_relevance_score: float = 4.5
    notice_min_text_chars: int = 16
    notice_near_duplicate_days: int = 30
    notice_max_per_day: int = 4
    notice_exclude_routine: bool = True
    notice_min_event_strength: float = 0.60
    notice_title_only_min_relevance: float = 7.0
    notice_min_keep_rows: int = 0
    notice_fallback_min_event_strength: float = 0.60
    notice_fallback_title_only_min_relevance: float = 7.0
    notice_global_dedupe_days: int = 3650
    notice_max_per_class_per_month: int = 8
    notice_drop_title_only_periodic_reports: bool = True
    notice_drop_progress_updates: bool = True
    text_similarity_dedupe_threshold: float = 0.86
    text_similarity_dedupe_window_days: int = 14
    text_min_source_quality: float = 0.25
    sentiment_winsor_lower: float = 0.01
    sentiment_winsor_upper: float = 0.99
    macro_cleaning_enabled: bool = True
    macro_cpi_release_lag_days: int = 10
    macro_pmi_release_lag_days: int = 1
    macro_m2_release_lag_days: int = 12
    macro_gdp_release_lag_days: int = 20
    text_extract_enabled: bool = True
    text_top_k_per_day: int = 8
    text_extract_max_chars: int = 20000
    text_extract_pdf_pages: int = -1
    text_extract_news_rows: int = -1
    text_extract_policy_rows: int = -1
    text_extract_notice_rows: int = -1
    text_extract_report_rows: int = -1
    text_extract_cache_dir: str = ""
    text_extract_miss_ttl_hours: int = 72
    lexicon_path: str = "data/resources/loughran_mcdonald_master_dictionary.csv"
    finbert_model: str = "data/models/yiyanghkust_finbert-tone-chinese"
    finbert_batch_size: int = 8
    enable_finbert: bool = True
    industry_proxy_top_n: int = 12


@dataclass
class FeatureConfig:
    lookback: int = 30
    horizon: int = 1
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    denoise: bool = True
    causal_denoise: bool = True
    denoise_window: int = 160
    denoise_stride: int = 5
    vmd_k: int = 4
    wavelet: str = "db4"
    wavelet_level: int = 2
    min_history_for_denoise: int = 64
    text_embedding_dim: int = 768
    text_candidate_slots: int = 0
    text_candidate_dim: int = 0
    text_selection_top_k: int = 4
    use_text_modality: bool = True
    use_tcn: bool = True
    use_static_covariates: bool = True
    use_known_future_decoder: bool = True
    use_cross_attention: bool = True
    use_modality_gate: bool = True
    use_soft_topk: bool = True
    raw_text_end_to_end_finetune: bool = False
    use_news: bool = True
    use_notices: bool = True
    use_sentiment: bool = True
    use_policy: bool = True
    use_financial: bool = True
    use_macro: bool = True
    use_industry: bool = True
    notice_event_weight: float = 1.0
    macro_feature_weight: float = 1.0
    macro_use_changes_only: bool = False
    macro_add_release_impulses: bool = True
    industry_feature_weight: float = 1.0


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda"
    batch_size: int = 256
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 32
    heads: int = 4
    dropout: float = 0.1
    early_stop_rounds: int = 5
    point_weight: float = 1.0
    quantile_weight: float = 0.3
    direction_weight: float = 1.6
    direction_focal_gamma: float = 1.5
    text_event_quality_gate_strength: float = 0.0
    modality_prior_strength: float = 0.0
    modality_numeric_prior: float = 0.60
    modality_text_prior: float = 0.20
    modality_cross_prior: float = 0.20
    amp: bool = True
    tf32: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    tensor_cache: bool = True
    compute_attributions: bool = False
    hyper_hidden_dims: list[int] = field(default_factory=lambda: [32, 48, 64])
    hyper_dropouts: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    hyper_direction_weights: list[float] = field(default_factory=lambda: [1.2, 1.6, 2.0, 2.5, 3.0])
    hyper_point_weights: list[float] = field(default_factory=lambda: [0.8, 1.0, 1.2])
    hyper_quantile_weights: list[float] = field(default_factory=lambda: [0.2, 0.3, 0.5])
    hyper_lrs: list[float] = field(default_factory=lambda: [5e-4, 8e-4, 1e-3])
    hyper_weight_decays: list[float] = field(default_factory=lambda: [5e-5, 1e-4, 3e-4])
    hyper_trial_specs: list[dict[str, Any]] = field(default_factory=list)
    hyper_max_trials: int = 24
    tune_metric: str = "target55"
    multi_seeds: list[int] = field(
        default_factory=lambda: [
            2024,
            2025,
            2026,
            2027,
            2028,
            2029,
            2030,
            2031,
            2032,
            2033,
            2034,
            2035,
            2036,
            2037,
            2038,
            2039,
            2040,
            2041,
            2042,
            2043,
        ]
    )


@dataclass
class BacktestConfig:
    train_days: int = 720
    val_days: int = 120
    test_days: int = 120
    step_days: int = 60
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 10.0
    direction_threshold: float = 0.55
    max_position: float = 0.6
    max_daily_turnover: float = 0.25
    execution_delay_days: int = 1
    block_limit_trades: bool = True
    limit_trade_buffer: float = 0.002
    price_limit_pct: float = 0.20
    strategy_return_step: int = 1
    strategy_signal: str = "return"
    strategy_threshold_mode: str = "validation_quantile"
    strategy_quantiles: list[float] = field(default_factory=lambda: [0.35, 0.45, 0.55, 0.65, 0.75])
    strategy_min_exposure: float = 0.15
    strategy_max_exposure: float = 0.65
    strategy_target_exposure: float = 0.35
    strategy_opt_metric: str = "balanced"
    strategy_scale_floor: float = 0.005
    strategy_use_external_baselines: bool = True
    strategy_robust_segments: int = 1
    strategy_min_positive_segment_ratio: float = 0.0
    strategy_worst_segment_weight: float = 0.0
    strategy_fit_validation_ensemble: bool = False
    strategy_fit_ridge_alpha: float = 0.25
    strategy_shadow_experts: list[str] = field(default_factory=list)
    strategy_preferred_sources: list[str] = field(default_factory=list)
    strategy_preferred_source_bonus: float = 0.0
    strategy_preferred_min_validation_score: float = 0.0
    strategy_preferred_min_positive_segment_ratio: float = 0.0
    strategy_position_ensemble: bool = False
    strategy_position_ensemble_top_k: int = 3
    strategy_position_ensemble_min_return: float = 0.0
    strategy_position_ensemble_sort_metric: str = "validation_return"
    strategy_position_ensemble_leverage: float = 1.0
    strategy_position_ensemble_allow_inverse: bool = False
    strategy_position_ensemble_inverse_min_return: float = 0.0
    strategy_position_ensemble_risk_overlay: bool = False
    strategy_position_pruning: bool = False
    strategy_position_pruning_quantiles: list[float] = field(default_factory=lambda: [0.65, 0.70, 0.75])
    strategy_position_pruning_min_return_ratio: float = 0.75
    strategy_risk_overlay_vol_quantiles: list[float] = field(default_factory=lambda: [0.80, 0.90])
    strategy_risk_overlay_drawdown_thresholds: list[float] = field(default_factory=lambda: [-0.03, -0.05])
    strategy_risk_overlay_scales: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.50])
    strategy_risk_overlay_min_improvement: float = 0.0
    strategy_final_risk_overlay: bool = False
    strategy_final_risk_overlay_min_return_ratio: float = 0.85
    strategy_position_ensemble_reserve_sources: list[str] = field(default_factory=list)
    strategy_position_ensemble_reserve_weight: float = 0.0
    strategy_position_ensemble_reserve_min_return: float = -0.05
    ablation_use_configured_strategy: bool = False
    strategy_disable_optimization: bool = False
    ablation_variant_disable_strategy_optimization: bool = False
    ablation_variant_strategy_signal: str = "hybrid"
    ablation_variant_strategy_target_exposure: float = 0.25
    require_positive_probability: bool = False
    require_positive_return: bool = False
    deduplicate_predictions: bool = True
    non_overlapping_label_eval: bool = True
    feature_importance_top_k: int = 12
    run_baselines: bool = True
    use_direction_calibration: bool = True
    direction_use_external_baselines: bool = True
    direction_selection_mode: str = "validation"


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def load_config(path: str | Path | None = None) -> AppConfig:
    config = AppConfig()
    if path is None:
        return config
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    user_payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    merged = _deep_update(config.to_dict(), user_payload)
    return AppConfig(
        data=DataConfig(**merged["data"]),
        features=FeatureConfig(**merged["features"]),
        train=TrainConfig(**merged["train"]),
        backtest=BacktestConfig(**merged["backtest"]),
    )
