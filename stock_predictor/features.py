from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pywt

try:
    from vmdpy import VMD
except Exception:
    VMD = None

from .config import AppConfig
from .data import DataBundle
from .text_processing import TextSentimentDatasetBuilder, build_text_corpus
from .utils import business_day_range, get_logger


@dataclass
class PreparedDataset:
    frame: pd.DataFrame
    numeric_features: list[str]
    text_features: list[str]
    observed_features: list[str]
    known_future_features: list[str]
    baseline_features: list[str]
    static_features: list[str]
    object_columns: list[str]
    quantiles: list[float]
    max_text_events: int
    text_event_feature_names: list[str]
    text_selection_top_k: int
    metadata: dict[str, Any]


class FeatureBuilder:
    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        self.logger = get_logger()
        if self.cfg.features.raw_text_end_to_end_finetune:
            raise NotImplementedError(
                "Full raw-text-to-prediction FinBERT fine-tuning is not implemented in this rolling-window pipeline. "
                "Use the local FinBERT event-embedding pipeline, or redesign training around a batched text encoder."
            )
        self.text_builder = TextSentimentDatasetBuilder(config, include_embeddings=True)
        self._event_scalar_feature_names = [
            "hybrid_sentiment",
            "finbert_positive",
            "finbert_negative",
            "finbert_neutral",
            "lexicon_positive_hits",
            "lexicon_negative_hits",
            "lexicon_sentiment",
            "text_length",
            "title_length",
            "news_relevance_score",
            "source_quality_score",
            "event_strength_score",
            "notice_event_code",
            "fulltext_flag",
            "is_official_source",
            "is_news_source",
            "is_notice_source",
            "is_policy_source",
            "is_report_source",
            "is_guba_source",
        ]
        self._known_future_feature_names = [
            "known_time_idx",
            "known_dow_sin",
            "known_dow_cos",
            "known_dom_sin",
            "known_dom_cos",
            "known_month_sin",
            "known_month_cos",
            "known_is_month_end",
            "known_is_quarter_end",
        ]
        self._business_start_day = np.datetime64(pd.Timestamp(self.cfg.data.start_date).date())
        end_day = np.datetime64((pd.Timestamp(self.cfg.data.end_date) + pd.Timedelta(days=1)).date())
        self._business_total_days = max(int(np.busday_count(self._business_start_day, end_day)), 1)

    def build(self, bundle: DataBundle) -> PreparedDataset:
        stock = self._build_stock_features(bundle.stock)
        financial_source = self._align_financial_availability(bundle.financial) if self.cfg.features.use_financial else pd.DataFrame()
        financial = self._prepare_factor_frame(financial_source, prefix="fin_")
        macro = self._prepare_macro_features(bundle.macro) if self.cfg.features.use_macro else pd.DataFrame()
        macro = self._scale_numeric_columns(macro, float(getattr(self.cfg.features, "macro_feature_weight", 1.0)))
        industry = self._prepare_factor_frame(bundle.industry, prefix="") if self.cfg.features.use_industry else pd.DataFrame()
        industry = self._normalize_industry_features(industry)
        industry = self._scale_numeric_columns(industry, float(getattr(self.cfg.features, "industry_feature_weight", 1.0)))
        lagged_sentiment = self._lag_text_frame(self._filter_investor_sentiment(bundle.sentiment)) if self.cfg.features.use_sentiment else pd.DataFrame()
        investor_sentiment = self._prepare_factor_frame(lagged_sentiment, prefix="sent_")
        text_sentiment_daily = (
            self._prepare_factor_frame(self._text_sentiment_daily(bundle), prefix="")
            if self.cfg.features.use_text_modality and self.cfg.features.use_sentiment
            else pd.DataFrame()
        )
        policy_source = self._lag_text_frame(bundle.policy) if self.cfg.features.use_policy else pd.DataFrame()
        policy = self._build_policy_features(policy_source) if self.cfg.features.use_text_modality else pd.DataFrame()
        if self.cfg.features.use_text_modality:
            text_pool_frame, max_text_events, text_event_feature_names = self._build_text_event_pool(
                news=self._lag_text_frame(bundle.news) if self.cfg.features.use_news else pd.DataFrame(),
                notices=self._lag_text_frame(bundle.notices) if self.cfg.features.use_notices else pd.DataFrame(),
                policy=policy_source,
            )
        else:
            text_pool_frame, max_text_events, text_event_feature_names = pd.DataFrame(), 1, self._event_scalar_feature_names

        frame = stock.copy()
        merge_frames = [financial, macro, industry, investor_sentiment, text_sentiment_daily, policy, text_pool_frame]
        for optional in merge_frames:
            if optional.empty:
                continue
            frame = frame.merge(optional, on="date", how="left")
        frame = frame.sort_values("date").reset_index(drop=True)
        frame = self._add_known_future_features(frame)
        frame = self._add_static_features(frame)
        frame = self._fill_scalar_features(frame)

        if self.cfg.features.denoise:
            frame = self._add_multisource_denoised_features(frame)

        max_horizon = max(1, int(self.cfg.features.horizon))
        target_columns: dict[str, pd.Series] = {}
        for step in range(1, max_horizon + 1):
            step_return = frame["close"].shift(-step) / frame["close"] - 1.0
            target_columns[f"target_return_h{step}"] = step_return
            target_columns[f"target_direction_h{step}"] = (step_return > 0).astype(int)
        frame = pd.concat([frame, pd.DataFrame(target_columns, index=frame.index)], axis=1)
        frame["target_return"] = frame[f"target_return_h{max_horizon}"]
        frame["target_direction"] = frame[f"target_direction_h{max_horizon}"]
        frame = frame.dropna(subset=[f"target_return_h{step}" for step in range(1, max_horizon + 1)]).reset_index(drop=True)

        object_columns = [
            "future_known_matrix",
            "text_event_vectors",
            "text_event_titles",
            "text_event_sources",
            "text_event_publishers",
            "text_event_urls",
            "text_event_uids",
            "text_event_types",
        ]
        for column in object_columns:
            if column not in frame.columns:
                frame[column] = frame["date"].map(lambda _: self._default_object_value(column, text_event_feature_names))
            else:
                frame[column] = frame[column].map(
                    lambda value, col=column: value
                    if self._is_valid_object_value(col, value)
                    else self._default_object_value(col, text_event_feature_names)
                )

        observed_features = [
            column
            for column in frame.columns
            if column not in {"date", "symbol", "target_return", "target_direction"}
            and not column.startswith("target_return_h")
            and not column.startswith("target_direction_h")
            and column not in object_columns
            and pd.api.types.is_numeric_dtype(frame[column])
            and not column.startswith("known_")
            and not column.startswith("static_")
        ]
        known_future_features = [column for column in self._known_future_feature_names if column in frame.columns]
        text_features = [column for column in observed_features if self.cfg.features.use_text_modality and (column.startswith("text_") or column.startswith("policy_"))]
        numeric_features = [column for column in observed_features if column not in text_features] + known_future_features
        static_features = [column for column in frame.columns if column.startswith("static_") and pd.api.types.is_numeric_dtype(frame[column])]
        baseline_features = numeric_features + text_features
        for column in baseline_features:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

        self.cfg.features.text_candidate_slots = max_text_events
        self.cfg.features.text_candidate_dim = len(text_event_feature_names)
        denoised_feature_count = sum(column.endswith("_denoised") for column in frame.columns)
        text_available_ratio = float(frame["text_event_vectors"].map(lambda values: len(values) > 0).mean()) if not frame.empty else 0.0

        metadata = {
            "finbert_enabled": True,
            "finbert_sentiment_enabled": True,
            "text_encoder": "finbert_local_only",
            "vmd_enabled": bool(VMD is not None and self.cfg.features.denoise),
            "causal_denoise": bool(getattr(self.cfg.features, "causal_denoise", True)),
            "denoise_window": int(getattr(self.cfg.features, "denoise_window", 160)),
            "denoise_stride": int(getattr(self.cfg.features, "denoise_stride", 5)),
            "differentiable_text_selection": True,
            "end_to_end_text_candidate_pool": True,
            "raw_text_to_prediction_end_to_end": False,
            "raw_text_end_to_end_note": (
                "disabled: rolling-window FinBERT fine-tuning over all raw news candidates is not computationally practical "
                "in this project pipeline; FinBERT embeddings remain local pretrained encodings."
            ),
            "strict_official_policy": bool(self.cfg.data.strict_official_policy),
            "guba_historical_pages": int(self.cfg.data.guba_pages),
            "guba_min_rows_per_year": int(self.cfg.data.guba_min_rows_per_year),
            "financial_alignment": "disclosure_date_or_report_date_plus_lag",
            "financial_report_lag_days": int(self.cfg.data.financial_report_lag_days),
            "text_lag_business_days": int(self.cfg.data.text_lag_business_days),
            "ablation_feature_switches": {
                "news": bool(self.cfg.features.use_news),
                "notices": bool(self.cfg.features.use_notices),
                "sentiment": bool(self.cfg.features.use_sentiment),
                "policy": bool(self.cfg.features.use_policy),
                "financial": bool(self.cfg.features.use_financial),
                "macro": bool(self.cfg.features.use_macro),
                "industry": bool(self.cfg.features.use_industry),
            },
            "modality_input_weights": {
                "notice_event_weight": float(getattr(self.cfg.features, "notice_event_weight", 1.0)),
                "macro_feature_weight": float(getattr(self.cfg.features, "macro_feature_weight", 1.0)),
                "industry_feature_weight": float(getattr(self.cfg.features, "industry_feature_weight", 1.0)),
            },
            "text_event_pool_enabled": bool(self.cfg.features.use_text_modality),
            "canonical_tft_decoder": True,
            "tcn_enabled": bool(self.cfg.features.use_tcn),
            "static_covariates_enabled": bool(self.cfg.features.use_static_covariates and static_features),
            "known_future_decoder_enabled": bool(self.cfg.features.use_known_future_decoder),
            "two_industry_sequences": True,
            "independent_text_sentiment_dataset": True,
            "lookback": self.cfg.features.lookback,
            "horizon": self.cfg.features.horizon,
            "max_text_events": max_text_events,
            "text_event_feature_dim": len(text_event_feature_names),
            "text_selection_top_k": min(self.cfg.features.text_selection_top_k, max_text_events),
            "known_future_feature_count": len(known_future_features),
            "static_feature_count": len(static_features),
            "denoised_feature_count": denoised_feature_count,
            "text_available_ratio": text_available_ratio,
        }
        return PreparedDataset(
            frame=frame,
            numeric_features=numeric_features,
            text_features=text_features,
            observed_features=observed_features,
            known_future_features=known_future_features,
            baseline_features=baseline_features,
            static_features=static_features,
            object_columns=object_columns,
            quantiles=self.cfg.features.quantiles,
            max_text_events=max_text_events,
            text_event_feature_names=text_event_feature_names,
            text_selection_top_k=min(self.cfg.features.text_selection_top_k, max_text_events),
            metadata=metadata,
        )

    def _normalize_industry_features(self, industry: pd.DataFrame) -> pd.DataFrame:
        if industry.empty:
            return industry
        out = industry.copy()
        legacy_map = {
            "industry_open": "industry_primary_open",
            "industry_high": "industry_primary_high",
            "industry_low": "industry_primary_low",
            "industry_close": "industry_primary_close",
            "industry_volume": "industry_primary_volume",
            "industry_amount": "industry_primary_amount",
            "industry_pe_weighted": "industry_primary_pe_weighted",
            "industry_pe_median": "industry_primary_pe_median",
            "industry_pe_mean": "industry_primary_pe_mean",
        }
        for old_name, new_name in legacy_map.items():
            if old_name in out.columns and new_name not in out.columns:
                out[new_name] = out[old_name]
        if "industry_primary_close" in out.columns:
            primary = pd.to_numeric(out["industry_primary_close"], errors="coerce")
            out["industry_primary_return_1d"] = primary.pct_change().fillna(0.0)
            out["industry_primary_ma_20"] = primary.rolling(20).mean().ffill()
        if "industry_secondary_close" in out.columns:
            secondary = pd.to_numeric(out["industry_secondary_close"], errors="coerce")
            out["industry_secondary_return_1d"] = secondary.pct_change().fillna(0.0)
            out["industry_secondary_ma_20"] = secondary.rolling(20).mean().ffill()
        else:
            for suffix in ["open", "high", "low", "close", "volume", "amount", "return_1d", "ma_20"]:
                primary_col = f"industry_primary_{suffix}"
                secondary_col = f"industry_secondary_{suffix}"
                if primary_col in out.columns and secondary_col not in out.columns:
                    out[secondary_col] = out[primary_col]
        if "industry_primary_close" in out.columns and "industry_secondary_close" in out.columns:
            primary = pd.to_numeric(out["industry_primary_close"], errors="coerce").replace(0, np.nan)
            secondary = pd.to_numeric(out["industry_secondary_close"], errors="coerce")
            out["industry_primary_secondary_spread"] = (primary - secondary).fillna(0.0)
            out["industry_secondary_to_primary"] = (secondary / primary).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        return out

    def _scale_numeric_columns(self, frame: pd.DataFrame, weight: float) -> pd.DataFrame:
        if frame.empty or abs(weight - 1.0) < 1e-12:
            return frame
        out = frame.copy()
        for column in out.columns:
            if column == "date" or str(column).endswith("_source") or str(column).endswith("_name"):
                continue
            if pd.api.types.is_numeric_dtype(out[column]):
                out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0) * weight
        return out

    def _align_financial_availability(self, financial: pd.DataFrame) -> pd.DataFrame:
        if financial.empty or "date" not in financial.columns:
            return financial
        out = financial.copy()
        report_date = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
        disclosure = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
        for column in ["disclosure_date", "announce_date", "announcement_date", "publish_date", "pub_date"]:
            if column in out.columns:
                parsed = pd.to_datetime(out[column], errors="coerce").dt.normalize()
                disclosure = disclosure.fillna(parsed)
        lag_days = max(0, int(self.cfg.data.financial_report_lag_days))
        fallback = report_date + pd.to_timedelta(lag_days, unit="D")
        out["report_date"] = report_date
        out["date"] = disclosure.fillna(fallback)
        return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    def _lag_text_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or "date" not in frame.columns:
            return frame
        lag_days = max(0, int(self.cfg.data.text_lag_business_days))
        if lag_days == 0:
            return frame
        out = frame.copy()
        out["source_date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
        out["date"] = out["source_date"] + pd.offsets.BDay(lag_days)
        start_date = pd.to_datetime(self.cfg.data.start_date)
        end_date = pd.to_datetime(self.cfg.data.end_date)
        out = out.dropna(subset=["date"])
        out = out[(out["date"] >= start_date) & (out["date"] <= end_date)]
        return out.sort_values("date").reset_index(drop=True)

    def _text_sentiment_daily(self, bundle: DataBundle) -> pd.DataFrame:
        events = bundle.text_sentiment_events
        if events.empty:
            return pd.DataFrame()
        out = events.copy()
        if "text_type" in out.columns:
            allowed_types: set[str] = set()
            if self.cfg.features.use_news:
                allowed_types.add("news")
                allowed_types.add("report")
            if self.cfg.features.use_notices:
                allowed_types.add("notice")
            out = out[out["text_type"].astype(str).isin(allowed_types)]
        if out.empty:
            return pd.DataFrame()
        out = self._lag_text_frame(out)
        if out.empty:
            return pd.DataFrame()
        default_columns = {
            "text_uid": "",
            "hybrid_sentiment": 0.0,
            "positive_flag": 0.0,
            "negative_flag": 0.0,
            "finbert_positive": 0.0,
            "finbert_negative": 0.0,
            "finbert_neutral": 1.0,
            "lexicon_positive_hits": 0.0,
            "lexicon_negative_hits": 0.0,
            "text_length": 0.0,
            "news_relevance_score": 0.0,
            "source_quality_score": 0.5,
            "event_strength_score": 0.0,
            "is_official_source": 0.0,
            "fulltext_flag": 0.0,
            "is_news_source": 0.0,
            "is_notice_source": 0.0,
            "is_report_source": 0.0,
            "source": "",
            "publisher": "",
        }
        for column, default in default_columns.items():
            if column not in out.columns:
                out[column] = default
        daily = (
            out.groupby("date")
            .agg(
                text_sent_event_count=("text_uid", "size"),
                text_sent_hybrid_mean=("hybrid_sentiment", "mean"),
                text_sent_hybrid_abs_mean=("hybrid_sentiment", lambda x: float(pd.Series(x).abs().mean())),
                text_sent_positive_ratio=("positive_flag", "mean"),
                text_sent_negative_ratio=("negative_flag", "mean"),
                text_sent_finbert_positive_mean=("finbert_positive", "mean"),
                text_sent_finbert_negative_mean=("finbert_negative", "mean"),
                text_sent_finbert_neutral_mean=("finbert_neutral", "mean"),
                text_sent_lexicon_positive_hits=("lexicon_positive_hits", "sum"),
                text_sent_lexicon_negative_hits=("lexicon_negative_hits", "sum"),
                text_sent_length_mean=("text_length", "mean"),
                text_sent_length_sum=("text_length", "sum"),
                text_sent_news_quality_mean=("news_relevance_score", "mean"),
                text_sent_source_quality_mean=("source_quality_score", "mean"),
                text_sent_event_strength_mean=("event_strength_score", "mean"),
                text_sent_official_ratio=("is_official_source", "mean"),
                text_sent_fulltext_ratio=("fulltext_flag", "mean"),
                text_sent_news_ratio=("is_news_source", "mean"),
                text_sent_notice_ratio=("is_notice_source", "mean"),
                text_sent_report_ratio=("is_report_source", "mean"),
                text_sent_source_diversity=("source", "nunique"),
                text_sent_publisher_diversity=("publisher", "nunique"),
            )
            .reset_index()
        )
        for column in [col for col in daily.columns if col != "date"]:
            daily[column] = pd.to_numeric(daily[column], errors="coerce").fillna(0.0)
        return daily

    def _filter_investor_sentiment(self, sentiment: pd.DataFrame) -> pd.DataFrame:
        if sentiment.empty:
            return sentiment
        out = sentiment.copy()
        source_enabled = {
            "news": bool(self.cfg.features.use_news),
            "notice": bool(self.cfg.features.use_notices),
            "policy": bool(self.cfg.features.use_policy),
        }
        disabled_prefixes = [prefix for prefix, enabled in source_enabled.items() if not enabled]
        if disabled_prefixes:
            drop_columns = [
                column
                for column in out.columns
                if any(str(column).startswith(f"{prefix}_proxy_") for prefix in disabled_prefixes)
            ]
            mixed_columns = [column for column in out.columns if str(column).startswith("sentiment_proxy_")]
            out = out.drop(columns=drop_columns + mixed_columns, errors="ignore")
        return out

    def _build_stock_features(self, stock: pd.DataFrame) -> pd.DataFrame:
        df = stock.copy().sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if self.cfg.features.denoise and VMD is not None and len(df) >= self.cfg.features.min_history_for_denoise:
            df["close_denoised"] = self._denoise_feature(df["close"])
        else:
            df["close_denoised"] = df["close"]
        df["return_1d"] = df["close"].pct_change()
        df["return_5d"] = df["close"].pct_change(5)
        df["log_return"] = np.log(df["close"]).diff()
        df["ma_5"] = df["close"].rolling(5).mean()
        df["ma_10"] = df["close"].rolling(10).mean()
        df["ma_20"] = df["close"].rolling(20).mean()
        df["volatility_10"] = df["return_1d"].rolling(10).std()
        df["volume_ma_10"] = df["volume"].rolling(10).mean()
        df["price_to_ma_20"] = df["close"] / df["ma_20"]
        df["rsi_14"] = self._rsi(df["close"], 14)
        df["bollinger_z"] = (df["close"] - df["ma_20"]) / df["close"].rolling(20).std()
        return self._fill_numeric(df)

    def _prepare_factor_frame(self, df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce").astype("datetime64[ns]")
        out = out.dropna(subset=["date"]).sort_values("date")
        calendar = pd.DataFrame({"date": business_day_range(self.cfg.data.start_date, self.cfg.data.end_date)})
        calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce").astype("datetime64[ns]")
        out = pd.merge_asof(calendar, out, on="date", direction="backward")
        rename = {
            column: f"{prefix}{column}"
            for column in out.columns
            if column != "date" and prefix and not column.startswith(prefix)
        }
        out = out.rename(columns=rename)
        for column in [col for col in out.columns if col != "date" and pd.api.types.is_numeric_dtype(out[col])]:
            out[column] = pd.to_numeric(out[column], errors="coerce")
        return out

    def _prepare_macro_features(self, macro: pd.DataFrame) -> pd.DataFrame:
        if macro.empty or "date" not in macro.columns:
            return pd.DataFrame()
        source = macro.copy()
        source["date"] = pd.to_datetime(source["date"], errors="coerce").dt.normalize()
        source = source.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        value_columns = [column for column in source.columns if column != "date"]
        for column in value_columns:
            source[column] = pd.to_numeric(source[column], errors="coerce")
        level_source = source[["date", *value_columns]].copy()
        levels = self._prepare_factor_frame(level_source, prefix="macro_")
        if bool(getattr(self.cfg.features, "macro_use_changes_only", False)):
            levels = levels[["date"]].copy()
        if not bool(getattr(self.cfg.features, "macro_add_release_impulses", True)):
            return levels
        impulse = source[["date"]].copy()
        impulse_columns: list[str] = []
        for column in value_columns:
            observed = source[["date", column]].dropna().copy()
            if observed.empty:
                continue
            observed[f"macro_{column}_release"] = 1.0
            observed[f"macro_{column}_change"] = observed[column].diff().fillna(0.0)
            cols = ["date", f"macro_{column}_release", f"macro_{column}_change"]
            impulse = impulse.merge(observed[cols], on="date", how="left")
            impulse_columns.extend(cols[1:])
        calendar = pd.DataFrame({"date": business_day_range(self.cfg.data.start_date, self.cfg.data.end_date)})
        calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce").dt.normalize()
        impulse = calendar.merge(impulse, on="date", how="left")
        for column in impulse_columns:
            impulse[column] = pd.to_numeric(impulse[column], errors="coerce").fillna(0.0)
            if column.endswith("_change"):
                impulse[f"{column}_decay5"] = impulse[column].ewm(halflife=5, adjust=False).mean()
                impulse[f"{column}_decay20"] = impulse[column].ewm(halflife=20, adjust=False).mean()
        return levels.merge(impulse, on="date", how="left")

    def _build_policy_features(self, policy: pd.DataFrame) -> pd.DataFrame:
        if policy.empty:
            return pd.DataFrame()
        corpus = build_text_corpus(self.cfg, None, None, policy, include_policy=True)
        if corpus.empty:
            return pd.DataFrame()
        scored = TextSentimentDatasetBuilder(self.cfg, include_embeddings=False).build(corpus)
        if scored.events.empty:
            return pd.DataFrame()
        daily = (
            scored.events.groupby("date")
            .agg(
                policy_count=("text_uid", "size"),
                policy_hybrid_sentiment=("hybrid_sentiment", "mean"),
                policy_positive_ratio=("positive_flag", "mean"),
                policy_negative_ratio=("negative_flag", "mean"),
                policy_official_ratio=("is_official_source", "mean"),
                policy_fulltext_ratio=("fulltext_flag", "mean"),
            )
            .reset_index()
        )
        for column in [col for col in daily.columns if col != "date"]:
            daily[column] = pd.to_numeric(daily[column], errors="coerce").fillna(0.0)
        return daily

    def _build_text_event_pool(
        self,
        *,
        news: pd.DataFrame,
        notices: pd.DataFrame,
        policy: pd.DataFrame,
    ) -> tuple[pd.DataFrame, int, list[str]]:
        corpus = build_text_corpus(self.cfg, news, notices, policy, include_policy=True)
        text_event_feature_names = self._event_scalar_feature_names + [
            f"embedding_{index}" for index in range(self.text_builder.embedding_dim)
        ]
        if corpus.empty:
            return pd.DataFrame(), 1, text_event_feature_names
        scored = self.text_builder.build(corpus).events
        if scored.empty:
            return pd.DataFrame(), 1, text_event_feature_names
        rows: list[dict[str, Any]] = []
        max_events = 1
        for date_value, group in scored.groupby("date", sort=True):
            vectors: list[np.ndarray] = []
            titles: list[str] = []
            sources: list[str] = []
            publishers: list[str] = []
            urls: list[str] = []
            uids: list[str] = []
            text_types: list[str] = []
            for _, row in group.iterrows():
                scalar_values = []
                for name in self._event_scalar_feature_names:
                    if not self.cfg.features.use_sentiment and name in {
                        "hybrid_sentiment",
                        "finbert_positive",
                        "finbert_negative",
                        "finbert_neutral",
                        "lexicon_positive_hits",
                        "lexicon_negative_hits",
                        "lexicon_sentiment",
                    }:
                        scalar_values.append(0.0)
                    else:
                        scalar_values.append(float(row.get(name, 0.0) or 0.0))
                embedding = np.asarray(row.get("embedding_vector", np.zeros(self.text_builder.embedding_dim)), dtype=np.float32)
                vector = np.concatenate([np.asarray(scalar_values, dtype=np.float32), embedding], axis=0)
                event_type = str(row.get("text_type", "") or "")
                if event_type == "notice":
                    vector = vector * float(getattr(self.cfg.features, "notice_event_weight", 1.0))
                vectors.append(vector)
                titles.append(str(row.get("title", "") or ""))
                sources.append(str(row.get("source", "") or ""))
                publishers.append(str(row.get("publisher", "") or ""))
                urls.append(str(row.get("url", "") or ""))
                uids.append(str(row.get("text_uid", "") or ""))
                text_types.append(str(row.get("text_type", "") or ""))
            max_events = max(max_events, len(vectors))
            rows.append(
                {
                    "date": pd.Timestamp(date_value),
                    "text_event_vectors": vectors,
                    "text_event_titles": titles,
                    "text_event_sources": sources,
                    "text_event_publishers": publishers,
                    "text_event_urls": urls,
                    "text_event_uids": uids,
                    "text_event_types": text_types,
                }
            )
        return pd.DataFrame(rows), max_events, text_event_feature_names

    def _add_known_future_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        known_feature_rows = [self._calendar_feature_row(date_value) for date_value in out["date"]]
        known_frame = pd.DataFrame(known_feature_rows)
        for column in self._known_future_feature_names:
            out[column] = known_frame[column].to_numpy(dtype=float)
        out["future_known_matrix"] = out["date"].map(self._future_known_matrix)
        return out

    def _add_static_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["static_symbol_hash"] = self._stable_hash_value(self.cfg.data.symbol)
        out["static_exchange_sz"] = float(str(self.cfg.data.exchange).lower() == "sz")
        out["static_industry_hash"] = self._stable_hash_value(self.cfg.data.industry_name)
        if "fin_parent_equity" in out.columns and "fin_net_profit" in out.columns:
            equity = pd.to_numeric(out["fin_parent_equity"], errors="coerce").replace(0, np.nan)
            out["fin_roe"] = pd.to_numeric(out["fin_net_profit"], errors="coerce") / equity
        return out

    def _stable_hash_value(self, value: object) -> float:
        digest = hashlib.blake2b(str(value or "").encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, "little") / float(2**32 - 1)

    def _calendar_feature_row(self, date_value: pd.Timestamp) -> dict[str, float]:
        ts = pd.Timestamp(date_value)
        current_day = np.datetime64(ts.date())
        current_idx = max(int(np.busday_count(self._business_start_day, current_day)), 0)
        day_of_week = ts.dayofweek
        day_of_month = ts.day
        month = ts.month
        return {
            "known_time_idx": current_idx / self._business_total_days,
            "known_dow_sin": float(np.sin(2 * np.pi * day_of_week / 7.0)),
            "known_dow_cos": float(np.cos(2 * np.pi * day_of_week / 7.0)),
            "known_dom_sin": float(np.sin(2 * np.pi * (day_of_month - 1) / 31.0)),
            "known_dom_cos": float(np.cos(2 * np.pi * (day_of_month - 1) / 31.0)),
            "known_month_sin": float(np.sin(2 * np.pi * (month - 1) / 12.0)),
            "known_month_cos": float(np.cos(2 * np.pi * (month - 1) / 12.0)),
            "known_is_month_end": float(ts.is_month_end),
            "known_is_quarter_end": float(ts.is_quarter_end),
        }

    def _future_known_matrix(self, date_value: pd.Timestamp) -> np.ndarray:
        rows: list[list[float]] = []
        current = pd.Timestamp(date_value)
        for step in range(1, self.cfg.features.horizon + 1):
            future_date = current + pd.offsets.BDay(step)
            row = self._calendar_feature_row(future_date)
            rows.append([float(row[name]) for name in self._known_future_feature_names])
        return np.asarray(rows, dtype=np.float32)

    def _fill_scalar_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        numeric_columns = [
            column
            for column in out.columns
            if column != "date" and pd.api.types.is_numeric_dtype(out[column]) and not column.startswith("known_")
        ]
        for column in numeric_columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
        out[numeric_columns] = out[numeric_columns].ffill().fillna(0.0)
        for column in [col for col in out.columns if col.startswith("known_")]:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
        return out

    def _fill_numeric(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for column in [col for col in out.columns if col != "date" and col != "symbol"]:
            out[column] = pd.to_numeric(out[column], errors="coerce")
        numeric_columns = [col for col in out.columns if col not in {"date", "symbol"}]
        out[numeric_columns] = out[numeric_columns].ffill().fillna(0.0)
        return out.dropna(subset=["date", "close"]).reset_index(drop=True)

    def _default_object_value(self, column: str, text_event_feature_names: list[str]) -> Any:
        if column == "future_known_matrix":
            return np.zeros((self.cfg.features.horizon, len(self._known_future_feature_names)), dtype=np.float32)
        if column == "text_event_vectors":
            return []
        if column.startswith("text_event_"):
            return []
        return None

    def _is_valid_object_value(self, column: str, value: Any) -> bool:
        if column == "future_known_matrix":
            return isinstance(value, np.ndarray)
        if column.startswith("text_event_"):
            return isinstance(value, list)
        return value is not None and not pd.isna(value)

    def _is_denoise_eligible_column(self, column: str) -> bool:
        if column in {"date", "symbol", "target_return", "target_direction"}:
            return False
        if column.startswith("target_return_h") or column.startswith("target_direction_h"):
            return False
        if column.startswith("known_") or column.endswith("_denoised"):
            return False
        if column.startswith("text_sent_") or column.startswith("policy_"):
            return False
        return pd.api.types.is_numeric_dtype

    def _add_multisource_denoised_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or len(frame) < self.cfg.features.min_history_for_denoise:
            return frame
        out = frame.copy()
        denoise_candidates = [
            "close",
            "volume",
            "return_1d",
            "ma_20",
            "volatility_10",
            "industry_close",
            "industry_volume",
            "macro_cpi_yoy",
            "macro_pmi_manufacturing",
            "macro_m2_yoy",
        ]
        ignore_columns = {
            "date",
            "symbol",
            "target_return",
            "target_direction",
            "future_known_matrix",
            "text_event_vectors",
            "text_event_titles",
            "text_event_sources",
            "text_event_publishers",
            "text_event_urls",
            "text_event_uids",
            "text_event_types",
        }
        for column in [name for name in denoise_candidates if name in out.columns]:
            if column in ignore_columns or column.startswith("known_") or column.endswith("_denoised"):
                continue
            if column.startswith("target_return_h") or column.startswith("target_direction_h"):
                continue
            if column.startswith("text_sent_") or column.startswith("policy_"):
                continue
            if not pd.api.types.is_numeric_dtype(out[column]):
                continue
            numeric = pd.to_numeric(out[column], errors="coerce")
            if numeric.dropna().shape[0] < self.cfg.features.min_history_for_denoise:
                continue
            if numeric.nunique(dropna=True) < 4:
                continue
            out[f"{column}_denoised"] = self._denoise_feature(numeric)
        return out

    def _denoise_feature(self, series: pd.Series) -> pd.Series:
        if not bool(getattr(self.cfg.features, "causal_denoise", True)):
            return self._denoise(series)
        return self._causal_denoise(series)

    def _causal_denoise(self, series: pd.Series) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce").ffill().fillna(0.0)
        n = len(values)
        min_history = max(8, int(self.cfg.features.min_history_for_denoise))
        window = max(min_history, int(getattr(self.cfg.features, "denoise_window", 160) or 160))
        stride = max(1, int(getattr(self.cfg.features, "denoise_stride", 5) or 5))
        if n < min_history:
            return values
        restored = pd.Series(np.nan, index=series.index, dtype=float)
        anchor_indexes = list(range(min_history - 1, n, stride))
        if anchor_indexes[-1] != n - 1:
            anchor_indexes.append(n - 1)
        for idx in anchor_indexes:
            start = max(0, idx + 1 - window)
            history = values.iloc[start : idx + 1]
            restored.iloc[idx] = float(self._denoise(history).iloc[-1])
        restored = restored.ffill().fillna(values)
        return restored

    def _rsi(self, series: pd.Series, window: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta.clip(upper=0)).rolling(window).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _denoise(self, series: pd.Series) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce").ffill().fillna(0.0).to_numpy(dtype=float)
        try:
            if VMD is None:
                raise RuntimeError("VMD unavailable")
            modes, _, _ = VMD(values, alpha=2000, tau=0, K=self.cfg.features.vmd_k, DC=0, init=1, tol=1e-7)
            reconstructed = modes[:-1].sum(axis=0) if modes.shape[0] > 1 else modes[0]
        except Exception:
            reconstructed = values
        try:
            coeffs = pywt.wavedec(reconstructed, self.cfg.features.wavelet, level=self.cfg.features.wavelet_level)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs) > 1 else 0.0
            threshold = sigma * np.sqrt(2 * np.log(len(reconstructed))) if sigma else 0.0
            coeffs[1:] = [pywt.threshold(item, threshold, mode="soft") for item in coeffs[1:]]
            restored = pywt.waverec(coeffs, self.cfg.features.wavelet)[: len(reconstructed)]
        except Exception:
            restored = reconstructed
        restored = self._match_length(restored, len(series))
        return pd.Series(restored, index=series.index)

    def _match_length(self, values: np.ndarray | list[float], target_len: int) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if len(arr) == target_len:
            return arr
        if len(arr) > target_len:
            return arr[:target_len]
        if len(arr) == 0:
            return np.zeros(target_len, dtype=float)
        return np.pad(arr, (0, target_len - len(arr)), constant_values=arr[-1])
