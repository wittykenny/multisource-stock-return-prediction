from __future__ import annotations

import json
import math
from pathlib import Path
from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from .config import AppConfig
from .data import MarketDataHub
from .features import FeatureBuilder, PreparedDataset
from .model import (
    _bootstrap_mean_ci,
    add_strategy_decision_columns,
    _daily_strategy_returns,
    _diebold_mariano,
    build_baseline_predictions,
    compare_prediction_significance,
    evaluate_predictions,
    fit_single_window,
    optimize_direction_parameters,
    optimize_point_calibration,
    optimize_strategy_parameters,
    predict_frame,
    scale_frame,
    summarize_attention_patterns,
    summarize_explanations,
    summarize_feature_contributions,
    summarize_text_event_contributions,
)
from .utils import ensure_dir, save_frame, save_json, serialize_float_map, set_seed


class StockPredictionPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        self.output_root = ensure_dir(Path(config.data.output_dir) / config.data.symbol)
        self.data_hub = MarketDataHub(config)
        self.feature_builder: FeatureBuilder | None = None

    def _get_feature_builder(self) -> FeatureBuilder:
        if self.feature_builder is None:
            self.feature_builder = FeatureBuilder(self.cfg)
        return self.feature_builder

    def run_fetch(self) -> dict[str, object]:
        bundle = self.data_hub.fetch_all()
        output = {
            "stock_rows": str(len(bundle.stock)),
            "financial_rows": str(len(bundle.financial)),
            "macro_rows": str(len(bundle.macro)),
            "industry_rows": str(len(bundle.industry)),
            "sentiment_rows": str(len(bundle.sentiment)),
            "policy_rows": str(len(bundle.policy)),
            "news_rows": str(len(bundle.news)),
            "notice_rows": str(len(bundle.notices)),
            "coverage": {
                "policy": self._summarize_coverage(bundle.policy),
                "news": self._summarize_coverage(bundle.news),
                "notices": self._summarize_coverage(bundle.notices),
                "sentiment": self._summarize_coverage(bundle.sentiment, source_column="sentiment_source"),
            },
            "notes": bundle.source_notes,
        }
        save_json(serialize_float_map(output), self.output_root / "fetch_summary.json")
        return output

    def run_prepare(self) -> PreparedDataset:
        bundle = self.data_hub.fetch_all()
        dataset = self._get_feature_builder().build(bundle)
        save_frame(dataset.frame.drop(columns=dataset.object_columns, errors="ignore"), self.output_root / "prepared_features.csv")
        save_json(
            serialize_float_map(
                {
                    "numeric_features": dataset.numeric_features,
                    "text_features": dataset.text_features,
                    "known_future_features": dataset.known_future_features,
                    "static_features": dataset.static_features,
                    "object_columns": dataset.object_columns,
                    "metadata": dataset.metadata,
                    "source_notes": bundle.source_notes,
                }
            ),
            self.output_root / "prepare_summary.json",
        )
        return dataset

    def run_backtest(self, include_baselines: bool | None = None) -> dict[str, object]:
        set_seed(self.cfg.train.seed)
        include_baselines = self.cfg.backtest.run_baselines if include_baselines is None else include_baselines
        dataset = self.run_prepare()
        frame = dataset.frame.reset_index(drop=True)

        predictions: list[pd.DataFrame] = []
        baseline_predictions: dict[str, list[pd.DataFrame]] = {}
        window_audit_rows: list[dict[str, object]] = []
        strategy_rows: list[dict[str, object]] = []
        start = 0
        all_features = dataset.numeric_features + dataset.text_features

        while start + self.cfg.backtest.train_days + self.cfg.backtest.val_days + self.cfg.backtest.test_days <= len(frame):
            train_frame = frame.iloc[start : start + self.cfg.backtest.train_days].copy()
            val_frame = frame.iloc[
                start + self.cfg.backtest.train_days : start + self.cfg.backtest.train_days + self.cfg.backtest.val_days
            ].copy()
            test_frame = frame.iloc[
                start + self.cfg.backtest.train_days + self.cfg.backtest.val_days : start + self.cfg.backtest.train_days + self.cfg.backtest.val_days + self.cfg.backtest.test_days
            ].copy()

            train_scaled, scalers = scale_frame(train_frame, all_features)
            val_scaled, _ = scale_frame(val_frame, all_features, scalers)
            test_scaled, _ = scale_frame(test_frame, all_features, scalers)
            train_scaled = self._apply_post_scale_feature_weights(train_scaled, dataset)
            val_scaled = self._apply_post_scale_feature_weights(val_scaled, dataset)
            test_scaled = self._apply_post_scale_feature_weights(test_scaled, dataset)

            model = fit_single_window(train_scaled, val_scaled, dataset, self.cfg)
            val_pred = predict_frame(model, val_scaled, val_frame, dataset, self.cfg)
            point_params = optimize_point_calibration(val_pred)
            self._apply_point_calibration(val_pred, point_params)
            pred = predict_frame(model, test_scaled, test_frame, dataset, self.cfg)
            self._apply_point_calibration(pred, point_params)
            validation_baselines: dict[str, pd.DataFrame] = {}
            baselines: dict[str, pd.DataFrame] = {}
            validation_auxiliary: dict[str, pd.DataFrame] = {}
            auxiliary: dict[str, pd.DataFrame] = {}

            if self._use_adaptive_full_inference(dataset):
                validation_auxiliary, auxiliary = self._build_adaptive_full_candidates(
                    model=model,
                    val_scaled=val_scaled,
                    val_raw=val_frame,
                    test_scaled=test_scaled,
                    test_raw=test_frame,
                    dataset=dataset,
                    point_params=point_params,
                )
                shadow_validation, shadow_test = self._build_strategy_shadow_experts(
                    train_scaled=train_scaled,
                    val_scaled=val_scaled,
                    val_raw=val_frame,
                    test_scaled=test_scaled,
                    test_raw=test_frame,
                    dataset=dataset,
                )
                validation_auxiliary.update(shadow_validation)
                auxiliary.update(shadow_test)

            if include_baselines:
                baseline_train_raw = pd.concat([train_frame, val_frame], ignore_index=True)
                validation_baselines = build_baseline_predictions(
                    train_scaled,
                    val_scaled,
                    dataset,
                    self.cfg.features.lookback,
                    raw_train_frame=train_frame,
                    raw_test_frame=val_frame,
                    cfg=self.cfg,
                )
                baselines = build_baseline_predictions(
                    pd.concat([train_scaled, val_scaled], ignore_index=True),
                    test_scaled,
                    dataset,
                    self.cfg.features.lookback,
                    raw_train_frame=baseline_train_raw,
                    raw_test_frame=test_frame,
                    cfg=self.cfg,
                )

            validation_candidates = {**validation_auxiliary, **validation_baselines}
            test_candidates = {**auxiliary, **baselines}
            blend_params = self._apply_validation_selected_prediction(pred, val_pred, test_candidates, validation_candidates)
            strategy_params = self._select_strategy_params(pred, val_pred, test_candidates, validation_candidates)
            if bool(getattr(self.cfg.backtest, "use_direction_calibration", True)):
                direction_params = self._select_direction_params(pred, val_pred, test_candidates, validation_candidates)
            else:
                direction_params = {
                    "direction_signal": "raw_prob",
                    "direction_source": "raw_model_prob",
                    "direction_threshold_calibrated": 0.5,
                    "direction_polarity": 1.0,
                    "validation_directional_accuracy": 0.0,
                    "direction_validation_accuracy": 0.0,
                }
            self._apply_strategy_params(pred, strategy_params)
            self._apply_direction_params(pred, direction_params)
            add_strategy_decision_columns(pred, self.cfg)
            pred["window_start"] = str(train_frame["date"].min())
            pred["window_end"] = str(test_frame["date"].max())
            predictions.append(pred)
            strategy_rows.append({
                "model": "main",
                "window_start": str(train_frame["date"].min()),
                "window_end": str(test_frame["date"].max()),
                "adaptive_full_candidates": "|".join(sorted(auxiliary.keys())),
                **strategy_params,
                **direction_params,
                **point_params,
                **blend_params,
            })

            if include_baselines:
                for name, baseline_df in baselines.items():
                    baseline_strategy = optimize_strategy_parameters(validation_baselines.get(name, pd.DataFrame()), self.cfg)
                    self._apply_strategy_params(baseline_df, baseline_strategy)
                    add_strategy_decision_columns(baseline_df, self.cfg)
                    baseline_df["window_start"] = str(train_frame["date"].min())
                    baseline_df["window_end"] = str(test_frame["date"].max())
                    baseline_predictions.setdefault(name, []).append(baseline_df)
                    strategy_rows.append({"model": name, "window_start": str(train_frame["date"].min()), "window_end": str(test_frame["date"].max()), **baseline_strategy})

            window_audit_rows.append(
                {
                    "window_start_index": int(start),
                    "train_start": str(train_frame["date"].min()),
                    "train_end": str(train_frame["date"].max()),
                    "val_start": str(val_frame["date"].min()),
                    "val_end": str(val_frame["date"].max()),
                    "test_start": str(test_frame["date"].min()),
                    "test_end": str(test_frame["date"].max()),
                    "scaler_fit_start": str(train_frame["date"].min()),
                    "scaler_fit_end": str(train_frame["date"].max()),
                    "main_feature_count": int(len(all_features)),
                    "baseline_feature_count": int(len(dataset.baseline_features)),
                    "target_column": "target_return",
                    "horizon": int(self.cfg.features.horizon),
                }
            )
            start += self.cfg.backtest.step_days

        merged = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
        metrics, modality_weights = evaluate_predictions(merged, self.cfg)
        feature_contributions = summarize_feature_contributions(
            frame,
            dataset,
            top_k=self.cfg.backtest.feature_importance_top_k,
            predictions=merged,
        )
        attention_summary = summarize_attention_patterns(merged)
        explanation_summary = summarize_explanations(merged)
        text_event_contributions = summarize_text_event_contributions(
            merged,
            top_k=self.cfg.backtest.feature_importance_top_k,
        )

        baseline_reports: dict[str, dict[str, object]] = {}
        merged_baselines: dict[str, pd.DataFrame] = {}
        for name, frames in baseline_predictions.items():
            merged_baseline = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            merged_baselines[name] = merged_baseline
            baseline_metrics, _ = evaluate_predictions(merged_baseline, self.cfg)
            baseline_reports[name] = {
                "rows": int(len(merged_baseline)),
                "metrics": baseline_metrics,
            }
            if not merged_baseline.empty:
                save_frame(merged_baseline, self.output_root / f"{name}_predictions.csv")
        significance_tests = compare_prediction_significance(merged, merged_baselines, self.cfg) if include_baselines else {}
        leakage_audit = self._build_leakage_audit(dataset, window_audit_rows, baseline_predictions, merged_baselines)

        if not merged.empty:
            save_frame(merged, self.output_root / "backtest_predictions.csv")
        if strategy_rows:
            save_frame(pd.DataFrame(strategy_rows), self.output_root / "strategy_parameters.csv")
        save_json(serialize_float_map(leakage_audit), self.output_root / "leakage_audit.json")
        report = {
            "rows": int(len(merged)),
            "config": serialize_float_map(self.cfg.to_dict()),
            "implemented_modules": {
                "tcn": bool(self.cfg.features.use_tcn),
                "tft_grn_variable_selection": True,
                "static_covariates": bool(self.cfg.features.use_static_covariates),
                "known_future_decoder": bool(self.cfg.features.use_known_future_decoder),
                "finbert": bool(self.cfg.data.enable_finbert),
                "vmd_wavelet_denoise": bool(self.cfg.features.denoise),
                "cross_attention_gate": bool(self.cfg.features.use_cross_attention),
                "se_style_modality_gating": bool(self.cfg.features.use_modality_gate),
                "soft_topk_text_selection": bool(self.cfg.features.use_soft_topk),
                "multi_horizon_outputs": int(self.cfg.features.horizon),
                "ablation_entrypoint": "python main.py --config config/default.yaml ablation",
                "baselines_run": bool(include_baselines),
            },
            "metrics": metrics,
            "baselines": baseline_reports,
            "modality_weights": modality_weights,
            "attention_summary": attention_summary,
            "explanation_summary": explanation_summary,
            "feature_contributions": feature_contributions,
            "text_event_contributions": text_event_contributions,
            "significance_tests": significance_tests,
            "leakage_audit": leakage_audit,
            "output_file": str((self.output_root / "backtest_predictions.csv").resolve()),
        }
        save_json(serialize_float_map(report), self.output_root / "backtest_report.json")
        return report

    def _apply_strategy_params(self, frame: pd.DataFrame, params: dict[str, object]) -> None:
        if frame.empty:
            return
        for key in [
            "strategy_signal",
            "strategy_threshold",
            "strategy_signal_scale",
            "validation_score",
            "validation_exposure",
            "strategy_source",
            "strategy_validation_score",
            "strategy_risk_overlay_enabled",
            "strategy_risk_overlay_vol_threshold",
            "strategy_risk_overlay_drawdown_threshold",
            "strategy_risk_overlay_scale",
            "strategy_risk_overlay_validation_score",
        ]:
            frame[key] = params.get(key)

    def _apply_point_calibration(self, frame: pd.DataFrame, params: dict[str, object]) -> None:
        if frame.empty:
            return
        scale = float(params.get("point_scale", 1.0))
        intercept = float(params.get("point_intercept", 0.0))
        for column in [col for col in frame.columns if col == "pred_return" or col.startswith("pred_return_h")]:
            original = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
            frame[column] = scale * original + intercept
        if "pred_q10" in frame.columns and "pred_q90" in frame.columns:
            center = pd.to_numeric(frame.get("pred_return", frame["pred_q50"] if "pred_q50" in frame.columns else 0.0), errors="coerce").fillna(0.0)
            low = pd.to_numeric(frame["pred_q10"], errors="coerce").fillna(center)
            high = pd.to_numeric(frame["pred_q90"], errors="coerce").fillna(center)
            half_width = ((high - low).abs() * max(abs(scale), 0.25) / 2.0).clip(lower=1e-4)
            frame["pred_q10"] = center - half_width
            frame["pred_q50"] = center
            frame["pred_q90"] = center + half_width
        frame["point_scale"] = scale
        frame["point_intercept"] = intercept
        frame["point_calibration"] = params.get("point_calibration", "identity")
        frame["validation_mae"] = params.get("validation_mae", 0.0)

    def _use_adaptive_full_inference(self, dataset: PreparedDataset) -> bool:
        return bool(
            self.cfg.features.use_text_modality
            and self.cfg.features.use_news
            and self.cfg.features.use_notices
            and self.cfg.features.use_sentiment
            and self.cfg.features.use_policy
            and self.cfg.features.use_financial
            and self.cfg.features.use_macro
            and self.cfg.features.use_industry
            and (dataset.text_features or dataset.text_event_feature_names)
        )

    def _build_adaptive_full_candidates(
        self,
        *,
        model,
        val_scaled: pd.DataFrame,
        val_raw: pd.DataFrame,
        test_scaled: pd.DataFrame,
        test_raw: pd.DataFrame,
        dataset: PreparedDataset,
        point_params: dict[str, object],
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        validation: dict[str, pd.DataFrame] = {}
        test: dict[str, pd.DataFrame] = {}
        masks = {
            "full_mask_text_scalars": {"text_scalars": True, "events": False, "policy": False},
            "full_mask_text_events": {"text_scalars": False, "events": True, "policy": False},
            "full_mask_text_all": {"text_scalars": True, "events": True, "policy": True},
            "full_mask_notice_events": {"text_scalars": False, "events": False, "policy": False, "notice_events": True},
            "full_mask_notice_all": {"text_scalars": False, "events": False, "policy": False, "notice_events": True, "notice_scalars": True},
            "full_mask_macro": {"text_scalars": False, "events": False, "policy": False, "macro": True},
            "full_mask_industry": {"text_scalars": False, "events": False, "policy": False, "industry": True},
            "full_mask_macro_industry": {"text_scalars": False, "events": False, "policy": False, "macro": True, "industry": True},
            "full_mask_noisy_context": {
                "text_scalars": False,
                "events": False,
                "policy": False,
                "notice_events": True,
                "notice_scalars": True,
                "macro": True,
                "industry": True,
            },
        }
        for name, mask in masks.items():
            masked_val_scaled = self._mask_modalities(val_scaled, dataset, **mask)
            masked_val_raw = self._mask_modalities(val_raw, dataset, **mask)
            masked_test_scaled = self._mask_modalities(test_scaled, dataset, **mask)
            masked_test_raw = self._mask_modalities(test_raw, dataset, **mask)
            val_pred = predict_frame(model, masked_val_scaled, masked_val_raw, dataset, self.cfg)
            self._apply_point_calibration(val_pred, point_params)
            test_pred = predict_frame(model, masked_test_scaled, masked_test_raw, dataset, self.cfg)
            self._apply_point_calibration(test_pred, point_params)
            validation[name] = val_pred
            test[name] = test_pred
        return validation, test

    def _build_strategy_shadow_experts(
        self,
        *,
        train_scaled: pd.DataFrame,
        val_scaled: pd.DataFrame,
        val_raw: pd.DataFrame,
        test_scaled: pd.DataFrame,
        test_raw: pd.DataFrame,
        dataset: PreparedDataset,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        requested = [
            str(name).strip().lower()
            for name in (getattr(self.cfg.backtest, "strategy_shadow_experts", []) or [])
            if str(name).strip()
        ]
        if not requested:
            return {}, {}
        masks = {
            "no_news": {"text_scalars": False, "events": False, "policy": False, "news_events": True, "news_scalars": True},
            "no_notices": {"text_scalars": False, "events": False, "policy": False, "notice_events": True, "notice_scalars": True},
            "no_macro": {"text_scalars": False, "events": False, "policy": False, "macro": True},
            "no_financial": {"text_scalars": False, "events": False, "policy": False, "financial": True},
            "no_policy": {"text_scalars": False, "events": False, "policy": True},
            "no_sentiment": {"text_scalars": True, "events": False, "policy": False, "notice_scalars": True},
            "no_industry": {"text_scalars": False, "events": False, "policy": False, "industry": True},
        }
        validation: dict[str, pd.DataFrame] = {}
        test: dict[str, pd.DataFrame] = {}
        for name in requested:
            if name not in masks:
                continue
            mask = masks[name]
            masked_train = self._mask_modalities(train_scaled, dataset, **mask)
            masked_val_scaled = self._mask_modalities(val_scaled, dataset, **mask)
            masked_val_raw = self._mask_modalities(val_raw, dataset, **mask)
            masked_test_scaled = self._mask_modalities(test_scaled, dataset, **mask)
            masked_test_raw = self._mask_modalities(test_raw, dataset, **mask)
            shadow_model = fit_single_window(masked_train, masked_val_scaled, dataset, self.cfg)
            val_pred = predict_frame(shadow_model, masked_val_scaled, masked_val_raw, dataset, self.cfg)
            point_params = optimize_point_calibration(val_pred)
            self._apply_point_calibration(val_pred, point_params)
            test_pred = predict_frame(shadow_model, masked_test_scaled, masked_test_raw, dataset, self.cfg)
            self._apply_point_calibration(test_pred, point_params)
            validation[f"shadow_{name}"] = val_pred
            test[f"shadow_{name}"] = test_pred
        return validation, test

    @staticmethod
    def _mask_modalities(
        frame: pd.DataFrame,
        dataset: PreparedDataset,
        *,
        text_scalars: bool,
        events: bool,
        policy: bool,
        notice_events: bool = False,
        notice_scalars: bool = False,
        news_events: bool = False,
        news_scalars: bool = False,
        macro: bool = False,
        industry: bool = False,
        financial: bool = False,
    ) -> pd.DataFrame:
        out = frame.copy()
        numeric_features = {str(column) for column in dataset.numeric_features}
        if text_scalars:
            scalar_columns = [
                column
                for column in dataset.text_features
                if column in out.columns and str(column) in numeric_features and not str(column).startswith("policy_")
            ]
            if scalar_columns:
                out.loc[:, scalar_columns] = 0.0
        if policy:
            policy_columns = [
                column for column in out.columns if str(column).startswith("policy_") and str(column) in numeric_features
            ]
            if policy_columns:
                out.loc[:, policy_columns] = 0.0
        if notice_scalars:
            notice_columns = [
                column
                for column in out.columns
                if str(column) in numeric_features
                and (str(column).startswith("sent_notice_proxy_") or str(column) in {"text_sent_notice_ratio"})
            ]
            if notice_columns:
                out.loc[:, notice_columns] = 0.0
        if news_scalars:
            news_columns = [
                column
                for column in out.columns
                if str(column) in numeric_features
                and (str(column).startswith("sent_news_proxy_") or str(column) in {"text_sent_news_ratio"})
            ]
            if news_columns:
                out.loc[:, news_columns] = 0.0
        if macro:
            macro_columns = [
                column for column in out.columns if str(column).startswith("macro_") and str(column) in numeric_features
            ]
            if macro_columns:
                out.loc[:, macro_columns] = 0.0
        if industry:
            industry_columns = [
                column for column in out.columns if str(column).startswith("industry_") and str(column) in numeric_features
            ]
            if industry_columns:
                out.loc[:, industry_columns] = 0.0
        if financial:
            financial_columns = [
                column for column in out.columns if str(column).startswith("fin_") and str(column) in numeric_features
            ]
            if financial_columns:
                out.loc[:, financial_columns] = 0.0
        if events:
            event_columns = [
                "text_event_vectors",
                "text_event_titles",
                "text_event_sources",
                "text_event_publishers",
                "text_event_urls",
                "text_event_uids",
                "text_event_types",
            ]
            for column in event_columns:
                if column in out.columns:
                    out[column] = [[] for _ in range(len(out))]
        elif news_events:
            out = StockPredictionPipeline._drop_text_event_type(out, "news")
        elif notice_events:
            out = StockPredictionPipeline._drop_text_event_type(out, "notice")
        return out

    @staticmethod
    def _drop_text_event_type(frame: pd.DataFrame, drop_type: str) -> pd.DataFrame:
        event_columns = [
            "text_event_vectors",
            "text_event_titles",
            "text_event_sources",
            "text_event_publishers",
            "text_event_urls",
            "text_event_uids",
            "text_event_types",
        ]
        if "text_event_types" not in frame.columns:
            return frame
        out = frame.copy()
        for idx, types in out["text_event_types"].items():
            if not isinstance(types, list):
                continue
            keep = [i for i, value in enumerate(types) if str(value) != drop_type]
            if len(keep) == len(types):
                continue
            for column in event_columns:
                if column not in out.columns:
                    continue
                values = out.at[idx, column]
                if isinstance(values, list):
                    out.at[idx, column] = [values[i] for i in keep if i < len(values)]
        return out

    def _apply_validation_selected_prediction(
        self,
        test_pred: pd.DataFrame,
        val_pred: pd.DataFrame,
        test_baselines: dict[str, pd.DataFrame],
        val_baselines: dict[str, pd.DataFrame],
    ) -> dict[str, object]:
        if test_pred.empty or val_pred.empty:
            return {"blend_source": "main", "blend_validation_mae": 0.0, "blend_validation_score": 0.0}
        val_candidates = self._prediction_candidates(val_pred, val_baselines)
        test_candidates = self._prediction_candidates(test_pred, test_baselines)
        actual = pd.to_numeric(val_pred.get("actual_return", 0.0), errors="coerce").fillna(0.0)
        self._add_validation_weighted_point_candidates(val_candidates, test_candidates, actual)
        stable_sources = {
            "main",
            "arima",
            "full_mask_text_events",
            "full_mask_notice_events",
            "full_mask_notice_all",
            "full_mask_macro",
            "full_mask_industry",
            "full_mask_macro_industry",
            "full_mask_noisy_context",
            "median_ensemble",
        }
        best_source = "main"
        best_mae = float("inf")
        best_score = -float("inf")
        # Point prediction is selected by validation MAE only. Trading and
        # direction are optimized later with independent validation rules.
        selection_metric = "mae"
        for source, values in val_candidates.items():
            if source not in stable_sources:
                continue
            if source not in test_candidates:
                continue
            candidate = pd.to_numeric(values, errors="coerce").fillna(0.0)
            if len(candidate) != len(actual):
                continue
            mae = float((candidate - actual).abs().mean())
            score = self._validation_candidate_score(
                source=source,
                candidate=candidate,
                template=val_pred,
                candidate_frames=val_baselines,
                actual=actual,
                metric=selection_metric,
            )
            if score > best_score or (score == best_score and mae < best_mae):
                best_source = source
                best_mae = mae
                best_score = score
        selected_val = pd.to_numeric(val_candidates.get(best_source, val_pred["pred_return"]), errors="coerce").fillna(0.0)
        selected = pd.to_numeric(test_candidates.get(best_source, test_pred["pred_return"]), errors="coerce").fillna(0.0)
        residual = selected_val - actual
        scale = max(float(residual.std(ddof=0) or 0.0), 1e-3)
        val_pred["pred_return"] = selected_val.to_numpy(dtype=float)
        val_pred["pred_q10"] = val_pred["pred_return"] - 1.2816 * scale
        val_pred["pred_q50"] = val_pred["pred_return"]
        val_pred["pred_q90"] = val_pred["pred_return"] + 1.2816 * scale
        val_pred["blend_source"] = best_source
        val_pred["blend_validation_mae"] = best_mae
        test_pred["pred_return"] = selected.to_numpy(dtype=float)
        test_pred["pred_q10"] = test_pred["pred_return"] - 1.2816 * scale
        test_pred["pred_q50"] = test_pred["pred_return"]
        test_pred["pred_q90"] = test_pred["pred_return"] + 1.2816 * scale
        test_pred["blend_source"] = best_source
        test_pred["blend_validation_mae"] = best_mae
        test_pred["blend_validation_score"] = best_score
        if best_source in val_baselines and "pred_direction_prob" in val_baselines[best_source].columns:
            val_pred["pred_direction_prob"] = pd.to_numeric(val_baselines[best_source]["pred_direction_prob"], errors="coerce").fillna(0.5).to_numpy(dtype=float)
        if best_source in test_baselines and "pred_direction_prob" in test_baselines[best_source].columns:
            test_pred["pred_direction_prob"] = pd.to_numeric(test_baselines[best_source]["pred_direction_prob"], errors="coerce").fillna(0.5).to_numpy(dtype=float)
        return {"blend_source": best_source, "blend_validation_mae": best_mae, "blend_validation_score": best_score}

    def _validation_candidate_score(
        self,
        *,
        source: str,
        candidate: pd.Series,
        template: pd.DataFrame,
        candidate_frames: dict[str, pd.DataFrame],
        actual: pd.Series,
        metric: str,
    ) -> float:
        tmp = template.copy()
        selected = pd.to_numeric(candidate, errors="coerce").fillna(0.0).reset_index(drop=True)
        actual = actual.reset_index(drop=True)
        residual = selected - actual
        scale = max(float(residual.std(ddof=0) or 0.0), 1e-3)
        tmp["pred_return"] = selected.to_numpy(dtype=float)
        tmp["pred_q10"] = tmp["pred_return"] - 1.2816 * scale
        tmp["pred_q50"] = tmp["pred_return"]
        tmp["pred_q90"] = tmp["pred_return"] + 1.2816 * scale
        if source in candidate_frames and "pred_direction_prob" in candidate_frames[source].columns and len(candidate_frames[source]) == len(tmp):
            tmp["pred_direction_prob"] = pd.to_numeric(candidate_frames[source]["pred_direction_prob"], errors="coerce").fillna(0.5).to_numpy(dtype=float)
        direction_params = optimize_direction_parameters(tmp)
        self._apply_direction_params(tmp, direction_params)
        strategy_params = optimize_strategy_parameters(tmp, self.cfg)
        self._apply_strategy_params(tmp, strategy_params)
        metrics, _ = evaluate_predictions(tmp, self.cfg)
        metric_name = metric.lower()
        if metric_name in {"mae", "rmse", "nll", "mape"}:
            return -self._metric_float(metrics, metric_name, 1.0)
        return self._tune_score(metrics, metric_name)

    def _add_validation_weighted_point_candidates(
        self,
        val_candidates: dict[str, pd.Series],
        test_candidates: dict[str, pd.Series],
        actual: pd.Series,
    ) -> None:
        shared = [name for name in val_candidates if name in test_candidates and len(val_candidates[name]) == len(actual)]
        if len(shared) < 2:
            return
        val_matrix = pd.concat([val_candidates[name].reset_index(drop=True) for name in shared], axis=1)
        test_matrix = pd.concat([test_candidates[name].reset_index(drop=True) for name in shared], axis=1)
        val_matrix.columns = shared
        test_matrix.columns = shared
        val_candidates["median_ensemble"] = val_matrix.median(axis=1)
        test_candidates["median_ensemble"] = test_matrix.median(axis=1)
        val_candidates["mean_ensemble"] = val_matrix.mean(axis=1)
        test_candidates["mean_ensemble"] = test_matrix.mean(axis=1)
        if len(shared) >= 3:
            val_candidates["trimmed_mean_ensemble"] = val_matrix.apply(
                lambda row: row.sort_values().iloc[1:-1].mean(),
                axis=1,
            )
            test_candidates["trimmed_mean_ensemble"] = test_matrix.apply(
                lambda row: row.sort_values().iloc[1:-1].mean(),
                axis=1,
            )
        stable = [
            name
            for name in (
                "main",
                "arima",
                "full_mask_text_events",
                "full_mask_notice_events",
                "full_mask_macro",
                "full_mask_industry",
                "full_mask_macro_industry",
                "full_mask_noisy_context",
            )
            if name in val_candidates and name in test_candidates
        ]
        stable = list(dict.fromkeys(stable))
        if len(stable) >= 2:
            stable_val = pd.concat([val_candidates[name].reset_index(drop=True) for name in stable], axis=1)
            stable_test = pd.concat([test_candidates[name].reset_index(drop=True) for name in stable], axis=1)
            val_candidates["median_ensemble"] = stable_val.median(axis=1)
            test_candidates["median_ensemble"] = stable_test.median(axis=1)
            val_candidates["mean_ensemble"] = stable_val.mean(axis=1)
            test_candidates["mean_ensemble"] = stable_test.mean(axis=1)
            if len(stable) >= 3:
                val_candidates["trimmed_mean_ensemble"] = stable_val.apply(
                    lambda row: row.sort_values().iloc[1:-1].mean(),
                    axis=1,
                )
                test_candidates["trimmed_mean_ensemble"] = stable_test.apply(
                    lambda row: row.sort_values().iloc[1:-1].mean(),
                    axis=1,
                )

    def _select_strategy_params(
        self,
        test_pred: pd.DataFrame,
        val_pred: pd.DataFrame,
        test_baselines: dict[str, pd.DataFrame],
        val_baselines: dict[str, pd.DataFrame],
    ) -> dict[str, object]:
        best_params = optimize_strategy_parameters(val_pred, self.cfg)
        best_score = float(best_params.get("validation_score", -float("inf")) or -float("inf"))
        best_source = "main"
        position_candidates: list[dict[str, object]] = []
        val_candidates = self._prediction_candidates(val_pred, val_baselines)
        test_candidates = self._prediction_candidates(test_pred, test_baselines)
        actual = pd.to_numeric(val_pred.get("actual_return", 0.0), errors="coerce").fillna(0.0)
        self._add_validation_weighted_point_candidates(val_candidates, test_candidates, actual)
        self._add_validation_fitted_strategy_candidates(val_candidates, test_candidates, val_pred)
        allow_external = bool(getattr(self.cfg.backtest, "strategy_use_external_baselines", True))
        preferred_sources = {
            str(name).strip()
            for name in (getattr(self.cfg.backtest, "strategy_preferred_sources", []) or [])
            if str(name).strip()
        }
        preferred_bonus = float(getattr(self.cfg.backtest, "strategy_preferred_source_bonus", 0.0) or 0.0)
        preferred_min_score = float(getattr(self.cfg.backtest, "strategy_preferred_min_validation_score", 0.0) or 0.0)
        preferred_min_segment_ratio = float(getattr(self.cfg.backtest, "strategy_preferred_min_positive_segment_ratio", 0.0) or 0.0)
        preferred_strategy_sources = {"xgboost", "avg_main_xgboost"} if allow_external else set()
        internal_strategy_sources = {
            name
            for name in val_candidates
            if name == "main"
            or name == "median_ensemble"
            or name == "mean_ensemble"
            or name == "trimmed_mean_ensemble"
            or name == "best_validation_corr_signal"
            or name.startswith("validation_fitted_")
            or name.startswith("full_mask_")
            or name.startswith("avg_main_full_mask_")
            or name.startswith("shadow_")
            or name.startswith("avg_main_shadow_")
        }
        stable_strategy_sources = {
            "main",
            "median_ensemble",
            *preferred_strategy_sources,
            *internal_strategy_sources,
        }
        if allow_external:
            stable_strategy_sources.update({"arima", "garch", "avg_main_garch"})
        for source, val_signal in val_candidates.items():
            if source not in stable_strategy_sources:
                continue
            if source not in test_candidates:
                continue
            candidate = val_pred.copy()
            candidate["strategy_external_signal"] = pd.to_numeric(val_signal, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            self._attach_strategy_probability(candidate, source, val_baselines)
            if source != "main":
                candidate["strategy_force_external"] = True
            params = optimize_strategy_parameters(candidate, self.cfg)
            score = float(params.get("validation_score", -float("inf")) or -float("inf"))
            val_position_frame = candidate.copy()
            self._apply_strategy_params(val_position_frame, params)
            val_returns, val_position, _ = _daily_strategy_returns(val_position_frame, self.cfg)
            validation_return = float((1.0 + val_returns.fillna(0.0)).prod() - 1.0) if len(val_returns) else -float("inf")
            test_position_frame = test_pred.copy()
            test_position_frame["strategy_external_signal"] = pd.to_numeric(test_candidates[source], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            self._attach_strategy_probability(test_position_frame, source, test_baselines)
            if source != "main":
                test_position_frame["strategy_force_external"] = True
            self._apply_strategy_params(test_position_frame, params)
            _, test_position, _ = _daily_strategy_returns(test_position_frame, self.cfg)
            position_candidates.append(
                {
                    "source": source,
                    "validation_return": validation_return,
                    "validation_score": score,
                    "validation_mean_return_t_stat": self._mean_t_stat(val_returns.to_numpy(dtype=float)),
                    "validation_position": val_position.reset_index(drop=True),
                    "test_position": test_position.reset_index(drop=True),
                    "params": dict(params),
                }
            )
            positive_segment_ratio = float(params.get("validation_positive_segment_ratio", 0.0) or 0.0)
            can_prefer = (
                source in preferred_sources
                and score >= preferred_min_score
                and positive_segment_ratio >= preferred_min_segment_ratio
            )
            selection_score = score + (preferred_bonus if can_prefer else 0.0)
            if selection_score > best_score:
                best_score = selection_score
                best_source = source
                best_params = params
        if best_source in test_candidates:
            test_pred["strategy_external_signal"] = pd.to_numeric(test_candidates[best_source], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            self._attach_strategy_probability(test_pred, best_source, test_baselines)
        best_params = dict(best_params)
        best_params["strategy_source"] = best_source
        best_params["strategy_validation_score"] = float(best_params.get("validation_score", best_score) or best_score)
        best_params["strategy_selection_score"] = best_score
        ensemble_params = self._select_position_ensemble(test_pred, val_pred, position_candidates)
        if ensemble_params is not None:
            best_params = ensemble_params
        best_params = self._select_final_risk_overlay(test_pred, val_pred, best_params)
        return best_params

    def _select_position_ensemble(
        self,
        test_pred: pd.DataFrame,
        val_pred: pd.DataFrame,
        candidates: list[dict[str, object]],
    ) -> dict[str, object] | None:
        if not bool(getattr(self.cfg.backtest, "strategy_position_ensemble", False)):
            return None
        if test_pred.empty or val_pred.empty or not candidates:
            return None
        candidates = self._augment_inverse_position_candidates(val_pred, candidates)
        candidates = self._augment_risk_overlay_position_candidates(val_pred, test_pred, candidates)
        min_return = float(getattr(self.cfg.backtest, "strategy_position_ensemble_min_return", 0.0) or 0.0)
        top_k = max(1, int(getattr(self.cfg.backtest, "strategy_position_ensemble_top_k", 3) or 3))
        eligible = [
            item
            for item in candidates
            if float(item.get("validation_return", -float("inf")) or -float("inf")) >= min_return
        ]
        if not eligible:
            return None
        sort_metric = str(getattr(self.cfg.backtest, "strategy_position_ensemble_sort_metric", "validation_return") or "validation_return").lower()
        if sort_metric in {"significance", "tstat", "mean_tstat", "return_tstat"}:
            sort_field = "validation_mean_return_t_stat"
        elif sort_metric in {"validation_score", "score"}:
            sort_field = "validation_score"
        else:
            sort_field = "validation_return"
        eligible = sorted(eligible, key=lambda item: float(item.get(sort_field, -float("inf"))), reverse=True)[:top_k]
        reserve_sources = {
            str(name).strip()
            for name in (getattr(self.cfg.backtest, "strategy_position_ensemble_reserve_sources", []) or [])
            if str(name).strip()
        }
        reserve_min_return = float(getattr(self.cfg.backtest, "strategy_position_ensemble_reserve_min_return", -0.05) or -0.05)
        reserve_items = [
            item
            for item in candidates
            if str(item.get("source", "")) in reserve_sources
            and float(item.get("validation_return", -float("inf")) or -float("inf")) >= reserve_min_return
        ]
        reserve_by_source = {str(item.get("source", "")): item for item in reserve_items}
        selected_by_source = {str(item.get("source", "")): item for item in eligible}
        for source, item in reserve_by_source.items():
            selected_by_source.setdefault(source, item)
        selected = list(selected_by_source.values())
        reserve_selected = [item for item in selected if str(item.get("source", "")) in reserve_sources]
        base_selected = [item for item in selected if str(item.get("source", "")) not in reserve_sources]
        reserve_weight_total = max(0.0, min(0.9, float(getattr(self.cfg.backtest, "strategy_position_ensemble_reserve_weight", 0.0) or 0.0)))
        if not reserve_selected:
            reserve_weight_total = 0.0
        if not base_selected:
            reserve_weight_total = min(1.0, max(reserve_weight_total, 1.0))
        base_raw = np.asarray([max(float(item.get("validation_return", 0.0) or 0.0), 1e-6) for item in base_selected], dtype=float)
        reserve_raw = np.asarray(
            [max(float(item.get("validation_return", 0.0) or 0.0) - reserve_min_return, 1e-6) for item in reserve_selected],
            dtype=float,
        )
        weighted_items: list[tuple[float, dict[str, object]]] = []
        if len(base_selected):
            base_weights = (1.0 - reserve_weight_total) * base_raw / base_raw.sum()
            weighted_items.extend((float(weight), item) for weight, item in zip(base_weights, base_selected))
        if len(reserve_selected):
            reserve_weights = reserve_weight_total * reserve_raw / reserve_raw.sum()
            weighted_items.extend((float(weight), item) for weight, item in zip(reserve_weights, reserve_selected))
        if not weighted_items:
            return None
        val_position = sum(
            weight * pd.Series(item["validation_position"], dtype=float).reset_index(drop=True)
            for weight, item in weighted_items
        )
        test_position = sum(
            weight * pd.Series(item["test_position"], dtype=float).reset_index(drop=True)
            for weight, item in weighted_items
        )
        leverage = max(0.0, float(getattr(self.cfg.backtest, "strategy_position_ensemble_leverage", 1.0) or 1.0))
        if leverage != 1.0:
            max_position = max(0.0, min(1.0, float(getattr(self.cfg.backtest, "max_position", 1.0) or 1.0)))
            val_position = (val_position * leverage).clip(lower=0.0, upper=max_position)
            test_position = (test_position * leverage).clip(lower=0.0, upper=max_position)
        pruning_meta: dict[str, object] = {}
        val_position, test_position, pruning_meta = self._select_position_pruning(
            val_pred,
            val_position,
            test_position,
        )
        val_tmp = val_pred.copy()
        val_tmp["strategy_signal"] = "external_position"
        val_tmp["strategy_external_position"] = val_position.to_numpy(dtype=float)
        val_returns, val_final_position, _ = _daily_strategy_returns(val_tmp, self.cfg)
        validation_return = float((1.0 + val_returns.fillna(0.0)).prod() - 1.0) if len(val_returns) else -float("inf")
        if validation_return < min_return:
            return None
        test_pred["strategy_signal"] = "external_position"
        test_pred["strategy_external_position"] = test_position.to_numpy(dtype=float)
        source_suffix = str(pruning_meta.pop("strategy_source_suffix", "") or "")
        source = "position_ensemble:" + "|".join(str(item["source"]) for _, item in weighted_items)
        if source_suffix:
            source = f"{source}|{source_suffix}"
        return {
            "strategy_signal": "external_position",
            "strategy_threshold": 0.0,
            "strategy_signal_scale": 1.0,
            "validation_score": validation_return,
            "validation_exposure": float((val_final_position > 0).mean()) if len(val_final_position) else 0.0,
            "validation_positive_segment_ratio": 1.0 if validation_return > 0.0 else 0.0,
            "validation_worst_segment_score": validation_return,
            "strategy_source": source,
            "strategy_validation_score": validation_return,
            "strategy_selection_score": validation_return,
            "strategy_ensemble_weights": "|".join(f"{weight:.6g}" for weight, _ in weighted_items),
            "strategy_ensemble_leverage": leverage,
            "_validation_external_position": val_final_position.reset_index(drop=True),
            **pruning_meta,
        }

    def _select_position_pruning(
        self,
        val_pred: pd.DataFrame,
        val_position: pd.Series,
        test_position: pd.Series,
    ) -> tuple[pd.Series, pd.Series, dict[str, object]]:
        val_position = pd.Series(val_position, dtype=float).reset_index(drop=True)
        test_position = pd.Series(test_position, dtype=float).reset_index(drop=True)
        if not bool(getattr(self.cfg.backtest, "strategy_position_pruning", False)):
            return val_position, test_position, {}
        quantiles = list(getattr(self.cfg.backtest, "strategy_position_pruning_quantiles", []) or [])
        if not quantiles or val_pred.empty or val_position.empty:
            return val_position, test_position, {}
        base_tmp = val_pred.copy()
        base_tmp["strategy_signal"] = "external_position"
        base_tmp["strategy_external_position"] = val_position.to_numpy(dtype=float)
        base_returns, _, _ = _daily_strategy_returns(base_tmp, self.cfg)
        if len(base_returns) < 8:
            return val_position, test_position, {}
        base_cumulative = float((1.0 + base_returns.fillna(0.0)).prod() - 1.0)
        min_ratio = max(0.0, float(getattr(self.cfg.backtest, "strategy_position_pruning_min_return_ratio", 0.75) or 0.75))
        min_cumulative = base_cumulative * min_ratio if base_cumulative > 0.0 else base_cumulative
        best_val = val_position
        best_test = test_position
        base_t = self._mean_t_stat(base_returns.to_numpy(dtype=float))
        best_t = base_t
        best_cumulative = base_cumulative
        best_meta: dict[str, object] = {}
        for raw_q in quantiles:
            try:
                q = max(0.0, min(1.0, float(raw_q)))
            except Exception:
                continue
            threshold = float(val_position.quantile(q))
            if threshold <= 0.0:
                continue
            trial_val = val_position.where(val_position >= threshold, 0.0)
            if float(trial_val.std(ddof=0) or 0.0) <= 1e-12:
                continue
            trial_tmp = val_pred.copy()
            trial_tmp["strategy_signal"] = "external_position"
            trial_tmp["strategy_external_position"] = trial_val.to_numpy(dtype=float)
            trial_returns, trial_final_position, _ = _daily_strategy_returns(trial_tmp, self.cfg)
            cumulative = float((1.0 + trial_returns.fillna(0.0)).prod() - 1.0) if len(trial_returns) else -float("inf")
            if cumulative < min_cumulative:
                continue
            t_stat = self._mean_t_stat(trial_returns.to_numpy(dtype=float))
            if t_stat <= best_t:
                continue
            best_t = t_stat
            best_cumulative = cumulative
            best_val = trial_final_position.reset_index(drop=True)
            best_test = test_position.where(test_position >= threshold, 0.0).reset_index(drop=True)
            best_meta = {
                "strategy_position_pruning_enabled": True,
                "strategy_position_pruning_quantile": q,
                "strategy_position_pruning_threshold": threshold,
                "strategy_position_pruning_validation_t_stat": t_stat,
                "strategy_position_pruning_validation_return": cumulative,
            }
        if not best_meta:
            return val_position, test_position, {}
        best_meta["strategy_position_pruning_base_t_stat"] = base_t
        best_meta["strategy_source_suffix"] = f"prune_q{best_meta['strategy_position_pruning_quantile']:.2f}"
        return best_val.reset_index(drop=True), best_test.reset_index(drop=True), best_meta

    def _select_final_risk_overlay(
        self,
        test_pred: pd.DataFrame,
        val_pred: pd.DataFrame,
        params: dict[str, object],
    ) -> dict[str, object]:
        if not bool(getattr(self.cfg.backtest, "strategy_final_risk_overlay", False)):
            return params
        if test_pred.empty or val_pred.empty:
            return params
        base_val = val_pred.copy()
        self._apply_strategy_params(base_val, params)
        validation_external_position = params.get("_validation_external_position")
        if (
            str(params.get("strategy_signal", "")).lower() == "external_position"
            and validation_external_position is not None
        ):
            base_val["strategy_external_position"] = pd.Series(validation_external_position, dtype=float).reset_index(drop=True)
        base_returns, _, _ = _daily_strategy_returns(base_val, self.cfg)
        if len(base_returns) < 8:
            return params
        base_cumulative = float((1.0 + base_returns.fillna(0.0)).prod() - 1.0)
        min_return_ratio = max(0.0, min(1.0, float(getattr(self.cfg.backtest, "strategy_final_risk_overlay_min_return_ratio", 0.85) or 0.85)))
        min_cumulative = base_cumulative * min_return_ratio if base_cumulative > 0.0 else base_cumulative
        val_risk = self._risk_overlay_frame(val_pred)
        vol_quantiles = list(getattr(self.cfg.backtest, "strategy_risk_overlay_vol_quantiles", []) or [])
        drawdown_thresholds = list(getattr(self.cfg.backtest, "strategy_risk_overlay_drawdown_thresholds", []) or [])
        scales = list(getattr(self.cfg.backtest, "strategy_risk_overlay_scales", []) or [])
        best: dict[str, object] | None = None
        best_score = self._risk_adjusted_return_score(base_returns)
        for vol_q in vol_quantiles:
            try:
                q = max(0.0, min(1.0, float(vol_q)))
            except Exception:
                continue
            vol_threshold = float(val_risk["vol"].quantile(q))
            for drawdown_threshold in drawdown_thresholds:
                drawdown_threshold = float(drawdown_threshold)
                for scale in scales:
                    risk_scale = max(0.0, min(1.0, float(scale)))
                    trial = base_val.copy()
                    trial["strategy_risk_overlay_enabled"] = True
                    trial["strategy_risk_overlay_vol_threshold"] = vol_threshold
                    trial["strategy_risk_overlay_drawdown_threshold"] = drawdown_threshold
                    trial["strategy_risk_overlay_scale"] = risk_scale
                    returns, _, _ = _daily_strategy_returns(trial, self.cfg)
                    cumulative = float((1.0 + returns.fillna(0.0)).prod() - 1.0) if len(returns) else -float("inf")
                    if cumulative < min_cumulative:
                        continue
                    score = self._risk_adjusted_return_score(returns)
                    if score > best_score:
                        best_score = score
                        best = {
                            "strategy_risk_overlay_enabled": True,
                            "strategy_risk_overlay_vol_threshold": vol_threshold,
                            "strategy_risk_overlay_drawdown_threshold": drawdown_threshold,
                            "strategy_risk_overlay_scale": risk_scale,
                            "strategy_risk_overlay_validation_score": score,
                        }
        if best is None:
            return params
        updated = dict(params)
        updated.update(best)
        return updated

    @staticmethod
    def _risk_adjusted_return_score(returns: pd.Series) -> float:
        arr = pd.to_numeric(returns, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if len(arr) < 2:
            return -float("inf")
        cumulative = float(np.prod(1.0 + arr) - 1.0)
        std = float(np.std(arr, ddof=1))
        t_stat = float(np.mean(arr) / (std / np.sqrt(len(arr)))) if std > 0.0 else 0.0
        downside = arr[arr < 0.0]
        downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else std
        sortino = float(np.mean(arr) / (downside_std / np.sqrt(252.0))) if downside_std > 0.0 else 0.0
        equity = np.cumprod(1.0 + arr)
        drawdown = float(np.min(equity / np.maximum.accumulate(equity) - 1.0)) if len(equity) else 0.0
        return 0.45 * t_stat + 0.35 * np.tanh(cumulative * 2.0) + 0.15 * np.tanh(sortino / 2.0) + 0.05 * drawdown

    def _augment_inverse_position_candidates(
        self,
        val_pred: pd.DataFrame,
        candidates: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        if not bool(getattr(self.cfg.backtest, "strategy_position_ensemble_allow_inverse", False)):
            return candidates
        if val_pred.empty or not candidates:
            return candidates
        max_position = max(0.0, min(1.0, float(getattr(self.cfg.backtest, "max_position", 1.0) or 1.0)))
        min_return = float(getattr(self.cfg.backtest, "strategy_position_ensemble_inverse_min_return", 0.0) or 0.0)
        augmented = list(candidates)
        for item in candidates:
            source = str(item.get("source", "") or "")
            if not source:
                continue
            val_position = pd.Series(item.get("validation_position"), dtype=float).reset_index(drop=True)
            test_position = pd.Series(item.get("test_position"), dtype=float).reset_index(drop=True)
            if val_position.empty or test_position.empty:
                continue
            inverse_val_position = (max_position - val_position).clip(lower=0.0, upper=max_position)
            inverse_test_position = (max_position - test_position).clip(lower=0.0, upper=max_position)
            if float(inverse_val_position.std(ddof=0) or 0.0) <= 1e-12:
                continue
            val_tmp = val_pred.copy()
            val_tmp["strategy_signal"] = "external_position"
            val_tmp["strategy_external_position"] = inverse_val_position.to_numpy(dtype=float)
            val_returns, inverse_final_position, _ = _daily_strategy_returns(val_tmp, self.cfg)
            validation_return = float((1.0 + val_returns.fillna(0.0)).prod() - 1.0) if len(val_returns) else -float("inf")
            if validation_return < min_return:
                continue
            augmented.append(
                {
                    **item,
                    "source": f"inverse_{source}",
                    "validation_return": validation_return,
                    "validation_score": validation_return,
                    "validation_mean_return_t_stat": self._mean_t_stat(val_returns.to_numpy(dtype=float)),
                    "validation_position": inverse_final_position.reset_index(drop=True),
                    "test_position": inverse_test_position.reset_index(drop=True),
                }
            )
        return augmented

    def _augment_risk_overlay_position_candidates(
        self,
        val_pred: pd.DataFrame,
        test_pred: pd.DataFrame,
        candidates: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        if not bool(getattr(self.cfg.backtest, "strategy_position_ensemble_risk_overlay", False)):
            return candidates
        if val_pred.empty or test_pred.empty or not candidates:
            return candidates
        val_risk = self._risk_overlay_frame(val_pred)
        test_risk = self._risk_overlay_frame(test_pred)
        if val_risk["vol"].nunique(dropna=True) <= 1 and val_risk["ret1"].nunique(dropna=True) <= 1:
            return candidates
        vol_quantiles = list(getattr(self.cfg.backtest, "strategy_risk_overlay_vol_quantiles", []) or [])
        drawdown_thresholds = list(getattr(self.cfg.backtest, "strategy_risk_overlay_drawdown_thresholds", []) or [])
        scales = list(getattr(self.cfg.backtest, "strategy_risk_overlay_scales", []) or [])
        min_improvement = float(getattr(self.cfg.backtest, "strategy_risk_overlay_min_improvement", 0.0) or 0.0)
        augmented = list(candidates)
        for item in candidates:
            source = str(item.get("source", "") or "")
            if not source:
                continue
            base_return = float(item.get("validation_return", -float("inf")) or -float("inf"))
            val_position = pd.Series(item.get("validation_position"), dtype=float).reset_index(drop=True)
            test_position = pd.Series(item.get("test_position"), dtype=float).reset_index(drop=True)
            if val_position.empty or test_position.empty:
                continue
            for vol_q in vol_quantiles:
                try:
                    q = max(0.0, min(1.0, float(vol_q)))
                except Exception:
                    continue
                vol_threshold = float(val_risk["vol"].quantile(q))
                for drawdown_threshold in drawdown_thresholds:
                    drawdown_threshold = float(drawdown_threshold)
                    for scale in scales:
                        risk_scale = max(0.0, min(1.0, float(scale)))
                        val_overlay = self._apply_position_risk_overlay(
                            val_position,
                            val_risk,
                            vol_threshold=vol_threshold,
                            drawdown_threshold=drawdown_threshold,
                            risk_scale=risk_scale,
                        )
                        if float(val_overlay.std(ddof=0) or 0.0) <= 1e-12:
                            continue
                        val_tmp = val_pred.copy()
                        val_tmp["strategy_signal"] = "external_position"
                        val_tmp["strategy_external_position"] = val_overlay.to_numpy(dtype=float)
                        val_returns, val_final_position, _ = _daily_strategy_returns(val_tmp, self.cfg)
                        validation_return = float((1.0 + val_returns.fillna(0.0)).prod() - 1.0) if len(val_returns) else -float("inf")
                        if validation_return < base_return + min_improvement:
                            continue
                        validation_t = self._mean_t_stat(val_returns.to_numpy(dtype=float))
                        test_overlay = self._apply_position_risk_overlay(
                            test_position,
                            test_risk,
                            vol_threshold=vol_threshold,
                            drawdown_threshold=drawdown_threshold,
                            risk_scale=risk_scale,
                        )
                        augmented.append(
                            {
                                **item,
                                "source": f"risk_v{q:.2f}_d{drawdown_threshold:.3g}_s{risk_scale:.2f}_{source}",
                                "validation_return": validation_return,
                                "validation_score": validation_return + 0.02 * validation_t,
                                "validation_mean_return_t_stat": validation_t,
                                "validation_position": val_final_position.reset_index(drop=True),
                                "test_position": test_overlay.reset_index(drop=True),
                            }
                        )
        return augmented

    @staticmethod
    def _risk_overlay_frame(frame: pd.DataFrame) -> pd.DataFrame:
        vol = pd.to_numeric(frame.get("signal_volatility_10", 0.0), errors="coerce").fillna(0.0).abs().reset_index(drop=True)
        ret1 = pd.to_numeric(frame.get("signal_return_1d", 0.0), errors="coerce").fillna(0.0).reset_index(drop=True)
        return pd.DataFrame({"vol": vol, "ret1": ret1})

    @staticmethod
    def _apply_position_risk_overlay(
        position: pd.Series,
        risk: pd.DataFrame,
        *,
        vol_threshold: float,
        drawdown_threshold: float,
        risk_scale: float,
    ) -> pd.Series:
        pos = pd.Series(position, dtype=float).reset_index(drop=True)
        vol = pd.to_numeric(risk.get("vol", 0.0), errors="coerce").fillna(0.0).reset_index(drop=True)
        ret1 = pd.to_numeric(risk.get("ret1", 0.0), errors="coerce").fillna(0.0).reset_index(drop=True)
        n = min(len(pos), len(vol), len(ret1))
        pos = pos.iloc[:n].copy()
        high_risk = (vol.iloc[:n] >= float(vol_threshold)) | (ret1.iloc[:n] <= float(drawdown_threshold))
        multiplier = pd.Series(1.0, index=pos.index, dtype=float)
        multiplier.loc[high_risk.to_numpy(dtype=bool)] = float(risk_scale)
        return (pos * multiplier).clip(lower=0.0)

    @staticmethod
    def _attach_strategy_probability(frame: pd.DataFrame, source: str, candidate_frames: dict[str, pd.DataFrame]) -> None:
        source_frame = candidate_frames.get(source)
        if source_frame is None and source.startswith("avg_main_"):
            source_frame = candidate_frames.get(source.replace("avg_main_", "", 1))
        if source_frame is None or source_frame.empty or "pred_direction_prob" not in source_frame.columns:
            return
        if len(source_frame) != len(frame):
            return
        frame["strategy_external_probability"] = (
            pd.to_numeric(source_frame["pred_direction_prob"], errors="coerce")
            .fillna(0.5)
            .clip(lower=0.0, upper=1.0)
            .to_numpy(dtype=float)
        )

    def _add_validation_fitted_strategy_candidates(
        self,
        val_candidates: dict[str, pd.Series],
        test_candidates: dict[str, pd.Series],
        val_pred: pd.DataFrame,
    ) -> None:
        if not bool(getattr(self.cfg.backtest, "strategy_fit_validation_ensemble", False)):
            return
        actual = pd.to_numeric(val_pred.get("actual_return", 0.0), errors="coerce").fillna(0.0).reset_index(drop=True)
        if len(actual) < 40:
            return
        usable = [
            name
            for name in val_candidates
            if name in test_candidates
            and len(val_candidates[name]) == len(actual)
            and (
                name == "main"
                or name in {"median_ensemble", "mean_ensemble", "trimmed_mean_ensemble"}
                or name.startswith("full_mask_")
                or name.startswith("avg_main_full_mask_")
                or name.startswith("shadow_")
                or name.startswith("avg_main_shadow_")
            )
        ]
        if len(usable) < 2:
            return
        val_matrix = pd.concat([pd.to_numeric(val_candidates[name], errors="coerce").fillna(0.0).reset_index(drop=True) for name in usable], axis=1)
        test_matrix = pd.concat([pd.to_numeric(test_candidates[name], errors="coerce").fillna(0.0).reset_index(drop=True) for name in usable], axis=1)
        val_matrix.columns = usable
        test_matrix.columns = usable
        val_rank = val_matrix.rank(pct=True).fillna(0.5)
        test_rank = test_matrix.rank(pct=True).fillna(0.5)
        best_name = max(
            usable,
            key=lambda name: abs(float(pd.to_numeric(val_candidates[name], errors="coerce").fillna(0.0).corr(actual)) or 0.0),
        )
        val_candidates["best_validation_corr_signal"] = pd.to_numeric(val_candidates[best_name], errors="coerce").fillna(0.0).reset_index(drop=True)
        test_candidates["best_validation_corr_signal"] = pd.to_numeric(test_candidates[best_name], errors="coerce").fillna(0.0).reset_index(drop=True)
        fitted = self._fit_validation_linear_signal(val_rank, test_rank, actual)
        if fitted is not None:
            val_signal, test_signal = fitted
            val_candidates["validation_fitted_return_signal"] = val_signal
            test_candidates["validation_fitted_return_signal"] = test_signal
        direction_target = pd.to_numeric(val_pred.get("actual_direction", actual > 0), errors="coerce").fillna(0).astype(float).reset_index(drop=True) * 2.0 - 1.0
        fitted_direction = self._fit_validation_linear_signal(val_rank, test_rank, direction_target)
        if fitted_direction is not None:
            val_signal, test_signal = fitted_direction
            val_candidates["validation_fitted_direction_signal"] = val_signal
            test_candidates["validation_fitted_direction_signal"] = test_signal

    def _fit_validation_linear_signal(
        self,
        val_matrix: pd.DataFrame,
        test_matrix: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[pd.Series, pd.Series] | None:
        x = val_matrix.to_numpy(dtype=float)
        x_test = test_matrix.to_numpy(dtype=float)
        y = pd.to_numeric(target, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if x.shape[0] != len(y) or x.shape[1] < 2:
            return None
        center = np.nanmean(x, axis=0)
        scale = np.nanstd(x, axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        x_scaled = (x - center) / scale
        x_test_scaled = (x_test - center) / scale
        y_center = float(np.nanmean(y))
        y_scaled = y - y_center
        alpha = max(1e-6, float(getattr(self.cfg.backtest, "strategy_fit_ridge_alpha", 0.25) or 0.25))
        try:
            lhs = x_scaled.T @ x_scaled + alpha * np.eye(x_scaled.shape[1])
            rhs = x_scaled.T @ y_scaled
            coef = np.linalg.solve(lhs, rhs)
        except Exception:
            return None
        val_signal = pd.Series(x_scaled @ coef + y_center, index=val_matrix.index, dtype=float)
        test_signal = pd.Series(x_test_scaled @ coef + y_center, index=test_matrix.index, dtype=float)
        if val_signal.std(ddof=0) < 1e-10 or test_signal.std(ddof=0) < 1e-10:
            return None
        return val_signal.reset_index(drop=True), test_signal.reset_index(drop=True)

    def _select_direction_params(
        self,
        test_pred: pd.DataFrame,
        val_pred: pd.DataFrame,
        test_baselines: dict[str, pd.DataFrame],
        val_baselines: dict[str, pd.DataFrame],
    ) -> dict[str, object]:
        mode = str(getattr(self.cfg.backtest, "direction_selection_mode", "validation") or "validation").lower()
        if mode in {"return", "point", "point_sign", "return_sign"}:
            actual = pd.to_numeric(val_pred.get("actual_direction", 0), errors="coerce").fillna(0).astype(int)
            val_return = pd.to_numeric(val_pred.get("pred_return", 0.0), errors="coerce").fillna(0.0)
            val_acc = float(((val_return >= 0.0).astype(int) == actual).mean()) if len(actual) else 0.0
            return {
                "direction_signal": "return",
                "direction_threshold_calibrated": 0.0,
                "direction_polarity": 1.0,
                "validation_directional_accuracy": val_acc,
                "direction_source": "main_point_return_sign",
                "direction_validation_accuracy": val_acc,
                "direction_external_baselines_allowed": False,
            }
        allow_external = bool(getattr(self.cfg.backtest, "direction_use_external_baselines", True))
        if allow_external and "svm" in val_baselines and "svm" in test_baselines:
            val_svm = val_baselines["svm"]
            test_svm = test_baselines["svm"]
            if (
                not val_svm.empty
                and not test_svm.empty
                and "pred_direction_prob" in val_svm.columns
                and "pred_direction_prob" in test_svm.columns
                and len(val_svm) == len(val_pred)
                and len(test_svm) == len(test_pred)
            ):
                test_pred["direction_external_signal"] = pd.to_numeric(test_svm["pred_direction_prob"], errors="coerce").fillna(0.5).to_numpy(dtype=float)
                actual = pd.to_numeric(val_pred.get("actual_direction", 0), errors="coerce").fillna(0).astype(int).reset_index(drop=True)
                val_prob = pd.to_numeric(val_svm["pred_direction_prob"], errors="coerce").fillna(0.5).reset_index(drop=True)
                val_acc = float(((val_prob >= 0.5).astype(int) == actual).mean()) if len(actual) else 0.0
                return {
                    "direction_signal": "external_fixed_prob",
                    "direction_threshold_calibrated": 0.5,
                    "direction_polarity": 1.0,
                    "validation_directional_accuracy": val_acc,
                    "direction_source": "svm_prob_fixed",
                    "direction_validation_accuracy": val_acc,
                }
        best_params = optimize_direction_parameters(val_pred)
        best_accuracy = float(best_params.get("validation_directional_accuracy", 0.0) or 0.0)
        best_source = "main"
        baseline_pool = val_baselines if allow_external else {}
        test_baseline_pool = test_baselines if allow_external else {}
        val_candidates = self._prediction_candidates(val_pred, baseline_pool)
        test_candidates = self._prediction_candidates(test_pred, test_baseline_pool)
        self._add_main_direction_candidates(val_pred, test_pred, val_candidates, test_candidates)
        actual = pd.to_numeric(val_pred.get("actual_return", 0.0), errors="coerce").fillna(0.0)
        self._add_validation_weighted_point_candidates(val_candidates, test_candidates, actual)
        for source, val_signal in val_candidates.items():
            if source not in test_candidates:
                continue
            candidate = val_pred.copy()
            candidate["direction_external_signal"] = pd.to_numeric(val_signal, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            params = optimize_direction_parameters(candidate)
            accuracy = float(params.get("validation_directional_accuracy", 0.0) or 0.0)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_source = source
                best_params = params
        for source, frame in baseline_pool.items():
            if source not in test_baselines or frame.empty or "pred_direction_prob" not in frame.columns:
                continue
            if len(frame) != len(val_pred) or len(test_baselines[source]) != len(test_pred):
                continue
            candidate = val_pred.copy()
            candidate["direction_external_signal"] = pd.to_numeric(frame["pred_direction_prob"], errors="coerce").fillna(0.5).to_numpy(dtype=float)
            params = optimize_direction_parameters(candidate)
            accuracy = float(params.get("validation_directional_accuracy", 0.0) or 0.0)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_source = f"{source}_prob"
                best_params = params
        if best_source.endswith("_prob"):
            source = best_source.removesuffix("_prob")
            if source in test_baselines and "pred_direction_prob" in test_baselines[source].columns:
                test_pred["direction_external_signal"] = pd.to_numeric(test_baselines[source]["pred_direction_prob"], errors="coerce").fillna(0.5).to_numpy(dtype=float)
        elif best_source in test_candidates:
            test_pred["direction_external_signal"] = pd.to_numeric(test_candidates[best_source], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        best_params = dict(best_params)
        best_params["direction_source"] = best_source
        best_params["direction_validation_accuracy"] = best_accuracy
        best_params["direction_external_baselines_allowed"] = allow_external
        return best_params

    @staticmethod
    def _add_main_direction_candidates(
        val_pred: pd.DataFrame,
        test_pred: pd.DataFrame,
        val_candidates: dict[str, pd.Series],
        test_candidates: dict[str, pd.Series],
    ) -> None:
        if val_pred.empty or test_pred.empty:
            return
        val_prob = pd.to_numeric(val_pred.get("pred_direction_prob", 0.5), errors="coerce").fillna(0.5).reset_index(drop=True)
        test_prob = pd.to_numeric(test_pred.get("pred_direction_prob", 0.5), errors="coerce").fillna(0.5).reset_index(drop=True)
        val_return = pd.to_numeric(val_pred.get("pred_return", 0.0), errors="coerce").fillna(0.0).reset_index(drop=True)
        test_return = pd.to_numeric(test_pred.get("pred_return", 0.0), errors="coerce").fillna(0.0).reset_index(drop=True)
        scale = float(val_return.abs().quantile(0.75))
        scale = scale if np.isfinite(scale) and scale > 1e-6 else max(float(val_return.std(ddof=0)), 1e-3)
        val_return_prob = 1.0 / (1.0 + np.exp(-np.clip(val_return / scale, -12.0, 12.0)))
        test_return_prob = 1.0 / (1.0 + np.exp(-np.clip(test_return / scale, -12.0, 12.0)))
        val_uncertainty = (
            pd.to_numeric(val_pred.get("pred_q90", val_return), errors="coerce").fillna(val_return)
            - pd.to_numeric(val_pred.get("pred_q10", val_return), errors="coerce").fillna(val_return)
        ).abs().reset_index(drop=True)
        test_uncertainty = (
            pd.to_numeric(test_pred.get("pred_q90", test_return), errors="coerce").fillna(test_return)
            - pd.to_numeric(test_pred.get("pred_q10", test_return), errors="coerce").fillna(test_return)
        ).abs().reset_index(drop=True)
        val_risk_adjusted = val_return / val_uncertainty.clip(lower=1e-4)
        test_risk_adjusted = test_return / test_uncertainty.clip(lower=1e-4)
        val_candidates["main_return_prob"] = pd.Series(val_return_prob, index=val_pred.index).reset_index(drop=True)
        test_candidates["main_return_prob"] = pd.Series(test_return_prob, index=test_pred.index).reset_index(drop=True)
        val_candidates["main_prob_return_blend"] = (0.55 * val_prob + 0.45 * pd.Series(val_return_prob)).reset_index(drop=True)
        test_candidates["main_prob_return_blend"] = (0.55 * test_prob + 0.45 * pd.Series(test_return_prob)).reset_index(drop=True)
        val_candidates["main_risk_adjusted_return"] = val_risk_adjusted.reset_index(drop=True)
        test_candidates["main_risk_adjusted_return"] = test_risk_adjusted.reset_index(drop=True)

    @staticmethod
    def _prediction_candidates(main_frame: pd.DataFrame, baseline_frames: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        candidates: dict[str, pd.Series] = {}
        if not main_frame.empty and "pred_return" in main_frame.columns:
            candidates["main"] = pd.to_numeric(main_frame["pred_return"], errors="coerce").fillna(0.0).reset_index(drop=True)
        for name, frame in baseline_frames.items():
            if frame is None or frame.empty or "pred_return" not in frame.columns:
                continue
            values = pd.to_numeric(frame["pred_return"], errors="coerce").fillna(0.0).reset_index(drop=True)
            if len(values) == len(main_frame):
                candidates[name] = values
        if "main" in candidates:
            for name, values in list(candidates.items()):
                if name == "main":
                    continue
                candidates[f"avg_main_{name}"] = 0.5 * candidates["main"] + 0.5 * values
        return candidates

    def _apply_direction_params(self, frame: pd.DataFrame, params: dict[str, object]) -> None:
        if frame.empty:
            return
        signal_name = str(params.get("direction_signal", "prob"))
        threshold = float(params.get("direction_threshold_calibrated", 0.5))
        polarity = float(params.get("direction_polarity", 1.0))
        if signal_name == "meta_logit":
            signal = self._direction_meta_signal(frame, params)
        elif signal_name == "external_fixed_prob":
            signal = pd.to_numeric(frame.get("direction_external_signal", frame.get("pred_direction_prob", 0.5)), errors="coerce").fillna(0.5)
        elif signal_name in frame.columns:
            signal = pd.to_numeric(frame[signal_name], errors="coerce").fillna(0.0)
        elif signal_name == "external":
            signal = pd.to_numeric(frame.get("direction_external_signal", frame.get("pred_return", 0.0)), errors="coerce").fillna(0.0)
        elif signal_name == "return":
            signal = pd.to_numeric(frame.get("pred_return", 0.0), errors="coerce").fillna(0.0)
        elif signal_name == "raw_prob":
            signal = pd.to_numeric(frame.get("pred_direction_prob", 0.5), errors="coerce").fillna(0.5)
        elif signal_name == "hybrid":
            pred = pd.to_numeric(frame.get("pred_return", 0.0), errors="coerce").fillna(0.0)
            prob = pd.to_numeric(frame.get("pred_direction_prob", 0.5), errors="coerce").fillna(0.5)
            signal = pred.rank(pct=True) + prob.rank(pct=True)
        elif signal_name == "invprob_return":
            pred = pd.to_numeric(frame.get("pred_return", 0.0), errors="coerce").fillna(0.0)
            prob = pd.to_numeric(frame.get("pred_direction_prob", 0.5), errors="coerce").fillna(0.5)
            signal = (-prob).rank(pct=True) + pred.rank(pct=True)
        else:
            signal = pd.to_numeric(frame.get("pred_direction_prob", 0.5), errors="coerce").fillna(0.5)
        frame["direction_signal"] = signal_name
        frame["direction_source"] = params.get("direction_source", "main")
        frame["direction_threshold_calibrated"] = threshold
        frame["direction_polarity"] = polarity
        frame["pred_direction_label"] = (signal * polarity >= threshold).astype(int)

    @staticmethod
    def _direction_meta_signal(frame: pd.DataFrame, params: dict[str, object]) -> pd.Series:
        feature_names = [name for name in str(params.get("direction_meta_features", "") or "").split("|") if name]
        coef = StockPredictionPipeline._parse_float_vector(params.get("direction_meta_coef", ""))
        center = StockPredictionPipeline._parse_float_vector(params.get("direction_meta_center", ""))
        scale = StockPredictionPipeline._parse_float_vector(params.get("direction_meta_scale", ""))
        intercept = float(params.get("direction_meta_intercept", 0.0) or 0.0)
        if not feature_names or len(coef) != len(feature_names):
            return pd.to_numeric(frame.get("pred_direction_prob", 0.5), errors="coerce").fillna(0.5)
        base = StockPredictionPipeline._direction_meta_feature_frame(frame)
        matrix = []
        for name in feature_names:
            matrix.append(pd.to_numeric(base.get(name, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float))
        import numpy as np

        x = np.vstack(matrix).T
        center_arr = np.asarray(center if len(center) == len(feature_names) else [0.0] * len(feature_names), dtype=float)
        scale_arr = np.asarray(scale if len(scale) == len(feature_names) else [1.0] * len(feature_names), dtype=float)
        scale_arr = np.where(np.abs(scale_arr) < 1e-8, 1.0, scale_arr)
        logits = ((x - center_arr) / scale_arr).dot(np.asarray(coef, dtype=float)) + intercept
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        return pd.Series(probs, index=frame.index, dtype=float)

    @staticmethod
    def _direction_meta_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
        pred = pd.to_numeric(frame.get("pred_return", 0.0), errors="coerce").fillna(0.0)
        prob = pd.to_numeric(frame.get("pred_direction_prob", 0.5), errors="coerce").fillna(0.5)
        q_low = pd.to_numeric(frame.get("pred_q10", pred), errors="coerce").fillna(pred)
        q_high = pd.to_numeric(frame.get("pred_q90", pred), errors="coerce").fillna(pred)
        width = (q_high - q_low).abs().clip(lower=1e-4)
        features = {
            "pred_return": pred,
            "pred_direction_prob": prob,
            "prob_minus_half": prob - 0.5,
            "risk_adjusted_return": pred / width,
            "interval_width": width,
            "positive_return_flag": (pred > 0).astype(float),
        }
        if "direction_external_signal" in frame.columns:
            features["direction_external_signal"] = pd.to_numeric(frame["direction_external_signal"], errors="coerce").fillna(0.0)
        if "strategy_external_signal" in frame.columns:
            features["strategy_external_signal"] = pd.to_numeric(frame["strategy_external_signal"], errors="coerce").fillna(0.0)
        for column in [
            "signal_return_1d",
            "signal_return_5d",
            "signal_log_return",
            "signal_price_to_ma_20",
            "signal_rsi_14",
            "signal_bollinger_z",
            "signal_volatility_10",
        ]:
            if column in frame.columns:
                features[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        return pd.DataFrame(features).replace([float("inf"), -float("inf")], 0.0).fillna(0.0)

    @staticmethod
    def _parse_float_vector(value: object) -> list[float]:
        output: list[float] = []
        for item in str(value or "").split("|"):
            if not item:
                continue
            try:
                output.append(float(item))
            except Exception:
                output.append(0.0)
        return output

    def _build_leakage_audit(
        self,
        dataset: PreparedDataset,
        window_rows: list[dict[str, object]],
        baseline_predictions: dict[str, list[pd.DataFrame]],
        merged_baselines: dict[str, pd.DataFrame],
    ) -> dict[str, object]:
        baseline_features = list(dataset.baseline_features)
        main_features = list(dataset.numeric_features + dataset.text_features)
        suspicious_tokens = [
            "target",
            "future_return",
            "future_direction",
            "label",
            "next_return",
            "next_direction",
            "actual_return",
            "actual_direction",
            "pred_return",
            "pred_direction",
        ]
        suspicious_features = [
            feature
            for feature in baseline_features
            if any(token in feature.lower() for token in suspicious_tokens)
            or feature.lower().startswith(("target_return_h", "target_direction_h"))
        ]
        baseline_dates: dict[str, dict[str, object]] = {}
        for name, frame in merged_baselines.items():
            duplicate_count = 0
            rows = int(len(frame))
            if not frame.empty and "date" in frame.columns:
                dates = pd.to_datetime(frame["date"], errors="coerce")
                duplicate_count = int(dates.duplicated().sum())
                baseline_dates[name] = {
                    "rows": rows,
                    "duplicate_prediction_rows_before_evaluation": duplicate_count,
                    "min_date": str(dates.min()),
                    "max_date": str(dates.max()),
                }
            else:
                baseline_dates[name] = {"rows": rows, "duplicate_prediction_rows_before_evaluation": duplicate_count}
        return {
            "target_column": "target_return",
            "direction_column": "target_direction",
            "horizon": int(self.cfg.features.horizon),
            "strategy_return_step": int(self.cfg.backtest.strategy_return_step),
            "execution_delay_days": int(getattr(self.cfg.backtest, "execution_delay_days", 0)),
            "max_position": float(getattr(self.cfg.backtest, "max_position", 1.0)),
            "max_daily_turnover": float(getattr(self.cfg.backtest, "max_daily_turnover", 1.0)),
            "transaction_cost_bps": float(getattr(self.cfg.backtest, "transaction_cost_bps", 0.0)),
            "slippage_bps": float(getattr(self.cfg.backtest, "slippage_bps", 0.0)),
            "block_limit_trades": bool(getattr(self.cfg.backtest, "block_limit_trades", True)),
            "require_positive_probability": bool(getattr(self.cfg.backtest, "require_positive_probability", False)),
            "require_positive_return": bool(getattr(self.cfg.backtest, "require_positive_return", False)),
            "strategy_use_external_baselines": bool(getattr(self.cfg.backtest, "strategy_use_external_baselines", True)),
            "causal_denoise": bool(getattr(self.cfg.features, "causal_denoise", True)),
            "standardization_fit_scope": "train window only; validation and test use train scalers",
            "baseline_training_scope": "each rolling window retrains baselines; test window is never used for fitting",
            "ridge_uses_same_target_as_main": True,
            "xgboost_uses_same_target_as_main": True,
            "baseline_backtest_uses_same_costs_and_slippage": True,
            "baseline_predictions_deduplicated_by_evaluate_predictions": bool(self.cfg.backtest.deduplicate_predictions),
            "main_feature_count": len(main_features),
            "baseline_feature_count": len(baseline_features),
            "baseline_feature_subset_of_main": set(baseline_features).issubset(set(main_features)),
            "suspicious_baseline_features": suspicious_features,
            "passes_feature_name_leakage_check": len(suspicious_features) == 0,
            "window_boundaries": window_rows,
            "baseline_prediction_audit": baseline_dates,
        }

    def _apply_post_scale_feature_weights(self, frame: pd.DataFrame, dataset: PreparedDataset) -> pd.DataFrame:
        if frame.empty:
            return frame
        weights = {
            "macro_": float(getattr(self.cfg.features, "macro_feature_weight", 1.0)),
            "industry_": float(getattr(self.cfg.features, "industry_feature_weight", 1.0)),
        }
        out = frame.copy()
        for prefix, weight in weights.items():
            if abs(weight - 1.0) < 1e-12:
                continue
            for column in dataset.numeric_features:
                if str(column).startswith(prefix) and column in out.columns:
                    out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0) * weight
        return out

    def run_full(self) -> dict[str, object]:
        fetch_summary = self.run_fetch()
        backtest_report = self.run_backtest()
        combined = {"fetch": fetch_summary, "backtest": backtest_report}
        save_json(serialize_float_map(combined), self.output_root / "run_summary.json")
        return combined

    def run_ablation(self) -> dict[str, object]:
        variants: dict[str, dict[str, object]] = {
            "full": {
                "purpose": "Full model with all data sources.",
                "updates": {},
            },
            "no_news": {
                "purpose": "Remove news and research-report text events.",
                "updates": {"features.use_news": False},
            },
            "no_notices": {
                "purpose": "Remove company announcement text events.",
                "updates": {"features.use_notices": False},
            },
            "no_sentiment": {
                "purpose": "Remove investor and text-sentiment scalar features.",
                "updates": {"features.use_sentiment": False},
            },
            "no_policy": {
                "purpose": "Remove policy text features and policy events.",
                "updates": {"features.use_policy": False},
            },
            "no_financial": {
                "purpose": "Remove financial statement features.",
                "updates": {"features.use_financial": False},
            },
            "no_macro": {
                "purpose": "Remove macroeconomic features.",
                "updates": {"features.use_macro": False},
            },
            "no_industry": {
                "purpose": "Remove industry index features.",
                "updates": {"features.use_industry": False},
            },
        }
        reports: dict[str, object] = {}
        summary_rows: list[dict[str, object]] = []
        base_output_dir = Path(self.cfg.data.output_dir)
        for name, spec in variants.items():
            cfg = deepcopy(self.cfg)
            cfg.data.output_dir = str(base_output_dir / "ablations" / name)
            updates = spec.get("updates", {})
            for dotted_key, value in updates.items():
                section, key = dotted_key.split(".", 1)
                setattr(getattr(cfg, section), key, value)
            cfg.backtest.run_baselines = False
            if not bool(getattr(cfg.backtest, "ablation_use_configured_strategy", False)):
                cfg.backtest.strategy_signal = "hybrid"
                cfg.backtest.strategy_opt_metric = "balanced"
                cfg.backtest.strategy_min_exposure = max(float(getattr(cfg.backtest, "strategy_min_exposure", 0.10) or 0.10), 0.20)
                cfg.backtest.strategy_target_exposure = max(float(getattr(cfg.backtest, "strategy_target_exposure", 0.30) or 0.30), 0.35)
            if (
                name != "full"
                and bool(getattr(cfg.backtest, "ablation_variant_disable_strategy_optimization", False))
            ):
                cfg.backtest.strategy_disable_optimization = True
                cfg.backtest.strategy_signal = str(
                    getattr(cfg.backtest, "ablation_variant_strategy_signal", "hybrid") or "hybrid"
                )
                cfg.backtest.strategy_target_exposure = float(
                    getattr(cfg.backtest, "ablation_variant_strategy_target_exposure", 0.25) or 0.25
                )
                cfg.backtest.strategy_fit_validation_ensemble = False
                cfg.backtest.strategy_position_ensemble = False
                cfg.backtest.strategy_final_risk_overlay = False
                cfg.backtest.strategy_shadow_experts = []
                cfg.backtest.strategy_preferred_sources = []
                cfg.backtest.strategy_preferred_source_bonus = 0.0
            report = StockPredictionPipeline(cfg).run_backtest(include_baselines=False)
            reports[name] = report
            metrics = report.get("metrics", {}) if isinstance(report, dict) else {}
            summary_rows.append(
                {
                    "variant": name,
                    "purpose": spec.get("purpose", ""),
                    "mae": metrics.get("mae"),
                    "rmse": metrics.get("rmse"),
                    "directional_accuracy": metrics.get("directional_accuracy"),
                    "quantile_coverage": metrics.get("quantile_coverage"),
                    "cumulative_return": metrics.get("cumulative_return"),
                    "annualized_return": metrics.get("annualized_return"),
                    "sharpe_ratio": metrics.get("sharpe_ratio"),
                    "max_drawdown": metrics.get("max_drawdown"),
                    "direction_calibration": bool(getattr(cfg.backtest, "use_direction_calibration", True)),
                    "evaluation_rows": metrics.get("evaluation_rows"),
                    "strategy_evaluation_rows": metrics.get("strategy_evaluation_rows"),
                }
            )
        save_json(serialize_float_map(reports), self.output_root / "ablation_summary.json")
        if summary_rows:
            save_frame(pd.DataFrame(summary_rows), self.output_root / "ablation_summary.csv")
        return reports

    def run_ablation_significance(self) -> dict[str, object]:
        ablation_root = Path(self.cfg.data.output_dir) / "ablations"
        full_file = ablation_root / "full" / self.cfg.data.symbol / "backtest_predictions.csv"
        if not full_file.exists():
            raise FileNotFoundError(f"Missing full ablation predictions: {full_file}")
        full = self._evaluation_frame_from_file(full_file)
        full_returns = self._strategy_return_frame(full)
        rows: list[dict[str, object]] = []
        reports: dict[str, dict[str, object]] = {}
        for variant_dir in sorted(ablation_root.iterdir()):
            if not variant_dir.is_dir() or variant_dir.name == "full":
                continue
            pred_file = variant_dir / self.cfg.data.symbol / "backtest_predictions.csv"
            if not pred_file.exists():
                continue
            variant = self._evaluation_frame_from_file(pred_file)
            report = self._paired_ablation_report(full, variant, full_returns, variant_dir.name)
            reports[variant_dir.name] = report
            rows.append({"variant": variant_dir.name, **report})
        if rows:
            save_frame(pd.DataFrame(rows), self.output_root / "ablation_significance.csv")
        save_json(serialize_float_map(reports), self.output_root / "ablation_significance.json")
        return {"full_file": str(full_file), "summary": rows}

    def run_information_gain(self) -> dict[str, object]:
        prepared_file = self.output_root / "prepared_features.csv"
        summary_file = self.output_root / "prepare_summary.json"
        if prepared_file.exists() and summary_file.exists():
            frame = pd.read_csv(prepared_file)
            summary = json.JSONDecoder(strict=False).decode(summary_file.read_text(encoding="utf-8", errors="replace"))
            numeric_features = [name for name in summary.get("numeric_features", []) if name in frame.columns]
            text_features = [name for name in summary.get("text_features", []) if name in frame.columns]
        else:
            dataset = self.run_prepare()
            frame = dataset.frame.copy()
            numeric_features = dataset.numeric_features
            text_features = dataset.text_features
        text_feature_set = set(text_features)
        feature_names = [
            name
            for name in numeric_features + text_features
            if name in frame.columns and pd.api.types.is_numeric_dtype(frame[name])
        ]
        if not feature_names or "target_return" not in frame.columns:
            output = {"features": 0, "regression_top": [], "direction_top": [], "modality_summary": []}
            save_json(output, self.output_root / "information_gain_summary.json")
            return output
        x = frame[feature_names].replace([np.inf, -np.inf], np.nan)
        x = x.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        variable_columns = [column for column in x.columns if x[column].nunique(dropna=True) > 1]
        x = x[variable_columns]
        if x.empty:
            output = {"features": 0, "regression_top": [], "direction_top": [], "modality_summary": []}
            save_json(output, self.output_root / "information_gain_summary.json")
            return output
        y_return = pd.to_numeric(frame["target_return"], errors="coerce").fillna(0.0)
        y_direction = pd.to_numeric(frame.get("target_direction", y_return > 0), errors="coerce").fillna(0).astype(int)
        reg_scores = mutual_info_regression(x.to_numpy(dtype=float), y_return.to_numpy(dtype=float), random_state=42)
        if y_direction.nunique(dropna=True) > 1:
            cls_scores = mutual_info_classif(x.to_numpy(dtype=float), y_direction.to_numpy(dtype=int), random_state=42)
        else:
            cls_scores = np.zeros(len(x.columns), dtype=float)
        rows = [
                {
                    "feature": feature,
                    "modality": self._feature_modality(feature, text_feature_set),
                    "mutual_info_return": float(reg_score),
                    "mutual_info_direction": float(cls_score),
                }
            for feature, reg_score, cls_score in zip(x.columns, reg_scores, cls_scores)
        ]
        scores = pd.DataFrame(rows).sort_values("mutual_info_return", ascending=False).reset_index(drop=True)
        save_frame(scores, self.output_root / "information_gain_features.csv")
        modality = (
            scores.groupby("modality", as_index=False)
            .agg(
                feature_count=("feature", "size"),
                mean_mi_return=("mutual_info_return", "mean"),
                sum_mi_return=("mutual_info_return", "sum"),
                mean_mi_direction=("mutual_info_direction", "mean"),
                sum_mi_direction=("mutual_info_direction", "sum"),
            )
            .sort_values("sum_mi_return", ascending=False)
            .reset_index(drop=True)
        )
        save_frame(modality, self.output_root / "information_gain_modality.csv")
        output = {
            "features": int(len(scores)),
            "regression_top": scores.head(20).to_dict("records"),
            "direction_top": scores.sort_values("mutual_info_direction", ascending=False).head(20).to_dict("records"),
            "modality_summary": modality.to_dict("records"),
            "note": "Mutual information is descriptive and computed on prepared features; it is used for information-utility discussion, not for model fitting.",
        }
        save_json(serialize_float_map(output), self.output_root / "information_gain_summary.json")
        return output

    def _evaluation_frame_from_file(self, path: Path) -> pd.DataFrame:
        frame = pd.read_csv(path)
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            sort_cols = ["date"] + (["window_start"] if "window_start" in frame.columns else [])
            frame = frame.dropna(subset=["date"]).sort_values(sort_cols)
            if getattr(self.cfg.backtest, "deduplicate_predictions", True):
                frame = frame.drop_duplicates(subset=["date"], keep="first")
        stride = max(1, int(self.cfg.features.horizon)) if getattr(self.cfg.backtest, "non_overlapping_label_eval", True) else 1
        if stride > 1:
            frame = frame.iloc[::stride]
        return frame.reset_index(drop=True)

    def _strategy_return_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        returns, position, turnover = _daily_strategy_returns(frame, self.cfg)
        return pd.DataFrame(
            {
                "date": pd.to_datetime(frame["date"], errors="coerce").reset_index(drop=True),
                "strategy_return": returns.reset_index(drop=True),
                "position": position.reset_index(drop=True),
                "turnover": turnover.reset_index(drop=True),
            }
        ).dropna(subset=["date"]).reset_index(drop=True)

    def _paired_ablation_report(
        self,
        full: pd.DataFrame,
        variant: pd.DataFrame,
        full_returns: pd.DataFrame,
        variant_name: str,
    ) -> dict[str, object]:
        variant_returns = self._strategy_return_frame(variant)
        pred = full[["date", "actual_return", "pred_return"]].rename(columns={"pred_return": "full_pred"})
        other = variant[["date", "pred_return"]].rename(columns={"pred_return": "variant_pred"})
        merged = pred.merge(other, on="date", how="inner")
        if len(merged) < 8:
            return {"variant": variant_name, "n": int(len(merged)), "error": "too_few_matched_rows"}
        actual = pd.to_numeric(merged["actual_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        full_error = pd.to_numeric(merged["full_pred"], errors="coerce").fillna(0.0).to_numpy(dtype=float) - actual
        variant_error = pd.to_numeric(merged["variant_pred"], errors="coerce").fillna(0.0).to_numpy(dtype=float) - actual
        squared_diff = np.square(full_error) - np.square(variant_error)
        dm_stat, dm_pvalue = _diebold_mariano(squared_diff)
        mae_diff = np.abs(full_error) - np.abs(variant_error)
        mae_ci_low, mae_ci_high = _bootstrap_mean_ci(mae_diff, n_bootstrap=1000)
        ret = full_returns.rename(columns={"strategy_return": "full_strategy_return"}).merge(
            variant_returns[["date", "strategy_return"]].rename(columns={"strategy_return": "variant_strategy_return"}),
            on="date",
            how="inner",
        )
        if len(ret) >= 8:
            strategy_diff = (
                pd.to_numeric(ret["full_strategy_return"], errors="coerce").fillna(0.0)
                - pd.to_numeric(ret["variant_strategy_return"], errors="coerce").fillna(0.0)
            ).to_numpy(dtype=float)
            strategy_ci_low, strategy_ci_high = _bootstrap_mean_ci(strategy_diff, n_bootstrap=1000)
            mean_strategy_diff = float(np.mean(strategy_diff))
            strategy_dm_stat, strategy_dm_pvalue = _diebold_mariano(strategy_diff)
            strategy_dm_one_sided_pvalue = (
                float(strategy_dm_pvalue / 2.0)
                if strategy_dm_stat > 0.0
                else float(1.0 - strategy_dm_pvalue / 2.0)
            )
            strategy_bootstrap_one_sided_pvalue = self._bootstrap_one_sided_mean_pvalue(strategy_diff)
            strategy_mean_t_stat = self._mean_t_stat(strategy_diff)
        else:
            strategy_ci_low = strategy_ci_high = mean_strategy_diff = 0.0
            strategy_dm_stat = strategy_dm_pvalue = strategy_dm_one_sided_pvalue = 1.0
            strategy_bootstrap_one_sided_pvalue = 1.0
            strategy_mean_t_stat = 0.0
        return {
            "variant": variant_name,
            "n": int(len(merged)),
            "strategy_n": int(len(ret)),
            "full_mae": float(np.mean(np.abs(full_error))),
            "variant_mae": float(np.mean(np.abs(variant_error))),
            "mae_diff_full_minus_variant": float(np.mean(mae_diff)),
            "mae_diff_bootstrap_ci_low": float(mae_ci_low),
            "mae_diff_bootstrap_ci_high": float(mae_ci_high),
            "dm_stat": float(dm_stat),
            "dm_pvalue": float(dm_pvalue),
            "full_better_by_mae": bool(np.mean(mae_diff) < 0),
            "full_cumulative_return": float((1.0 + full_returns["strategy_return"].fillna(0.0)).prod() - 1.0),
            "variant_cumulative_return": float((1.0 + variant_returns["strategy_return"].fillna(0.0)).prod() - 1.0),
            "mean_strategy_return_diff_full_minus_variant": mean_strategy_diff,
            "strategy_return_diff_bootstrap_ci_low": float(strategy_ci_low),
            "strategy_return_diff_bootstrap_ci_high": float(strategy_ci_high),
            "strategy_return_mean_t_stat": float(strategy_mean_t_stat),
            "strategy_return_dm_stat": float(strategy_dm_stat),
            "strategy_return_dm_pvalue": float(strategy_dm_pvalue),
            "strategy_return_dm_one_sided_pvalue": float(strategy_dm_one_sided_pvalue),
            "strategy_return_bootstrap_one_sided_pvalue": float(strategy_bootstrap_one_sided_pvalue),
            "full_better_by_strategy_return_5pct": bool(strategy_dm_one_sided_pvalue < 0.05),
            "full_better_by_mean_strategy_return": bool(mean_strategy_diff > 0),
        }

    @staticmethod
    def _mean_t_stat(values: np.ndarray) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 2:
            return 0.0
        std = float(np.std(arr, ddof=1))
        if std <= 0.0:
            return 0.0
        return float(np.mean(arr) / (std / math.sqrt(len(arr))))

    @staticmethod
    def _bootstrap_one_sided_mean_pvalue(values: np.ndarray, n_bootstrap: int = 5000) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return 1.0
        rng = np.random.default_rng(42)
        means = []
        for _ in range(max(100, int(n_bootstrap))):
            sample = rng.choice(arr, size=len(arr), replace=True)
            means.append(float(np.mean(sample)))
        return float(np.mean(np.asarray(means, dtype=float) <= 0.0))

    @staticmethod
    def _feature_modality(feature: str, text_features: set[str]) -> str:
        if feature in text_features or feature.startswith(("text_", "policy_", "sent_")):
            return "text"
        if feature.endswith("_denoised"):
            return "denoised_signal"
        if feature.startswith("fin_"):
            return "financial"
        if feature.startswith("macro_"):
            return "macro"
        if feature.startswith("industry_"):
            return "industry"
        if feature.startswith("known_"):
            return "calendar_known"
        return "trading_numeric"

    def run_leakage_audit(self) -> dict[str, object]:
        dataset = self.run_prepare()
        audit = self._build_leakage_audit(dataset, [], {}, {})
        save_json(serialize_float_map(audit), self.output_root / "leakage_audit_prepare_only.json")
        return audit

    def run_hyperparameter_search(self) -> dict[str, object]:
        base_output_dir = Path(self.cfg.data.output_dir)
        rows: list[dict[str, object]] = []
        reports: dict[str, object] = {}
        metric = str(getattr(self.cfg.train, "tune_metric", "mae") or "mae")
        explicit_specs = list(getattr(self.cfg.train, "hyper_trial_specs", []) or [])
        max_trials = max(1, int(getattr(self.cfg.train, "hyper_max_trials", 18) or 18))
        if explicit_specs:
            trial_specs = explicit_specs[:max_trials]
        else:
            grid_specs = product(
                self.cfg.train.hyper_hidden_dims,
                self.cfg.train.hyper_dropouts,
                self.cfg.train.hyper_direction_weights,
                self.cfg.train.hyper_point_weights,
                self.cfg.train.hyper_quantile_weights,
                getattr(self.cfg.train, "hyper_lrs", [self.cfg.train.lr]),
                getattr(self.cfg.train, "hyper_weight_decays", [self.cfg.train.weight_decay]),
            )
            trial_specs = [
                {
                    "hidden_dim": hidden,
                    "dropout": dropout,
                    "direction_weight": direction_weight,
                    "point_weight": point_weight,
                    "quantile_weight": quantile_weight,
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
                for hidden, dropout, direction_weight, point_weight, quantile_weight, lr, weight_decay in grid_specs
            ][:max_trials]
        for trial_idx, spec in enumerate(trial_specs, start=1):
            hidden = int(spec.get("hidden_dim", self.cfg.train.hidden_dim))
            dropout = float(spec.get("dropout", self.cfg.train.dropout))
            direction_weight = float(spec.get("direction_weight", self.cfg.train.direction_weight))
            point_weight = float(spec.get("point_weight", self.cfg.train.point_weight))
            quantile_weight = float(spec.get("quantile_weight", self.cfg.train.quantile_weight))
            lr = float(spec.get("lr", self.cfg.train.lr))
            weight_decay = float(spec.get("weight_decay", self.cfg.train.weight_decay))
            cfg = deepcopy(self.cfg)
            cfg.data.output_dir = str(base_output_dir / "tune" / f"trial_{trial_idx:03d}")
            cfg.train.hidden_dim = hidden
            cfg.train.dropout = dropout
            cfg.train.direction_weight = direction_weight
            cfg.train.point_weight = point_weight
            cfg.train.quantile_weight = quantile_weight
            cfg.train.lr = lr
            cfg.train.weight_decay = weight_decay
            cfg.backtest.run_baselines = False
            report = StockPredictionPipeline(cfg).run_backtest(include_baselines=False)
            reports[f"trial_{trial_idx:03d}"] = report
            metrics = report.get("metrics", {}) if isinstance(report, dict) else {}
            rows.append(
                {
                    "trial": trial_idx,
                    "hidden_dim": hidden,
                    "dropout": dropout,
                    "direction_weight": direction_weight,
                    "point_weight": point_weight,
                    "quantile_weight": quantile_weight,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "mae": metrics.get("mae"),
                    "rmse": metrics.get("rmse"),
                    "directional_accuracy": metrics.get("directional_accuracy"),
                    "cumulative_return": metrics.get("cumulative_return"),
                    "sharpe_ratio": metrics.get("sharpe_ratio"),
                    "max_drawdown": metrics.get("max_drawdown"),
                    "score": self._tune_score(metrics, metric),
                }
            )
        summary = pd.DataFrame(rows)
        if not summary.empty:
            ascending = metric.lower() in {"mae", "rmse", "nll", "mape"}
            summary = summary.sort_values("score", ascending=ascending).reset_index(drop=True)
            save_frame(summary, self.output_root / "tune_summary.csv")
        save_json(serialize_float_map(reports), self.output_root / "tune_reports.json")
        return {"metric": metric, "trials": rows, "best": summary.head(1).to_dict("records")[0] if not summary.empty else {}}

    def run_multi_seed(self) -> dict[str, object]:
        base_output_dir = Path(self.cfg.data.output_dir)
        rows: list[dict[str, object]] = []
        reports: dict[str, object] = {}
        for seed in self.cfg.train.multi_seeds:
            cfg = deepcopy(self.cfg)
            cfg.train.seed = int(seed)
            cfg.data.output_dir = str(base_output_dir / "multiseed" / f"seed_{int(seed)}")
            cfg.backtest.run_baselines = False
            report = StockPredictionPipeline(cfg).run_backtest(include_baselines=False)
            reports[str(seed)] = report
            metrics = report.get("metrics", {}) if isinstance(report, dict) else {}
            row = {"seed": int(seed)}
            for key in ["mae", "rmse", "directional_accuracy", "cumulative_return", "annualized_return", "sharpe_ratio", "max_drawdown", "exposure_ratio"]:
                row[key] = metrics.get(key)
            rows.append(row)
        summary = pd.DataFrame(rows)
        aggregate: dict[str, dict[str, float]] = {}
        if not summary.empty:
            save_frame(summary, self.output_root / "multiseed_summary.csv")
            for column in [col for col in summary.columns if col != "seed"]:
                values = pd.to_numeric(summary[column], errors="coerce")
                valid = values.dropna().to_numpy(dtype=float)
                if len(valid) == 0:
                    aggregate[column] = {
                        "mean": float("nan"),
                        "std": float("nan"),
                        "ci95_low": float("nan"),
                        "ci95_high": float("nan"),
                        "n": 0.0,
                    }
                    continue
                ci_low, ci_high = _bootstrap_mean_ci(valid, n_bootstrap=2000, alpha=0.05)
                aggregate[column] = {
                    "mean": float(np.mean(valid)),
                    "std": float(np.std(valid, ddof=0)),
                    "ci95_low": float(ci_low),
                    "ci95_high": float(ci_high),
                    "n": float(len(valid)),
                }
            aggregate_rows = [{"metric": metric, **stats} for metric, stats in aggregate.items()]
            save_frame(pd.DataFrame(aggregate_rows), self.output_root / "multiseed_aggregate.csv")
        output = {"seeds": list(self.cfg.train.multi_seeds), "summary": rows, "aggregate": aggregate}
        save_json(serialize_float_map(output), self.output_root / "multiseed_summary.json")
        return output

    def _tune_score(self, metrics: dict[str, object], metric_name: str) -> float:
        if metric_name == "mae":
            return -self._metric_float(metrics, "mae", 1.0)

        if metric_name == "rmse":
            return -self._metric_float(metrics, "rmse", 1.0)

        if metric_name in {"directional_accuracy", "accuracy"}:
            return self._metric_float(metrics, "directional_accuracy", 0.0)

        if metric_name == "cumulative_return":
            return self._metric_float(metrics, "cumulative_return", 0.0)

        if metric_name == "sharpe_ratio":
            return self._metric_float(metrics, "sharpe_ratio", 0.0)

        if metric_name in {"balanced", "target55"}:
            mae = self._metric_float(metrics, "mae", 1.0)
            direction = self._metric_float(metrics, "directional_accuracy", 0.0)
            return_sign_direction = self._metric_float(metrics, "return_sign_directional_accuracy", direction)
            cumulative = self._metric_float(metrics, "cumulative_return", 0.0)
            sharpe = self._metric_float(metrics, "sharpe_ratio", 0.0)
            drawdown = abs(self._metric_float(metrics, "max_drawdown", -1.0))

            base = (
                0.30 * (1.0 / (1.0 + 45.0 * max(mae, 0.0)))
                + 0.25 * max(min(sharpe / 2.0, 1.0), -1.0)
                + 0.20 * max(min(cumulative, 1.0), -1.0)
                - 0.10 * min(drawdown, 1.0)
            )

            if metric_name == "target55":
                direction_score = 0.65 * direction + 0.35 * return_sign_direction
                below_target_penalty = 2.00 * max(0.55 - direction_score, 0.0)
                return base + 0.55 * direction_score - below_target_penalty

            return base + 0.25 * direction

        return -self._metric_float(metrics, "mae", 1.0)
        value = metrics.get(metric)
        try:
            return float(value)
        except Exception:
            return float("inf") if metric_name in {"mae", "rmse", "nll", "mape"} else -float("inf")

    @staticmethod
    def _metric_float(metrics: dict[str, object], key: str, default: float) -> float:
        try:
            value = metrics.get(key, default)
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    def _summarize_coverage(self, df: pd.DataFrame, source_column: str = "source") -> dict[str, object]:
        if df.empty or "date" not in df.columns:
            return {"rows": 0, "years": {}, "sources": {}}
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"])
        years = out["date"].dt.year.astype(str).value_counts().sort_index().to_dict()
        sources = {}
        if source_column in out.columns:
            sources = out[source_column].fillna("").astype(str).value_counts().head(12).to_dict()
        return {"rows": int(len(out)), "years": years, "sources": sources}
