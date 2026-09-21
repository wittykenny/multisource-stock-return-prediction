import unittest

import pandas as pd

from stock_predictor.config import AppConfig
from stock_predictor.pipeline import StockPredictionPipeline


class StrategySignificanceSelectionTest(unittest.TestCase):
    def test_position_ensemble_significance_sort_prefers_higher_validation_t_stat(self):
        cfg = AppConfig()
        cfg.backtest.strategy_position_ensemble = True
        cfg.backtest.strategy_position_ensemble_top_k = 1
        cfg.backtest.strategy_position_ensemble_min_return = -1.0
        cfg.backtest.strategy_position_ensemble_sort_metric = "significance"
        cfg.backtest.strategy_position_ensemble_leverage = 1.0
        cfg.backtest.strategy_position_ensemble_allow_inverse = False
        cfg.backtest.strategy_position_ensemble_risk_overlay = False
        cfg.backtest.max_position = 1.0
        cfg.backtest.max_daily_turnover = 1.0
        cfg.backtest.transaction_cost_bps = 0.0
        cfg.backtest.slippage_bps = 0.0

        frame = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=12),
                "actual_return": [0.01] * 12,
                "pred_return": [0.0] * 12,
                "pred_direction_prob": [0.5] * 12,
            }
        )
        volatile = pd.Series([1.0, 0.0] * 6, dtype=float)
        stable = pd.Series([0.4] * 12, dtype=float)
        candidates = [
            {
                "source": "volatile",
                "validation_return": 0.08,
                "validation_score": 0.08,
                "validation_mean_return_t_stat": 1.0,
                "validation_position": volatile,
                "test_position": volatile,
                "params": {},
            },
            {
                "source": "stable",
                "validation_return": 0.05,
                "validation_score": 0.05,
                "validation_mean_return_t_stat": 3.0,
                "validation_position": stable,
                "test_position": stable,
                "params": {},
            },
        ]

        params = StockPredictionPipeline(cfg)._select_position_ensemble(frame.copy(), frame.copy(), candidates)

        self.assertIsNotNone(params)
        self.assertEqual(params["strategy_source"], "position_ensemble:stable")

    def test_position_ensemble_prunes_low_validation_positions_for_significance(self):
        cfg = AppConfig()
        cfg.backtest.strategy_position_ensemble = True
        cfg.backtest.strategy_position_ensemble_top_k = 1
        cfg.backtest.strategy_position_ensemble_min_return = -1.0
        cfg.backtest.strategy_position_ensemble_sort_metric = "significance"
        cfg.backtest.strategy_position_ensemble_leverage = 1.0
        cfg.backtest.strategy_position_ensemble_allow_inverse = False
        cfg.backtest.strategy_position_ensemble_risk_overlay = False
        cfg.backtest.strategy_position_pruning = True
        cfg.backtest.strategy_position_pruning_quantiles = [0.5]
        cfg.backtest.strategy_position_pruning_min_return_ratio = 0.0
        cfg.backtest.max_position = 1.0
        cfg.backtest.max_daily_turnover = 1.0
        cfg.backtest.transaction_cost_bps = 0.0
        cfg.backtest.slippage_bps = 0.0

        frame = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=8),
                "actual_return": [-0.01, -0.01, -0.01, -0.01, 0.02, 0.02, 0.02, 0.02],
                "pred_return": [0.0] * 8,
                "pred_direction_prob": [0.5] * 8,
            }
        )
        position = pd.Series([0.1, 0.1, 0.1, 0.1, 0.8, 0.8, 0.8, 0.8], dtype=float)
        candidates = [
            {
                "source": "main",
                "validation_return": 0.02,
                "validation_score": 0.02,
                "validation_mean_return_t_stat": 0.5,
                "validation_position": position,
                "test_position": position,
                "params": {},
            },
        ]

        params = StockPredictionPipeline(cfg)._select_position_ensemble(frame.copy(), frame.copy(), candidates)

        self.assertIsNotNone(params)
        self.assertEqual(params["strategy_position_pruning_quantile"], 0.5)
        self.assertEqual(params["strategy_position_pruning_threshold"], 0.45)
        self.assertEqual(params["strategy_source"], "position_ensemble:main|prune_q0.50")


if __name__ == "__main__":
    unittest.main()
