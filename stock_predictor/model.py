from __future__ import annotations

import ast
import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVC, SVR
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .config import AppConfig
from .features import PreparedDataset
from .utils import annualized_return, max_drawdown, safe_divide, sharpe_ratio


@dataclass
class FittedWindowModel:
    net: "TFTLite"
    feature_names: list[str]
    numeric_features: list[str]
    text_features: list[str]
    static_features: list[str]
    quantiles: list[float]
    lookback: int
    horizon: int
    event_dim: int
    text_top_k: int
    device: torch.device
    residual_scale: float


class WindowDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, dataset: PreparedDataset, cfg: AppConfig, predict_all: bool = False) -> None:
        self.frame = frame.reset_index(drop=True)
        self.dataset = dataset
        self.cfg = cfg
        self.lookback = max(1, int(cfg.features.lookback))
        self.predict_all = predict_all
        first = 0 if predict_all else self.lookback - 1
        self.indices = list(range(first, len(self.frame)))
        self.use_cache = bool(getattr(cfg.train, "tensor_cache", True))
        if self.use_cache:
            self._numeric_all = _feature_matrix(self.frame, dataset.numeric_features)
            self._text_all = _feature_matrix(self.frame, dataset.text_features)
            self._events_all = np.stack([_event_matrix(row, dataset) for _, row in self.frame.iterrows()]).astype(np.float32)
            self._static_all = np.stack([_static_vector(row, dataset) for _, row in self.frame.iterrows()]).astype(np.float32)
            self._future_all = np.stack([_future_known_matrix(row, dataset, cfg) for _, row in self.frame.iterrows()]).astype(np.float32)
            self._target_all = np.stack([_target_vector(row, cfg, "target_return_h") for _, row in self.frame.iterrows()]).astype(np.float32)
            self._direction_all = np.stack([_target_vector(row, cfg, "target_direction_h") for _, row in self.frame.iterrows()]).astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        idx = self.indices[item]
        if self.use_cache:
            return self._cached_tensor_sample(idx)
        return _tensor_sample(self.frame, idx, self.dataset, self.cfg)

    def _cached_tensor_sample(self, idx: int) -> dict[str, torch.Tensor]:
        start = max(0, idx - self.lookback + 1)
        numeric = self._numeric_all[start : idx + 1]
        text = self._text_all[start : idx + 1]
        if len(numeric) < self.lookback:
            pad_len = self.lookback - len(numeric)
            numeric = np.concatenate([np.repeat(numeric[:1], pad_len, axis=0), numeric], axis=0)
            text = np.concatenate([np.repeat(text[:1], pad_len, axis=0), text], axis=0)
        return {
            "numeric": torch.from_numpy(np.ascontiguousarray(numeric, dtype=np.float32)),
            "text": torch.from_numpy(np.ascontiguousarray(text, dtype=np.float32)),
            "events": torch.from_numpy(np.ascontiguousarray(self._events_all[idx], dtype=np.float32)),
            "static": torch.from_numpy(np.ascontiguousarray(self._static_all[idx], dtype=np.float32)),
            "future_known": torch.from_numpy(np.ascontiguousarray(self._future_all[idx], dtype=np.float32)),
            "target": torch.tensor(float(self.frame.iloc[idx].get("target_return", 0.0) or 0.0), dtype=torch.float32),
            "direction": torch.tensor(float(self.frame.iloc[idx].get("target_direction", 0.0) or 0.0), dtype=torch.float32),
            "target_vector": torch.from_numpy(np.ascontiguousarray(self._target_all[idx], dtype=np.float32)),
            "direction_vector": torch.from_numpy(np.ascontiguousarray(self._direction_all[idx], dtype=np.float32)),
            "row_index": torch.tensor(idx, dtype=torch.long),
        }


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        output_dim = output_dim or input_dim
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(output_dim, output_dim)
        self.skip = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.out(self.dropout(F.elu(self.proj(x))))
        gated = z * torch.sigmoid(self.gate(z))
        return self.norm(self.skip(x) + gated)


class GateAddNorm(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        gated = self.dropout(x) * torch.sigmoid(self.gate(x))
        return self.norm(residual + gated)


class TemporalConvBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.net(x.transpose(1, 2)).transpose(1, 2)
        y = y[:, : x.shape[1], :]
        return self.norm(residual + y)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.n_features = max(1, n_features)
        self.selector = nn.Linear(self.n_features, self.n_features)
        self.project = nn.Linear(self.n_features, hidden_dim)
        self.grn = GatedResidualNetwork(hidden_dim, hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.selector(x), dim=-1)
        selected = self.project(x * weights)
        return self.grn(selected), weights


def scale_frame(
    frame: pd.DataFrame,
    features: list[str],
    scalers: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    out = frame.copy()
    fitted = scalers or {}
    for feature in features:
        if feature not in out.columns:
            out[feature] = 0.0
        values = pd.to_numeric(out[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if scalers is None:
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            fitted[feature] = (mean, std if std > 1e-12 else 1.0)
        mean, std = fitted.get(feature, (0.0, 1.0))
        out[feature] = (values - mean) / (std if std > 1e-12 else 1.0)
    return out, fitted


class TFTLite(nn.Module):
    """Temporal Fusion Transformer with multimodal cross-attention and modality gating."""

    def __init__(
        self,
        n_numeric: int,
        n_text: int,
        n_static: int,
        n_known_future: int,
        event_dim: int,
        quantiles: list[float],
        cfg: AppConfig,
    ) -> None:
        super().__init__()
        hidden = int(cfg.train.hidden_dim)
        heads = max(1, min(int(cfg.train.heads), hidden))
        while hidden % heads:
            heads -= 1
        self.n_numeric = n_numeric
        self.n_text = n_text
        self.n_static = n_static
        self.n_known_future = n_known_future
        self.event_dim = max(1, event_dim)
        self.quantiles = quantiles
        self.horizon = max(1, int(cfg.features.horizon))
        self.text_top_k = max(1, int(cfg.features.text_selection_top_k))
        self.use_tcn = bool(cfg.features.use_tcn)
        self.use_static = bool(cfg.features.use_static_covariates)
        self.use_known_future = bool(cfg.features.use_known_future_decoder)
        self.use_cross_attention = bool(cfg.features.use_cross_attention)
        self.use_modality_gate = bool(cfg.features.use_modality_gate)
        self.use_soft_topk = bool(cfg.features.use_soft_topk)
        self.use_text_modality = bool(cfg.features.use_text_modality)
        self.event_quality_gate_strength = max(0.0, float(getattr(cfg.train, "text_event_quality_gate_strength", 0.0) or 0.0))
        self.modality_prior_strength = max(0.0, min(1.0, float(getattr(cfg.train, "modality_prior_strength", 0.0) or 0.0)))
        prior = torch.tensor(
            [
                float(getattr(cfg.train, "modality_numeric_prior", 0.60) or 0.60),
                float(getattr(cfg.train, "modality_text_prior", 0.20) or 0.20),
                float(getattr(cfg.train, "modality_cross_prior", 0.20) or 0.20),
            ],
            dtype=torch.float32,
        ).clamp_min(1e-6)
        self.register_buffer("modality_prior", prior / prior.sum())

        self.numeric_vsn = VariableSelectionNetwork(max(1, n_numeric), hidden, cfg.train.dropout)
        self.text_vsn = VariableSelectionNetwork(max(1, n_text), hidden, cfg.train.dropout)
        self.tcn = TemporalConvBlock(hidden * 2, cfg.train.dropout)
        self.input_fusion = GatedResidualNetwork(hidden * 2, hidden, hidden, cfg.train.dropout)

        self.static_encoder = nn.Sequential(
            nn.Linear(max(1, n_static), hidden),
            nn.ReLU(),
            GatedResidualNetwork(hidden, hidden, dropout=cfg.train.dropout),
        )
        self.static_feature_selector = nn.Linear(max(1, n_static), max(1, n_static))
        self.static_selection_context = GatedResidualNetwork(max(1, n_static), hidden, hidden, cfg.train.dropout)
        self.static_enrichment_context = GatedResidualNetwork(max(1, n_static), hidden, hidden, cfg.train.dropout)
        self.static_state_h = nn.Linear(hidden, hidden)
        self.static_state_c = nn.Linear(hidden, hidden)

        self.encoder = nn.LSTM(hidden, hidden, batch_first=True, dropout=0.0)
        self.future_feature_selector = nn.Linear(max(1, n_known_future), max(1, n_known_future))
        self.future_proj = nn.Linear(max(1, n_known_future), hidden)
        self.future_grn = GatedResidualNetwork(hidden, hidden, dropout=cfg.train.dropout)
        self.future_decoder = nn.LSTM(hidden, hidden, batch_first=True, dropout=0.0)
        self.decoder_seed = GatedResidualNetwork(hidden, hidden, dropout=cfg.train.dropout)
        self.post_lstm_gate = GateAddNorm(hidden, cfg.train.dropout)
        self.static_enrichment = GatedResidualNetwork(hidden * 2, hidden, hidden, cfg.train.dropout)
        self.temporal_attention = nn.MultiheadAttention(hidden, heads, dropout=cfg.train.dropout, batch_first=True)
        self.post_attention_gate = GateAddNorm(hidden, cfg.train.dropout)
        self.temporal_grn = GatedResidualNetwork(hidden, hidden, dropout=cfg.train.dropout)

        self.event_proj = nn.Linear(self.event_dim, hidden)
        self.event_score = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.cross_attention = nn.MultiheadAttention(hidden, heads, dropout=cfg.train.dropout, batch_first=True)
        self.se_numeric = nn.Sequential(nn.Linear(hidden, max(1, hidden // 4)), nn.ReLU(), nn.Linear(max(1, hidden // 4), hidden), nn.Sigmoid())
        self.se_text = nn.Sequential(nn.Linear(hidden, max(1, hidden // 4)), nn.ReLU(), nn.Linear(max(1, hidden // 4), hidden), nn.Sigmoid())
        self.se_cross = nn.Sequential(nn.Linear(hidden, max(1, hidden // 4)), nn.ReLU(), nn.Linear(max(1, hidden // 4), hidden), nn.Sigmoid())
        self.modality_gate = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Linear(hidden, 3))

        self.decoder = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(cfg.train.dropout), nn.Linear(hidden, hidden), nn.ReLU())
        self.point_head = nn.Linear(hidden, 1)
        self.quantile_head = nn.Linear(hidden, len(quantiles))
        self.direction_head = nn.Linear(hidden, 1)

    def forward(
        self,
        numeric: torch.Tensor,
        text: torch.Tensor,
        events: torch.Tensor,
        static: torch.Tensor | None = None,
        future_known: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if self.n_numeric == 0:
            numeric = numeric.new_zeros(numeric.shape[0], numeric.shape[1], 1)
        if self.n_text == 0:
            text = text.new_zeros(text.shape[0], text.shape[1], 1)
        if static is None or static.shape[-1] == 0:
            static = numeric.new_zeros(numeric.shape[0], max(1, self.n_static))
        if future_known is None or future_known.shape[-1] == 0:
            future_known = numeric.new_zeros(numeric.shape[0], self.horizon, max(1, self.n_known_future))

        num_h, num_weights = self.numeric_vsn(numeric)
        text_h, text_weights = self.text_vsn(text)
        fused_inputs = torch.cat([num_h, text_h], dim=-1)
        if self.use_tcn:
            fused_inputs = self.tcn(fused_inputs)
        past_inputs = self.input_fusion(fused_inputs)

        batch_size = numeric.shape[0]
        static_feature_weights = static.new_zeros(batch_size, max(1, self.n_static))
        future_feature_weights = future_known.new_zeros(batch_size, self.horizon, max(1, self.n_known_future))
        if self.use_static:
            static_feature_weights = torch.softmax(self.static_feature_selector(static), dim=-1)
            selected_static = static * static_feature_weights
            static_ctx = self.static_encoder(selected_static)
            selection_ctx = self.static_selection_context(selected_static)
            enrichment_ctx = self.static_enrichment_context(selected_static)
            state_h = torch.tanh(self.static_state_h(static_ctx)).unsqueeze(0)
            state_c = torch.tanh(self.static_state_c(static_ctx)).unsqueeze(0)
            past_inputs = past_inputs + selection_ctx.unsqueeze(1)
        else:
            static_ctx = past_inputs.new_zeros(batch_size, past_inputs.shape[-1])
            enrichment_ctx = static_ctx
            state_h = past_inputs.new_zeros(1, batch_size, past_inputs.shape[-1])
            state_c = past_inputs.new_zeros(1, batch_size, past_inputs.shape[-1])

        encoded, encoder_state = self.encoder(past_inputs, (state_h, state_c))
        decoder_context = encoded[:, -1:, :].expand(-1, self.horizon, -1)
        future_weights = past_inputs.new_zeros(batch_size, self.horizon)
        if self.use_known_future and future_known.numel() > 0:
            future_feature_weights = torch.softmax(self.future_feature_selector(future_known), dim=-1)
            future_h = self.future_grn(self.future_proj(future_known * future_feature_weights))
            decoder_context = self.decoder_seed(decoder_context + future_h)
            future_weights = future_h.norm(dim=-1)
            future_weights = future_weights / future_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        decoded_future, _ = self.future_decoder(decoder_context, encoder_state)
        temporal_sequence = torch.cat([encoded, decoded_future], dim=1)
        input_sequence = torch.cat([past_inputs, decoder_context], dim=1)
        temporal_sequence = self.post_lstm_gate(temporal_sequence, input_sequence)
        enrich = enrichment_ctx.unsqueeze(1).expand(-1, temporal_sequence.shape[1], -1)
        temporal_sequence = self.static_enrichment(torch.cat([temporal_sequence, enrich], dim=-1))
        total_steps = temporal_sequence.shape[1]
        attention_mask = torch.triu(
            torch.ones(total_steps, total_steps, dtype=torch.bool, device=temporal_sequence.device),
            diagonal=1,
        )
        temporal, temporal_weights = self.temporal_attention(
            temporal_sequence,
            temporal_sequence,
            temporal_sequence,
            attn_mask=attention_mask,
            need_weights=True,
        )
        temporal = self.post_attention_gate(temporal, temporal_sequence)
        horizon_states = self.temporal_grn(temporal[:, -self.horizon :, :])
        temporal_ctx = horizon_states.mean(dim=1)

        event_ctx, cross_ctx, event_weights, cross_weights = self._text_context(temporal_ctx, events)
        if not self.use_text_modality:
            event_ctx = event_ctx.new_zeros(event_ctx.shape)
            cross_ctx = cross_ctx.new_zeros(cross_ctx.shape)
            event_weights = event_weights.new_zeros(event_weights.shape)
            cross_weights = cross_weights.new_zeros(cross_weights.shape)
        if not self.use_cross_attention:
            cross_ctx = cross_ctx.new_zeros(cross_ctx.shape)
            cross_weights = cross_weights.new_zeros(cross_weights.shape)
        horizon_states = horizon_states * self.se_numeric(horizon_states)
        event_ctx = event_ctx * self.se_text(event_ctx)
        cross_ctx = cross_ctx * self.se_cross(cross_ctx)
        gate = torch.softmax(self.modality_gate(torch.cat([temporal_ctx, event_ctx, cross_ctx], dim=-1)), dim=-1)
        if not self.use_text_modality:
            gate = torch.zeros_like(gate)
            gate[:, 0] = 1.0
        elif not self.use_modality_gate:
            gate = torch.zeros_like(gate)
            gate[:, :] = 1.0 / 3.0
        elif self.modality_prior_strength > 0.0:
            prior = self.modality_prior.to(dtype=gate.dtype, device=gate.device).unsqueeze(0)
            gate = (1.0 - self.modality_prior_strength) * gate + self.modality_prior_strength * prior
        fused = (
            gate[:, 0:1].unsqueeze(1) * horizon_states
            + gate[:, 1:2].unsqueeze(1) * event_ctx.unsqueeze(1)
            + gate[:, 2:3].unsqueeze(1) * cross_ctx.unsqueeze(1)
        )
        decoded = self.decoder(fused)
        quantiles = self.quantile_head(decoded)
        quantiles = torch.sort(quantiles, dim=-1).values
        pred = {
            "point": self.point_head(decoded).squeeze(-1),
            "quantiles": quantiles,
            "direction": torch.sigmoid(self.direction_head(decoded)).squeeze(-1),
        }
        aux = {
            "numeric_weights": num_weights,
            "text_scalar_weights": text_weights,
            "event_weights": event_weights,
            "temporal_weights": temporal_weights,
            "cross_weights": cross_weights,
            "modality_gate": gate,
            "static_context_norm": static_ctx.norm(dim=-1),
            "future_weights": future_weights,
            "static_feature_weights": static_feature_weights,
            "future_feature_weights": future_feature_weights,
        }
        return pred, aux

    def _text_context(self, query: torch.Tensor, events: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if events.shape[-1] == 0:
            events = events.new_zeros(events.shape[0], 1, self.event_dim)
        event_mask = events.abs().sum(dim=-1) > 0
        no_event = ~event_mask.any(dim=1)
        safe_mask = event_mask.clone()
        safe_mask[no_event, 0] = True

        event_h = self.event_proj(events)
        query_expanded = query.unsqueeze(1).expand_as(event_h)
        logits = self.event_score(torch.cat([event_h, query_expanded], dim=-1)).squeeze(-1)
        if self.event_quality_gate_strength > 0.0 and events.shape[-1] > 11:
            source_quality = events[..., 10].clamp(0.0, 1.0)
            event_strength = events[..., 11].clamp(0.0, 1.0)
            if events.shape[-1] > 13:
                fulltext = events[..., 13].clamp(0.0, 1.0)
            else:
                fulltext = torch.zeros_like(source_quality)
            quality = (0.35 * source_quality + 0.50 * event_strength + 0.15 * fulltext).clamp_min(1e-4)
            logits = logits + self.event_quality_gate_strength * torch.log(quality)
        logits = logits.masked_fill(~safe_mask, -1e4)
        k = min(self.text_top_k, logits.shape[1])
        if not self.use_soft_topk:
            event_weights = safe_mask.float()
        elif k < logits.shape[1]:
            top_values, top_indices = torch.topk(logits, k=k, dim=1)
            top_weights = torch.softmax(top_values, dim=1).to(dtype=logits.dtype)
            hard_weights = torch.zeros_like(logits).scatter(1, top_indices, top_weights)
            soft_weights = torch.softmax(logits / 0.7, dim=1).to(dtype=logits.dtype)
            event_weights = soft_weights + (hard_weights - soft_weights).detach()
        else:
            event_weights = torch.softmax(logits, dim=1)
        event_weights = event_weights * safe_mask.float()
        event_weights = event_weights / event_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        event_ctx = torch.bmm(event_weights.unsqueeze(1), event_h).squeeze(1)

        cross, cross_weights = self.cross_attention(
            query.unsqueeze(1),
            event_h,
            event_h,
            key_padding_mask=~safe_mask,
            need_weights=True,
        )
        return event_ctx, cross.squeeze(1), event_weights, cross_weights.squeeze(1)


def fit_single_window(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    dataset: PreparedDataset,
    cfg: AppConfig,
) -> FittedWindowModel:
    device = _resolve_torch_device(cfg.train.device)
    _configure_torch_runtime(cfg, device)
    net = TFTLite(
        len(dataset.numeric_features),
        len(dataset.text_features),
        len(dataset.static_features),
        len(dataset.known_future_features),
        len(dataset.text_event_feature_names),
        dataset.quantiles,
        cfg,
    ).to(device)
    train_ds = WindowDataset(train_frame, dataset, cfg)
    val_ds = WindowDataset(val_frame, dataset, cfg)
    loader_kwargs = _loader_kwargs(cfg, device, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, **_loader_kwargs(cfg, device, shuffle=False))
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    use_amp = _use_amp(cfg, device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(enabled=use_amp)

    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    best_loss = float("inf")
    stale = 0
    for _ in range(max(1, int(cfg.train.epochs))):
        net.train()
        for batch in train_loader:
            batch = _to_device(batch, device)
            opt.zero_grad()
            with _autocast_context(cfg, device):
                pred, _ = net(batch["numeric"], batch["text"], batch["events"], batch["static"], batch["future_known"])
                loss = _multi_task_loss(pred, batch["target_vector"], batch["direction_vector"], dataset.quantiles, cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

        val_loss = _evaluate_loss(net, val_loader, dataset.quantiles, cfg, device)
        if val_loss < best_loss:
            best_loss = val_loss
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= max(1, int(cfg.train.early_stop_rounds)):
                break
    net.load_state_dict(best_state)
    residual_scale = _residual_scale(net, train_loader, cfg, device)
    return FittedWindowModel(
        net=net,
        feature_names=dataset.baseline_features,
        numeric_features=dataset.numeric_features,
        text_features=dataset.text_features,
        static_features=dataset.static_features,
        quantiles=dataset.quantiles,
        lookback=cfg.features.lookback,
        horizon=cfg.features.horizon,
        event_dim=len(dataset.text_event_feature_names),
        text_top_k=dataset.text_selection_top_k,
        device=device,
        residual_scale=residual_scale,
    )


def _resolve_torch_device(requested: str) -> torch.device:
    value = str(requested or "auto").strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda"):
        return torch.device(value if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _configure_torch_runtime(cfg: AppConfig, device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    if bool(getattr(cfg.train, "tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def _use_amp(cfg: AppConfig, device: torch.device) -> bool:
    return bool(getattr(cfg.train, "amp", True)) and device.type == "cuda"


def _autocast_context(cfg: AppConfig, device: torch.device):
    if _use_amp(cfg, device):
        return torch.amp.autocast("cuda") if hasattr(torch, "amp") else torch.cuda.amp.autocast()
    return nullcontext()


def _loader_kwargs(cfg: AppConfig, device: torch.device, shuffle: bool) -> dict[str, Any]:
    num_workers = max(0, int(getattr(cfg.train, "num_workers", 0) or 0))
    kwargs: dict[str, Any] = {
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": bool(getattr(cfg.train, "pin_memory", True)) and device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def predict_frame(
    model: FittedWindowModel,
    scaled_frame: pd.DataFrame,
    raw_frame: pd.DataFrame,
    dataset: PreparedDataset,
    cfg: AppConfig,
) -> pd.DataFrame:
    ds = WindowDataset(scaled_frame, dataset, cfg, predict_all=True)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, **_loader_kwargs(cfg, model.device, shuffle=False))
    rows: list[dict[str, Any]] = []
    model.net.eval()
    for batch in loader:
        batch_device = _to_device(batch, model.device)
        with torch.no_grad():
            with _autocast_context(cfg, model.device):
                pred, aux = model.net(
                    batch_device["numeric"],
                    batch_device["text"],
                    batch_device["events"],
                    batch_device["static"],
                    batch_device["future_known"],
                )
        attributions = _input_gradient_attributions(model.net, batch_device) if bool(getattr(cfg.train, "compute_attributions", False)) else None
        for i in range(pred["point"].shape[0]):
            idx = int(batch["row_index"][i].item())
            rows.append(_prediction_row(raw_frame.iloc[idx], pred, aux, i, dataset, model.residual_scale, attributions))
    return pd.DataFrame(rows)


def build_baseline_predictions(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    dataset: PreparedDataset,
    lookback: int,
    raw_train_frame: pd.DataFrame | None = None,
    raw_test_frame: pd.DataFrame | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, pd.DataFrame]:
    raw_train = raw_train_frame if raw_train_frame is not None else train_frame
    raw_test = raw_test_frame if raw_test_frame is not None else test_frame
    features = [feature for feature in dataset.baseline_features if feature in train_frame.columns and feature in test_frame.columns]
    x_train = _feature_matrix(train_frame, features)
    y_train = pd.to_numeric(train_frame["target_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    x_test = _feature_matrix(test_frame, features)
    out = {"naive_last_return": _naive_baseline(raw_train, raw_test)}
    seed = int(getattr(cfg.train, "seed", 0)) if cfg is not None else 0
    out["ridge_logistic"] = _sklearn_baseline(Ridge(alpha=1.0), LogisticRegression(max_iter=500), x_train, y_train, x_test, raw_test)
    out["arima"] = _ar_baseline(y_train, raw_test)
    out["garch"] = _garch_like_baseline(y_train, raw_test)

    selected = _select_columns_by_variance(x_train, max_features=80)
    svm_reg = _select_sklearn_regressor(
        [SVR(C=c, epsilon=e) for c in (0.5, 1.0, 5.0) for e in (0.005, 0.01, 0.02)],
        x_train[:, selected],
        y_train,
    )
    svm_clf = _select_sklearn_classifier(
        [SVC(C=c, probability=True) for c in (0.5, 1.0, 5.0)],
        x_train[:, selected],
        y_train > 0,
    )
    out["svm"] = _sklearn_baseline(svm_reg, svm_clf, x_train[:, selected], y_train, x_test[:, selected], raw_test)
    try:
        from xgboost import XGBClassifier, XGBRegressor

        reg_candidates = [
            XGBRegressor(n_estimators=n, max_depth=d, learning_rate=lr, objective="reg:squarederror", verbosity=0, random_state=seed, n_jobs=-1)
            for n in (60, 120)
            for d in (2, 3)
            for lr in (0.03, 0.08)
        ]
        clf_candidates = [
            XGBClassifier(n_estimators=n, max_depth=d, learning_rate=lr, eval_metric="logloss", verbosity=0, random_state=seed, n_jobs=-1)
            for n in (60, 120)
            for d in (2, 3)
            for lr in (0.03, 0.08)
        ]
    except Exception:
        reg_candidates = [GradientBoostingRegressor(random_state=0, n_estimators=n, max_depth=d) for n in (60, 120) for d in (2, 3)]
        clf_candidates = [GradientBoostingClassifier(random_state=0, n_estimators=n, max_depth=d) for n in (60, 120) for d in (2, 3)]
    reg = _select_sklearn_regressor(reg_candidates, x_train[:, selected], y_train)
    clf = _select_sklearn_classifier(clf_candidates, x_train[:, selected], y_train > 0)
    out["xgboost"] = _sklearn_baseline(reg, clf, x_train[:, selected], y_train, x_test[:, selected], raw_test)
    return out


def evaluate_predictions(predictions: pd.DataFrame, cfg: AppConfig) -> tuple[dict[str, float], dict[str, float]]:
    if predictions.empty:
        return {}, {"numeric": 0.0, "text": 0.0, "cross": 0.0}
    eval_frame, duplicate_count = _prepare_evaluation_frame(predictions, cfg)
    label_frame = _non_overlapping_frame(eval_frame, max(1, int(cfg.features.horizon))) if getattr(cfg.backtest, "non_overlapping_label_eval", True) else eval_frame
    if label_frame.empty:
        label_frame = eval_frame
    actual = pd.to_numeric(label_frame["actual_return"], errors="coerce").fillna(0.0)
    pred = pd.to_numeric(label_frame["pred_return"], errors="coerce").fillna(0.0)
    prob = pd.to_numeric(label_frame.get("pred_direction_prob", pred > 0), errors="coerce").fillna(0.5)
    direction = pd.to_numeric(label_frame["actual_direction"], errors="coerce").fillna(0).astype(int)
    if "pred_direction_label" in label_frame.columns:
        pred_direction = pd.to_numeric(label_frame["pred_direction_label"], errors="coerce").fillna(0).astype(int)
    else:
        pred_direction = (prob >= 0.5).astype(int)
    error = pred - actual
    q_low = pd.to_numeric(label_frame.get("pred_q10", pred), errors="coerce").fillna(pred)
    q_high = pd.to_numeric(label_frame.get("pred_q90", pred), errors="coerce").fillna(pred)
    sigma = ((q_high - q_low).abs() / (2 * 1.2816)).clip(lower=1e-4)

    strategy_returns, position, turnover = _daily_strategy_returns(eval_frame, cfg)
    equity_curve = (1 + strategy_returns.fillna(0)).cumprod()
    ic = float(actual.corr(pred)) if len(actual) > 1 and actual.std(ddof=0) > 1e-12 and pred.std(ddof=0) > 1e-12 else 0.0
    rank_actual = actual.rank()
    rank_pred = pred.rank()
    rank_ic = float(rank_actual.corr(rank_pred)) if len(actual) > 1 and rank_actual.std(ddof=0) > 1e-12 and rank_pred.std(ddof=0) > 1e-12 else 0.0
    metrics = {
        "raw_prediction_rows": int(len(predictions)),
        "evaluation_rows": int(len(label_frame)),
        "strategy_evaluation_rows": int(len(eval_frame)),
        "duplicate_predictions_dropped": int(duplicate_count),
        "label_evaluation_stride": int(max(1, int(cfg.features.horizon))) if getattr(cfg.backtest, "non_overlapping_label_eval", True) else 1,
        "strategy_return_step": int(getattr(cfg.backtest, "strategy_return_step", 1)),
        "execution_delay_days": int(getattr(cfg.backtest, "execution_delay_days", 0)),
        "max_position": float(getattr(cfg.backtest, "max_position", 1.0)),
        "max_daily_turnover": float(getattr(cfg.backtest, "max_daily_turnover", 1.0)),
        "transaction_cost_bps": float(getattr(cfg.backtest, "transaction_cost_bps", 0.0)),
        "slippage_bps": float(getattr(cfg.backtest, "slippage_bps", 0.0)),
        "block_limit_trades": bool(getattr(cfg.backtest, "block_limit_trades", True)),
        "require_positive_probability": bool(getattr(cfg.backtest, "require_positive_probability", False)),
        "require_positive_return": bool(getattr(cfg.backtest, "require_positive_return", False)),
        "strategy_use_external_baselines": bool(getattr(cfg.backtest, "strategy_use_external_baselines", True)),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mape": float((error.abs() / actual.abs().clip(lower=1e-6)).mean()),
        "r2": float(1.0 - safe_divide(float(np.square(error).sum()), float(np.square(actual - actual.mean()).sum()))),
        "nrmse": float(safe_divide(float(np.sqrt(np.mean(np.square(error)))), float(actual.std(ddof=0)))),
        "accuracy": float(accuracy_score(direction, pred_direction)),
        "precision": float(precision_score(direction, pred_direction, zero_division=0)),
        "recall": float(recall_score(direction, pred_direction, zero_division=0)),
        "f1": float(f1_score(direction, pred_direction, zero_division=0)),
        "mcc": float(matthews_corrcoef(direction, pred_direction)) if direction.nunique() > 1 else 0.0,
        "directional_accuracy": float((pred_direction == direction).mean()),
        "return_sign_directional_accuracy": float(((pred >= 0).astype(int) == direction).mean()),
        "mean_direction_probability": float(prob.mean()),
        "direction_probability_std": float(prob.std(ddof=0)),
        "direction_positive_label_ratio": float((pred_direction == 1).mean()),
        "direction_threshold_mean": float(pd.to_numeric(label_frame.get("direction_threshold_calibrated", pd.Series(0.5, index=label_frame.index)), errors="coerce").fillna(0.5).mean()),
        "quantile_coverage": float(((actual >= q_low) & (actual <= q_high)).mean()),
        "interval_width": float((q_high - q_low).abs().mean()),
        "nll": float((0.5 * np.log(2 * np.pi * np.square(sigma)) + np.square(error) / (2 * np.square(sigma))).mean()),
        "cumulative_return": float((1 + strategy_returns.fillna(0)).prod() - 1),
        "equity_curve_final": float(equity_curve.iloc[-1]) if not equity_curve.empty else 1.0,
        "equity_curve_min": float(equity_curve.min()) if not equity_curve.empty else 1.0,
        "equity_curve_max": float(equity_curve.max()) if not equity_curve.empty else 1.0,
        "annualized_return": annualized_return(strategy_returns),
        "sharpe_ratio": sharpe_ratio(strategy_returns),
        "max_drawdown": max_drawdown(strategy_returns),
        "win_rate": float((strategy_returns > 0).mean()),
        "exposure_ratio": float((position > 0).mean()),
        "avg_turnover": float(turnover.mean()),
        "ic": ic,
        "rank_ic": rank_ic,
    }
    metrics["icir"] = float(safe_divide(metrics["ic"], float(error.std(ddof=0))))
    for step in range(1, max(1, int(cfg.features.horizon)) + 1):
        actual_col = f"actual_return_h{step}"
        pred_col = f"pred_return_h{step}"
        if actual_col in eval_frame.columns and pred_col in eval_frame.columns:
            step_frame = _non_overlapping_frame(eval_frame, step) if getattr(cfg.backtest, "non_overlapping_label_eval", True) else eval_frame
            step_actual = pd.to_numeric(step_frame[actual_col], errors="coerce").fillna(0.0)
            step_pred = pd.to_numeric(step_frame[pred_col], errors="coerce").fillna(0.0)
            step_error = step_pred - step_actual
            metrics[f"h{step}_mae"] = float(step_error.abs().mean())
            metrics[f"h{step}_rmse"] = float(np.sqrt(np.mean(np.square(step_error))))
            metrics[f"h{step}_directional_accuracy"] = float(((step_pred >= 0).astype(int) == (step_actual > 0).astype(int)).mean())
    weights = {
        "numeric": float(_numeric_eval_series(eval_frame, "modality_numeric", 0.0).mean()),
        "text": float(_numeric_eval_series(eval_frame, "modality_text", 0.0).mean()),
        "cross": float(_numeric_eval_series(eval_frame, "modality_cross", 0.0).mean()),
    }
    return metrics, weights


def optimize_direction_parameters(validation_predictions: pd.DataFrame) -> dict[str, float | str]:
    if validation_predictions.empty:
        return {"direction_signal": "prob", "direction_threshold_calibrated": 0.5, "direction_polarity": 1.0, "validation_directional_accuracy": 0.0}
    frame, _ = _prepare_evaluation_frame(validation_predictions, None)
    if frame.empty or "actual_direction" not in frame.columns:
        return {"direction_signal": "prob", "direction_threshold_calibrated": 0.5, "direction_polarity": 1.0, "validation_directional_accuracy": 0.0}
    actual = pd.to_numeric(frame["actual_direction"], errors="coerce").fillna(0).astype(int)
    signals = {
        "prob": pd.to_numeric(frame.get("pred_direction_prob", 0.5), errors="coerce").fillna(0.5),
        "return": pd.to_numeric(frame.get("pred_return", 0.0), errors="coerce").fillna(0.0),
    }
    if "direction_external_signal" in frame.columns:
        signals["external"] = pd.to_numeric(frame["direction_external_signal"], errors="coerce").fillna(0.0)
    for column in [
        "signal_return_1d",
        "signal_return_5d",
        "signal_log_return",
        "signal_price_to_ma_20",
        "signal_rsi_14",
        "signal_bollinger_z",
    ]:
        if column in frame.columns:
            signals[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    meta_params = _fit_direction_meta_logit(frame, actual)
    if meta_params:
        signals["meta_logit"] = pd.Series(meta_params["validation_signal"], index=frame.index, dtype=float)
    signals["hybrid"] = signals["prob"].rank(pct=True) + signals["return"].rank(pct=True)
    signals["invprob_return"] = (-signals["prob"]).rank(pct=True) + signals["return"].rank(pct=True)
    best = {"direction_signal": "prob", "direction_threshold_calibrated": 0.5, "direction_polarity": 1.0, "validation_directional_accuracy": 0.0}
    for name, signal in signals.items():
        clean_signal = pd.to_numeric(signal, errors="coerce").fillna(0.0)
        for polarity in (1.0, -1.0):
            calibrated_signal = clean_signal * polarity
            if calibrated_signal.nunique(dropna=True) <= 1:
                thresholds = [float(calibrated_signal.iloc[0]) if len(calibrated_signal) else 0.0]
            else:
                thresholds = sorted(
                    {
                        float(calibrated_signal.quantile(q))
                        for q in np.linspace(0.02, 0.98, 33)
                    }
                )
                if name == "prob":
                    thresholds.append(0.5 * polarity)
                elif name == "return":
                    thresholds.append(0.0)
            for threshold in thresholds:
                pred_direction = (calibrated_signal >= threshold).astype(int)
                accuracy = float((pred_direction == actual).mean())
                if accuracy > float(best["validation_directional_accuracy"]):
                    best = {
                        "direction_signal": name,
                        "direction_threshold_calibrated": float(threshold),
                        "direction_polarity": float(polarity),
                        "validation_directional_accuracy": accuracy,
                    }
                    if name == "meta_logit":
                        best.update({key: value for key, value in meta_params.items() if key != "validation_signal"})
    return best


def _fit_direction_meta_logit(frame: pd.DataFrame, actual: pd.Series) -> dict[str, Any]:
    if len(frame) < 30 or actual.nunique(dropna=True) < 2:
        return {}
    feature_frame = _direction_meta_feature_frame(frame)
    if feature_frame.empty:
        return {}
    x = feature_frame.to_numpy(dtype=float)
    y = actual.to_numpy(dtype=int)
    center = np.nanmean(x, axis=0)
    scale = np.nanstd(x, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    x_scaled = (x - center) / scale
    best: dict[str, Any] = {}
    best_accuracy = -1.0
    for c_value in (0.05, 0.1, 0.25, 0.5, 1.0):
        try:
            clf = LogisticRegression(
                C=float(c_value),
                class_weight="balanced",
                max_iter=1000,
                solver="lbfgs",
            )
            clf.fit(x_scaled, y)
            probability = clf.predict_proba(x_scaled)[:, 1]
            direction = (probability >= 0.5).astype(int)
            accuracy = float((direction == y).mean())
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best = {
                    "direction_meta_features": "|".join(feature_frame.columns),
                    "direction_meta_coef": "|".join(f"{value:.12g}" for value in clf.coef_[0]),
                    "direction_meta_intercept": float(clf.intercept_[0]),
                    "direction_meta_center": "|".join(f"{value:.12g}" for value in center),
                    "direction_meta_scale": "|".join(f"{value:.12g}" for value in scale),
                    "direction_meta_c": float(c_value),
                    "validation_signal": probability,
                }
        except Exception:
            continue
    return best


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
    return pd.DataFrame(features).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def optimize_point_calibration(validation_predictions: pd.DataFrame) -> dict[str, float | str]:
    if validation_predictions.empty:
        return {"point_scale": 1.0, "point_intercept": 0.0, "point_calibration": "identity", "validation_mae": 0.0}
    frame, _ = _prepare_evaluation_frame(validation_predictions, None)
    if frame.empty or "actual_return" not in frame.columns or "pred_return" not in frame.columns:
        return {"point_scale": 1.0, "point_intercept": 0.0, "point_calibration": "identity", "validation_mae": 0.0}
    actual = pd.to_numeric(frame["actual_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    pred = pd.to_numeric(frame["pred_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(actual) < 8 or float(np.nanstd(pred)) < 1e-10:
        intercept = float(np.nanmedian(actual - pred)) if len(actual) else 0.0
        mae = float(np.mean(np.abs((pred + intercept) - actual))) if len(actual) else 0.0
        return {"point_scale": 1.0, "point_intercept": intercept, "point_calibration": "bias", "validation_mae": mae}

    candidates: list[tuple[str, float, float]] = [("identity", 1.0, 0.0)]
    for scale in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25):
        intercept = float(np.nanmedian(actual - scale * pred))
        candidates.append((f"shrink_{scale:g}", float(scale), intercept))
    try:
        reg = LinearRegression().fit(pred.reshape(-1, 1), actual)
        slope = float(np.clip(reg.coef_[0], -2.0, 2.0))
        intercept = float(reg.intercept_)
        candidates.append(("linear", slope, intercept))
    except Exception:
        pass

    best_name, best_scale, best_intercept, best_mae = "identity", 1.0, 0.0, float("inf")
    for name, scale, intercept in candidates:
        calibrated = scale * pred + intercept
        mae = float(np.mean(np.abs(calibrated - actual)))
        if mae < best_mae:
            best_name, best_scale, best_intercept, best_mae = name, float(scale), float(intercept), mae
    return {
        "point_scale": best_scale,
        "point_intercept": best_intercept,
        "point_calibration": best_name,
        "validation_mae": best_mae,
    }


def _numeric_eval_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def _prepare_evaluation_frame(predictions: pd.DataFrame, cfg: AppConfig | None) -> tuple[pd.DataFrame, int]:
    frame = predictions.copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"]).sort_values(["date", "window_start"] if "window_start" in frame.columns else ["date"])
    before = len(frame)
    if (cfg is None or getattr(cfg.backtest, "deduplicate_predictions", True)) and "date" in frame.columns:
        frame = frame.drop_duplicates(subset=["date"], keep="first")
    return frame.reset_index(drop=True), before - len(frame)


def _non_overlapping_frame(frame: pd.DataFrame, stride: int) -> pd.DataFrame:
    stride = max(1, int(stride))
    if frame.empty or stride == 1:
        return frame
    if "date" not in frame.columns:
        return frame.iloc[::stride].reset_index(drop=True)
    ordered = frame.sort_values("date").reset_index(drop=True)
    return ordered.iloc[::stride].reset_index(drop=True)


def _daily_strategy_returns(predictions: pd.DataFrame, cfg: AppConfig) -> tuple[pd.Series, pd.Series, pd.Series]:
    if predictions.empty:
        empty = pd.Series(dtype=float)
        return empty, empty, empty
    step = max(1, int(getattr(cfg.backtest, "strategy_return_step", 1)))
    actual_col = f"actual_return_h{step}"
    if actual_col not in predictions.columns:
        actual_col = "actual_return"
    actual = pd.to_numeric(predictions[actual_col], errors="coerce").fillna(0.0).reset_index(drop=True)
    if step != 1:
        actual = actual / float(step)
    price_limit = float(getattr(cfg.backtest, "price_limit_pct", 0.20) or 0.20)
    actual = actual.clip(lower=-abs(price_limit), upper=abs(price_limit))
    configured_signal = str(getattr(cfg.backtest, "strategy_signal", "hybrid") or "hybrid").lower()
    if "strategy_signal" in predictions.columns and predictions["strategy_signal"].notna().any():
        configured_signal = str(predictions["strategy_signal"].dropna().astype(str).iloc[0]).lower()
    if configured_signal == "external_position" and "strategy_external_position" in predictions.columns:
        max_position = max(0.0, min(1.0, float(getattr(cfg.backtest, "max_position", 1.0) or 1.0)))
        position = (
            pd.to_numeric(predictions["strategy_external_position"], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0, upper=max_position)
            .reset_index(drop=True)
        )
        position = _apply_strategy_risk_overlay(position, predictions, cfg)
        turnover = position.diff().fillna(position).abs()
        max_turnover = float(getattr(cfg.backtest, "max_daily_turnover", max_position) or max_position)
        turnover = turnover.clip(upper=max(0.0, min(max_position, max_turnover)))
        cost_rate = (float(cfg.backtest.transaction_cost_bps) + float(getattr(cfg.backtest, "slippage_bps", 0.0))) / 10000.0
        strategy_returns = (position * actual - turnover * cost_rate).clip(lower=-1.0 + 1e-6)
        return strategy_returns.reset_index(drop=True), position.reset_index(drop=True), turnover.reset_index(drop=True)
    prob = _strategy_probability(predictions).reset_index(drop=True)
    pred = pd.to_numeric(predictions.get("pred_return", 0.0), errors="coerce").fillna(0.0).reset_index(drop=True)
    signal = _strategy_signal(predictions, cfg).reset_index(drop=True)
    max_position = max(0.0, min(1.0, float(getattr(cfg.backtest, "max_position", 1.0) or 1.0)))
    default_threshold = _default_strategy_threshold(signal, prob, cfg)
    threshold = _prediction_strategy_value(predictions, "strategy_threshold", default_threshold).reset_index(drop=True)
    scale_default = max(float(getattr(cfg.backtest, "strategy_scale_floor", 0.005) or 0.005), float(signal.std(ddof=0) or 0.0))
    scale = _prediction_strategy_value(predictions, "strategy_signal_scale", scale_default).reset_index(drop=True).abs().clip(lower=1e-8)
    raw_position = ((signal - threshold) / scale).clip(lower=0.0, upper=1.0)
    desired_position = raw_position * max_position
    if bool(getattr(cfg.backtest, "require_positive_probability", False)):
        desired_position = desired_position * (prob >= 0.5).astype(float)
    if bool(getattr(cfg.backtest, "require_positive_return", False)):
        desired_position = desired_position * (pred >= 0).astype(float)
    position, turnover = _apply_execution_constraints(desired_position, actual, cfg)
    position = _apply_strategy_risk_overlay(position, predictions, cfg)
    turnover = position.diff().fillna(position).abs()
    max_turnover = float(getattr(cfg.backtest, "max_daily_turnover", max_position) or max_position)
    turnover = turnover.clip(upper=max(0.0, min(max_position, max_turnover)))
    cost_rate = (float(cfg.backtest.transaction_cost_bps) + float(getattr(cfg.backtest, "slippage_bps", 0.0))) / 10000.0
    strategy_returns = position * actual - turnover * cost_rate
    strategy_returns = strategy_returns.clip(lower=-1.0 + 1e-6)
    return strategy_returns.reset_index(drop=True), position.reset_index(drop=True), turnover.reset_index(drop=True)


def _apply_strategy_risk_overlay(position: pd.Series, predictions: pd.DataFrame, cfg: AppConfig) -> pd.Series:
    if predictions.empty or "strategy_risk_overlay_enabled" not in predictions.columns:
        return position.reset_index(drop=True)
    enabled = pd.Series(predictions["strategy_risk_overlay_enabled"]).fillna(False).astype(bool)
    if not bool(enabled.any()):
        return position.reset_index(drop=True)
    pos = pd.Series(position, dtype=float).reset_index(drop=True)
    vol = pd.to_numeric(predictions.get("signal_volatility_10", 0.0), errors="coerce").fillna(0.0).abs().reset_index(drop=True)
    ret1 = pd.to_numeric(predictions.get("signal_return_1d", 0.0), errors="coerce").fillna(0.0).reset_index(drop=True)
    vol_threshold = _prediction_strategy_value(predictions, "strategy_risk_overlay_vol_threshold", float("inf")).reset_index(drop=True)
    drawdown_threshold = _prediction_strategy_value(predictions, "strategy_risk_overlay_drawdown_threshold", -float("inf")).reset_index(drop=True)
    risk_scale = _prediction_strategy_value(predictions, "strategy_risk_overlay_scale", 1.0).reset_index(drop=True).clip(lower=0.0, upper=1.0)
    n = min(len(pos), len(vol), len(ret1), len(enabled), len(vol_threshold), len(drawdown_threshold), len(risk_scale))
    if n <= 0:
        return pos
    high_risk = (vol.iloc[:n] >= vol_threshold.iloc[:n]) | (ret1.iloc[:n] <= drawdown_threshold.iloc[:n])
    active = enabled.reset_index(drop=True).iloc[:n] & high_risk
    adjusted = pos.copy()
    adjusted.iloc[:n] = adjusted.iloc[:n] * np.where(active.to_numpy(dtype=bool), risk_scale.iloc[:n].to_numpy(dtype=float), 1.0)
    return adjusted.clip(lower=0.0).reset_index(drop=True)


def add_strategy_decision_columns(predictions: pd.DataFrame, cfg: AppConfig) -> None:
    if predictions.empty:
        return
    signal = _strategy_signal(predictions, cfg).reset_index(drop=True)
    prob = _strategy_probability(predictions).reset_index(drop=True)
    pred = pd.to_numeric(predictions.get("pred_return", 0.0), errors="coerce").fillna(0.0).reset_index(drop=True)
    default_threshold = _default_strategy_threshold(signal, prob, cfg)
    threshold = _prediction_strategy_value(predictions, "strategy_threshold", default_threshold).reset_index(drop=True)
    scale_default = max(float(getattr(cfg.backtest, "strategy_scale_floor", 0.005) or 0.005), float(signal.std(ddof=0) or 0.0))
    scale = _prediction_strategy_value(predictions, "strategy_signal_scale", scale_default).reset_index(drop=True).abs().clip(lower=1e-8)
    raw_position = ((signal - threshold) / scale).clip(lower=0.0, upper=1.0)
    desired_position = raw_position * max(0.0, min(1.0, float(getattr(cfg.backtest, "max_position", 1.0) or 1.0)))
    if bool(getattr(cfg.backtest, "require_positive_probability", False)):
        desired_position = desired_position * (prob >= 0.5).astype(float)
    if bool(getattr(cfg.backtest, "require_positive_return", False)):
        desired_position = desired_position * (pred >= 0).astype(float)
    _, position, turnover = _daily_strategy_returns(predictions, cfg)
    delta = position.diff().fillna(position)
    action = np.select(
        [delta > 1e-8, delta < -1e-8, position > 1e-8],
        ["buy_or_add", "sell_or_reduce", "hold_long"],
        default="hold_cash",
    )
    configured_signal = str(predictions.get("strategy_signal", pd.Series([getattr(cfg.backtest, "strategy_signal", "hybrid")])).dropna().astype(str).iloc[0])
    rule = (
        f"Buy/add when strategy_signal_value >= strategy_threshold"
        f" and optional filters pass; sell/reduce when target position falls."
        f" max_position={float(getattr(cfg.backtest, 'max_position', 1.0) or 1.0):.4g},"
        f" max_daily_turnover={float(getattr(cfg.backtest, 'max_daily_turnover', 1.0) or 1.0):.4g},"
        f" execution_delay_days={int(getattr(cfg.backtest, 'execution_delay_days', 0) or 0)},"
        f" cost_bps={float(getattr(cfg.backtest, 'transaction_cost_bps', 0.0) or 0.0):.4g},"
        f" slippage_bps={float(getattr(cfg.backtest, 'slippage_bps', 0.0) or 0.0):.4g},"
        f" signal={configured_signal}."
    )
    predictions["strategy_signal_value"] = signal.to_numpy(dtype=float)
    predictions["strategy_desired_position"] = desired_position.to_numpy(dtype=float)
    predictions["strategy_position"] = position.to_numpy(dtype=float)
    predictions["strategy_turnover"] = turnover.to_numpy(dtype=float)
    predictions["trade_action"] = action
    predictions["trade_rule"] = rule


def _apply_execution_constraints(desired_position: pd.Series, actual_return: pd.Series, cfg: AppConfig) -> tuple[pd.Series, pd.Series]:
    max_position = max(0.0, min(1.0, float(getattr(cfg.backtest, "max_position", 1.0) or 1.0)))
    delay = max(0, int(getattr(cfg.backtest, "execution_delay_days", 0) or 0))
    target = desired_position.astype(float).clip(lower=0.0, upper=max_position).reset_index(drop=True)
    if delay:
        target = target.shift(delay).fillna(0.0)
    actual = actual_return.astype(float).reset_index(drop=True)
    max_turnover = float(getattr(cfg.backtest, "max_daily_turnover", max_position) or max_position)
    max_turnover = max(0.0, min(max_position, max_turnover))
    block_limit_trades = bool(getattr(cfg.backtest, "block_limit_trades", True))
    price_limit = abs(float(getattr(cfg.backtest, "price_limit_pct", 0.20) or 0.20))
    buffer = abs(float(getattr(cfg.backtest, "limit_trade_buffer", 0.0) or 0.0))
    upper_limit = max(0.0, price_limit - buffer)
    lower_limit = -max(0.0, price_limit - buffer)

    positions: list[float] = []
    turnovers: list[float] = []
    previous = 0.0
    for i, desired in enumerate(target):
        next_position = float(desired)
        realized = float(actual.iloc[i]) if i < len(actual) else 0.0
        if block_limit_trades:
            if realized >= upper_limit and next_position > previous:
                next_position = previous
            if realized <= lower_limit and next_position < previous:
                next_position = previous
        delta = max(-max_turnover, min(max_turnover, next_position - previous))
        current = max(0.0, min(max_position, previous + delta))
        positions.append(current)
        turnovers.append(abs(current - previous))
        previous = current
    return pd.Series(positions, dtype=float), pd.Series(turnovers, dtype=float)


def _strategy_signal(predictions: pd.DataFrame, cfg: AppConfig) -> pd.Series:
    configured = str(getattr(cfg.backtest, "strategy_signal", "hybrid") or "hybrid").lower()
    if "strategy_signal" in predictions.columns:
        configured = str(predictions["strategy_signal"].dropna().astype(str).iloc[0]).lower() if predictions["strategy_signal"].notna().any() else configured
    pred = pd.to_numeric(predictions.get("pred_return", 0.0), errors="coerce").fillna(0.0)
    prob = _strategy_probability(predictions)
    external = pd.to_numeric(predictions.get("strategy_external_signal", pred), errors="coerce").fillna(0.0)
    width = (
        pd.to_numeric(predictions.get("pred_q90", pred), errors="coerce").fillna(pred)
        - pd.to_numeric(predictions.get("pred_q10", pred), errors="coerce").fillna(pred)
    ).abs().clip(lower=1e-4)
    if configured == "external":
        return external.astype(float)
    if configured == "inv_external":
        return (-external).astype(float)
    if configured == "external_prob_return":
        return (external.rank(pct=True) + prob.rank(pct=True)).astype(float)
    if configured == "inv_external_prob_return":
        return ((-external).rank(pct=True) + (1.0 - prob).rank(pct=True)).astype(float)
    if configured == "external_hybrid":
        return (external.clip(lower=0.0) * (prob - 0.5).clip(lower=0.0)).astype(float)
    if configured == "inv_external_hybrid":
        return ((-external).clip(lower=0.0) * (0.5 - prob).clip(lower=0.0)).astype(float)
    if configured == "external_position" and "strategy_external_position" in predictions.columns:
        return pd.to_numeric(predictions["strategy_external_position"], errors="coerce").fillna(0.0).astype(float)
    if configured == "prob":
        return prob.astype(float)
    if configured == "return":
        return pred.astype(float)
    if configured == "risk_adjusted":
        return (pred / width).astype(float)
    if configured == "inv_return":
        return (-pred).astype(float)
    if configured == "inv_prob":
        return (-prob).astype(float)
    if configured == "inv_hybrid":
        return ((-pred).clip(lower=0.0) * (0.5 - prob).clip(lower=0.0)).astype(float)
    if configured == "prob_return":
        return (pred.rank(pct=True) + prob.rank(pct=True)).astype(float)
    return (pred.clip(lower=0.0) * (prob - 0.5).clip(lower=0.0)).astype(float)


def _strategy_probability(predictions: pd.DataFrame) -> pd.Series:
    default = pd.to_numeric(predictions.get("pred_direction_prob", 0.5), errors="coerce").fillna(0.5)
    if "strategy_external_probability" not in predictions.columns:
        return default.astype(float)
    values = pd.to_numeric(predictions["strategy_external_probability"], errors="coerce")
    if values.notna().any():
        return values.fillna(default).clip(lower=0.0, upper=1.0).astype(float)
    return default.astype(float)


def _default_strategy_threshold(signal: pd.Series, prob: pd.Series, cfg: AppConfig) -> float:
    mode = str(getattr(cfg.backtest, "strategy_threshold_mode", "validation_quantile") or "").lower()
    if mode == "fixed":
        configured = str(getattr(cfg.backtest, "strategy_signal", "hybrid") or "hybrid").lower()
        return float(cfg.backtest.direction_threshold) if configured == "prob" else 0.0
    target_exposure = float(getattr(cfg.backtest, "strategy_target_exposure", getattr(cfg.backtest, "strategy_min_exposure", 0.25)) or 0.25)
    q = 1.0 - max(0.0, min(1.0, target_exposure))
    return float(signal.quantile(q)) if len(signal) else 0.0


def _prediction_strategy_value(predictions: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column not in predictions.columns:
        return pd.Series(default, index=predictions.index, dtype=float)
    values = pd.to_numeric(predictions[column], errors="coerce")
    if values.notna().any():
        return values.ffill().bfill().fillna(default).astype(float)
    return pd.Series(default, index=predictions.index, dtype=float)


def optimize_strategy_parameters(validation_predictions: pd.DataFrame, cfg: AppConfig) -> dict[str, float | str]:
    if validation_predictions.empty:
        return _fallback_strategy_parameters(cfg)
    if bool(getattr(cfg.backtest, "strategy_disable_optimization", False)):
        return _fallback_strategy_parameters(cfg, validation_predictions)
    best: dict[str, float | str] | None = None
    best_score = -float("inf")
    configured_signal = str(getattr(cfg.backtest, "strategy_signal", "hybrid") or "hybrid")
    signals = list(dict.fromkeys([
        configured_signal,
        "return",
        "risk_adjusted",
        "external",
        "external_prob_return",
        "external_hybrid",
        "inv_external",
        "inv_external_prob_return",
        "inv_external_hybrid",
        "prob_return",
        "hybrid",
        "prob",
        "inv_return",
        "inv_prob",
        "inv_hybrid",
    ]))
    if "strategy_force_external" in validation_predictions.columns:
        force_external = bool(pd.Series(validation_predictions["strategy_force_external"]).fillna(False).astype(bool).any())
        if force_external:
            signals = [
                "external",
                "external_prob_return",
                "external_hybrid",
                "inv_external",
                "inv_external_prob_return",
                "inv_external_hybrid",
            ]
    quantiles = list(getattr(cfg.backtest, "strategy_quantiles", [0.55, 0.65, 0.75]) or [0.55, 0.65, 0.75])
    min_exposure = float(getattr(cfg.backtest, "strategy_min_exposure", 0.25) or 0.25)
    max_exposure = float(getattr(cfg.backtest, "strategy_max_exposure", 0.70) or 0.70)
    target_exposure = max(min_exposure, min(max_exposure, float(getattr(cfg.backtest, "strategy_target_exposure", 0.45) or 0.45)))
    metric = str(getattr(cfg.backtest, "strategy_opt_metric", "sharpe") or "sharpe").lower()
    for signal_name in signals:
        candidate_base = validation_predictions.copy()
        candidate_base["strategy_signal"] = signal_name
        signal = _strategy_signal(candidate_base, cfg)
        if signal.nunique(dropna=True) <= 1:
            thresholds = [float(signal.iloc[0]) if len(signal) else 0.0]
        else:
            thresholds = [float(signal.quantile(max(0.0, min(1.0, q)))) for q in quantiles]
        scale = max(float(signal.std(ddof=0) or 0.0), float(getattr(cfg.backtest, "strategy_scale_floor", 0.005) or 0.005))
        for threshold in thresholds:
            candidate = candidate_base.copy()
            candidate["strategy_threshold"] = threshold
            candidate["strategy_signal_scale"] = scale
            returns, position, _ = _daily_strategy_returns(candidate, cfg)
            exposure = float((position > 0).mean()) if len(position) else 0.0
            if exposure < min_exposure or exposure > max_exposure:
                continue
            exposure_penalty = 2.0 * abs(exposure - target_exposure)
            score, positive_segment_ratio, worst_segment_score = _robust_strategy_score(returns, metric, cfg)
            min_positive_ratio = float(getattr(cfg.backtest, "strategy_min_positive_segment_ratio", 0.0) or 0.0)
            segment_penalty = 4.0 * max(0.0, min_positive_ratio - positive_segment_ratio)
            score -= exposure_penalty + segment_penalty
            if score > best_score:
                best_score = score
                best = {
                    "strategy_signal": signal_name,
                    "strategy_threshold": float(threshold),
                    "strategy_signal_scale": float(scale),
                    "validation_score": float(score),
                    "validation_exposure": float(exposure),
                    "validation_positive_segment_ratio": float(positive_segment_ratio),
                    "validation_worst_segment_score": float(worst_segment_score),
                }
    return best or _fallback_strategy_parameters(cfg, validation_predictions)


def _fallback_strategy_parameters(cfg: AppConfig, validation_predictions: pd.DataFrame | None = None) -> dict[str, float | str]:
    signal_name = str(getattr(cfg.backtest, "strategy_signal", "hybrid") or "hybrid")
    threshold = 0.0
    scale = float(getattr(cfg.backtest, "strategy_scale_floor", 0.005) or 0.005)
    if validation_predictions is not None and not validation_predictions.empty:
        tmp = validation_predictions.copy()
        if "strategy_force_external" in tmp.columns and bool(pd.Series(tmp["strategy_force_external"]).fillna(False).astype(bool).any()):
            signal_name = "external"
        tmp["strategy_signal"] = signal_name
        signal = _strategy_signal(tmp, cfg)
        target_exposure = max(0.0, min(1.0, float(getattr(cfg.backtest, "strategy_target_exposure", 0.45) or 0.45)))
        threshold = float(signal.quantile(1.0 - target_exposure))
        scale = max(scale, float(signal.std(ddof=0) or 0.0))
    return {
        "strategy_signal": signal_name,
        "strategy_threshold": float(threshold),
        "strategy_signal_scale": float(scale),
        "validation_score": 0.0,
        "validation_exposure": 0.0,
    }


def _strategy_score(returns: pd.Series, metric: str) -> float:
    if returns.empty:
        return -float("inf")
    clean = pd.to_numeric(returns, errors="coerce").fillna(0.0).astype(float)
    cumulative = float((1 + returns.fillna(0.0)).prod() - 1.0)
    ann = annualized_return(returns)
    sharpe = sharpe_ratio(returns)
    dd = abs(max_drawdown(returns))
    n = int(len(clean))
    mean_return = float(clean.mean()) if n else 0.0
    return_std = float(clean.std(ddof=1)) if n > 1 else 0.0
    t_stat = safe_divide(mean_return, return_std / math.sqrt(max(n, 1)))
    if metric in {"return", "cumulative_return", "annualized_return"}:
        return cumulative
    if metric in {"tstat", "mean_tstat", "return_tstat"}:
        return t_stat
    if metric in {"significance", "stable_return", "return_significance"}:
        calmar = safe_divide(ann, dd)
        return (
            0.45 * math.tanh(t_stat / 2.0)
            + 0.25 * math.tanh(cumulative * 3.0)
            + 0.20 * math.tanh(sharpe / 2.0)
            + 0.10 * math.tanh(calmar / 3.0)
            - 0.15 * min(dd, 1.0)
        )
    if metric == "calmar":
        return safe_divide(annualized_return(returns), dd)
    if metric == "balanced":
        calmar = safe_divide(ann, dd)
        return (
            0.40 * math.tanh(sharpe / 2.0)
            + 0.30 * math.tanh(cumulative * 2.0)
            + 0.20 * math.tanh(calmar / 3.0)
            - 0.10 * min(dd, 1.0)
        )
    return sharpe_ratio(returns)


def _robust_strategy_score(returns: pd.Series, metric: str, cfg: AppConfig) -> tuple[float, float, float]:
    base = _strategy_score(returns, metric)
    segments = max(1, int(getattr(cfg.backtest, "strategy_robust_segments", 1) or 1))
    if segments <= 1 or len(returns) < segments * 12:
        return base, float(base > 0.0), base
    segment_scores = [
        _strategy_score(pd.Series(chunk, dtype=float), metric)
        for chunk in np.array_split(returns.to_numpy(dtype=float), segments)
        if len(chunk)
    ]
    if not segment_scores:
        return base, float(base > 0.0), base
    positive_ratio = float(np.mean(np.asarray(segment_scores) > 0.0))
    worst = float(np.min(segment_scores))
    median = float(np.median(segment_scores))
    worst_weight = max(0.0, min(1.0, float(getattr(cfg.backtest, "strategy_worst_segment_weight", 0.0) or 0.0)))
    robust = (1.0 - worst_weight) * (0.5 * base + 0.5 * median) + worst_weight * worst
    return float(robust), positive_ratio, worst


def summarize_attention_patterns(predictions: pd.DataFrame) -> dict[str, float]:
    if predictions.empty:
        return {}
    fields = ["temporal_peak_lag", "temporal_attention_entropy", "text_peak_weight", "cross_peak_weight", "cross_attention_entropy"]
    return {field: float(pd.to_numeric(predictions.get(field, 0), errors="coerce").fillna(0).mean()) for field in fields}


def summarize_explanations(predictions: pd.DataFrame) -> dict[str, Any]:
    if predictions.empty:
        return {}
    top_numeric = predictions.get("top_numeric_feature", pd.Series(dtype=str)).fillna("").astype(str)
    top_text = predictions.get("top_text_title", pd.Series(dtype=str)).fillna("").astype(str)
    return {
        "dominant_numeric_features": top_numeric.value_counts().head(10).to_dict(),
        "dominant_text_events": top_text[top_text.str.len() > 0].value_counts().head(10).to_dict(),
        "dominant_static_features": predictions.get("top_static_feature", pd.Series(dtype=str)).fillna("").astype(str).value_counts().head(10).to_dict(),
        "dominant_known_future_features": predictions.get("top_known_future_feature", pd.Series(dtype=str)).fillna("").astype(str).value_counts().head(10).to_dict(),
        "mean_bullish_probability": float(pd.to_numeric(predictions.get("pred_direction_prob", 0), errors="coerce").fillna(0).mean()),
        "sample_explanations": predictions.get("prediction_explanation", pd.Series(dtype=str)).head(5).tolist(),
    }


def summarize_text_event_contributions(predictions: pd.DataFrame, top_k: int = 20) -> list[dict[str, Any]]:
    if predictions.empty or "text_event_contributions" not in predictions.columns:
        return []
    scores: dict[str, dict[str, Any]] = {}
    for value in predictions["text_event_contributions"]:
        events = value
        if isinstance(value, str):
            try:
                events = ast.literal_eval(value)
            except Exception:
                events = []
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            key = str(event.get("uid") or event.get("title") or "").strip()
            if not key:
                continue
            weight = float(event.get("contribution_score", event.get("weight") or 0.0) or 0.0)
            current = scores.setdefault(
                key,
                {
                    "title": str(event.get("title") or ""),
                    "source": str(event.get("source") or ""),
                    "publisher": str(event.get("publisher") or ""),
                    "url": str(event.get("url") or ""),
                    "type": str(event.get("type") or ""),
                    "mean_weight": 0.0,
                    "mean_input_gradient_attribution": 0.0,
                    "count": 0,
                },
            )
            current["mean_weight"] += weight
            current["mean_input_gradient_attribution"] += float(event.get("input_gradient_attribution") or 0.0)
            current["count"] += 1
    for item in scores.values():
        item["mean_weight"] = float(item["mean_weight"] / max(1, int(item["count"])))
        item["mean_input_gradient_attribution"] = float(item["mean_input_gradient_attribution"] / max(1, int(item["count"])))
    return sorted(scores.values(), key=lambda item: item["mean_weight"], reverse=True)[:top_k]


def summarize_feature_contributions(
    frame: pd.DataFrame,
    dataset: PreparedDataset,
    top_k: int = 12,
    predictions: pd.DataFrame | None = None,
) -> dict[str, float]:
    if predictions is not None and not predictions.empty:
        attribution_columns = [column for column in predictions.columns if column.startswith("attribution__")]
        if attribution_columns:
            scores = {
                column.replace("attribution__", "", 1): float(pd.to_numeric(predictions[column], errors="coerce").abs().mean())
                for column in attribution_columns
            }
            return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k])
        weight_columns = [column for column in predictions.columns if column.startswith("feature_weight__")]
        if weight_columns:
            scores = {
                column.replace("feature_weight__", "", 1): float(pd.to_numeric(predictions[column], errors="coerce").abs().mean())
                for column in weight_columns
            }
            return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k])
    if frame.empty:
        return {}
    y = pd.to_numeric(frame["target_return"], errors="coerce").fillna(0.0)
    scores: dict[str, float] = {}
    for feature in dataset.baseline_features:
        if feature not in frame.columns:
            continue
        x = pd.to_numeric(frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        score = abs(float(x.corr(y))) if x.nunique() > 1 and y.nunique() > 1 else 0.0
        scores[feature] = 0.0 if np.isnan(score) else score
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k])


def compare_prediction_significance(
    model_predictions: pd.DataFrame,
    baseline_predictions: dict[str, pd.DataFrame],
    cfg: AppConfig | None = None,
    *,
    n_bootstrap: int = 1000,
) -> dict[str, dict[str, float]]:
    if model_predictions.empty:
        return {}
    if cfg is not None:
        model_source, _ = _prepare_evaluation_frame(model_predictions, cfg)
        if getattr(cfg.backtest, "non_overlapping_label_eval", True):
            model_source = _non_overlapping_frame(model_source, max(1, int(cfg.features.horizon)))
    else:
        model_source = model_predictions.copy()
    model = model_source[["date", "actual_return", "pred_return"]].copy()
    model["date"] = pd.to_datetime(model["date"], errors="coerce")
    model = model.dropna(subset=["date"])
    model = model.rename(columns={"pred_return": "model_pred"})
    reports: dict[str, dict[str, float]] = {}
    for name, baseline in baseline_predictions.items():
        if baseline.empty:
            continue
        if cfg is not None:
            other_source, _ = _prepare_evaluation_frame(baseline, cfg)
            if getattr(cfg.backtest, "non_overlapping_label_eval", True):
                other_source = _non_overlapping_frame(other_source, max(1, int(cfg.features.horizon)))
        else:
            other_source = baseline.copy()
        other = other_source[["date", "pred_return"]].copy()
        other["date"] = pd.to_datetime(other["date"], errors="coerce")
        other = other.dropna(subset=["date"]).rename(columns={"pred_return": "baseline_pred"})
        merged = model.merge(other, on="date", how="inner")
        if len(merged) < 8:
            continue
        actual = pd.to_numeric(merged["actual_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        model_error = pd.to_numeric(merged["model_pred"], errors="coerce").fillna(0.0).to_numpy(dtype=float) - actual
        baseline_error = pd.to_numeric(merged["baseline_pred"], errors="coerce").fillna(0.0).to_numpy(dtype=float) - actual
        diff = np.square(model_error) - np.square(baseline_error)
        dm_stat, dm_pvalue = _diebold_mariano(diff)
        mae_diff = np.abs(model_error) - np.abs(baseline_error)
        ci_low, ci_high = _bootstrap_mean_ci(mae_diff, n_bootstrap=n_bootstrap)
        reports[name] = {
            "n": int(len(merged)),
            "dm_stat": float(dm_stat),
            "dm_pvalue": float(dm_pvalue),
            "mean_squared_error_diff": float(np.mean(diff)),
            "mean_absolute_error_diff": float(np.mean(mae_diff)),
            "mae_diff_bootstrap_ci_low": float(ci_low),
            "mae_diff_bootstrap_ci_high": float(ci_high),
            "model_better_by_mae": bool(np.mean(mae_diff) < 0),
        }
    return reports


def _diebold_mariano(loss_diff: np.ndarray, max_lag: int | None = None) -> tuple[float, float]:
    values = np.asarray(loss_diff, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return 0.0, 1.0
    centered = values - values.mean()
    if max_lag is None:
        max_lag = min(5, max(0, int(round(n ** (1 / 3)))))
    gamma0 = float(np.mean(centered * centered))
    variance = gamma0
    for lag in range(1, max_lag + 1):
        cov = float(np.mean(centered[lag:] * centered[:-lag]))
        variance += 2.0 * (1.0 - lag / (max_lag + 1.0)) * cov
    variance = max(variance, 1e-12)
    stat = float(values.mean() / math.sqrt(variance / n))
    pvalue = float(math.erfc(abs(stat) / math.sqrt(2.0)))
    return stat, pvalue


def _bootstrap_mean_ci(values: np.ndarray, n_bootstrap: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(42)
    means = []
    for _ in range(max(100, int(n_bootstrap))):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def _tensor_sample(frame: pd.DataFrame, idx: int, dataset: PreparedDataset, cfg: AppConfig) -> dict[str, torch.Tensor]:
    lookback = max(1, int(cfg.features.lookback))
    start = max(0, idx - lookback + 1)
    window = frame.iloc[start : idx + 1]
    if len(window) < lookback:
        pad = pd.concat([window.iloc[[0]]] * (lookback - len(window)), ignore_index=True)
        window = pd.concat([pad, window], ignore_index=True)
    numeric = _feature_matrix(window, dataset.numeric_features)
    text = _feature_matrix(window, dataset.text_features)
    events = _event_matrix(frame.iloc[idx], dataset)
    static = _static_vector(frame.iloc[idx], dataset)
    future_known = _future_known_matrix(frame.iloc[idx], dataset, cfg)
    target_vector = _target_vector(frame.iloc[idx], cfg, "target_return_h")
    direction_vector = _target_vector(frame.iloc[idx], cfg, "target_direction_h")
    return {
        "numeric": torch.tensor(numeric, dtype=torch.float32),
        "text": torch.tensor(text, dtype=torch.float32),
        "events": torch.tensor(events, dtype=torch.float32),
        "static": torch.tensor(static, dtype=torch.float32),
        "future_known": torch.tensor(future_known, dtype=torch.float32),
        "target": torch.tensor(float(frame.iloc[idx].get("target_return", 0.0) or 0.0), dtype=torch.float32),
        "direction": torch.tensor(float(frame.iloc[idx].get("target_direction", 0.0) or 0.0), dtype=torch.float32),
        "target_vector": torch.tensor(target_vector, dtype=torch.float32),
        "direction_vector": torch.tensor(direction_vector, dtype=torch.float32),
        "row_index": torch.tensor(idx, dtype=torch.long),
    }


def _feature_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    if not features:
        return np.zeros((len(frame), 0), dtype=np.float32)
    cols = []
    for feature in features:
        values = pd.to_numeric(frame[feature] if feature in frame.columns else 0.0, errors="coerce")
        cols.append(np.asarray(values, dtype=np.float32).reshape(-1, 1))
    return np.nan_to_num(np.concatenate(cols, axis=1), nan=0.0, posinf=0.0, neginf=0.0)


def _event_matrix(row: pd.Series, dataset: PreparedDataset) -> np.ndarray:
    event_dim = max(1, len(dataset.text_event_feature_names))
    max_events = max(1, int(dataset.max_text_events))
    raw = row.get("text_event_vectors", [])
    vectors = raw if isinstance(raw, list) else []
    matrix = np.zeros((max_events, event_dim), dtype=np.float32)
    for i, vector in enumerate(vectors[:max_events]):
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        width = min(event_dim, arr.shape[0])
        matrix[i, :width] = arr[:width]
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def _static_vector(row: pd.Series, dataset: PreparedDataset) -> np.ndarray:
    if not dataset.static_features:
        return np.zeros(1, dtype=np.float32)
    values = [float(pd.to_numeric(row.get(feature, 0.0), errors="coerce") or 0.0) for feature in dataset.static_features]
    return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _future_known_matrix(row: pd.Series, dataset: PreparedDataset, cfg: AppConfig) -> np.ndarray:
    horizon = max(1, int(cfg.features.horizon))
    n_features = max(1, len(dataset.known_future_features))
    value = row.get("future_known_matrix")
    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=np.float32)
    else:
        arr = np.zeros((horizon, n_features), dtype=np.float32)
    out = np.zeros((horizon, n_features), dtype=np.float32)
    rows = min(horizon, arr.shape[0]) if arr.ndim == 2 else 0
    cols = min(n_features, arr.shape[1]) if arr.ndim == 2 else 0
    if rows and cols:
        out[:rows, :cols] = arr[:rows, :cols]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _target_vector(row: pd.Series, cfg: AppConfig, prefix: str) -> np.ndarray:
    values = []
    for step in range(1, max(1, int(cfg.features.horizon)) + 1):
        values.append(float(row.get(f"{prefix}{step}", row.get("target_return", 0.0)) or 0.0))
    return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _input_gradient_attributions(net: TFTLite, batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    inputs: dict[str, torch.Tensor] = {}
    for key in ["numeric", "text", "events", "static", "future_known"]:
        value = batch[key].detach().clone().requires_grad_(True)
        inputs[key] = value
    net.zero_grad(set_to_none=True)
    with torch.backends.cudnn.flags(enabled=False):
        pred, _ = net(inputs["numeric"], inputs["text"], inputs["events"], inputs["static"], inputs["future_known"])
        score = pred["point"][:, -1].sum()
        score.backward()

    def attr(name: str) -> torch.Tensor:
        value = inputs[name]
        grad = value.grad if value.grad is not None else torch.zeros_like(value)
        return (value * grad).abs().detach()

    numeric = attr("numeric").mean(dim=1).cpu().numpy()
    text = attr("text").mean(dim=1).cpu().numpy()
    events = attr("events").sum(dim=-1).cpu().numpy()
    static = attr("static").cpu().numpy()
    future = attr("future_known").mean(dim=1).cpu().numpy()
    net.zero_grad(set_to_none=True)
    return {
        "numeric": numeric,
        "text": text,
        "events": events,
        "static": static,
        "future_known": future,
    }


def _multi_task_loss(pred: dict[str, torch.Tensor], target: torch.Tensor, direction: torch.Tensor, quantiles: list[float], cfg: AppConfig) -> torch.Tensor:
    point_pred = pred["point"].float()
    target = target.float()
    direction = direction.float()
    point = nn.functional.smooth_l1_loss(point_pred, target, beta=0.01)
    q_loss = 0.0
    for i, quantile in enumerate(quantiles):
        error = target - pred["quantiles"][:, :, i].float()
        q_loss = q_loss + torch.maximum((quantile - 1) * error, quantile * error).mean()
    direction_prob = pred["direction"].float().clamp(1e-4, 1 - 1e-4)

    pos = direction.mean().clamp(1e-4, 1 - 1e-4)
    pos_weight = ((1.0 - pos) / pos).clamp(0.5, 2.0)
    bce = -(pos_weight * direction * torch.log(direction_prob) + (1.0 - direction) * torch.log(1.0 - direction_prob))

    pt = torch.where(direction > 0.5, direction_prob, 1.0 - direction_prob)
    gamma = float(getattr(cfg.train, "direction_focal_gamma", 1.5) or 0.0)
    direction_loss = (((1.0 - pt).clamp_min(0.0) ** gamma) * bce).mean()
    return cfg.train.point_weight * point + cfg.train.quantile_weight * q_loss + cfg.train.direction_weight * direction_loss


def _evaluate_loss(net: TFTLite, loader: DataLoader, quantiles: list[float], cfg: AppConfig, device: torch.device) -> float:
    if len(loader.dataset) == 0:
        return float("inf")
    net.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            with _autocast_context(cfg, device):
                pred, _ = net(batch["numeric"], batch["text"], batch["events"], batch["static"], batch["future_known"])
                loss = _multi_task_loss(pred, batch["target_vector"], batch["direction_vector"], quantiles, cfg)
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("inf")


def _residual_scale(net: TFTLite, loader: DataLoader, cfg: AppConfig, device: torch.device) -> float:
    residuals = []
    net.eval()
    with torch.no_grad():
        for batch in loader:
            batch_device = _to_device(batch, device)
            with _autocast_context(cfg, device):
                pred, _ = net(batch_device["numeric"], batch_device["text"], batch_device["events"], batch_device["static"], batch_device["future_known"])
            residuals.extend((pred["point"][:, -1].cpu() - batch["target_vector"][:, -1]).numpy().tolist())
    scale = float(np.std(residuals)) if residuals else 0.02
    return max(scale, 1e-3)


def _safe_attr_row(values: np.ndarray | None, batch_idx: int) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0 or batch_idx >= arr.shape[0]:
        return np.asarray([], dtype=float)
    row = arr[batch_idx]
    return np.nan_to_num(np.asarray(row, dtype=float).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)


def _combine_event_scores(weights: np.ndarray, attributions: np.ndarray) -> np.ndarray:
    weight_arr = np.nan_to_num(np.asarray(weights, dtype=float).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    attr_arr = np.nan_to_num(np.asarray(attributions, dtype=float).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if len(weight_arr) == 0:
        return attr_arr
    if len(attr_arr) == 0:
        return weight_arr
    width = min(len(weight_arr), len(attr_arr))
    attr = attr_arr[:width]
    if attr.max() > 0:
        attr = attr / attr.max()
    return 0.5 * weight_arr[:width] + 0.5 * attr


def _prediction_row(
    raw: pd.Series,
    pred: dict[str, torch.Tensor],
    aux: dict[str, torch.Tensor],
    batch_idx: int,
    dataset: PreparedDataset,
    residual_scale: float,
    attributions: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    horizon_points = pred["point"][batch_idx].cpu().numpy().astype(float)
    horizon_probs = pred["direction"][batch_idx].cpu().numpy().astype(float)
    horizon_quantiles = pred["quantiles"][batch_idx].cpu().numpy().astype(float)
    report_idx = len(horizon_points) - 1
    point = float(horizon_points[report_idx])
    direction_prob = float(horizon_probs[report_idx])
    quantile_values = horizon_quantiles[report_idx]
    if len(quantile_values) >= 3:
        q10, q50, q90 = quantile_values[0], quantile_values[len(quantile_values) // 2], quantile_values[-1]
    else:
        q10, q50, q90 = point - residual_scale, point, point + residual_scale
    q10 = min(q10, point - 0.1 * residual_scale)
    q90 = max(q90, point + 0.1 * residual_scale)

    gate = aux["modality_gate"][batch_idx].cpu().numpy()
    numeric_weights = aux["numeric_weights"][batch_idx, -1].cpu().numpy()
    text_scalar_weights = aux["text_scalar_weights"][batch_idx, -1].cpu().numpy()
    static_feature_weights = aux.get("static_feature_weights", torch.zeros(1))[batch_idx].cpu().numpy()
    future_feature_weights = aux.get("future_feature_weights", torch.zeros(1))[batch_idx].cpu().numpy()
    event_weights = aux["event_weights"][batch_idx].cpu().numpy()
    temporal_weights = aux["temporal_weights"][batch_idx, -1].cpu().numpy()
    cross_weights = aux["cross_weights"][batch_idx].cpu().numpy()
    attributions = attributions or {}
    numeric_attr = _safe_attr_row(attributions.get("numeric"), batch_idx)
    text_attr = _safe_attr_row(attributions.get("text"), batch_idx)
    event_attr = _safe_attr_row(attributions.get("events"), batch_idx)
    static_attr = _safe_attr_row(attributions.get("static"), batch_idx)
    future_attr = _safe_attr_row(attributions.get("future_known"), batch_idx)

    if len(numeric_attr) and dataset.numeric_features:
        top_num_idx = int(np.argmax(numeric_attr[: len(dataset.numeric_features)]))
    elif len(numeric_weights):
        top_num_idx = int(np.argmax(numeric_weights))
    else:
        top_num_idx = 0
    combined_event_score = _combine_event_scores(event_weights, event_attr)
    if len(combined_event_score):
        top_event_idx = int(np.argmax(combined_event_score))
    elif len(event_weights):
        top_event_idx = int(np.argmax(event_weights))
    else:
        top_event_idx = 0
    top_numeric = dataset.numeric_features[top_num_idx] if top_num_idx < len(dataset.numeric_features) else ""
    titles = raw.get("text_event_titles", [])
    sources = raw.get("text_event_sources", [])
    publishers = raw.get("text_event_publishers", [])
    urls = raw.get("text_event_urls", [])
    uids = raw.get("text_event_uids", [])
    types = raw.get("text_event_types", [])
    selected_title = titles[top_event_idx] if isinstance(titles, list) and top_event_idx < len(titles) else ""
    selected_source = sources[top_event_idx] if isinstance(sources, list) and top_event_idx < len(sources) else ""
    selected_url = urls[top_event_idx] if isinstance(urls, list) and top_event_idx < len(urls) else ""
    top_static_idx = int(np.argmax(static_feature_weights)) if len(static_feature_weights) else 0
    top_future_idx = int(np.argmax(np.nanmean(future_feature_weights, axis=0))) if getattr(future_feature_weights, "ndim", 0) == 2 and future_feature_weights.size else 0
    top_static = dataset.static_features[top_static_idx] if top_static_idx < len(dataset.static_features) else ""
    top_future = dataset.known_future_features[top_future_idx] if top_future_idx < len(dataset.known_future_features) else ""
    event_contributions = []
    if isinstance(titles, list):
        event_order = sorted(enumerate(combined_event_score if len(combined_event_score) else event_weights), key=lambda item: float(item[1]), reverse=True)
        for idx, score in event_order[:10]:
            if idx >= len(titles):
                continue
            event_contributions.append(
                {
                    "rank": len(event_contributions) + 1,
                    "weight": float(event_weights[idx]) if idx < len(event_weights) else 0.0,
                    "input_gradient_attribution": float(event_attr[idx]) if idx < len(event_attr) else 0.0,
                    "contribution_score": float(score),
                    "title": str(titles[idx] or ""),
                    "source": str(sources[idx] if isinstance(sources, list) and idx < len(sources) else ""),
                    "publisher": str(publishers[idx] if isinstance(publishers, list) and idx < len(publishers) else ""),
                    "url": str(urls[idx] if isinstance(urls, list) and idx < len(urls) else ""),
                    "uid": str(uids[idx] if isinstance(uids, list) and idx < len(uids) else ""),
                    "type": str(types[idx] if isinstance(types, list) and idx < len(types) else ""),
                }
            )
    row = {
        "date": raw.get("date"),
        "actual_return": float(raw.get("target_return", 0.0) or 0.0),
        "actual_direction": int(raw.get("target_direction", 0) or 0),
        "pred_return": point,
        "pred_direction_prob": direction_prob,
        "pred_q10": float(q10),
        "pred_q50": float(q50),
        "pred_q90": float(q90),
        "modality_numeric": float(gate[0]),
        "modality_text": float(gate[1]),
        "modality_cross": float(gate[2]),
        "top_numeric_feature": top_numeric,
        "top_numeric_feature_weight": float(numeric_weights[top_num_idx]) if len(numeric_weights) else 0.0,
        "top_numeric_feature_attribution": float(numeric_attr[top_num_idx]) if top_num_idx < len(numeric_attr) else 0.0,
        "top_text_slot": top_event_idx,
        "top_text_slot_weight": float(event_weights[top_event_idx]) if len(event_weights) else 0.0,
        "top_text_slot_attribution": float(event_attr[top_event_idx]) if top_event_idx < len(event_attr) else 0.0,
        "top_text_title": selected_title,
        "top_text_source": selected_source,
        "top_text_url": selected_url,
        "top_static_feature": top_static,
        "top_static_feature_weight": float(static_feature_weights[top_static_idx]) if len(static_feature_weights) else 0.0,
        "top_known_future_feature": top_future,
        "top_known_future_feature_weight": float(np.nanmean(future_feature_weights, axis=0)[top_future_idx]) if getattr(future_feature_weights, "ndim", 0) == 2 and future_feature_weights.size else 0.0,
        "text_event_contributions": event_contributions,
        "feature_attribution_method": "input_x_gradient",
        "temporal_peak_lag": float(len(temporal_weights) - 1 - int(np.argmax(temporal_weights))) if len(temporal_weights) else 0.0,
        "temporal_peak_weight": float(np.max(temporal_weights)) if len(temporal_weights) else 0.0,
        "temporal_attention_entropy": _entropy(temporal_weights),
        "text_peak_lag": float(top_event_idx),
        "text_peak_weight": float(np.max(event_weights)) if len(event_weights) else 0.0,
        "text_attention_entropy": _entropy(event_weights),
        "cross_peak_lag": float(int(np.argmax(cross_weights))) if len(cross_weights) else 0.0,
        "cross_peak_weight": float(np.max(cross_weights)) if len(cross_weights) else 0.0,
        "cross_attention_entropy": _entropy(cross_weights),
        "static_context_norm": float(aux.get("static_context_norm", torch.zeros(1))[batch_idx].cpu()),
    }
    for raw_feature in [
        "return_1d",
        "return_5d",
        "log_return",
        "price_to_ma_20",
        "rsi_14",
        "bollinger_z",
        "volatility_10",
        "volume_ma_10",
    ]:
        row[f"signal_{raw_feature}"] = float(raw.get(raw_feature, 0.0) or 0.0)
    for step, value in enumerate(horizon_points, start=1):
        row[f"pred_return_h{step}"] = float(value)
        row[f"pred_direction_prob_h{step}"] = float(horizon_probs[step - 1])
        row[f"actual_return_h{step}"] = float(raw.get(f"target_return_h{step}", 0.0) or 0.0)
        row[f"actual_direction_h{step}"] = int(raw.get(f"target_direction_h{step}", 0) or 0)
        if horizon_quantiles.ndim == 2:
            row[f"pred_q10_h{step}"] = float(horizon_quantiles[step - 1, 0])
            row[f"pred_q50_h{step}"] = float(horizon_quantiles[step - 1, len(dataset.quantiles) // 2])
            row[f"pred_q90_h{step}"] = float(horizon_quantiles[step - 1, -1])
    row["prediction_explanation"] = (
        f"prob={direction_prob:.2f}, return={point:.4f}; "
        f"modalities numeric/text/cross={gate[0]:.2f}/{gate[1]:.2f}/{gate[2]:.2f}; "
        f"top numeric={top_numeric}, text={selected_title or top_event_idx}."
    )
    for feature, weight in zip(dataset.numeric_features, numeric_weights):
        row[f"sel_num__{feature}"] = float(weight)
        row[f"feature_weight__{feature}"] = float(weight)
    for feature, value in zip(dataset.numeric_features, numeric_attr):
        row[f"attribution__{feature}"] = float(value)
    for feature, weight in zip(dataset.text_features, text_scalar_weights):
        row[f"sel_text__{feature}"] = float(weight)
        row[f"feature_weight__{feature}"] = float(weight)
    for feature, value in zip(dataset.text_features, text_attr):
        row[f"attribution__{feature}"] = float(value)
    for feature, weight in zip(dataset.static_features, static_feature_weights):
        row[f"sel_static__{feature}"] = float(weight)
        row[f"feature_weight__static:{feature}"] = float(weight)
    for feature, value in zip(dataset.static_features, static_attr):
        row[f"attribution__static:{feature}"] = float(value)
    if getattr(future_feature_weights, "ndim", 0) == 2 and future_feature_weights.size:
        for feature, weight in zip(dataset.known_future_features, np.nanmean(future_feature_weights, axis=0)):
            row[f"sel_future__{feature}"] = float(weight)
            row[f"feature_weight__known_future:{feature}"] = float(weight)
    for feature, value in zip(dataset.known_future_features, future_attr):
        row[f"attribution__known_future:{feature}"] = float(value)
    return row


def _entropy(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=float)
    values = values[values > 1e-12]
    return float(-(values * np.log(values)).sum()) if len(values) else 0.0


def _prediction_frame(test_frame: pd.DataFrame, pred_return: np.ndarray, pred_prob: np.ndarray, residuals: np.ndarray | float) -> pd.DataFrame:
    residual_scale = float(np.std(residuals)) if np.ndim(residuals) else float(residuals)
    residual_scale = max(residual_scale, 1e-3)
    out = pd.DataFrame(
        {
            "date": test_frame["date"].to_numpy(),
            "actual_return": pd.to_numeric(test_frame["target_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            "actual_direction": pd.to_numeric(test_frame["target_direction"], errors="coerce").fillna(0).astype(int).to_numpy(),
            "pred_return": pred_return,
            "pred_direction_prob": pred_prob,
            "pred_q10": pred_return - 1.2816 * residual_scale,
            "pred_q50": pred_return,
            "pred_q90": pred_return + 1.2816 * residual_scale,
        }
    )
    out["modality_numeric"] = 1.0
    out["modality_text"] = 0.0
    out["modality_cross"] = 0.0
    horizon = _infer_horizon_from_frame(test_frame)
    for step in range(1, horizon + 1):
        actual_col = f"target_return_h{step}"
        direction_col = f"target_direction_h{step}"
        out[f"actual_return_h{step}"] = pd.to_numeric(test_frame.get(actual_col, test_frame["target_return"]), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        out[f"actual_direction_h{step}"] = pd.to_numeric(test_frame.get(direction_col, test_frame["target_direction"]), errors="coerce").fillna(0).astype(int).to_numpy()
        out[f"pred_return_h{step}"] = pred_return
        out[f"pred_direction_prob_h{step}"] = pred_prob
        out[f"pred_q10_h{step}"] = pred_return - 1.2816 * residual_scale
        out[f"pred_q50_h{step}"] = pred_return
        out[f"pred_q90_h{step}"] = pred_return + 1.2816 * residual_scale
    return out


def _infer_horizon_from_frame(frame: pd.DataFrame) -> int:
    steps = []
    for column in frame.columns:
        if column.startswith("target_return_h"):
            try:
                steps.append(int(column.replace("target_return_h", "")))
            except ValueError:
                continue
    return max(steps) if steps else 1


def _naive_baseline(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> pd.DataFrame:
    horizon = _infer_horizon_from_frame(test_frame)
    pred = _past_observed_return(train_frame, test_frame, horizon)
    actual = pd.to_numeric(test_frame["target_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    prob = 1.0 / (1.0 + np.exp(-pred / max(np.std(pred), 1e-3)))
    out = _prediction_frame(test_frame, pred, prob, actual - pred)
    for step in range(1, horizon + 1):
        step_pred = _past_observed_return(train_frame, test_frame, step)
        step_prob = 1.0 / (1.0 + np.exp(-step_pred / max(np.std(step_pred), 1e-3)))
        out[f"pred_return_h{step}"] = step_pred
        out[f"pred_direction_prob_h{step}"] = step_prob
        residual_scale = max(float(np.std(pd.to_numeric(test_frame.get(f"target_return_h{step}", test_frame["target_return"]), errors="coerce").fillna(0.0).to_numpy(dtype=float) - step_pred)), 1e-3)
        out[f"pred_q10_h{step}"] = step_pred - 1.2816 * residual_scale
        out[f"pred_q50_h{step}"] = step_pred
        out[f"pred_q90_h{step}"] = step_pred + 1.2816 * residual_scale
    return out


def _past_observed_return(train_frame: pd.DataFrame, test_frame: pd.DataFrame, step: int) -> np.ndarray:
    step = max(1, int(step))
    candidate = f"return_{step}d"
    if candidate in test_frame.columns:
        values = pd.to_numeric(test_frame[candidate], errors="coerce").replace([np.inf, -np.inf], np.nan)
        return values.fillna(0.0).to_numpy(dtype=float)
    if step == 1 and "return_1d" in test_frame.columns:
        values = pd.to_numeric(test_frame["return_1d"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        return values.fillna(0.0).to_numpy(dtype=float)
    if "close" in train_frame.columns and "close" in test_frame.columns:
        combined = pd.concat(
            [
                pd.to_numeric(train_frame["close"], errors="coerce"),
                pd.to_numeric(test_frame["close"], errors="coerce"),
            ],
            ignore_index=True,
        ).replace([np.inf, -np.inf], np.nan).ffill()
        returns = combined.pct_change(step).tail(len(test_frame)).fillna(0.0)
        return returns.to_numpy(dtype=float)
    return np.zeros(len(test_frame), dtype=float)


def _sklearn_baseline(reg: Any, clf: Any, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, test_frame: pd.DataFrame) -> pd.DataFrame:
    try:
        direction = (y_train > 0).astype(int)
        reg.fit(x_train, y_train)
        pred = np.asarray(reg.predict(x_test), dtype=float)
        if len(np.unique(direction)) > 1:
            clf.fit(x_train, direction)
            if hasattr(clf, "predict_proba"):
                prob = np.asarray(clf.predict_proba(x_test))[:, 1]
            else:
                score = np.asarray(clf.decision_function(x_test), dtype=float)
                prob = 1.0 / (1.0 + np.exp(-score))
        else:
            prob = np.full(len(x_test), float(direction[0]) if len(direction) else 0.5)
        residuals = y_train - np.asarray(reg.predict(x_train), dtype=float)
    except Exception:
        pred = np.full(len(test_frame), float(np.mean(y_train)) if len(y_train) else 0.0)
        prob = np.full(len(test_frame), float(np.mean(y_train > 0)) if len(y_train) else 0.5)
        residuals = y_train - np.mean(y_train) if len(y_train) else np.array([0.02])
    return _prediction_frame(test_frame, pred, prob, residuals)


def _select_sklearn_regressor(candidates: list[Any], x: np.ndarray, y: np.ndarray) -> Any:
    if len(y) < 40 or len(candidates) == 1:
        return candidates[0]
    splits = min(3, max(2, len(y) // 80))
    tscv = TimeSeriesSplit(n_splits=splits)
    best_model = candidates[0]
    best_score = float("inf")
    for candidate in candidates:
        fold_scores = []
        for train_idx, val_idx in tscv.split(x):
            try:
                model = clone(candidate)
                model.fit(x[train_idx], y[train_idx])
                pred = np.asarray(model.predict(x[val_idx]), dtype=float)
                fold_scores.append(float(np.mean(np.square(pred - y[val_idx]))))
            except Exception:
                fold_scores.append(float("inf"))
        score = float(np.mean(fold_scores)) if fold_scores else float("inf")
        if score < best_score:
            best_score = score
            best_model = candidate
    return best_model


def _select_sklearn_classifier(candidates: list[Any], x: np.ndarray, direction: np.ndarray) -> Any:
    if len(np.unique(direction)) < 2 or len(direction) < 40 or len(candidates) == 1:
        return candidates[0]
    splits = min(3, max(2, len(direction) // 80))
    tscv = TimeSeriesSplit(n_splits=splits)
    best_model = candidates[0]
    best_score = float("inf")
    for candidate in candidates:
        fold_scores = []
        for train_idx, val_idx in tscv.split(x):
            try:
                if len(np.unique(direction[train_idx])) < 2:
                    continue
                model = clone(candidate)
                model.fit(x[train_idx], direction[train_idx])
                if hasattr(model, "predict_proba"):
                    prob = np.asarray(model.predict_proba(x[val_idx]))[:, 1]
                else:
                    score = np.asarray(model.decision_function(x[val_idx]), dtype=float)
                    prob = 1.0 / (1.0 + np.exp(-score))
                prob = np.clip(prob, 1e-5, 1 - 1e-5)
                y_val = direction[val_idx].astype(float)
                fold_scores.append(float(-(y_val * np.log(prob) + (1 - y_val) * np.log(1 - prob)).mean()))
            except Exception:
                fold_scores.append(float("inf"))
        score = float(np.mean(fold_scores)) if fold_scores else float("inf")
        if score < best_score:
            best_score = score
            best_model = candidate
    return best_model


def _ar_baseline(y_train: np.ndarray, test_frame: pd.DataFrame) -> pd.DataFrame:
    try:
        from statsmodels.tsa.arima.model import ARIMA

        best_fit = None
        best_order = (1, 0, 1)
        best_aic = float("inf")
        for p in range(0, 4):
            for d in (0, 1):
                for q in range(0, 4):
                    if p == 0 and d == 0 and q == 0:
                        continue
                    try:
                        fit = ARIMA(y_train, order=(p, d, q)).fit()
                        aic = float(getattr(fit, "aic", np.inf))
                        if np.isfinite(aic) and aic < best_aic:
                            best_fit = fit
                            best_order = (p, d, q)
                            best_aic = aic
                    except Exception:
                        continue
        fitted = best_fit or ARIMA(y_train, order=best_order).fit()
        pred = np.asarray(fitted.forecast(steps=len(test_frame)), dtype=float)
        residuals = np.asarray(fitted.resid, dtype=float)
        prob = 1.0 / (1.0 + np.exp(-pred / max(float(np.std(residuals)), 1e-3)))
        out = _prediction_frame(test_frame, pred, prob, residuals)
        out["baseline_order"] = str(best_order)
        out["baseline_aic"] = best_aic
        return out
    except Exception:
        pass
    if len(y_train) < 3:
        pred = np.full(len(test_frame), float(np.mean(y_train)) if len(y_train) else 0.0)
        residuals = y_train - pred[0] if len(y_train) else np.array([0.02])
    else:
        model = LinearRegression().fit(y_train[:-1].reshape(-1, 1), y_train[1:])
        last = float(y_train[-1])
        pred = []
        for _ in range(len(test_frame)):
            last = float(model.predict(np.array([[last]]))[0])
            pred.append(last)
        pred = np.asarray(pred, dtype=float)
        residuals = y_train[1:] - model.predict(y_train[:-1].reshape(-1, 1))
    prob = 1.0 / (1.0 + np.exp(-pred / max(float(np.std(residuals)), 1e-3)))
    return _prediction_frame(test_frame, pred, prob, residuals)


def _garch_like_baseline(y_train: np.ndarray, test_frame: pd.DataFrame) -> pd.DataFrame:
    try:
        from arch import arch_model

        best_fit = None
        best_order = (1, 1)
        best_aic = float("inf")
        for p in (1, 2):
            for q in (1, 2):
                try:
                    fit = arch_model(y_train * 100.0, mean="AR", lags=1, vol="GARCH", p=p, q=q, rescale=False).fit(disp="off")
                    aic = float(getattr(fit, "aic", np.inf))
                    if np.isfinite(aic) and aic < best_aic:
                        best_fit = fit
                        best_order = (p, q)
                        best_aic = aic
                except Exception:
                    continue
        fitted = best_fit or arch_model(y_train * 100.0, mean="Constant", vol="GARCH", p=1, q=1, rescale=False).fit(disp="off")
        forecast = fitted.forecast(horizon=len(test_frame), reindex=False)
        mean = np.asarray(forecast.mean.iloc[-1], dtype=float) / 100.0
        variance = np.asarray(forecast.variance.iloc[-1], dtype=float) / 10000.0
        scale = np.sqrt(np.maximum(variance, 1e-8))
        prob = 1.0 / (1.0 + np.exp(-mean / np.maximum(scale, 1e-3)))
        out = _prediction_frame(test_frame, mean, prob, max(float(np.mean(scale)), 1e-3))
        out["pred_q10"] = mean - 1.2816 * scale
        out["pred_q90"] = mean + 1.2816 * scale
        out["baseline_order"] = str(best_order)
        out["baseline_aic"] = best_aic
        return out
    except Exception:
        pass
    if len(y_train) == 0:
        pred, scale = np.zeros(len(test_frame)), 0.02
    else:
        alpha = 0.08
        variance = float(np.var(y_train))
        for value in y_train:
            variance = alpha * float(value * value) + (1 - alpha) * variance
        pred = np.full(len(test_frame), float(np.mean(y_train[-20:])))
        scale = max(float(np.sqrt(variance)), 1e-3)
    prob = 1.0 / (1.0 + np.exp(-pred / scale))
    return _prediction_frame(test_frame, pred, prob, scale)


def _select_columns_by_variance(x: np.ndarray, max_features: int) -> np.ndarray:
    if x.shape[1] <= max_features:
        return np.arange(x.shape[1])
    return np.argsort(np.var(x, axis=0))[-max_features:]
