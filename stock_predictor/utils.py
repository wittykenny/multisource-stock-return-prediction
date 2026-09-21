from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import torch


class DataUnavailableError(RuntimeError):
    pass


def get_logger(name: str = "stock-predictor") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_frame(df: pd.DataFrame, path: str | Path) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return out


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return out


def retry_call(
    fn: Callable[[], Any],
    *,
    attempts: int = 3,
    wait_seconds: float = 1.5,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Any:
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except exceptions as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(wait_seconds * attempt)
    raise last_error or RuntimeError("retry_call failed unexpectedly")


def business_day_range(start_date: str, end_date: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start_date, end=end_date, freq="B")


def normalize_frame_dates(df: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out[column] = pd.to_datetime(out[column], errors="coerce").dt.tz_localize(None)
    out = out.dropna(subset=[column]).sort_values(column).drop_duplicates(column)
    out.reset_index(drop=True, inplace=True)
    return out


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.empty:
        return 0.0
    compounded = (1 + returns.fillna(0)).prod()
    years = max(len(returns) / periods_per_year, 1e-9)
    return float(compounded ** (1 / years) - 1)


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    std = returns.std(ddof=0)
    if std == 0 or math.isnan(std):
        return 0.0
    return float((returns.mean() / std) * math.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    curve = (1 + returns.fillna(0)).cumprod()
    peak = curve.cummax()
    drawdown = curve / peak - 1
    return float(drawdown.min())


def safe_divide(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def serialize_float_map(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            out[key] = serialize_float_map(value)
        elif isinstance(value, (np.floating, np.integer)):
            out[key] = value.item()
        elif isinstance(value, np.ndarray):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def top_n_text(values: Iterable[str], n: int = 5, max_chars: int | None = None) -> str:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    selected = cleaned[:n]
    if max_chars is None or max_chars <= 0:
        return " ".join(selected)
    parts: list[str] = []
    total = 0
    for item in selected:
        remaining = max_chars - total
        if remaining <= 0:
            break
        chunk = item[:remaining].strip()
        if not chunk:
            continue
        parts.append(chunk)
        total += len(chunk) + 1
    return " ".join(parts)
