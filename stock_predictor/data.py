from __future__ import annotations

import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html import unescape
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse

import akshare as ak
import numpy as np
import pandas as pd
import requests
import urllib3

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

from .config import AppConfig
from .text_processing import (
    _TEXT_QUALITY_VERSION,
    TextSentimentDatasetBuilder,
    build_text_corpus,
    clean_news_corpus,
    clean_notice_corpus,
)
from .utils import (
    DataUnavailableError,
    business_day_range,
    ensure_dir,
    get_logger,
    normalize_frame_dates,
    retry_call,
    save_frame,
)


@dataclass
class DataBundle:
    stock: pd.DataFrame
    financial: pd.DataFrame
    macro: pd.DataFrame
    industry: pd.DataFrame
    sentiment: pd.DataFrame
    text_sentiment_events: pd.DataFrame
    text_sentiment_daily: pd.DataFrame
    policy: pd.DataFrame
    news: pd.DataFrame
    notices: pd.DataFrame
    source_notes: list[str]


class MarketDataHub:
    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        self.logger = get_logger()
        self.cache_root = ensure_dir(Path(self.cfg.data.cache_dir) / self.cfg.data.symbol)
        cache_dir = str(self.cfg.data.text_extract_cache_dir or "").strip()
        self.content_cache_root = ensure_dir(Path(cache_dir) if cache_dir else self.cache_root / "content_text")
        self._miss_cache_ttl_seconds = max(1, int(self.cfg.data.text_extract_miss_ttl_hours)) * 60 * 60
        self._sina_rate_limited = False
        self._gdelt_rate_limited = False
        self._gdelt_unavailable = False
        self._gdelt_session = requests.Session()
        self._gdelt_session.trust_env = False
        self._content_cache: dict[str, str] = {}

    def fetch_all(self, refresh: bool | None = None) -> DataBundle:
        refresh = self.cfg.data.refresh if refresh is None else refresh
        notes: list[str] = []

        stock = self._cached("stock.csv", self.fetch_stock_history, refresh, dedupe_dates=True)
        if stock.empty:
            raise DataUnavailableError(
                "Stock history is required but unavailable. Both the primary and fallback interfaces failed."
            )

        financial = self._cached("financial.csv", self.fetch_financials, refresh, dedupe_dates=True)
        if financial.empty:
            notes.append("Financial interfaces returned no rows; quarterly financial factors are missing.")

        macro = self._cached("macro.csv", self.fetch_macro, refresh, dedupe_dates=True)
        if not macro.empty and bool(getattr(self.cfg.data, "macro_cleaning_enabled", True)):
            raw_macro_rows = len(macro)
            macro = self._clean_macro_release_frame(macro)
            if not macro.empty:
                save_frame(macro, self.cache_root / "macro_cleaned.csv")
                notes.append(f"Macro release-date cleaning kept {len(macro)} release rows from {raw_macro_rows} raw rows.")
        if macro.empty:
            notes.append("Macro interfaces returned no rows; macro factors are missing.")

        industry = self._cached(
            "industry.csv",
            self.fetch_industry,
            refresh,
            dedupe_dates=True,
        )
        if industry.empty:
            notes.append("Industry interfaces returned no rows; industry factors are missing.")
        else:
            if "industry_primary_source" in industry.columns and not industry["industry_primary_source"].dropna().empty:
                notes.append(f"Primary industry history source: {industry['industry_primary_source'].dropna().iloc[0]}")
            if "industry_secondary_source" in industry.columns and not industry["industry_secondary_source"].dropna().empty:
                notes.append(f"Secondary industry history source: {industry['industry_secondary_source'].dropna().iloc[0]}")
            if "industry_primary_pe_source" in industry.columns and not industry["industry_primary_pe_source"].dropna().empty:
                notes.append(f"Industry PE source: {industry['industry_primary_pe_source'].dropna().iloc[0]}")

        policy = self._cached("policy.csv", self.fetch_policy, refresh, dedupe_dates=False)
        if self.cfg.data.strict_official_policy and not policy.empty:
            policy = self._filter_strict_official_policy_rows(policy)
        if self.cfg.data.strict_official_policy and policy.empty and not refresh:
            try:
                policy = self._normalize_frame(self.fetch_policy(), dedupe_dates=False)
                if not policy.empty:
                    save_frame(policy, self.cache_root / "policy.csv")
            except Exception as exc:
                self.logger.warning("Failed to refresh strict official policy corpus: %s", exc)
                policy = pd.DataFrame()
        if policy.empty:
            notes.append("Policy text interface returned no rows; policy factors are missing.")
        elif "source" in policy.columns:
            sources = ", ".join(sorted(policy["source"].dropna().astype(str).unique()))
            notes.append(f"Policy corpus sources: {sources}")
            missing_policy_years = self._detect_missing_years(policy, min_rows=1)
            if missing_policy_years:
                notes.append(
                    "Official policy corpus is incomplete for years: "
                    + ", ".join(str(year) for year in sorted(missing_policy_years))
                    + ". Public ministry archive access/search coverage may be incomplete in the current network environment."
                )

        news = self._cached("news.csv", self.fetch_news, refresh, dedupe_dates=False)
        if not news.empty:
            cleaned_news = clean_news_corpus(
                self.cfg,
                news,
                enforce_daily_cap=True,
                logger=self.logger,
                stage="cached_news",
            )
            if len(cleaned_news) != len(news) or "news_relevance_score" not in news.columns:
                news = self._normalize_frame(cleaned_news, dedupe_dates=False)
                if not news.empty:
                    save_frame(news, self.cache_root / "news.csv")
        if news.empty:
            notes.append("Text interfaces returned no rows; news and report text factors are missing.")
        elif "source" in news.columns:
            sources = ", ".join(sorted(news["source"].dropna().astype(str).unique()))
            notes.append(f"Text corpus sources: {sources}")
            if news["source"].astype(str).str.contains("eastmoney_search", regex=False).any():
                notes.append("Eastmoney historical news search is available but effectively capped to the most recent result window per keyword.")
            if news["source"].astype(str).str.contains("gdelt_search", regex=False).any():
                notes.append("GDELT historical company news is fetched in quarter-sliced mode to extend 2020-2025 coverage.")
            if news["source"].astype(str).str.contains("sina_search", regex=False).any():
                notes.append("Sina historical news search is fetched in year-sliced mode to extend 2020-2025 company news coverage.")
            if news["source"].astype(str).str.contains("google_news_rss", regex=False).any():
                notes.append("Google News RSS is enabled in month-like time windows as a historical news supplement for 2020-2025 coverage.")
            if news["source"].astype(str).str.contains("caixin", regex=False).any():
                notes.append("Caixin market-news crawl is enabled as the second newsroom source for company and industry text supplementation.")
            if news["source"].astype(str).str.contains("guba", regex=False).any():
                notes.append("Eastmoney Guba board posts are crawled as an investor-discussion text supplement.")
        if self._gdelt_rate_limited:
            notes.append("GDELT historical news source hit its public rate limit; only partial GDELT rows may be available.")
        if self._gdelt_unavailable:
            notes.append("GDELT historical news source is temporarily unavailable in the current network environment.")
        if self._sina_rate_limited:
            notes.append("Sina historical news source is currently blocked by site rate limiting or login gating; only partial or zero Sina rows may be available.")

        notices = self._cached("notices.csv", self.fetch_notices, refresh, dedupe_dates=False)
        if notices.empty:
            notes.append("Notice interface returned no rows.")
        else:
            cleaned_notices = clean_notice_corpus(
                self.cfg,
                notices,
                enforce_daily_cap=True,
                logger=self.logger,
                stage="cached_notices",
            )
            if len(cleaned_notices) != len(notices) or "news_relevance_score" not in notices.columns:
                notices = self._normalize_frame(cleaned_notices, dedupe_dates=False)
                if not notices.empty:
                    save_frame(notices, self.cache_root / "notices_cleaned.csv")
            if "source" in notices.columns and notices["source"].astype(str).str.contains("cninfo", regex=False).any():
                notes.append("Company notices are fetched from CNInfo official fulltext search.")
            else:
                notes.append("Company notices are using the Eastmoney historical notice API fallback.")

        text_sentiment_events, text_sentiment_daily = self._load_or_build_text_sentiment_datasets(
            news=news,
            notices=notices,
            refresh=refresh,
        )
        if text_sentiment_events.empty or text_sentiment_daily.empty:
            notes.append("Independent news/notice sentiment dataset returned no rows.")
        else:
            notes.append(
                "Independent news/notice sentiment dataset is available with "
                f"{len(text_sentiment_events)} event rows and {len(text_sentiment_daily)} daily rows."
            )

        sentiment_path = self.cache_root / "sentiment.csv"
        if sentiment_path.exists() and not refresh:
            sentiment = self._normalize_frame(pd.read_csv(sentiment_path), dedupe_dates=True)
            if not sentiment.empty and "sentiment_source" in sentiment.columns:
                sources = ",".join(sentiment["sentiment_source"].dropna().astype(str).unique())
                needs_proxy_refresh = self.cfg.data.news_quality_filter and "news_proxy_quality_mean" not in sentiment.columns
                if "eastmoney_guba_board" not in sources or needs_proxy_refresh:
                    try:
                        refreshed = self.fetch_investor_sentiment(news=news, notices=notices, policy=policy)
                        if not refreshed.empty:
                            save_frame(refreshed, sentiment_path)
                            sentiment = self._normalize_frame(refreshed, dedupe_dates=True)
                    except Exception as exc:
                        self.logger.warning("Failed to refresh direct investor sentiment corpus: %s", exc)
        else:
            try:
                sentiment = self.fetch_investor_sentiment(news=news, notices=notices, policy=policy)
                if not sentiment.empty:
                    save_frame(sentiment, sentiment_path)
                sentiment = self._normalize_frame(sentiment, dedupe_dates=True)
            except Exception as exc:
                self.logger.warning("Failed to fetch sentiment.csv: %s", exc)
                sentiment = pd.DataFrame()
        if sentiment.empty:
            notes.append("Investor-sentiment interfaces returned no rows inside the configured date window; direct forum sentiment falls back to historical text-sentiment proxies.")
        elif "sentiment_source" in sentiment.columns:
            sources = ", ".join(sorted(sentiment["sentiment_source"].dropna().astype(str).unique()))
            notes.append(f"Investor sentiment sources: {sources}")
            missing_guba_years = self._detect_missing_years(sentiment, min_rows=max(1, int(self.cfg.data.guba_min_rows_per_year)))
            if "eastmoney_guba_board" not in sources or missing_guba_years:
                notes.append(
                    "Direct Eastmoney Guba historical coverage is incomplete for years: "
                    + ", ".join(str(year) for year in sorted(missing_guba_years))
                    + ". Text-sentiment proxy fields are retained only as a supplement."
                )

        return DataBundle(
            stock=stock,
            financial=financial,
            macro=macro,
            industry=industry,
            sentiment=sentiment,
            text_sentiment_events=text_sentiment_events,
            text_sentiment_daily=text_sentiment_daily,
            policy=policy,
            news=news,
            notices=notices,
            source_notes=notes,
        )

    def _cached(
        self,
        filename: str,
        loader,
        refresh: bool,
        dedupe_dates: bool,
        required_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        path = self.cache_root / filename
        cached_frame = self._read_cached_frame(path, dedupe_dates=dedupe_dates, required_columns=required_columns)
        if path.exists() and not refresh:
            if cached_frame is not None:
                return cached_frame
        try:
            df = loader()
            normalized = self._normalize_frame(df, dedupe_dates=dedupe_dates)
            if not normalized.empty:
                save_frame(normalized, path)
                return normalized
            if cached_frame is not None and not cached_frame.empty:
                self.logger.warning("Live fetch for %s returned no rows; falling back to cached data.", filename)
                return cached_frame
            return normalized
        except Exception as exc:
            self.logger.warning("Failed to fetch %s: %s", filename, exc)
            if cached_frame is not None and not cached_frame.empty:
                self.logger.warning("Falling back to cached %s because live fetch failed.", filename)
                return cached_frame
            return pd.DataFrame()

    def _read_cached_frame(
        self,
        path: Path,
        *,
        dedupe_dates: bool,
        required_columns: list[str] | None = None,
    ) -> pd.DataFrame | None:
        if not path.exists():
            return None
        try:
            frame = self._normalize_frame(pd.read_csv(path), dedupe_dates=dedupe_dates)
            if required_columns:
                missing = [column for column in required_columns if column not in frame.columns]
                if missing:
                    self.logger.warning("Cached frame %s is missing required columns: %s", path, ", ".join(missing))
                    return None
            return frame
        except Exception as exc:
            self.logger.warning("Failed to read cached frame %s: %s", path, exc)
            return None

    def _load_or_build_text_sentiment_datasets(
        self,
        *,
        news: pd.DataFrame,
        notices: pd.DataFrame,
        refresh: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        event_path = self.cache_root / "text_sentiment_events.csv"
        daily_path = self.cache_root / "text_sentiment_daily.csv"
        required_event_columns = [
            "text_uid",
            "hybrid_sentiment",
            "finbert_sentiment",
            "text_type",
            "news_relevance_score",
            "source_quality_score",
            "event_strength_score",
            "text_quality_version",
        ]
        required_daily_columns = [
            "text_sent_event_count",
            "text_sent_hybrid_mean",
            "text_sent_news_quality_mean",
            "text_sent_source_quality_mean",
            "text_sent_event_strength_mean",
        ]
        cached_events = self._read_cached_frame(
            event_path,
            dedupe_dates=False,
            required_columns=required_event_columns,
        )
        cached_daily = self._read_cached_frame(
            daily_path,
            dedupe_dates=True,
            required_columns=required_daily_columns,
        )
        if cached_events is not None:
            versions = set(cached_events.get("text_quality_version", pd.Series(dtype=str)).dropna().astype(str).unique())
            if versions != {_TEXT_QUALITY_VERSION}:
                self.logger.warning("Cached text sentiment events use stale quality version: %s", ", ".join(sorted(versions)) or "missing")
                cached_events = None
        if not refresh and cached_events is not None and cached_daily is not None:
            return cached_events, cached_daily
        corpus = build_text_corpus(self.cfg, news, notices, include_policy=False)
        if corpus.empty:
            return pd.DataFrame(), pd.DataFrame()
        try:
            builder = TextSentimentDatasetBuilder(self.cfg, include_embeddings=False)
            scored = builder.build(corpus)
            if not scored.events.empty:
                save_frame(scored.events.drop(columns=["embedding_vector"], errors="ignore"), event_path)
            if not scored.daily.empty:
                save_frame(scored.daily, daily_path)
            events = self._normalize_frame(scored.events, dedupe_dates=False)
            daily = self._normalize_frame(scored.daily, dedupe_dates=True)
            return events, daily
        except Exception as exc:
            self.logger.warning("Failed to build independent text sentiment datasets: %s", exc)
            if cached_events is not None and cached_daily is not None:
                self.logger.warning("Falling back to cached text sentiment datasets.")
                return cached_events, cached_daily
            return pd.DataFrame(), pd.DataFrame()

    def _clip_frame_by_date_range(self, df: pd.DataFrame, *, lookback_days: int = 0) -> pd.DataFrame:
        if df.empty or "date" not in df.columns:
            return df
        start_date = pd.to_datetime(self.cfg.data.start_date) - pd.Timedelta(days=max(0, lookback_days))
        end_date = pd.to_datetime(self.cfg.data.end_date)
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
        out = out.dropna(subset=["date"])
        out = out[(out["date"] >= start_date) & (out["date"] <= end_date)]
        return out.sort_values("date").reset_index(drop=True)

    def _normalize_frame(self, df: pd.DataFrame, dedupe_dates: bool) -> pd.DataFrame:
        if df.empty or "date" not in df.columns:
            return df
        if dedupe_dates:
            return normalize_frame_dates(df)
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
        out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return out

    def _resolve_row_limit(self, row_limit: int) -> int | None:
        limit = int(row_limit or 0)
        return None if limit <= 0 else limit

    def _focus_terms(self) -> list[str]:
        return [
            self.cfg.data.company_name,
            self.cfg.data.symbol,
            self.cfg.data.industry_name,
            "\u52a8\u529b\u7535\u6c60",
            "\u50a8\u80fd",
            "\u65b0\u80fd\u6e90",
            "\u65b0\u80fd\u6e90\u6c7d\u8f66",
            "\u9502\u7535\u6c60",
            "\u7535\u6c60\u56de\u6536",
        ]

    def _policy_title_terms(self) -> list[str]:
        return [
            "\u901a\u77e5",
            "\u610f\u89c1",
            "\u65b9\u6848",
            "\u89c4\u8303",
            "\u6761\u4ef6",
            "\u5b9e\u65bd",
            "\u653f\u7b56",
            "\u529e\u6cd5",
            "\u6307\u5f15",
            "\u89c4\u5212",
            "\u516c\u544a",
            "\u901a\u544a",
            "\u7ec6\u5219",
        ]

    def _official_policy_publishers(self) -> list[str]:
        return [
            "\u5de5\u4e1a\u548c\u4fe1\u606f\u5316\u90e8",
            "\u5de5\u4fe1\u90e8",
            "\u56fd\u5bb6\u80fd\u6e90\u5c40",
            "\u56fd\u5bb6\u53d1\u5c55\u6539\u9769\u59d4",
            "\u53d1\u6539\u59d4",
            "\u56fd\u52a1\u9662",
            "\u8d22\u653f\u90e8",
            "\u5546\u52a1\u90e8",
            "gov.cn",
            "miit.gov.cn",
            "nea.gov.cn",
            "ndrc.gov.cn",
        ]

    def _publisher_hint_tokens(self) -> set[str]:
        return set(self._official_policy_publishers())

    def _policy_query_terms(self) -> list[str]:
        values = list(self.cfg.data.policy_keywords) + [
            self.cfg.data.industry_name,
            "\u52a8\u529b\u7535\u6c60",
            "\u50a8\u80fd",
            "\u65b0\u80fd\u6e90\u6c7d\u8f66",
        ]
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            clean = self._compact_text(value)
            if not clean or clean in seen:
                continue
            seen.add(clean)
            deduped.append(clean)
        return deduped

    def _year_set(self, years: set[int] | None = None) -> list[int]:
        if years:
            return sorted(int(year) for year in years)
        start_year = pd.to_datetime(self.cfg.data.start_date).year
        end_year = pd.to_datetime(self.cfg.data.end_date).year
        return list(range(start_year, end_year + 1))

    def _detect_missing_years(self, df: pd.DataFrame, *, min_rows: int) -> set[int]:
        years = set(self._year_set())
        if df.empty or "date" not in df.columns:
            return years
        tmp = df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
        counts = tmp.dropna(subset=["date"]).groupby(tmp["date"].dt.year).size().to_dict()
        return {year for year in years if int(counts.get(year, 0)) < min_rows}

    def _filter_frame_years(self, df: pd.DataFrame, years: set[int] | None = None) -> pd.DataFrame:
        if df.empty or years is None or "date" not in df.columns:
            return df
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        return out.loc[out["date"].dt.year.isin(years)].reset_index(drop=True)

    def _response_text(self, response: requests.Response) -> str:
        try:
            response.encoding = response.apparent_encoding or response.encoding
        except Exception:
            pass
        return response.text

    def _extract_date_candidates(self, *values: object) -> pd.Timestamp:
        patterns = [
            r"(20\d{2})[-/\.](\d{1,2})[-/\.](\d{1,2})",
            r"(20\d{2})(\d{2})(\d{2})",
            r"(20\d{2})\u5e74(\d{1,2})\u6708(\d{1,2})\u65e5",
        ]
        for value in values:
            text = str(value or "")
            for pattern in patterns:
                match = re.search(pattern, text)
                if not match:
                    continue
                parts = [item.zfill(2) for item in match.groups()]
                dt = pd.to_datetime("-".join([parts[0], parts[1], parts[2]]), errors="coerce")
                if pd.notna(dt):
                    return dt
        return pd.NaT

    def _build_archive_page_urls(self, seed_url: str, max_pages: int) -> list[str]:
        clean = str(seed_url or "").strip()
        if not clean:
            return []
        urls = [clean]
        if max_pages <= 1:
            return urls
        suffix = ".html" if clean.endswith(".html") else ".htm"
        if clean.endswith(f"index{suffix}"):
            base = clean[: -len(f"index{suffix}")]
            urls.extend(f"{base}index_{page}{suffix}" for page in range(1, max_pages))
            return urls
        if clean.endswith("/"):
            urls.extend(f"{clean}index_{page}.html" for page in range(1, max_pages))
            return urls
        urls.extend(f"{clean.rstrip('/')}/index_{page}.html" for page in range(1, max_pages))
        return urls

    def _infer_publisher_from_url(self, url: object) -> str:
        target = str(url or "").lower()
        mapping = {
            "gov.cn": "\u56fd\u52a1\u9662",
            "miit.gov.cn": "\u5de5\u4e1a\u548c\u4fe1\u606f\u5316\u90e8",
            "nea.gov.cn": "\u56fd\u5bb6\u80fd\u6e90\u5c40",
            "ndrc.gov.cn": "\u56fd\u5bb6\u53d1\u5c55\u6539\u9769\u59d4",
        }
        for domain, publisher in mapping.items():
            if domain in target:
                return publisher
        return urlparse(target).netloc or ""

    def _looks_like_policy_row(self, title: object, url: object = "", extra_text: object = "") -> bool:
        text = f"{title or ''} {url or ''} {extra_text or ''}"
        has_title_term = any(token in text for token in self._policy_title_terms())
        has_focus_term = any(term and term in text for term in self._focus_terms())
        return has_title_term and has_focus_term

    def fetch_stock_history(self) -> pd.DataFrame:
        start = self.cfg.data.start_date.replace("-", "")
        end = self.cfg.data.end_date.replace("-", "")
        rename_map = {
            "\u65e5\u671f": "date",
            "\u5f00\u76d8": "open",
            "\u6536\u76d8": "close",
            "\u6700\u9ad8": "high",
            "\u6700\u4f4e": "low",
            "\u6210\u4ea4\u91cf": "volume",
            "\u6210\u4ea4\u989d": "amount",
            "\u632f\u5e45": "amplitude",
            "\u6da8\u8dcc\u5e45": "pct_change",
            "\u6da8\u8dcc\u989d": "change",
            "\u6362\u624b\u7387": "turnover_rate",
            "date": "date",
            "open": "open",
            "close": "close",
            "high": "high",
            "low": "low",
            "volume": "volume",
            "amount": "amount",
        }

        def primary() -> pd.DataFrame:
            return ak.stock_zh_a_hist(
                symbol=self.cfg.data.symbol,
                period="daily",
                start_date=start,
                end_date=end,
                adjust="",
            )

        def fallback() -> pd.DataFrame:
            return ak.stock_zh_a_hist_tx(
                symbol=f"{self.cfg.data.exchange}{self.cfg.data.symbol}",
                start_date=start,
                end_date=end,
                adjust="",
            )

        last_error: Exception | None = None
        for loader in [primary, fallback]:
            try:
                df = retry_call(loader, attempts=2)
                if df.empty:
                    continue
                df = df.rename(columns=rename_map)
                if "volume" not in df.columns and "amount" in df.columns:
                    df["volume"] = pd.to_numeric(df["amount"], errors="coerce")
                    df["amount"] = pd.NA
                for column in ["amount", "amplitude", "pct_change", "change", "turnover_rate"]:
                    if column not in df.columns:
                        df[column] = pd.NA
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                for column in [col for col in df.columns if col != "date"]:
                    df[column] = pd.to_numeric(df[column], errors="coerce")
                df["symbol"] = self.cfg.data.symbol
                keep = [
                    "date",
                    "symbol",
                    "open",
                    "close",
                    "high",
                    "low",
                    "volume",
                    "amount",
                    "amplitude",
                    "pct_change",
                    "change",
                    "turnover_rate",
                ]
                out = df[[col for col in keep if col in df.columns]]
                return out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
            except Exception as exc:
                last_error = exc
        raise DataUnavailableError(f"stock history fetch failed: {last_error}")

    def fetch_financials(self) -> pd.DataFrame:
        symbol = f"{self.cfg.data.symbol}.{self.cfg.data.exchange.upper()}"
        frames: list[pd.DataFrame] = []

        try:
            indicator = retry_call(
                lambda: ak.stock_financial_analysis_indicator_em(symbol=symbol, indicator="\u6309\u62a5\u544a\u671f"),
                attempts=2,
            ).rename(
                columns={
                    "REPORT_DATE": "date",
                    "NOTICE_DATE": "disclosure_date",
                    "ANNOUNCE_DATE": "disclosure_date",
                    "DISCLOSE_DATE": "disclosure_date",
                    "UPDATE_DATE": "disclosure_date",
                    "EPSJB": "eps",
                    "BPS": "bps",
                    "TOTALOPERATEREVE": "revenue",
                    "PARENTNETPROFIT": "net_profit",
                    "ROE_DILUTED": "roe",
                    "GROSS_PROFIT_RATIO": "gross_margin",
                    "NET_PROFIT_RATIO": "net_margin",
                }
            )
            frames.append(self._select_numeric_statement_fields(indicator))
        except Exception:
            self.logger.warning("Financial indicator interface unavailable.")

        try:
            balance = retry_call(lambda: ak.stock_balance_sheet_by_report_em(symbol=symbol), attempts=2).rename(
                columns={
                    "REPORT_DATE": "date",
                    "NOTICE_DATE": "disclosure_date",
                    "ANNOUNCE_DATE": "disclosure_date",
                    "DISCLOSE_DATE": "disclosure_date",
                    "UPDATE_DATE": "disclosure_date",
                    "MONETARYFUNDS": "cash",
                    "INVENTORY": "inventory",
                    "TOTAL_ASSETS": "total_assets",
                    "TOTAL_LIABILITIES": "total_liabilities",
                    "TOTAL_PARENT_EQUITY": "parent_equity",
                    "ACCOUNTS_PAYABLE": "accounts_payable",
                    "FIXED_ASSET": "fixed_asset",
                    "INTANGIBLE_ASSET": "intangible_asset",
                }
            )
            frames.append(self._select_numeric_statement_fields(balance))
        except Exception:
            self.logger.warning("Balance sheet interface unavailable.")

        try:
            profit = retry_call(lambda: ak.stock_profit_sheet_by_report_em(symbol=symbol), attempts=2).rename(
                columns={
                    "REPORT_DATE": "date",
                    "NOTICE_DATE": "disclosure_date",
                    "ANNOUNCE_DATE": "disclosure_date",
                    "DISCLOSE_DATE": "disclosure_date",
                    "UPDATE_DATE": "disclosure_date",
                    "TOTAL_OPERATE_COST": "total_operate_cost",
                    "OPERATE_PROFIT": "operate_profit",
                    "TOTAL_PROFIT": "total_profit",
                    "NETPROFIT": "statement_netprofit",
                    "PARENT_NETPROFIT": "statement_parent_netprofit",
                    "RESEARCH_EXPENSE": "research_expense",
                }
            )
            frames.append(self._select_numeric_statement_fields(profit))
        except Exception:
            self.logger.warning("Profit statement interface unavailable.")

        try:
            cashflow = retry_call(lambda: ak.stock_cash_flow_sheet_by_report_em(symbol=symbol), attempts=2).rename(
                columns={
                    "REPORT_DATE": "date",
                    "NOTICE_DATE": "disclosure_date",
                    "ANNOUNCE_DATE": "disclosure_date",
                    "DISCLOSE_DATE": "disclosure_date",
                    "UPDATE_DATE": "disclosure_date",
                    "NETCASH_OPERATE": "netcash_operate",
                    "NETCASH_INVEST": "netcash_invest",
                    "NETCASH_FINANCE": "netcash_finance",
                    "END_CCE": "end_cash_equivalents",
                    "SALES_SERVICES": "cash_from_sales",
                    "BUY_SERVICES": "cash_to_buy_services",
                }
            )
            frames.append(self._select_numeric_statement_fields(cashflow))
        except Exception:
            self.logger.warning("Cashflow statement interface unavailable.")

        if frames:
            merged: pd.DataFrame | None = None
            for frame in frames:
                tmp = frame.copy()
                tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
                for column in [col for col in tmp.columns if col != "date"]:
                    tmp[column] = pd.to_numeric(tmp[column], errors="coerce")
                merged = tmp if merged is None else merged.merge(tmp, on="date", how="outer")
            assert merged is not None
            merged = merged.sort_values("date").reset_index(drop=True)
            disclosure_columns = [col for col in merged.columns if str(col).startswith("disclosure_date")]
            if disclosure_columns:
                disclosure = pd.Series(pd.NaT, index=merged.index, dtype="datetime64[ns]")
                for column in disclosure_columns:
                    disclosure = disclosure.fillna(pd.to_datetime(merged[column], errors="coerce"))
                merged["disclosure_date"] = disclosure
                merged = merged.drop(columns=[col for col in disclosure_columns if col != "disclosure_date"], errors="ignore")
            financial_start = pd.to_datetime(self.cfg.data.financial_min_date)
            merged = merged[(pd.to_datetime(merged["date"], errors="coerce") >= financial_start)]
            return self._clip_frame_by_date_range(merged, lookback_days=0)

        abstract = retry_call(lambda: ak.stock_financial_abstract(symbol=self.cfg.data.symbol), attempts=2)
        value_map = {
            "\u6bcf\u80a1\u6536\u76ca": "eps",
            "\u6bcf\u80a1\u51c0\u8d44\u4ea7": "bps",
            "\u8425\u4e1a\u603b\u6536\u5165": "revenue",
            "\u5f52\u6bcd\u51c0\u5229\u6da6": "net_profit",
            "\u51c0\u8d44\u4ea7\u6536\u76ca\u7387": "roe",
            "\u9500\u552e\u6bdb\u5229\u7387": "gross_margin",
            "\u51c0\u5229\u7387": "net_margin",
        }
        melted = abstract.melt(
            id_vars=[col for col in ["\u9009\u9879", "\u6307\u6807"] if col in abstract.columns],
            var_name="report_date",
            value_name="value",
        )
        metric_col = "\u6307\u6807" if "\u6307\u6807" in melted.columns else "\u9009\u9879"
        melted = melted[melted[metric_col].isin(value_map)]
        out = (
            melted.assign(metric=melted[metric_col].map(value_map))
            .pivot_table(index="report_date", columns="metric", values="value", aggfunc="first")
            .reset_index()
            .rename(columns={"report_date": "date"})
        )
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        financial_start = pd.to_datetime(self.cfg.data.financial_min_date)
        out = out[(pd.to_datetime(out["date"], errors="coerce") >= financial_start)]
        return self._clip_frame_by_date_range(out, lookback_days=0)

    def fetch_investor_sentiment(
        self,
        *,
        news: pd.DataFrame | None = None,
        notices: pd.DataFrame | None = None,
        policy: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if not self.cfg.data.sentiment_enabled:
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        source_names: list[str] = []
        detail_specs = [
            (
                lambda: ak.stock_comment_detail_scrd_focus_em(symbol=self.cfg.data.symbol),
                {
                    "\u4ea4\u6613\u65e5": "date",
                    "\u7528\u6237\u5173\u6ce8\u6307\u6570": "investor_focus_index",
                },
                "eastmoney_comment_focus",
            ),
            (
                lambda: ak.stock_comment_detail_scrd_desire_em(symbol=self.cfg.data.symbol),
                {
                    "\u4ea4\u6613\u65e5\u671f": "date",
                    "\u53c2\u4e0e\u610f\u613f": "investor_participation",
                    "5\u65e5\u5e73\u5747\u53c2\u4e0e\u610f\u613f": "investor_participation_ma5",
                    "\u53c2\u4e0e\u610f\u613f\u53d8\u5316": "investor_participation_change",
                    "5\u65e5\u5e73\u5747\u53d8\u5316": "investor_participation_change_ma5",
                },
                "eastmoney_comment_desire",
            ),
            (
                lambda: ak.stock_comment_detail_zhpj_lspf_em(symbol=self.cfg.data.symbol),
                {
                    "\u4ea4\u6613\u65e5": "date",
                    "\u8bc4\u5206": "investor_rating_score",
                },
                "eastmoney_comment_rating",
            ),
            (
                lambda: ak.stock_comment_detail_zlkp_jgcyd_em(symbol=self.cfg.data.symbol),
                {
                    "\u4ea4\u6613\u65e5": "date",
                    "\u673a\u6784\u53c2\u4e0e\u5ea6": "institution_participation",
                },
                "eastmoney_comment_institution",
            ),
        ]

        for loader, rename_map, source in detail_specs:
            try:
                frame = retry_call(loader, attempts=2).rename(columns=rename_map)
            except Exception:
                continue
            if frame.empty or "date" not in frame.columns:
                continue
            out = frame[[col for col in frame.columns if col == "date" or col in rename_map.values()]].copy()
            out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
            for column in [col for col in out.columns if col != "date"]:
                out[column] = pd.to_numeric(out[column], errors="coerce")
            frames.append(out)
            source_names.append(source)

        guba_posts = self._fetch_guba_board_posts()
        if not guba_posts.empty:
            guba_posts["guba_sentiment_proxy"] = guba_posts["title"].map(self._proxy_sentiment_score)
            guba_daily = (
                guba_posts.groupby("date")
                .agg(
                    guba_post_count=("title", "size"),
                    guba_click_count=("click_count", "sum"),
                    guba_comment_count=("comment_count", "sum"),
                    guba_sentiment_proxy=("guba_sentiment_proxy", "mean"),
                )
                .reset_index()
            )
            frames.append(guba_daily)
            source_names.append("eastmoney_guba_board")

        if not frames:
            proxy = self._build_text_sentiment_proxy(news, notices, policy)
            return proxy

        merged: pd.DataFrame | None = None
        for frame in frames:
            merged = frame if merged is None else merged.merge(frame, on="date", how="outer")
        assert merged is not None
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.normalize()
        merged = merged.dropna(subset=["date"])
        merged = merged[
            (merged["date"] >= pd.to_datetime(self.cfg.data.start_date))
            & (merged["date"] <= pd.to_datetime(self.cfg.data.end_date))
        ]
        if merged.empty:
            proxy = self._build_text_sentiment_proxy(news, notices, policy)
            return proxy
        proxy = self._build_text_sentiment_proxy(news, notices, policy)
        if not proxy.empty:
            merged = merged.merge(proxy.drop(columns=["sentiment_source"], errors="ignore"), on="date", how="outer")
        merged["sentiment_source"] = ",".join(sorted(set(source_names + (proxy.get("sentiment_source", pd.Series(dtype=str)).dropna().astype(str).unique().tolist() if not proxy.empty else []))))
        return merged.sort_values("date").reset_index(drop=True)

    def fetch_macro(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []

        try:
            cpi = retry_call(ak.macro_china_cpi, attempts=2).rename(
                columns={
                    "\u6708\u4efd": "date",
                    "\u5168\u56fd-\u540c\u6bd4\u589e\u957f": "cpi_yoy",
                    "\u5168\u56fd-\u73af\u6bd4\u589e\u957f": "cpi_mom",
                }
            )
            cpi["date"] = cpi["date"].map(self._monthly)
            frames.append(cpi[[col for col in ["date", "cpi_yoy", "cpi_mom"] if col in cpi.columns]])
        except Exception:
            self.logger.warning("CPI interface unavailable.")

        try:
            pmi = retry_call(ak.macro_china_pmi, attempts=2).rename(
                columns={
                    "\u6708\u4efd": "date",
                    "\u5236\u9020\u4e1a-\u6307\u6570": "pmi_manufacturing",
                    "\u975e\u5236\u9020\u4e1a-\u6307\u6570": "pmi_non_manufacturing",
                }
            )
            pmi["date"] = pmi["date"].map(self._monthly)
            frames.append(
                pmi[[col for col in ["date", "pmi_manufacturing", "pmi_non_manufacturing"] if col in pmi.columns]]
            )
        except Exception:
            self.logger.warning("PMI interface unavailable.")

        try:
            money = retry_call(ak.macro_china_money_supply, attempts=2).rename(
                columns={
                    "\u6708\u4efd": "date",
                    "\u8d27\u5e01\u548c\u51c6\u8d27\u5e01(M2)-\u540c\u6bd4\u589e\u957f": "m2_yoy",
                }
            )
            money["date"] = money["date"].map(self._monthly)
            frames.append(money[[col for col in ["date", "m2_yoy"] if col in money.columns]])
        except Exception:
            self.logger.warning("Money supply interface unavailable.")

        try:
            gdp = retry_call(ak.macro_china_gdp, attempts=2).rename(
                columns={
                    "\u5b63\u5ea6": "date",
                    "\u56fd\u5185\u751f\u4ea7\u603b\u503c-\u540c\u6bd4\u589e\u957f": "gdp_yoy",
                }
            )
            gdp["date"] = gdp["date"].map(self._quarterly)
            frames.append(gdp[[col for col in ["date", "gdp_yoy"] if col in gdp.columns]])
        except Exception:
            self.logger.warning("GDP interface unavailable.")

        try:
            lpr = retry_call(ak.macro_china_lpr, attempts=2).rename(
                columns={
                    "TRADE_DATE": "date",
                    "LPR1Y": "lpr_1y",
                    "LPR5Y": "lpr_5y",
                }
            )
            frames.append(lpr[[col for col in ["date", "lpr_1y", "lpr_5y"] if col in lpr.columns]])
        except Exception:
            self.logger.warning("LPR interface unavailable.")

        merged: pd.DataFrame | None = None
        for frame in frames:
            tmp = frame.copy()
            tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
            for column in [col for col in tmp.columns if col != "date"]:
                tmp[column] = pd.to_numeric(tmp[column], errors="coerce")
            merged = tmp if merged is None else merged.merge(tmp, on="date", how="outer")
        return merged.sort_values("date").reset_index(drop=True) if merged is not None else pd.DataFrame()

    def _clean_macro_release_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or "date" not in frame.columns:
            return frame
        source = frame.copy()
        source["date"] = pd.to_datetime(source["date"], errors="coerce").dt.normalize()
        source = source.dropna(subset=["date"]).sort_values("date")
        start = pd.Timestamp(self.cfg.data.start_date) - pd.DateOffset(months=3)
        end = pd.Timestamp(self.cfg.data.end_date)
        source = source[(source["date"] >= start) & (source["date"] <= end)]
        lag_map = {
            "cpi_yoy": ("monthly", int(getattr(self.cfg.data, "macro_cpi_release_lag_days", 10))),
            "cpi_mom": ("monthly", int(getattr(self.cfg.data, "macro_cpi_release_lag_days", 10))),
            "pmi_manufacturing": ("monthly", int(getattr(self.cfg.data, "macro_pmi_release_lag_days", 1))),
            "pmi_non_manufacturing": ("monthly", int(getattr(self.cfg.data, "macro_pmi_release_lag_days", 1))),
            "m2_yoy": ("monthly", int(getattr(self.cfg.data, "macro_m2_release_lag_days", 12))),
            "gdp_yoy": ("quarterly", int(getattr(self.cfg.data, "macro_gdp_release_lag_days", 20))),
            "lpr_1y": ("actual", 0),
            "lpr_5y": ("actual", 0),
        }
        records: list[pd.DataFrame] = []
        for column, (frequency, lag_days) in lag_map.items():
            if column not in source.columns:
                continue
            values = pd.to_numeric(source[column], errors="coerce")
            valid = source.loc[values.notna(), ["date"]].copy()
            if valid.empty:
                continue
            valid[column] = values.loc[valid.index].astype(float)
            if frequency == "monthly":
                valid["date"] = valid["date"] + pd.offsets.MonthEnd(0) + pd.to_timedelta(lag_days, unit="D")
            elif frequency == "quarterly":
                valid["date"] = valid["date"] + pd.to_timedelta(lag_days, unit="D")
            valid["date"] = valid["date"] + pd.offsets.BDay(0)
            records.append(valid)
        if not records:
            return pd.DataFrame()
        merged: pd.DataFrame | None = None
        for item in records:
            item = item.groupby("date", as_index=False).last()
            merged = item if merged is None else merged.merge(item, on="date", how="outer")
        assert merged is not None
        value_columns = [column for column in lag_map if column in merged.columns]
        merged = merged.dropna(subset=value_columns, how="all")
        merged = merged[(merged["date"] >= pd.Timestamp(self.cfg.data.start_date)) & (merged["date"] <= end)]
        return merged.sort_values("date").reset_index(drop=True)

    def fetch_industry(self) -> pd.DataFrame:
        primary = self._fetch_industry_branch(
            names=[self.cfg.data.industry_name],
            code=self.cfg.data.industry_code,
            prefix="industry_primary",
        )
        secondary = self._fetch_industry_branch(
            names=list(self.cfg.data.secondary_industry_names),
            code=self.cfg.data.secondary_industry_code,
            prefix="industry_secondary",
        )
        if primary.empty:
            return pd.DataFrame()
        if secondary.empty:
            secondary = self._empty_industry_branch("industry_secondary")
        pe_history = self._fetch_industry_pe_history()
        if not pe_history.empty:
            pe_history = pe_history.rename(
                columns={
                    "industry_pe_weighted": "industry_primary_pe_weighted",
                    "industry_pe_median": "industry_primary_pe_median",
                    "industry_pe_mean": "industry_primary_pe_mean",
                    "industry_pe_name": "industry_primary_pe_name",
                    "industry_pe_code": "industry_primary_pe_code",
                    "industry_pe_source": "industry_primary_pe_source",
                }
            )
        frames = [frame for frame in [primary, secondary, pe_history] if not frame.empty]
        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on="date", how="outer")
        pe_cols = [col for col in merged.columns if col.startswith("industry_primary_pe_")]
        if pe_cols:
            merged[pe_cols] = merged[pe_cols].ffill()
        return merged.sort_values("date").reset_index(drop=True)

    def _fetch_industry_branch(self, *, names: list[str], code: str, prefix: str) -> pd.DataFrame:
        cleaned_names = [str(name).strip() for name in names if str(name).strip()]
        last_error: Exception | None = None
        for name in cleaned_names:
            try:
                return self._fetch_named_industry_direct(name=name, prefix=prefix)
            except Exception as exc:
                last_error = exc
        if str(code or "").strip():
            try:
                return self._fetch_named_industry_proxy(code=str(code).strip(), prefix=prefix, name_hint=cleaned_names[0] if cleaned_names else "")
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            self.logger.warning("%s history fetch failed: %s", prefix, last_error)
        return pd.DataFrame()

    def _fetch_named_industry_direct(self, *, name: str, prefix: str) -> pd.DataFrame:
        start = self.cfg.data.start_date.replace("-", "")
        end = self.cfg.data.end_date.replace("-", "")
        df = retry_call(
            lambda: ak.stock_board_industry_index_ths(
                symbol=name,
                start_date=start,
                end_date=end,
            ),
            attempts=2,
        ).rename(
            columns={
                "\u65e5\u671f": "date",
                "\u5f00\u76d8\u4ef7": f"{prefix}_open",
                "\u6536\u76d8\u4ef7": f"{prefix}_close",
                "\u6700\u9ad8\u4ef7": f"{prefix}_high",
                "\u6700\u4f4e\u4ef7": f"{prefix}_low",
                "\u6210\u4ea4\u91cf": f"{prefix}_volume",
                "\u6210\u4ea4\u989d": f"{prefix}_amount",
            }
        )
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        numeric_columns = [
            f"{prefix}_open",
            f"{prefix}_close",
            f"{prefix}_high",
            f"{prefix}_low",
            f"{prefix}_volume",
            f"{prefix}_amount",
        ]
        for column in [col for col in numeric_columns if col in df.columns]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df[f"{prefix}_name"] = name
        df[f"{prefix}_source"] = "direct_board_history"
        return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    def _fetch_named_industry_proxy(self, *, code: str, prefix: str, name_hint: str) -> pd.DataFrame:
        start = self.cfg.data.start_date.replace("-", "")
        end = self.cfg.data.end_date.replace("-", "")
        components = retry_call(lambda: ak.stock_board_industry_cons_em(symbol=code), attempts=2)
        components = components.copy()
        components["\u6210\u4ea4\u989d"] = pd.to_numeric(components.get("\u6210\u4ea4\u989d"), errors="coerce")
        top_components = (
            components.sort_values("\u6210\u4ea4\u989d", ascending=False)
            .head(self.cfg.data.industry_proxy_top_n)["\u4ee3\u7801"]
            .astype(str)
            .tolist()
        )
        frames: list[pd.DataFrame] = []
        for component_code in top_components:
            history = self._fetch_single_stock_history(component_code, start, end)
            if history.empty:
                continue
            history["component"] = component_code
            frames.append(history)
        if not frames:
            return pd.DataFrame()
        merged = pd.concat(frames, ignore_index=True)
        grouped = (
            merged.groupby("date")
            .agg(
                **{
                    f"{prefix}_open": ("open", "mean"),
                    f"{prefix}_close": ("close", "mean"),
                    f"{prefix}_high": ("high", "mean"),
                    f"{prefix}_low": ("low", "mean"),
                    f"{prefix}_volume": ("volume", "sum"),
                    f"{prefix}_amount": ("amount", "sum"),
                    f"{prefix}_component_count": ("component", "nunique"),
                    f"{prefix}_up_ratio": (
                        "pct_change",
                        lambda values: float((pd.to_numeric(values, errors="coerce") > 0).mean()),
                    ),
                }
            )
            .reset_index()
        )
        grouped[f"{prefix}_name"] = name_hint or code
        grouped[f"{prefix}_source"] = "component_proxy_history"
        return grouped

    def _empty_industry_branch(self, prefix: str) -> pd.DataFrame:
        calendar = pd.DataFrame({"date": business_day_range(self.cfg.data.start_date, self.cfg.data.end_date)})
        for suffix in ["open", "close", "high", "low", "volume", "amount", "component_count", "up_ratio"]:
            calendar[f"{prefix}_{suffix}"] = np.nan
        calendar[f"{prefix}_name"] = ""
        calendar[f"{prefix}_source"] = "unavailable"
        return calendar

    def _fetch_industry_pe_history(self) -> pd.DataFrame:
        month_ends = pd.date_range(self.cfg.data.start_date, self.cfg.data.end_date, freq="ME")
        rows: list[dict[str, object]] = []
        selected_name = self.cfg.data.industry_pe_name
        selected_code: str | None = None
        source = f"cninfo_{self.cfg.data.industry_pe_classification}"

        for ts in month_ends:
            snapshot = self._fetch_industry_pe_snapshot(ts.strftime("%Y%m%d"))
            if snapshot.empty:
                continue
            if selected_code is not None:
                match = snapshot[snapshot["\u884c\u4e1a\u7f16\u7801"] == selected_code]
            else:
                match = snapshot[snapshot["\u884c\u4e1a\u540d\u79f0"] == selected_name]
                if match.empty and self.cfg.data.industry_pe_name:
                    match = snapshot[
                        snapshot["\u884c\u4e1a\u540d\u79f0"].astype(str).str.contains(
                            self.cfg.data.industry_pe_name, regex=False, na=False
                        )
                    ]
                if match.empty and self.cfg.data.industry_pe_classification == "\u8bc1\u76d1\u4f1a\u884c\u4e1a\u5206\u7c7b":
                    match = snapshot[snapshot["\u884c\u4e1a\u540d\u79f0"] == "\u7535\u6c14\u673a\u68b0\u548c\u5668\u6750\u5236\u9020\u4e1a"]
                if match.empty and self.cfg.data.industry_pe_classification == "\u56fd\u8bc1\u884c\u4e1a\u5206\u7c7b":
                    match = snapshot[snapshot["\u884c\u4e1a\u540d\u79f0"] == "\u7535\u6c14\u90e8\u4ef6\u4e0e\u8bbe\u5907"]
                if match.empty:
                    continue
                selected_code = str(match.iloc[0]["\u884c\u4e1a\u7f16\u7801"])
                selected_name = str(match.iloc[0]["\u884c\u4e1a\u540d\u79f0"])
            if match.empty:
                continue
            row = match.iloc[0]
            rows.append(
                {
                    "date": pd.to_datetime(ts),
                    "industry_pe_weighted": pd.to_numeric(row["\u9759\u6001\u5e02\u76c8\u7387-\u52a0\u6743\u5e73\u5747"], errors="coerce"),
                    "industry_pe_median": pd.to_numeric(row["\u9759\u6001\u5e02\u76c8\u7387-\u4e2d\u4f4d\u6570"], errors="coerce"),
                    "industry_pe_mean": pd.to_numeric(row["\u9759\u6001\u5e02\u76c8\u7387-\u7b97\u672f\u5e73\u5747"], errors="coerce"),
                    "industry_pe_name": selected_name,
                    "industry_pe_code": selected_code,
                    "industry_pe_source": source,
                }
            )

        if not rows:
            return pd.DataFrame()
        out = pd.DataFrame(rows)
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        return out.sort_values("date").reset_index(drop=True)

    def _fetch_industry_pe_snapshot(self, date: str) -> pd.DataFrame:
        from akshare.stock.stock_industry_pe_cninfo import _get_file_content_ths, py_mini_racer

        sort_code_map = {
            "\u8bc1\u76d1\u4f1a\u884c\u4e1a\u5206\u7c7b": "008001",
            "\u56fd\u8bc1\u884c\u4e1a\u5206\u7c7b": "008200",
        }
        js_code = py_mini_racer.MiniRacer()
        js_code.eval(_get_file_content_ths("cninfo.js"))
        enckey = js_code.call("getResCode1")
        response = requests.post(
            "http://webapi.cninfo.com.cn/api/sysapi/p_sysapi1087",
            params={
                "tdate": f"{date[:4]}-{date[4:6]}-{date[6:]}",
                "sortcode": sort_code_map[self.cfg.data.industry_pe_classification],
            },
            headers={
                "Accept": "*/*",
                "Accept-Enckey": enckey,
                "Origin": "http://webapi.cninfo.com.cn",
                "Referer": "http://webapi.cninfo.com.cn/",
                "User-Agent": "Mozilla/5.0",
                "X-Requested-With": "XMLHttpRequest",
            },
            timeout=20,
        )
        response.raise_for_status()
        records = response.json().get("records", [])
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records).rename(
            columns={
                "F004N": "\u884c\u4e1a\u5c42\u7ea7",
                "F013N": "\u9759\u6001\u5e02\u76c8\u7387-\u7b97\u672f\u5e73\u5747",
                "F012N": "\u9759\u6001\u5e02\u76c8\u7387-\u4e2d\u4f4d\u6570",
                "F011N": "\u9759\u6001\u5e02\u76c8\u7387-\u52a0\u6743\u5e73\u5747",
                "F010N": "\u51c0\u5229\u6da6-\u9759\u6001",
                "F006V": "\u884c\u4e1a\u540d\u79f0",
                "F005V": "\u884c\u4e1a\u7f16\u7801",
                "F003V": "\u884c\u4e1a\u5206\u7c7b",
                "F009N": "\u603b\u5e02\u503c-\u9759\u6001",
                "F008N": "\u7eb3\u5165\u8ba1\u7b97\u516c\u53f8\u6570\u91cf",
                "VARYDATE": "\u53d8\u52a8\u65e5\u671f",
                "F007N": "\u516c\u53f8\u6570\u91cf",
            }
        )
        for column in [col for col in df.columns if col not in {"\u884c\u4e1a\u540d\u79f0", "\u884c\u4e1a\u7f16\u7801", "\u884c\u4e1a\u5206\u7c7b", "\u53d8\u52a8\u65e5\u671f"}]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        return df

    def fetch_policy(self) -> pd.DataFrame:
        if not self.cfg.data.policy_enabled:
            return pd.DataFrame()
        frames: list[pd.DataFrame] = []
        direct = self._fetch_policy_official_direct()
        if not direct.empty:
            frames.append(
                self._enrich_frame_from_urls(
                    direct,
                    row_limit=self.cfg.data.text_extract_policy_rows,
                    prefer_pdf=False,
                    success_level="body",
                )
            )
        missing_years = self._detect_missing_years(
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
            min_rows=1,
        )
        if missing_years:
            official = self._filter_frame_years(self._fetch_policy_official_search(), missing_years)
            if not official.empty:
                official = self._enrich_frame_from_urls(
                    official,
                    row_limit=self.cfg.data.text_extract_policy_rows,
                    prefer_pdf=False,
                    success_level="body",
                )
                frames.append(official)
                missing_years = self._detect_missing_years(pd.concat(frames, ignore_index=True), min_rows=1)
        if missing_years and not self.cfg.data.strict_official_policy:
            for keyword in self.cfg.data.policy_keywords:
                if not keyword:
                    continue
                frame = self._filter_frame_years(
                    self._fetch_news_keyword(keyword, "policy_search_keyword", max_pages=self.cfg.data.policy_pages),
                    missing_years,
                )
                if frame.empty:
                    continue
                frame = self._filter_official_policy_rows(frame)
                if frame.empty:
                    continue
                frame["source"] = frame.get("source", pd.Series("policy_search_keyword", index=frame.index)).replace("", "policy_search_keyword")
                frame["content_level"] = frame.get("content_level", pd.Series("summary", index=frame.index))
                frame = self._enrich_frame_from_urls(
                    frame,
                    row_limit=self.cfg.data.text_extract_policy_rows,
                    prefer_pdf=False,
                    success_level="body",
                )
                frames.append(frame)
            fallback = self._fetch_policy_from_cached_text(missing_years)
            if not fallback.empty:
                frames.append(fallback)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        if self.cfg.data.strict_official_policy:
            out = self._filter_strict_official_policy_rows(out)
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
        out = out.dropna(subset=["date", "title"])
        out = out[(out["date"] >= pd.to_datetime(self.cfg.data.start_date)) & (out["date"] <= pd.to_datetime(self.cfg.data.end_date))]
        out = out.drop_duplicates(subset=["date", "title", "url"]).sort_values("date").reset_index(drop=True)
        return out

    def _fetch_policy_from_cached_text(self, years: set[int] | None = None) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for filename, source in [("news.csv", "policy_cached_news_proxy"), ("notices.csv", "policy_cached_notice_proxy")]:
            path = self.cache_root / filename
            if not path.exists():
                continue
            try:
                frame = pd.read_csv(path)
            except Exception:
                continue
            if frame.empty or "date" not in frame.columns:
                continue
            tmp = frame.copy()
            tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.normalize()
            tmp = tmp.dropna(subset=["date"])
            if years:
                tmp = tmp[tmp["date"].dt.year.isin(years)]
            if tmp.empty:
                continue
            blob = (
                tmp.get("title", pd.Series("", index=tmp.index)).fillna("").astype(str)
                + " "
                + tmp.get("content", pd.Series("", index=tmp.index)).fillna("").astype(str)
                + " "
                + tmp.get("category", pd.Series("", index=tmp.index)).fillna("").astype(str)
            )
            term_pattern = "|".join(re.escape(term) for term in self._policy_query_terms() + self._policy_title_terms() if term)
            tmp = tmp[blob.str.contains(term_pattern, regex=True, na=False)].copy()
            if tmp.empty:
                continue
            tmp["source"] = source
            tmp["publisher"] = tmp.get("publisher", pd.Series("", index=tmp.index)).replace("", np.nan).fillna(
                tmp.get("url", pd.Series("", index=tmp.index)).map(self._infer_publisher_from_url)
            )
            tmp["content_level"] = tmp.get("content_level", pd.Series("summary", index=tmp.index)).fillna("summary")
            frames.append(tmp)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        keep = ["date", "title", "content", "source", "publisher", "url", "content_level", "category"]
        for column in keep:
            if column not in out.columns:
                out[column] = ""
        return out[keep].drop_duplicates(subset=["date", "title", "url"]).sort_values("date").reset_index(drop=True)

    def fetch_news(self) -> pd.DataFrame:
        direct_frames = [
            self._fetch_news_gdelt(),
            self._fetch_news_google_rss(),
            self._fetch_recent_stock_news(),
            self._fetch_research_reports(),
            self._fetch_caixin_market_news(),
            self._fetch_guba_board_posts(),
        ]
        direct_frames = [frame for frame in direct_frames if not frame.empty]
        direct_news = pd.concat(direct_frames, ignore_index=True) if direct_frames else pd.DataFrame()
        missing_years = self._detect_missing_years(direct_news, min_rows=max(1, int(self.cfg.data.news_min_rows_per_year)))
        frames = list(direct_frames)
        if direct_news.empty or missing_years:
            frames.extend(
                frame
                for frame in [
                    self._fetch_news_eastmoney_search(years=missing_years or None),
                    self._fetch_news_sina_search(years=missing_years or None),
                ]
                if not frame.empty
            )
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
        out = out.dropna(subset=["date", "title"])
        out = out[(out["date"] >= pd.to_datetime(self.cfg.data.start_date)) & (out["date"] <= pd.to_datetime(self.cfg.data.end_date))]
        out = out.drop_duplicates(subset=["date", "title", "source", "url"]).sort_values("date").reset_index(drop=True)
        out = clean_news_corpus(self.cfg, out, enforce_daily_cap=False, logger=self.logger, stage="raw_fetch")
        enriched = self._enrich_news_text(out)
        return clean_news_corpus(self.cfg, enriched, enforce_daily_cap=True, logger=self.logger, stage="enriched_fetch")

    def _build_search_queries(self, keyword: str, years: set[int] | None = None) -> list[str]:
        queries = [str(keyword).strip()]
        if not self.cfg.data.news_yearly_expansion:
            return [query for query in queries if query]
        for year in self._year_set(years):
            queries.append(f"{keyword} {year}")
            if str(keyword) == self.cfg.data.company_name:
                queries.append(f"{keyword} {year} \u5e74\u62a5")
                queries.append(f"{keyword} {year} \u4e1a\u7ee9")
            elif str(keyword) == self.cfg.data.industry_name:
                queries.append(f"{keyword} {year} \u653f\u7b56")
                queries.append(f"{keyword} {year} \u884c\u4e1a")
        deduped: list[str] = []
        seen: set[str] = set()
        for query in queries:
            clean = self._compact_text(query)
            if not clean or clean in seen:
                continue
            seen.add(clean)
            deduped.append(clean)
        return deduped

    def _enrich_news_text(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        out["content"] = out.get("content", pd.Series("", index=out.index)).fillna("").astype(str)
        out["content_level"] = out.get("content_level", pd.Series("summary", index=out.index)).fillna("summary").astype(str)
        priority = (
            out.get("title", pd.Series("", index=out.index)).astype(str)
            + " "
            + out.get("publisher", pd.Series("", index=out.index)).astype(str)
            + " "
            + out.get("source", pd.Series("", index=out.index)).astype(str)
        )
        high_value = priority.str.contains(
            "\u5b81\u5fb7\u65f6\u4ee3|\u52a8\u529b\u7535\u6c60|\u50a8\u80fd|\u7814\u62a5|\u516c\u544a|\u8d22\u62a5|\u4e1a\u7ee9|\u653f\u7b56|cninfo|report|research",
            regex=True,
            na=False,
        )
        out = out.assign(_priority=high_value.astype(int)).sort_values(["_priority", "date"], ascending=[False, False])
        out = self._enrich_frame_from_urls(
            out.drop(columns=["_priority"]),
            row_limit=self.cfg.data.text_extract_news_rows,
            prefer_pdf=False,
            success_level="body",
        )
        return out.sort_values("date").reset_index(drop=True)

    def _is_official_policy_url(self, url: object) -> bool:
        target = str(url or "").strip().lower()
        if not target:
            return False
        netloc = urlparse(target).netloc
        return any(domain.lower() in netloc or domain.lower() in target for domain in self.cfg.data.policy_official_domains)

    def _fetch_news_gdelt(self) -> pd.DataFrame:
        if not self.cfg.data.gdelt_enabled:
            return pd.DataFrame()
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        query_specs = [
            (self.cfg.data.gdelt_company_query, "gdelt_search_company"),
            (self.cfg.data.gdelt_industry_query, "gdelt_search_industry"),
        ]
        frames: list[pd.DataFrame] = []
        for query, source in query_specs:
            if not query:
                continue
            try:
                frame = self._fetch_news_gdelt_query(query, source)
            except Exception as exc:
                self._gdelt_unavailable = True
                self.logger.warning("GDELT historical news fetch failed for %s: %s", source, exc)
                continue
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _fetch_news_gdelt_query(self, query: str, source: str) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        start_date = pd.to_datetime(self.cfg.data.start_date)
        end_date = pd.to_datetime(self.cfg.data.end_date)
        max_records = max(1, int(self.cfg.data.gdelt_max_records_per_window))
        sleep_seconds = max(0.0, float(self.cfg.data.gdelt_sleep_seconds))
        windows = self._quarter_windows(start_date, end_date)

        for index, (window_start, window_end) in enumerate(windows):
            payload = self._fetch_gdelt_news_window(query, window_start, window_end, max_records=max_records)
            if payload is None:
                break
            for item in payload.get("articles", []):
                dt = pd.to_datetime(item.get("seendate"), format="%Y%m%dT%H%M%SZ", errors="coerce")
                if pd.isna(dt) or dt < start_date or dt > end_date:
                    continue
                title = self._strip_html(item.get("title"))
                if not title:
                    continue
                rows.append(
                    {
                        "date": dt.normalize(),
                        "title": title,
                        "content": self._build_gdelt_content(item),
                        "source": source,
                        "publisher": item.get("domain"),
                        "url": item.get("url"),
                        "content_level": "metadata",
                    }
                )
            if index < len(windows) - 1 and sleep_seconds > 0:
                time.sleep(sleep_seconds)

        return pd.DataFrame(rows)

    def _fetch_gdelt_news_window(
        self,
        query: str,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
        max_records: int,
    ) -> dict[str, object] | None:
        response = self._gdelt_session.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params={
                "query": query,
                "mode": "artlist",
                "maxrecords": str(max_records),
                "format": "json",
                "startdatetime": self._to_gdelt_datetime(window_start),
                "enddatetime": self._to_gdelt_datetime(window_end),
            },
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=40,
            verify=False,
        )
        if response.status_code == 429:
            self._gdelt_rate_limited = True
            self.logger.warning("GDELT historical news search rate-limited; keeping partial results.")
            return None
        response.raise_for_status()
        text = response.text.strip()
        if not text:
            return {}
        if text.startswith("Please limit requests"):
            self._gdelt_rate_limited = True
            self.logger.warning("GDELT historical news search rate-limited by response body; keeping partial results.")
            return None
        if not text.startswith("{"):
            self._gdelt_unavailable = True
            self.logger.warning("Unexpected GDELT historical news response: %s", text[:120])
            return {}
        return response.json()

    def _fetch_sina_news_page(
        self,
        base_params: dict[str, object],
        page: int,
        page_size: int,
        headers: dict[str, str],
    ) -> dict[str, object]:
        try:
            response = requests.get(
                "http://search.sina.com.cn/api/news",
                params={**base_params, "page": page, "size": page_size},
                headers=headers,
                timeout=20,
            )
            response.raise_for_status()
            time.sleep(0.2)
            return response.json()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                self._sina_rate_limited = True
                self.logger.warning("Sina historical news search rate-limited for page %s; keeping partial results.", page)
                return {}
            raise
        except requests.RequestException as exc:
            self.logger.warning("Sina historical news search request failed for page %s: %s", page, exc)
            return {}

    def _fetch_news_keyword(self, keyword: str, source: str, max_pages: int | None = None) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        end_date = pd.to_datetime(self.cfg.data.end_date)
        start_date = pd.to_datetime(self.cfg.data.start_date)
        max_pages = max(1, int(self.cfg.data.news_pages if max_pages is None else max_pages))
        page_size = 20

        for page in range(1, max_pages + 1):
            payload = {
                "uid": "",
                "keyword": keyword,
                "type": ["cmsArticleWebOld"],
                "client": "web",
                "clientType": "web",
                "clientVersion": "curr",
                "param": {
                    "cmsArticleWebOld": {
                        "pageSize": page_size,
                        "pageIndex": page,
                        "preTag": "<em>",
                        "postTag": "</em>",
                        "sort": "time",
                    }
                },
            }
            response = requests.get(
                "https://search-api-web.eastmoney.com/search/jsonp",
                params={"cb": "jQuery", "param": json.dumps(payload, ensure_ascii=False)},
                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://so.eastmoney.com/"},
                timeout=20,
            )
            response.raise_for_status()
            data = self._parse_jsonp_payload(response.text)
            article_list = data.get("result", {}).get("cmsArticleWebOld", []) if isinstance(data, dict) else []
            if not article_list:
                break

            page_dates: list[pd.Timestamp] = []
            for item in article_list:
                dt = pd.to_datetime(item.get("date"), errors="coerce")
                if pd.isna(dt):
                    continue
                page_dates.append(dt)
                if dt < start_date or dt > end_date:
                    continue
                rows.append(
                    {
                        "date": dt.normalize(),
                        "title": self._strip_html(item.get("title")),
                        "content": self._strip_html(item.get("content")),
                        "source": source,
                        "publisher": item.get("mediaName"),
                        "url": item.get("url"),
                        "content_level": "summary",
                    }
                )
            if page_dates and min(page_dates) < start_date:
                break

        return pd.DataFrame(rows)

    def _parse_sina_news_items(
        self,
        items: list[dict[str, object]],
        source: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for item in items:
            dt = pd.to_datetime(item.get("dataTime"), errors="coerce")
            if pd.isna(dt):
                dt = pd.to_datetime(item.get("ctime"), unit="s", errors="coerce")
            if pd.isna(dt) or dt < start_date or dt > end_date:
                continue
            rows.append(
                {
                    "date": dt.normalize(),
                    "title": self._strip_html(item.get("title")),
                    "content": self._strip_html(item.get("searchSummary") or item.get("intro")),
                    "source": source,
                    "publisher": self._strip_html(item.get("media_show") or item.get("author")),
                    "url": item.get("url"),
                    "content_level": "summary",
                }
            )
        return rows

    def _build_gdelt_content(self, item: dict[str, object]) -> str:
        title = self._strip_html(item.get("title"))
        domain = self._strip_html(item.get("domain"))
        language = self._strip_html(item.get("language"))
        country = self._strip_html(item.get("sourcecountry"))
        return f"title: {title} domain: {domain} language: {language} source_country: {country}".strip()

    def _fetch_recent_stock_news(self) -> pd.DataFrame:
        try:
            df = retry_call(lambda: ak.stock_news_em(symbol=self.cfg.data.symbol), attempts=2).rename(
                columns={
                    "\u65b0\u95fb\u6807\u9898": "title",
                    "\u65b0\u95fb\u5185\u5bb9": "content",
                    "\u53d1\u5e03\u65f6\u95f4": "date",
                    "\u6587\u7ae0\u6765\u6e90": "publisher",
                    "\u65b0\u95fb\u94fe\u63a5": "url",
                    "\u65b0\u95fb\u94fe\u63a5 ": "url",
                    "\u6765\u6e90": "publisher",
                }
            )
        except Exception:
            return pd.DataFrame()
        out = df[[col for col in ["date", "title", "content", "publisher", "url"] if col in df.columns]].copy()
        out["source"] = "eastmoney_stock_news"
        out["content_level"] = "body"
        return out

    def _fetch_caixin_market_news(self) -> pd.DataFrame:
        try:
            df = retry_call(ak.stock_news_main_cx, attempts=2)
        except Exception:
            return pd.DataFrame()
        if df.empty:
            return df
        out = df.rename(columns={"tag": "publisher", "summary": "content", "url": "url"}).copy()
        if "content" not in out.columns:
            return pd.DataFrame()
        out["title"] = out["content"].astype(str).str.slice(0, 60)
        out["date"] = out.get("url", pd.Series("", index=out.index)).map(self._extract_date_from_text)
        out["publisher"] = out.get("publisher", pd.Series("\u8d22\u65b0\u6570\u636e\u901a", index=out.index)).fillna("\u8d22\u65b0\u6570\u636e\u901a")
        keywords = [self.cfg.data.company_name, self.cfg.data.symbol, self.cfg.data.industry_name, "\u52a8\u529b\u7535\u6c60", "\u50a8\u80fd", "\u65b0\u80fd\u6e90"]
        text_blob = (out["title"].fillna("") + " " + out["content"].fillna("")).str.strip()
        mask = pd.Series(False, index=out.index)
        for keyword in keywords:
            if keyword:
                mask = mask | text_blob.str.contains(str(keyword), regex=False, na=False)
        out = out.loc[mask].copy()
        if out.empty:
            return out
        out["source"] = "caixin_market_news"
        out["content_level"] = "summary"
        out = out.dropna(subset=["date", "title"])
        return out[[col for col in ["date", "title", "content", "source", "publisher", "url", "content_level"] if col in out.columns]]

    def _fetch_guba_board_posts(self) -> pd.DataFrame:
        pages = max(1, int(self.cfg.data.guba_pages))
        start_date = pd.to_datetime(self.cfg.data.start_date)
        end_date = pd.to_datetime(self.cfg.data.end_date)
        rows: list[dict[str, object]] = []
        seen_urls: set[str] = set()
        for page in range(1, pages + 1):
            suffix = f"list,{self.cfg.data.symbol}.html" if page == 1 else f"list,{self.cfg.data.symbol}_{page}.html"
            page_dates: list[pd.Timestamp] = []
            try:
                response = requests.get(
                    f"https://guba-insight.eastmoney.com/{suffix}",
                    headers={"User-Agent": "Mozilla/5.0", "Referer": "https://guba.eastmoney.com/"},
                    timeout=20,
                )
                response.raise_for_status()
            except Exception:
                continue
            if BeautifulSoup is None:
                continue
            soup = BeautifulSoup(response.text, "lxml")
            for row in soup.select("tbody tr"):
                cells = row.select("td")
                title_link = row.select_one("td:nth-of-type(3) a[href]")
                author_link = row.select_one("td:nth-of-type(4) a")
                if len(cells) < 5 or title_link is None:
                    continue
                title = self._compact_text(title_link.get_text(" ", strip=True))
                if not title:
                    continue
                url = title_link.get("href") or ""
                if url.startswith("/"):
                    url = f"https://guba.eastmoney.com{url}"
                elif url.startswith("//"):
                    url = f"https:{url}"
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                dt = self._parse_guba_time(cells[-1].get_text(" ", strip=True))
                if pd.isna(dt):
                    dt = self._extract_date_candidates(url, title)
                if pd.notna(dt):
                    page_dates.append(pd.Timestamp(dt))
                if pd.isna(dt) or dt < start_date or dt > end_date:
                    continue
                rows.append(
                    {
                        "date": dt,
                        "title": title,
                        "content": title,
                        "source": "eastmoney_guba_board",
                        "publisher": self._compact_text(author_link.get_text(" ", strip=True) if author_link else ""),
                        "url": url,
                        "click_count": pd.to_numeric(cells[0].get_text(" ", strip=True), errors="coerce"),
                        "comment_count": pd.to_numeric(cells[1].get_text(" ", strip=True), errors="coerce"),
                        "content_level": "summary",
                    }
                )
            if page_dates and min(page_dates) < start_date:
                break
        if not rows:
            return pd.DataFrame()
        out = pd.DataFrame(rows)
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
        out = out.dropna(subset=["date", "title"])
        out = out[(out["date"] >= start_date) & (out["date"] <= end_date)]
        return out.sort_values("date").reset_index(drop=True)

    def _filter_official_policy_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        official_domain_pattern = "|".join(re.escape(domain) for domain in self.cfg.data.policy_official_domains if domain)
        publisher_pattern = "|".join(re.escape(token) for token in self._official_policy_publishers())
        title_pattern = "|".join(re.escape(token) for token in self._policy_title_terms())
        official_mask = (
            out.get("publisher", pd.Series("", index=out.index)).astype(str).str.contains(
                publisher_pattern,
                regex=True,
                na=False,
            )
            | out.get("url", pd.Series("", index=out.index)).astype(str).str.contains(
                official_domain_pattern,
                regex=True,
                na=False,
            )
            | out.get("title", pd.Series("", index=out.index)).astype(str).str.contains(
                title_pattern,
                regex=True,
                na=False,
            )
        )
        out = out.loc[official_mask].copy()
        if out.empty:
            return out
        out = out.loc[
            out.apply(lambda row: self._looks_like_policy_row(row.get("title"), row.get("url"), row.get("content")), axis=1)
        ].copy()
        if out.empty:
            return out
        out["source"] = np.where(
            out.get("url", pd.Series("", index=out.index)).astype(str).map(self._is_official_policy_url),
            "policy_official_fulltext",
            out.get("source", pd.Series("policy_search_keyword", index=out.index)),
        )
        inferred_publisher = out.get("url", pd.Series("", index=out.index)).map(self._infer_publisher_from_url)
        out["publisher"] = (
            out.get("publisher", pd.Series("", index=out.index))
            .replace("", np.nan)
            .fillna(inferred_publisher)
            .fillna("\u8d22\u65b0\u6570\u636e\u901a")
        )
        return out

    def _filter_strict_official_policy_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        out = out.loc[
            out.get("url", pd.Series("", index=out.index)).astype(str).map(self._is_official_policy_url)
        ].copy()
        if out.empty:
            return out
        out = out.loc[
            out.apply(lambda row: self._looks_like_policy_row(row.get("title"), row.get("url"), row.get("content")), axis=1)
        ].copy()
        if out.empty:
            return out
        out["source"] = "policy_official_fulltext"
        inferred_publisher = out.get("url", pd.Series("", index=out.index)).map(self._infer_publisher_from_url)
        out["publisher"] = (
            out.get("publisher", pd.Series("", index=out.index))
            .replace("", np.nan)
            .fillna(inferred_publisher)
            .fillna("official_policy_site")
        )
        return out

    def _fetch_policy_official_direct(self) -> pd.DataFrame:
        frames = [
            self._fetch_policy_gov_search_api(),
            self._fetch_policy_archive_seeds(),
        ]
        frames = [self._filter_official_policy_rows(frame) for frame in frames if not frame.empty]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        if "content_level" not in out.columns:
            out["content_level"] = "summary"
        out = out.drop_duplicates(subset=["date", "title", "url"]).sort_values("date").reset_index(drop=True)
        return out

    def _fetch_policy_gov_search_api(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        max_pages = max(1, int(self.cfg.data.policy_pages))
        start_date = pd.to_datetime(self.cfg.data.start_date)
        end_date = pd.to_datetime(self.cfg.data.end_date)
        for query in self._policy_query_terms():
            for page in range(max_pages):
                try:
                    response = requests.get(
                        "https://sousuo.www.gov.cn/search-gov/data",
                        params={
                            "t": "zhengce",
                            "timetype": "timeqb",
                            "sort": "pubtime",
                            "sortType": "1",
                            "searchfield": "title",
                            "q": query,
                            "p": page,
                            "n": 20,
                            "pubmintime": self.cfg.data.start_date,
                            "pubmaxtime": self.cfg.data.end_date,
                        },
                        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.gov.cn/"},
                        timeout=30,
                        verify=False,
                    )
                    response.raise_for_status()
                    payload_text = response.text.strip()
                    payload = response.json() if payload_text.startswith("{") else self._parse_jsonp_payload(payload_text)
                except Exception:
                    break
                if not isinstance(payload, dict):
                    break
                items = payload.get("searchVO") or payload.get("searchVOList") or payload.get("results") or payload.get("items") or []
                if isinstance(items, dict):
                    items = items.get("list") or items.get("items") or items.get("docs") or []
                if not items:
                    break
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    title = self._strip_html(item.get("title") or item.get("titleText") or item.get("docTitle"))
                    url = self._compact_text(item.get("url") or item.get("link") or item.get("docUrl"))
                    summary = self._compact_text(
                        self._strip_html(item.get("content") or item.get("summary") or item.get("brief") or item.get("description"))
                    )
                    dt = pd.to_datetime(
                        item.get("pubtime") or item.get("pubTime") or item.get("publishDate") or item.get("date"),
                        errors="coerce",
                    )
                    if pd.isna(dt):
                        dt = self._extract_date_candidates(url, summary, title)
                    if pd.isna(dt) or dt < start_date or dt > end_date or not title or not url:
                        continue
                    rows.append(
                        {
                            "date": dt.normalize(),
                            "title": title,
                            "content": summary or title,
                            "source": "policy_gov_search_api",
                            "publisher": self._infer_publisher_from_url(url),
                            "url": url,
                            "content_level": "summary",
                        }
                    )
        return pd.DataFrame(rows)

    def _fetch_policy_archive_seeds(self) -> pd.DataFrame:
        if BeautifulSoup is None:
            return pd.DataFrame()
        rows: list[dict[str, object]] = []
        max_pages = max(1, int(self.cfg.data.policy_archive_pages))
        for seed in self.cfg.data.policy_official_seed_urls:
            if isinstance(seed, dict):
                seed_url = str(seed.get("url") or "").strip()
                source = str(seed.get("source") or "policy_official_archive")
            else:
                seed_url = str(seed).strip()
                source = "policy_official_archive"
            if not seed_url:
                continue
            for page_url in self._build_archive_page_urls(seed_url, max_pages):
                try:
                    response = requests.get(
                        page_url,
                        headers={"User-Agent": "Mozilla/5.0", "Referer": seed_url},
                        timeout=30,
                        verify=False,
                    )
                    response.raise_for_status()
                except Exception:
                    continue
                rows.extend(self._parse_policy_archive_page(self._response_text(response), page_url, source))
        return pd.DataFrame(rows)

    def _parse_policy_archive_page(self, html: str, base_url: str, source: str) -> list[dict[str, object]]:
        if BeautifulSoup is None or not html:
            return []
        soup = BeautifulSoup(html, "lxml")
        rows: list[dict[str, object]] = []
        seen: set[tuple[str, str]] = set()
        for anchor in soup.select("a[href]"):
            title = self._compact_text(anchor.get_text(" ", strip=True))
            if len(title) < 8:
                continue
            url = self._compact_text(urljoin(base_url, anchor.get("href") or ""))
            if not url or not self._is_official_policy_url(url):
                continue
            surrounding = self._compact_text(anchor.parent.get_text(" ", strip=True) if anchor.parent else "")
            if not self._looks_like_policy_row(title, url, surrounding):
                continue
            date_value = self._extract_date_candidates(surrounding, title, url)
            key = (title, url)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "date": pd.to_datetime(date_value, errors="coerce").normalize() if pd.notna(date_value) else pd.NaT,
                    "title": title,
                    "content": surrounding or title,
                    "source": source,
                    "publisher": self._infer_publisher_from_url(url),
                    "url": url,
                    "content_level": "summary",
                }
            )
        return rows

    def _fetch_policy_official_search(self) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        for query in list(self.cfg.data.policy_official_queries):
            frame = self._fetch_google_news_rss_query(
                query=query,
                source="policy_official_rss_search",
                publisher_hint=self._publisher_hint_tokens(),
            )
            if not frame.empty:
                rows.append(frame)
        if not rows:
            return pd.DataFrame()
        out = pd.concat(rows, ignore_index=True)
        out = self._filter_official_policy_rows(out)
        if out.empty:
            return out
        out["content_level"] = out.get("content_level", pd.Series("summary", index=out.index)).fillna("summary")
        return out.drop_duplicates(subset=["date", "title", "publisher"]).sort_values("date").reset_index(drop=True)

    def _fetch_news_eastmoney_search(self, years: set[int] | None = None) -> pd.DataFrame:
        keyword_specs = [
            (self.cfg.data.company_name, "eastmoney_search_company_name"),
            (self.cfg.data.symbol, "eastmoney_search_company_code"),
            (self.cfg.data.industry_name, "eastmoney_search_industry"),
            ("\u65b0\u80fd\u6e90\u7535\u6c60", "eastmoney_search_industry_theme"),
        ]
        frames: list[pd.DataFrame] = []
        for keyword, source in keyword_specs:
            if not keyword:
                continue
            for query in self._build_search_queries(keyword, years=years):
                frame = self._fetch_news_keyword(query, source)
                if frame.empty:
                    continue
                frame["search_query"] = query
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _fetch_news_sina_search(self, years: set[int] | None = None) -> pd.DataFrame:
        keyword_specs = [
            (self.cfg.data.symbol, "sina_search_company_code"),
            (self.cfg.data.company_name, "sina_search_company_name"),
        ]
        frames: list[pd.DataFrame] = []
        for keyword, source in keyword_specs:
            if not keyword:
                continue
            try:
                frame = self._fetch_news_sina_keyword(keyword, source, years=years)
            except Exception as exc:
                self.logger.warning("Sina historical news fetch failed for %s: %s", source, exc)
                continue
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _fetch_news_sina_keyword(self, keyword: str, source: str, years: set[int] | None = None) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        start_date = pd.to_datetime(self.cfg.data.start_date)
        end_date = pd.to_datetime(self.cfg.data.end_date)
        max_pages = min(max(1, int(self.cfg.data.news_pages)), 10)
        page_size = 10
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://search.sina.com.cn/search?tp=news",
        }
        for window_start, window_end in self._year_windows(start_date, end_date):
            if years and int(window_start.year) not in years:
                continue
            base_params = {
                "q": keyword,
                "sort": 1,
                "from": "advanced_search",
                "stime": int(window_start.timestamp()),
                "etime": int(window_end.timestamp()),
            }
            payload = self._fetch_sina_news_page(base_params, page=1, page_size=page_size, headers=headers)
            if not payload:
                continue
            data = payload.get("data") or {}
            items = data.get("list") or []
            total = int(data.get("total") or 0)
            total_pages = min(max_pages, (total + page_size - 1) // page_size) if total else 0
            rows.extend(self._parse_sina_news_items(items, source, start_date, end_date))
            for page in range(2, total_pages + 1):
                page_payload = self._fetch_sina_news_page(base_params, page=page, page_size=page_size, headers=headers)
                if not page_payload:
                    break
                page_items = (page_payload.get("data") or {}).get("list") or []
                if not page_items:
                    break
                rows.extend(self._parse_sina_news_items(page_items, source, start_date, end_date))
        return pd.DataFrame(rows)

    def _fetch_news_google_rss(self) -> pd.DataFrame:
        if not self.cfg.data.google_news_enabled:
            return pd.DataFrame()
        queries = [
            (self.cfg.data.company_name, "google_news_rss_company"),
            (self.cfg.data.symbol, "google_news_rss_code"),
            (self.cfg.data.industry_name, "google_news_rss_industry"),
            ("\u52a8\u529b\u7535\u6c60", "google_news_rss_battery"),
            ("\u50a8\u80fd \u7535\u6c60", "google_news_rss_storage"),
        ]
        frames: list[pd.DataFrame] = []
        for query, source in queries:
            if not query:
                continue
            frame = self._fetch_google_news_rss_query(query=query, source=source)
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _fetch_google_news_rss_query(
        self,
        *,
        query: str,
        source: str,
        publisher_hint: set[str] | None = None,
    ) -> pd.DataFrame:
        start_date = pd.to_datetime(self.cfg.data.start_date)
        end_date = pd.to_datetime(self.cfg.data.end_date)
        window_days = max(7, int(self.cfg.data.google_news_window_days))
        rows: list[dict[str, object]] = []
        current = start_date.normalize()
        seen: set[tuple[str, str, str]] = set()
        while current <= end_date:
            window_end = min(current + pd.Timedelta(days=window_days - 1), end_date)
            rss_query = f'{query} after:{current.date()} before:{(window_end + pd.Timedelta(days=1)).date()}'
            url = "https://news.google.com/rss/search"
            try:
                response = requests.get(
                    url,
                    params={"q": rss_query, "hl": "zh-CN", "gl": "CN", "ceid": "CN:zh-Hans"},
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=30,
                )
                response.raise_for_status()
                root = ET.fromstring(response.content)
            except Exception:
                current = window_end + pd.Timedelta(days=1)
                continue

            for item in root.findall(".//item"):
                title = self._strip_google_title(item.findtext("title"))
                publisher = self._compact_text(item.findtext("source"))
                dt = pd.to_datetime(item.findtext("pubDate"), errors="coerce")
                description = self._compact_text(self._strip_html(unescape(item.findtext("description") or "")))
                link = self._compact_text(item.findtext("link"))
                if not title or pd.isna(dt):
                    continue
                if publisher_hint and not any(token in publisher for token in publisher_hint):
                    continue
                key = (dt.normalize().strftime("%Y-%m-%d"), title, publisher)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "date": dt.normalize(),
                        "title": title,
                        "content": description or title,
                        "source": source,
                        "publisher": publisher,
                        "url": link,
                        "content_level": "summary",
                    }
                )
            current = window_end + pd.Timedelta(days=1)
        return pd.DataFrame(rows)

    def _fetch_research_reports(self) -> pd.DataFrame:
        try:
            df = retry_call(lambda: ak.stock_research_report_em(symbol=self.cfg.data.symbol), attempts=2).rename(
                columns={
                    "\u65e5\u671f": "date",
                    "\u62a5\u544a\u540d\u79f0": "title",
                    "\u673a\u6784": "publisher",
                    "\u884c\u4e1a": "industry",
                    "\u4e1c\u8d22\u8bc4\u7ea7": "rating",
                    "\u62a5\u544aPDF\u94fe\u63a5": "url",
                }
            )
        except Exception:
            return pd.DataFrame()
        out = df[[col for col in ["date", "title", "publisher", "industry", "rating", "url"] if col in df.columns]].copy()
        out["content"] = (
            "report_title: "
            + out["title"].fillna("")
            + " institution: "
            + out.get("publisher", pd.Series("", index=out.index)).fillna("")
            + " rating: "
            + out.get("rating", pd.Series("", index=out.index)).fillna("")
            + " industry: "
            + out.get("industry", pd.Series("", index=out.index)).fillna("")
        ).str.strip()
        out["source"] = "eastmoney_research_report"
        out["content_level"] = "metadata"
        out = self._enrich_frame_from_urls(
            out,
            row_limit=self.cfg.data.text_extract_report_rows,
            prefer_pdf=True,
            success_level="pdf_excerpt",
        )
        return out[[col for col in ["date", "title", "content", "source", "publisher", "url", "content_level"] if col in out.columns]]

    def _fetch_single_stock_history(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        market_symbol = ("sz" if symbol.startswith(("0", "3")) else "sh") + symbol
        try:
            df = retry_call(
                lambda: ak.stock_zh_a_hist_tx(symbol=market_symbol, start_date=start, end_date=end, adjust=""),
                attempts=2,
            )
        except Exception:
            return pd.DataFrame()
        out = df.rename(
            columns={
                "date": "date",
                "open": "open",
                "close": "close",
                "high": "high",
                "low": "low",
                "amount": "volume",
            }
        )
        out["amount"] = pd.NA
        out["pct_change"] = pd.to_numeric(out["close"], errors="coerce").pct_change()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        for column in [col for col in out.columns if col != "date"]:
            out[column] = pd.to_numeric(out[column], errors="coerce")
        return out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    def fetch_notices(self) -> pd.DataFrame:
        try:
            cninfo = self._fetch_notices_cninfo()
            if not cninfo.empty:
                return self._enrich_notice_text(cninfo)
        except Exception as exc:
            self.logger.warning("CNInfo official notice fetch failed, switching to Eastmoney fallback: %s", exc)
        return self._enrich_notice_text(self._fetch_notices_eastmoney())

    def _fetch_notices_cninfo(self) -> pd.DataFrame:
        page_size = self.cfg.data.notice_page_size
        params = {
            "searchkey": self.cfg.data.symbol,
            "sdate": self.cfg.data.start_date,
            "edate": self.cfg.data.end_date,
            "isfulltext": "true",
            "sortName": "pubdate",
            "sortType": "desc",
            "pageNum": "1",
            "pageSize": str(page_size),
            "type": "",
        }
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": f"https://www.cninfo.com.cn/new/fulltextSearch?notautosubmit=&keyWord={self.cfg.data.symbol}",
        }
        response = requests.get("https://www.cninfo.com.cn/new/fulltextSearch/full", params=params, headers=headers, timeout=20)
        response.raise_for_status()
        payload = response.json()
        total = int(payload.get("totalAnnouncement", 0))
        if total == 0:
            return pd.DataFrame()
        total_pages = (total + page_size - 1) // page_size
        rows: list[dict[str, object]] = []

        for page in range(1, total_pages + 1):
            params["pageNum"] = str(page)
            page_response = requests.get(
                "https://www.cninfo.com.cn/new/fulltextSearch/full",
                params=params,
                headers=headers,
                timeout=20,
            )
            page_response.raise_for_status()
            announcements = page_response.json().get("announcements", [])
            if not announcements:
                continue
            for item in announcements:
                title = self._strip_html(item.get("announcementTitle"))
                if not title:
                    continue
                rows.append(
                    {
                        "date": pd.to_datetime(item.get("announcementTime"), unit="ms", errors="coerce"),
                        "title": title,
                        "category": item.get("announcementType") or item.get("pageColumn"),
                        "source": "cninfo_fulltext_search",
                        "url": f"https://static.cninfo.com.cn/{item.get('adjunctUrl')}" if item.get("adjunctUrl") else None,
                        "content": f"notice_title: {title} category: {item.get('announcementType') or item.get('pageColumn') or ''}".strip(),
                        "content_level": "metadata",
                    }
                )

        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
        out = out.dropna(subset=["date", "title"]).drop_duplicates(subset=["date", "title", "url"])
        out = out[(out["date"] >= pd.to_datetime(self.cfg.data.start_date)) & (out["date"] <= pd.to_datetime(self.cfg.data.end_date))]
        return out.sort_values("date").reset_index(drop=True)

    def _fetch_notices_eastmoney(self) -> pd.DataFrame:
        url = "https://np-anotice-stock.eastmoney.com/api/security/ann"
        page_size = self.cfg.data.notice_page_size
        params = {
            "sr": "-1",
            "page_size": str(page_size),
            "page_index": "1",
            "ann_type": "A",
            "client_source": "web",
            "f_node": "0",
            "s_node": "0",
            "begin_time": self.cfg.data.start_date,
            "end_time": self.cfg.data.end_date,
            "stock_list": self.cfg.data.symbol,
        }
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        total_hits = int(response.json().get("data", {}).get("total_hits", 0))
        if total_hits == 0:
            return pd.DataFrame()
        total_pages = (total_hits + page_size - 1) // page_size
        rows: list[dict[str, object]] = []

        for page in range(1, total_pages + 1):
            params["page_index"] = str(page)
            page_response = requests.get(url, params=params, timeout=20)
            page_response.raise_for_status()
            notice_list = page_response.json().get("data", {}).get("list", [])
            if not notice_list:
                continue
            for item in notice_list:
                codes = item.get("codes", [])
                target_code = next((code for code in codes if code.get("stock_code") == self.cfg.data.symbol), codes[0] if codes else {})
                columns = item.get("columns", [])
                category = ",".join(
                    str(column.get("column_name", "")).strip() for column in columns if str(column.get("column_name", "")).strip()
                )
                stock_code = str(target_code.get("stock_code", self.cfg.data.symbol))
                art_code = str(item.get("art_code", ""))
                rows.append(
                    {
                        "date": item.get("notice_date"),
                        "title": item.get("title"),
                        "category": category,
                        "source": "eastmoney_notice_api",
                        "url": f"https://data.eastmoney.com/notices/detail/{stock_code}/{art_code}.html" if art_code else None,
                        "content": f"notice_title: {item.get('title') or ''} category: {category}".strip(),
                        "content_level": "metadata",
                    }
                )

        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
        out = out.dropna(subset=["date", "title"]).drop_duplicates(subset=["date", "title", "url"])
        out = out[(out["date"] >= pd.to_datetime(self.cfg.data.start_date)) & (out["date"] <= pd.to_datetime(self.cfg.data.end_date))]
        return out.sort_values("date").reset_index(drop=True)

    def _enrich_notice_text(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        if "content" not in out.columns:
            out["content"] = ""
        if "content_level" not in out.columns:
            out["content_level"] = "title_only"
        priority = (
            out.get("title", pd.Series("", index=out.index)).astype(str)
            + " "
            + out.get("category", pd.Series("", index=out.index)).astype(str)
        )
        important = priority.str.contains(
            "\u5e74\u5ea6\u62a5\u544a|\u534a\u5e74\u5ea6|\u4e00\u5b63\u62a5|\u4e09\u5b63\u62a5|\u5b9a\u671f\u62a5\u544a|\u4e1a\u7ee9|\u95ee\u8be2|\u5ba1\u8ba1",
            regex=True,
            na=False,
        )
        out = out.assign(_important=important.astype(int)).sort_values(["_important", "date"], ascending=[False, False])
        out = self._enrich_frame_from_urls(
            out.drop(columns=["_important"]),
            row_limit=self.cfg.data.text_extract_notice_rows,
            prefer_pdf=True,
            success_level="pdf_excerpt",
        )
        return out.sort_values("date").reset_index(drop=True)

    def _enrich_frame_from_urls(
        self,
        df: pd.DataFrame,
        row_limit: int,
        *,
        prefer_pdf: bool,
        success_level: str,
    ) -> pd.DataFrame:
        if df.empty or not self.cfg.data.text_extract_enabled:
            return df
        out = df.copy()
        if "content" not in out.columns:
            out["content"] = ""
        if "content_level" not in out.columns:
            out["content_level"] = "summary"
        url_groups: dict[str, list[int]] = {}
        for idx in out.index:
            url = str(out.at[idx, "url"] or "").strip()
            if not url or not self._needs_text_enrichment(out.at[idx, "content"]):
                continue
            url_groups.setdefault(url, []).append(idx)

        candidates = list(url_groups.items())
        limit = self._resolve_row_limit(row_limit)
        if limit is not None:
            candidates = candidates[:limit]
        for url, indexes in candidates:
            text = self._extract_text_from_url(url, prefer_pdf=prefer_pdf)
            if not text:
                continue
            for idx in indexes:
                existing = str(out.at[idx, "content"] or "").strip()
                out.at[idx, "content"] = f"{existing}\n{text}".strip() if existing else text
                out.at[idx, "content_level"] = success_level
        return out

    def _extract_text_from_url(self, url: str, *, prefer_pdf: bool) -> str:
        if not url:
            return ""
        cached = self._read_cached_content(url)
        if cached is not None:
            return cached
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30, verify=False)
            response.raise_for_status()
        except Exception:
            self._write_miss_cache(url)
            return ""
        content_type = str(response.headers.get("Content-Type", "")).lower()
        html_text = self._response_text(response) if any(token in content_type for token in ["text", "html", "xml", "json"]) else ""
        strategies = ["pdf", "html"] if prefer_pdf else ["html", "pdf"]
        text = ""
        for strategy in strategies:
            if strategy == "pdf":
                if url.lower().endswith(".pdf") or "pdf" in content_type or prefer_pdf:
                    text = self._extract_pdf_text(response.content)
            else:
                text = self._extract_html_text(html_text)
            if text:
                break
        text = self._compact_text(text)
        if text:
            self._write_cached_content(url, text)
        else:
            self._write_miss_cache(url)
        return text

    def _needs_text_enrichment(self, content: object) -> bool:
        existing = str(content or "").strip()
        return len(existing) < max(400, self.cfg.data.text_extract_max_chars // 3)

    def _content_cache_key(self, url: str) -> str:
        return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()

    def _content_text_path(self, url: str) -> Path:
        return self.content_cache_root / f"{self._content_cache_key(url)}.txt"

    def _content_miss_path(self, url: str) -> Path:
        return self.content_cache_root / f"{self._content_cache_key(url)}.miss"

    def _read_cached_content(self, url: str) -> str | None:
        if url in self._content_cache:
            return self._content_cache[url]

        text_path = self._content_text_path(url)
        if text_path.exists():
            try:
                text = self._compact_text(text_path.read_text(encoding="utf-8"))
            except Exception:
                text = ""
            self._content_cache[url] = text
            return text

        miss_path = self._content_miss_path(url)
        if miss_path.exists():
            age_seconds = time.time() - miss_path.stat().st_mtime
            if age_seconds < self._miss_cache_ttl_seconds:
                self._content_cache[url] = ""
                return ""
            miss_path.unlink(missing_ok=True)
        return None

    def _write_cached_content(self, url: str, text: str) -> None:
        compacted = self._compact_text(text)
        self._content_cache[url] = compacted
        text_path = self._content_text_path(url)
        text_path.write_text(compacted, encoding="utf-8")
        self._content_miss_path(url).unlink(missing_ok=True)

    def _write_miss_cache(self, url: str) -> None:
        self._content_cache[url] = ""
        miss_path = self._content_miss_path(url)
        miss_path.write_text("", encoding="utf-8")

    def _extract_pdf_text(self, content: bytes) -> str:
        if PdfReader is None or not content:
            return ""
        try:
            reader = PdfReader(BytesIO(content))
            parts: list[str] = []
            pages = reader.pages if self.cfg.data.text_extract_pdf_pages <= 0 else reader.pages[: self.cfg.data.text_extract_pdf_pages]
            for page in pages:
                parts.append(page.extract_text() or "")
            return "\n".join(parts)
        except Exception:
            return ""

    def _extract_html_text(self, html: str) -> str:
        if BeautifulSoup is None or not html:
            return ""
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        selectors = [
            "#ContentBody",
            ".newsContent",
            ".txtinfos",
            ".article-body",
            ".artibody",
            ".main-content",
            "article",
        ]
        chunks: list[str] = []
        for selector in selectors:
            nodes = soup.select(selector)
            if not nodes:
                continue
            for node in nodes:
                paragraphs = [self._compact_text(item.get_text(" ", strip=True)) for item in node.find_all("p")]
                paragraphs = [item for item in paragraphs if len(item) > 20]
                if paragraphs:
                    chunks.extend(paragraphs)
            if chunks:
                break
        if not chunks:
            paragraphs = [self._compact_text(item.get_text(" ", strip=True)) for item in soup.find_all("p")]
            chunks = [item for item in paragraphs if len(item) > 20]
        if not chunks and soup.body is not None:
            body_text = self._compact_text(soup.body.get_text(" ", strip=True))
            if len(body_text) > 50:
                chunks = [body_text]
        return "\n".join(chunks)

    def _compact_text(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        return cleaned[: self.cfg.data.text_extract_max_chars]

    def _sanitize_feature_name(self, value: object) -> str:
        text = re.sub(r"[^0-9a-zA-Z_]+", "_", str(value or "").strip().lower())
        text = re.sub(r"_+", "_", text).strip("_")
        return text or "field"

    def _select_numeric_statement_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        renamed: dict[str, str] = {}
        seen = {"date", "disclosure_date"}
        skip_tokens = ("date", "time", "name", "code", "currency", "type", "remark")
        for column in out.columns:
            if column == "date":
                continue
            clean = self._sanitize_feature_name(column)
            if any(token in clean for token in skip_tokens):
                continue
            if clean in seen:
                continue
            renamed[column] = clean
            seen.add(clean)
        out = out.rename(columns=renamed)
        keep = ["date"]
        if "disclosure_date" in out.columns:
            out["disclosure_date"] = pd.to_datetime(out["disclosure_date"], errors="coerce")
            keep.append("disclosure_date")
        for column in [col for col in out.columns if col != "date"]:
            numeric = pd.to_numeric(out[column], errors="coerce")
            if numeric.notna().any():
                out[column] = numeric
                keep.append(column)
        return out[keep]

    def _extract_date_from_text(self, value: object) -> pd.Timestamp:
        match = re.search(r"(20\d{2})-(\d{2})-(\d{2})", str(value or ""))
        if not match:
            return pd.NaT
        return pd.to_datetime("-".join(match.groups()), errors="coerce")

    def _parse_guba_time(self, value: object) -> pd.Timestamp:
        text = str(value or "").strip()
        if not text:
            return pd.NaT
        if re.match(r"20\d{2}-\d{2}-\d{2}", text):
            return pd.to_datetime(text[:10], errors="coerce")
        match = re.match(r"(\d{2})-(\d{2})", text)
        if not match:
            return pd.NaT
        month = int(match.group(1))
        day = int(match.group(2))
        now = pd.Timestamp.now()
        year = now.year
        if (month, day) > (now.month, now.day):
            year -= 1
        return pd.Timestamp(year=year, month=month, day=day)

    def _proxy_sentiment_score(self, text: object) -> float:
        lowered = str(text or "").lower()
        positive_terms = ["\u6da8", "\u5229\u597d", "\u7a81\u7834", "\u56de\u8d2d", "\u589e\u6301", "\u9ad8\u666f\u6c14", "bull", "buy"]
        negative_terms = ["\u8dcc", "\u51cf\u6301", "\u98ce\u9669", "\u4e8f\u635f", "\u95ee\u8be2", "\u8fdd\u89c4", "sell", "risk"]
        positive = sum(term in lowered for term in positive_terms)
        negative = sum(term in lowered for term in negative_terms)
        return float(positive - negative)

    def _build_text_sentiment_proxy(
        self,
        news: pd.DataFrame | None,
        notices: pd.DataFrame | None,
        policy: pd.DataFrame | None,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for source_name, frame in [
            ("news", news),
            ("notice", notices),
            ("policy", policy),
        ]:
            if frame is None or frame.empty or "date" not in frame.columns:
                continue
            tmp = frame.copy()
            tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.normalize()
            tmp["blob"] = (
                tmp.get("title", pd.Series("", index=tmp.index)).fillna("")
                + " "
                + tmp.get("content", pd.Series("", index=tmp.index)).fillna("")
                + " "
                + tmp.get("category", pd.Series("", index=tmp.index)).fillna("")
            ).str.strip()
            tmp["sentiment_proxy"] = tmp["blob"].map(self._proxy_sentiment_score)
            tmp["text_len"] = tmp["blob"].astype(str).str.len()
            tmp["news_quality"] = pd.to_numeric(
                tmp.get("news_relevance_score", pd.Series(0.0, index=tmp.index)),
                errors="coerce",
            ).fillna(0.0)
            tmp["positive_flag"] = (tmp["sentiment_proxy"] > 0).astype(float)
            tmp["negative_flag"] = (tmp["sentiment_proxy"] < 0).astype(float)
            frames.append(
                tmp.assign(text_source=source_name)[
                    ["date", "sentiment_proxy", "text_len", "news_quality", "positive_flag", "negative_flag", "text_source"]
                ]
            )
        if not frames:
            return pd.DataFrame()
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.dropna(subset=["date"])
        out = (
            merged.groupby("date")
            .agg(
                sentiment_proxy_score=("sentiment_proxy", "mean"),
                sentiment_proxy_abs=("sentiment_proxy", lambda x: float(pd.Series(x).abs().mean())),
                sentiment_proxy_count=("sentiment_proxy", "size"),
                sentiment_proxy_text_len=("text_len", "sum"),
                sentiment_proxy_positive_ratio=("positive_flag", "mean"),
                sentiment_proxy_negative_ratio=("negative_flag", "mean"),
                sentiment_proxy_source_diversity=("text_source", "nunique"),
            )
            .reset_index()
        )
        source_daily: list[pd.DataFrame] = []
        for source_name, group in merged.groupby("text_source"):
            prefix = str(source_name).strip().lower()
            daily = (
                group.groupby("date")
                .agg(
                    **{
                        f"{prefix}_proxy_score": ("sentiment_proxy", "mean"),
                        f"{prefix}_proxy_abs": ("sentiment_proxy", lambda x: float(pd.Series(x).abs().mean())),
                        f"{prefix}_proxy_count": ("sentiment_proxy", "size"),
                        f"{prefix}_proxy_text_len": ("text_len", "sum"),
                        f"{prefix}_proxy_positive_ratio": ("positive_flag", "mean"),
                        f"{prefix}_proxy_negative_ratio": ("negative_flag", "mean"),
                        f"{prefix}_proxy_quality_mean": ("news_quality", "mean"),
                    }
                )
                .reset_index()
            )
            source_daily.append(daily)
        for daily in source_daily:
            out = out.merge(daily, on="date", how="left")
        out["sentiment_source"] = "historical_text_proxy"
        return out

    def _strip_google_title(self, value: object) -> str:
        title = self._compact_text(str(value or ""))
        return re.sub(r"\s*-\s*[^-]{1,40}$", "", title).strip()

    def _parse_jsonp_payload(self, text: str) -> dict[str, object]:
        match = re.search(r"^[^(]+\((.*)\)\s*$", text, flags=re.S)
        if not match:
            raise ValueError("Invalid JSONP payload")
        return json.loads(match.group(1))

    def _strip_html(self, value: object) -> str:
        text = re.sub(r"<[^>]+>", "", str(value or ""))
        return text.replace("&nbsp;", " ").strip()

    def _monthly(self, value: object) -> pd.Timestamp:
        text = str(value).strip().replace("\u5e74", "-").replace("\u6708\u4efd", "").replace("\u6708", "")
        parts = [item for item in text.split("-") if item]
        if len(parts) >= 2:
            return pd.Timestamp(year=int(parts[0]), month=int(parts[1]), day=1)
        return pd.NaT

    def _quarterly(self, value: object) -> pd.Timestamp:
        text = str(value).strip()
        if "\u7b2c1\u5b63\u5ea6" in text:
            return pd.Timestamp(year=int(text[:4]), month=3, day=31)
        quarter_map = {
            "\u7b2c1-2\u5b63\u5ea6": (6, 30),
            "\u7b2c1-3\u5b63\u5ea6": (9, 30),
            "\u7b2c1-4\u5b63\u5ea6": (12, 31),
        }
        for key, (month, day) in quarter_map.items():
            if key in text:
                return pd.Timestamp(year=int(text[:4]), month=month, day=day)
        return pd.NaT

    def _year_windows(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        current = pd.Timestamp(year=start_date.year, month=1, day=1)
        window_start = start_date.normalize()
        window_end = end_date.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
        while current <= window_end:
            year_end = pd.Timestamp(year=current.year, month=12, day=31, hour=23, minute=59, second=59)
            windows.append((max(window_start, current), min(window_end, year_end)))
            current = pd.Timestamp(year=current.year + 1, month=1, day=1)
        return windows

    def _quarter_windows(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        current = start_date.normalize()
        window_end = end_date.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
        while current <= window_end:
            quarter_end = (current + pd.offsets.QuarterEnd(0)).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
            if quarter_end < current:
                quarter_end = (current + pd.offsets.QuarterEnd(1)).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
            windows.append((current, min(window_end, quarter_end)))
            current = (quarter_end + pd.Timedelta(seconds=1)).normalize()
        return windows

    def _to_gdelt_datetime(self, value: pd.Timestamp) -> str:
        return pd.Timestamp(value).strftime("%Y%m%d%H%M%S")


def expand_to_business_days(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    out = out.dropna(subset=["date"]).sort_values("date")
    calendar = pd.DataFrame({"date": business_day_range(start_date, end_date)})
    out = calendar.merge(out, on="date", how="left").sort_values("date").ffill()
    return out
