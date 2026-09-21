from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
except Exception:
    AutoModel = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

from .config import AppConfig
from .utils import get_logger


@dataclass
class TextScoredDataset:
    events: pd.DataFrame
    daily: pd.DataFrame


_NEWS_EVENT_TERMS = [
    "业绩",
    "营收",
    "利润",
    "净利",
    "订单",
    "扩产",
    "产能",
    "合作",
    "投资",
    "回购",
    "减持",
    "增持",
    "评级",
    "目标价",
    "研报",
    "财报",
    "公告",
    "诉讼",
    "风险",
    "涨价",
    "降价",
    "召回",
    "出口",
    "海外",
    "供应",
    "装机",
    "市占率",
    "新技术",
    "麒麟电池",
    "earnings",
    "profit",
    "revenue",
    "capacity",
    "order",
    "guidance",
    "rating",
    "target price",
    "lawsuit",
    "recall",
]

_NEWS_COMPANY_SPECIFIC_TERMS = [
    "宁德时代",
    "宁德",
    "catl",
    "300750",
    "麒麟电池",
    "神行电池",
    "时代电服",
    "contemporary amperex",
]

_NEWS_GENERIC_TERMS = [
    "收评",
    "早评",
    "午评",
    "大盘",
    "三大指数",
    "龙虎榜",
    "etf",
    "基金",
    "板块",
    "个股",
    "快讯",
    "滚动",
]

_NOTICE_MATERIAL_TERMS = [
    "年度报告",
    "季度报告",
    "半年度报告",
    "业绩预告",
    "业绩快报",
    "利润分配",
    "权益分派",
    "回购",
    "增持",
    "减持",
    "股权激励",
    "限制性股票",
    "可转债",
    "定增",
    "非公开发行",
    "募集资金",
    "投资",
    "合作",
    "项目",
    "产能",
    "订单",
    "合同",
    "担保",
    "质押",
    "诉讼",
    "仲裁",
    "风险",
    "关联交易",
    "重大资产",
    "分拆",
    "海外",
    "电池",
]

_NOTICE_LOW_SIGNAL_TERMS = [
    "董事会决议",
    "监事会决议",
    "股东大会决议",
    "股东大会通知",
    "独立董事",
    "法律意见书",
    "章程",
    "制度",
    "规则",
    "募集说明书摘要",
    "保荐",
    "审计报告",
    "内部控制",
    "社会责任",
    "环境、社会及治理",
    "英文版",
]

_TEXT_QUALITY_VERSION = "20260605_release_aligned_v4"

_HIGH_QUALITY_SOURCE_PATTERN = (
    "cninfo|szse|sse|gov.cn|miit.gov.cn|nea.gov.cn|ndrc.gov.cn|"
    "证券时报|上海证券报|中国证券报|证券日报|财联社|财新|中证|券商|研究|研报"
)

_MEDIUM_QUALITY_SOURCE_PATTERN = "eastmoney|sina|同花顺|东方财富|新浪|google_news|gdelt"

_LOW_QUALITY_SOURCE_PATTERN = "guba|股吧|forum|bbs|theme|industry|storage"


def clean_news_corpus(
    config: AppConfig,
    frame: pd.DataFrame | None,
    *,
    enforce_daily_cap: bool = True,
    logger=None,
    stage: str = "news",
) -> pd.DataFrame:
    if frame is None or frame.empty or not bool(getattr(config.data, "news_quality_filter", True)):
        return pd.DataFrame() if frame is None else frame
    out = frame.copy()
    out["date"] = pd.to_datetime(out.get("date"), errors="coerce").dt.normalize()
    out = out.dropna(subset=["date"]).reset_index(drop=True)
    if out.empty:
        return out
    for column in ["title", "content", "source", "publisher", "url", "content_level", "category"]:
        default = "summary" if column == "content_level" else ""
        out[column] = out.get(column, pd.Series(default, index=out.index)).fillna(default).astype(str)

    before = len(out)
    out["news_blob"] = (
        out["title"].str.strip()
        + " "
        + out["content"].str.strip()
        + " "
        + out["category"].str.strip()
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    out["news_relevance_score"] = _news_relevance_score(config, out)
    out["news_title_fingerprint"] = out["title"].map(_news_title_fingerprint)
    out["news_url_key"] = out["url"].map(_canonical_url_key)
    out["news_text_chars"] = out["news_blob"].str.len().astype(int)
    out["news_garbled_ratio"] = out["news_blob"].map(_garbled_ratio).astype(float)
    out["source_quality_score"] = _source_quality_score(config, out)
    out["event_strength_score"] = _event_strength_score(config, out, "news")

    min_chars = max(0, int(getattr(config.data, "news_min_text_chars", 12)))
    min_score = float(getattr(config.data, "news_min_relevance_score", 3.0))
    industry_only_min_score = float(getattr(config.data, "news_industry_only_min_score", 8.0))
    max_garbled = float(getattr(config.data, "news_max_garbled_ratio", 0.02))
    min_source_quality = float(getattr(config.data, "text_min_source_quality", 0.25))
    blob_lower = out["news_blob"].astype(str).str.lower()
    title_lower = out["title"].astype(str).str.lower()
    source_lower = out["source"].astype(str).str.lower()
    company_hit = _contains_any(blob_lower, _company_terms(config)) > 0
    title_company_hit = _contains_any(title_lower, _company_terms(config) + _NEWS_COMPANY_SPECIFIC_TERMS) > 0
    event_hit = (_contains_any(title_lower, _NEWS_EVENT_TERMS) + _contains_any(blob_lower, _NEWS_EVENT_TERMS)) > 0
    generic_title = _contains_any(title_lower, _NEWS_GENERIC_TERMS) > 0
    weak_industry_source = source_lower.str.contains(
        "industry|theme|storage|battery|google_news_rss_industry|google_news_rss_storage|eastmoney_search_industry",
        regex=True,
        na=False,
    )
    official_or_report = source_lower.str.contains("cninfo|notice|report|research|公告|研报", regex=True, na=False)
    strict_industry_pass = title_company_hit | official_or_report | (
        event_hit & (out["news_relevance_score"] >= industry_only_min_score + 1.5)
    )
    out = out[
        (out["news_text_chars"] >= min_chars)
        & (out["news_relevance_score"] >= min_score)
        & (company_hit | (out["news_relevance_score"] >= industry_only_min_score))
        & (~weak_industry_source | strict_industry_pass)
        & (~generic_title | company_hit | official_or_report)
        & (out["news_garbled_ratio"] <= max_garbled)
        & (out["source_quality_score"] >= min_source_quality)
    ].copy()
    if out.empty:
        _log_news_cleaning(logger, stage, before, 0)
        return out.drop(columns=["news_blob"], errors="ignore")

    out = out.sort_values(
        ["date", "source_quality_score", "news_relevance_score", "event_strength_score", "news_text_chars"],
        ascending=[True, False, False, False, False],
    )
    out = out.drop_duplicates(subset=["date", "title", "source", "url"])
    out = out.drop_duplicates(subset=["date", "news_title_fingerprint", "source"])
    non_empty_url = out["news_url_key"].astype(str).str.len() > 0
    with_url = out.loc[non_empty_url].drop_duplicates(subset=["news_url_key"])
    without_url = out.loc[~non_empty_url]
    out = pd.concat([with_url, without_url], ignore_index=True)

    out = _drop_near_duplicate_news(out, int(getattr(config.data, "news_near_duplicate_days", 7)))
    out = _drop_similar_texts(
        out,
        int(getattr(config.data, "text_similarity_dedupe_window_days", 14)),
        float(getattr(config.data, "text_similarity_dedupe_threshold", 0.86)),
    )
    if enforce_daily_cap:
        per_source = int(getattr(config.data, "news_max_per_source_per_day", 4))
        per_day = int(getattr(config.data, "news_max_per_day", 24))
        out = out.sort_values(["date", "source", "source_quality_score", "news_relevance_score"], ascending=[True, True, False, False])
        if per_source > 0:
            out = out.groupby(["date", "source"], group_keys=False).head(per_source)
        out = out.sort_values(
            ["date", "source_quality_score", "news_relevance_score", "event_strength_score", "news_text_chars"],
            ascending=[True, False, False, False, False],
        )
        if per_day > 0:
            out = out.groupby("date", group_keys=False).head(per_day)

    out = out.sort_values(["date", "source_quality_score", "news_relevance_score", "title"], ascending=[True, False, False, True]).reset_index(drop=True)
    _log_news_cleaning(logger, stage, before, len(out))
    return out.drop(columns=["news_blob"], errors="ignore")


def clean_notice_corpus(
    config: AppConfig,
    frame: pd.DataFrame | None,
    *,
    enforce_daily_cap: bool = True,
    logger=None,
    stage: str = "notice",
) -> pd.DataFrame:
    if frame is None or frame.empty or not bool(getattr(config.data, "notice_quality_filter", True)):
        return pd.DataFrame() if frame is None else frame
    out = frame.copy()
    out["date"] = pd.to_datetime(out.get("date"), errors="coerce").dt.normalize()
    out = out.dropna(subset=["date"]).reset_index(drop=True)
    if out.empty:
        return out
    for column in ["title", "content", "source", "publisher", "url", "content_level", "category"]:
        default = "summary" if column == "content_level" else ""
        out[column] = out.get(column, pd.Series(default, index=out.index)).fillna(default).astype(str)

    before = len(out)
    out["news_blob"] = (
        out["title"].str.strip()
        + " "
        + out["content"].str.strip()
        + " "
        + out["category"].str.strip()
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    out["news_relevance_score"] = _notice_relevance_score(config, out)
    out["news_title_fingerprint"] = out["title"].map(_news_title_fingerprint)
    out["news_url_key"] = out["url"].map(_canonical_url_key)
    out["news_text_chars"] = out["news_blob"].str.len().astype(int)
    out["news_garbled_ratio"] = out["news_blob"].map(_garbled_ratio).astype(float)
    out["source_quality_score"] = _source_quality_score(config, out)
    out["event_strength_score"] = _event_strength_score(config, out, "notice")
    out["notice_event_class"] = _notice_event_class(out)
    content_text = out["content"].astype(str).str.strip()
    has_fulltext = out["content_level"].str.lower().isin({"body", "pdf_excerpt"}) & (content_text.str.len() >= 80)

    title_lower = out["title"].astype(str).str.lower()
    blob_lower = out["news_blob"].astype(str).str.lower()
    material_hit = (_contains_any(title_lower, _NOTICE_MATERIAL_TERMS) + _contains_any(blob_lower, _NOTICE_MATERIAL_TERMS)) > 0
    company_hit = (_contains_any(title_lower, _company_terms(config)) + _contains_any(blob_lower, _company_terms(config))) > 0
    routine_title = _contains_any(title_lower, _NOTICE_LOW_SIGNAL_TERMS) > 0
    min_chars = max(0, int(getattr(config.data, "notice_min_text_chars", 16)))
    min_score = float(getattr(config.data, "notice_min_relevance_score", 4.5))
    max_garbled = float(getattr(config.data, "news_max_garbled_ratio", 0.02))
    min_source_quality = float(getattr(config.data, "text_min_source_quality", 0.25))
    out = out[
        (out["news_text_chars"] >= min_chars)
        & (out["news_relevance_score"] >= min_score)
        & (company_hit | material_hit)
        & (~routine_title | material_hit | (out["news_relevance_score"] >= min_score + 2.0))
        & (out["news_garbled_ratio"] <= max_garbled)
        & (out["source_quality_score"] >= min_source_quality)
    ].copy()
    if bool(getattr(config.data, "notice_exclude_routine", True)):
        out = out[out["notice_event_class"] != "routine"].copy()
    title_only_mask = ~has_fulltext.reindex(out.index, fill_value=False)
    if bool(getattr(config.data, "notice_drop_title_only_periodic_reports", True)):
        periodic_pattern = r"年度报告$|半年度报告$|季度报告$|一季度报告$|三季度报告$|权益分派|利润分配实施"
        out = out[~(title_only_mask & out["title"].str.contains(periodic_pattern, regex=True, na=False))].copy()
        title_only_mask = ~has_fulltext.reindex(out.index, fill_value=False)
    if bool(getattr(config.data, "notice_drop_progress_updates", True)):
        progress_pattern = r"进展公告|实施公告|完成公告|提示性公告|监管协议|鉴证报告|专项报告|上市流通|解除质押|调剂担保额度"
        out = out[~(title_only_mask & out["title"].str.contains(progress_pattern, regex=True, na=False))].copy()
    quality_pool = out.copy()
    min_strength = float(getattr(config.data, "notice_min_event_strength", 0.0) or 0.0)
    if min_strength > 0.0:
        out = out[out["event_strength_score"] >= min_strength].copy()
    title_only_min = float(getattr(config.data, "notice_title_only_min_relevance", min_score) or min_score)
    fulltext_mask = has_fulltext.reindex(out.index, fill_value=False)
    out = out[fulltext_mask | (out["news_relevance_score"] >= title_only_min)].copy()
    min_keep = int(getattr(config.data, "notice_min_keep_rows", 0) or 0)
    if min_keep > 0 and len(out) < min_keep and not quality_pool.empty:
        fallback_strength = min(
            min_strength if min_strength > 0.0 else float("inf"),
            float(getattr(config.data, "notice_fallback_min_event_strength", 0.60) or 0.60),
        )
        if not np.isfinite(fallback_strength):
            fallback_strength = float(getattr(config.data, "notice_fallback_min_event_strength", 0.60) or 0.60)
        fallback_title_min = min(
            title_only_min,
            float(getattr(config.data, "notice_fallback_title_only_min_relevance", 7.0) or 7.0),
        )
        fallback = quality_pool.copy()
        fallback_fulltext = has_fulltext.reindex(fallback.index, fill_value=False)
        fallback = fallback[
            (fallback["event_strength_score"] >= fallback_strength)
            & (fallback_fulltext | (fallback["news_relevance_score"] >= fallback_title_min))
        ].copy()
        if not fallback.empty:
            out = pd.concat([out, fallback], ignore_index=False)
            out = out.sort_values(
                ["source_quality_score", "news_relevance_score", "event_strength_score", "news_text_chars", "date"],
                ascending=[False, False, False, False, False],
            )
            out = out.drop_duplicates(subset=["date", "title", "source", "url"]).head(min_keep).copy()
    if out.empty:
        _log_text_cleaning(logger, stage, before, 0)
        return out.drop(columns=["news_blob"], errors="ignore")

    out = out.sort_values(
        ["date", "source_quality_score", "news_relevance_score", "event_strength_score", "news_text_chars"],
        ascending=[True, False, False, False, False],
    )
    out = out.drop_duplicates(subset=["date", "title", "source", "url"])
    out = out.drop_duplicates(subset=["date", "news_title_fingerprint", "source"])
    non_empty_url = out["news_url_key"].astype(str).str.len() > 0
    out = pd.concat(
        [
            out.loc[non_empty_url].drop_duplicates(subset=["news_url_key"]),
            out.loc[~non_empty_url],
        ],
        ignore_index=True,
    )
    dedupe_days = max(
        int(getattr(config.data, "notice_near_duplicate_days", 30)),
        int(getattr(config.data, "notice_global_dedupe_days", 0) or 0),
    )
    out = _drop_near_duplicate_news(out, dedupe_days)
    out = _drop_similar_texts(
        out,
        int(getattr(config.data, "text_similarity_dedupe_window_days", 14)),
        float(getattr(config.data, "text_similarity_dedupe_threshold", 0.86)),
    )
    if enforce_daily_cap:
        class_month_cap = int(getattr(config.data, "notice_max_per_class_per_month", 0) or 0)
        if class_month_cap > 0 and not out.empty:
            out["_notice_month"] = out["date"].dt.to_period("M").astype(str)
            out = (
                out.sort_values(
                    ["date", "source_quality_score", "news_relevance_score", "event_strength_score", "news_text_chars"],
                    ascending=[True, False, False, False, False],
                )
                .groupby(["_notice_month", "notice_event_class"], group_keys=False)
                .head(class_month_cap)
                .drop(columns=["_notice_month"], errors="ignore")
            )
        per_day = int(getattr(config.data, "notice_max_per_day", 4))
        if per_day > 0:
            out = (
                out.sort_values(
                    ["date", "source_quality_score", "news_relevance_score", "event_strength_score", "news_text_chars"],
                    ascending=[True, False, False, False, False],
                )
                .groupby("date", group_keys=False)
                .head(per_day)
            )
    out = out.sort_values(["date", "source_quality_score", "news_relevance_score", "title"], ascending=[True, False, False, True]).reset_index(drop=True)
    _log_text_cleaning(logger, stage, before, len(out))
    return out.drop(columns=["news_blob"], errors="ignore")


def _news_relevance_score(config: AppConfig, frame: pd.DataFrame) -> pd.Series:
    company_terms = _company_terms(config)
    industry_terms = _industry_terms(config)
    title = frame["title"].astype(str).str.lower()
    blob = frame["news_blob"].astype(str).str.lower()
    source = frame["source"].astype(str).str.lower()
    publisher = frame["publisher"].astype(str).str.lower()
    level = frame["content_level"].astype(str).str.lower()
    score = pd.Series(0.0, index=frame.index)
    score += 4.0 * _contains_any(title, company_terms) + 2.0 * _contains_any(blob, company_terms)
    score += 2.0 * _contains_any(title, industry_terms) + 1.0 * _contains_any(blob, industry_terms)
    score += 1.5 * _contains_any(title, _NEWS_EVENT_TERMS) + 0.75 * _contains_any(blob, _NEWS_EVENT_TERMS)
    score += 1.0 * level.isin({"body", "pdf_excerpt"}).astype(float)
    score += 0.75 * source.str.contains("report|research|cninfo|eastmoney|sina|google_news|gdelt", regex=True, na=False).astype(float)
    score += 0.5 * publisher.str.contains("证券|时报|财联社|上证|中证|证券报|券商|研究|财经|财新|sina|eastmoney", regex=True, na=False).astype(float)
    score += np.minimum(frame["news_blob"].astype(str).str.len().astype(float) / 240.0, 1.0) * 0.75
    generic = _contains_any(title, _NEWS_GENERIC_TERMS)
    company_hit = _contains_any(blob, company_terms)
    company_title_hit = _contains_any(title, company_terms + _NEWS_COMPANY_SPECIFIC_TERMS)
    score -= 2.0 * ((generic > 0) & (company_hit == 0)).astype(float)
    weak_industry_source = source.str.contains("industry|theme|storage|google_news_rss_industry|google_news_rss_storage|eastmoney_search_industry", regex=True, na=False)
    score -= 2.5 * (weak_industry_source & (company_hit == 0)).astype(float)
    score += 1.0 * (company_title_hit > 0).astype(float)
    score -= 1.0 * source.str.contains("guba", regex=False, na=False).astype(float)
    return score.astype(float)


def _notice_relevance_score(config: AppConfig, frame: pd.DataFrame) -> pd.Series:
    company_terms = _company_terms(config)
    title = frame["title"].astype(str).str.lower()
    blob = frame["news_blob"].astype(str).str.lower()
    source = frame["source"].astype(str).str.lower()
    level = frame["content_level"].astype(str).str.lower()
    score = pd.Series(0.0, index=frame.index)
    score += 3.0 * _contains_any(title, company_terms) + 1.0 * _contains_any(blob, company_terms)
    score += 2.5 * _contains_any(title, _NOTICE_MATERIAL_TERMS) + 0.75 * _contains_any(blob, _NOTICE_MATERIAL_TERMS)
    score += 1.0 * level.isin({"body", "pdf_excerpt"}).astype(float)
    score += 0.75 * source.str.contains("cninfo|notice|eastmoney", regex=True, na=False).astype(float)
    score += np.minimum(frame["news_blob"].astype(str).str.len().astype(float) / 360.0, 1.0) * 0.5
    low_signal = _contains_any(title, _NOTICE_LOW_SIGNAL_TERMS)
    material = _contains_any(title, _NOTICE_MATERIAL_TERMS) + _contains_any(blob, _NOTICE_MATERIAL_TERMS)
    score -= 2.0 * ((low_signal > 0) & (material == 0)).astype(float)
    return score.astype(float)


def _source_quality_score(config: AppConfig, frame: pd.DataFrame) -> pd.Series:
    source = frame.get("source", pd.Series("", index=frame.index)).fillna("").astype(str).str.lower()
    publisher = frame.get("publisher", pd.Series("", index=frame.index)).fillna("").astype(str).str.lower()
    url = frame.get("url", pd.Series("", index=frame.index)).fillna("").astype(str).str.lower()
    joined = source + " " + publisher + " " + url
    score = pd.Series(0.50, index=frame.index, dtype=float)
    official_domains = [str(domain).lower() for domain in getattr(config.data, "policy_official_domains", [])]
    if official_domains:
        official_pattern = "|".join(re.escape(domain) for domain in official_domains if domain)
        if official_pattern:
            score = score.mask(joined.str.contains(official_pattern, regex=True, na=False), 0.95)
    score = score.mask(joined.str.contains(_HIGH_QUALITY_SOURCE_PATTERN, regex=True, na=False), 0.90)
    score = score.mask(joined.str.contains(_MEDIUM_QUALITY_SOURCE_PATTERN, regex=True, na=False), score.clip(lower=0.65))
    score = score.mask(joined.str.contains(_LOW_QUALITY_SOURCE_PATTERN, regex=True, na=False), score.clip(upper=0.35))
    score = score.mask(source.str.contains("cninfo|notice|policy", regex=True, na=False), score.clip(lower=0.85))
    score = score.mask(source.str.contains("report|research", regex=True, na=False), score.clip(lower=0.80))
    return score.clip(0.0, 1.0).astype(float)


def _event_strength_score(config: AppConfig, frame: pd.DataFrame, text_type: str) -> pd.Series:
    title = frame.get("title", pd.Series("", index=frame.index)).fillna("").astype(str).str.lower()
    blob_source = frame["news_blob"] if "news_blob" in frame.columns else frame.get("blob", pd.Series("", index=frame.index))
    blob = blob_source.fillna("").astype(str).str.lower()
    company_hits = _contains_any(title, _company_terms(config)) + _contains_any(blob, _company_terms(config))
    if text_type == "notice":
        strong_hits = _contains_any(title, _NOTICE_MATERIAL_TERMS) + 0.5 * _contains_any(blob, _NOTICE_MATERIAL_TERMS)
        weak_hits = _contains_any(title, _NOTICE_LOW_SIGNAL_TERMS)
    else:
        strong_hits = _contains_any(title, _NEWS_EVENT_TERMS) + 0.5 * _contains_any(blob, _NEWS_EVENT_TERMS)
        weak_hits = _contains_any(title, _NEWS_GENERIC_TERMS)
    length_bonus = np.minimum(blob_source.astype(str).str.len().astype(float) / 500.0, 1.0)
    raw = 1.5 * company_hits + 2.0 * strong_hits + 0.5 * length_bonus - 1.25 * weak_hits
    return np.tanh(raw / 4.0).clip(0.0, 1.0).astype(float)


def _notice_event_class(frame: pd.DataFrame) -> pd.Series:
    title = frame.get("title", pd.Series("", index=frame.index)).fillna("").astype(str)
    rules = [
        ("performance", "年度报告|季度报告|半年度报告|业绩预告|业绩快报|利润分配|权益分派"),
        ("capital", "回购|增持|减持|股权激励|限制性股票|可转债|定增|非公开发行|募集资金"),
        ("operation", "投资|合作|项目|产能|订单|合同|海外|电池"),
        ("risk", "担保|质押|诉讼|仲裁|风险|关联交易|重大资产|分拆"),
        ("routine", "|".join(re.escape(term) for term in _NOTICE_LOW_SIGNAL_TERMS)),
    ]
    out = pd.Series("other", index=frame.index, dtype="object")
    for label, pattern in rules:
        out = out.mask(title.str.contains(pattern, regex=True, na=False), label)
    return out


def _company_terms(config: AppConfig) -> list[str]:
    return _unique_terms(
        [
            config.data.company_name,
            config.data.symbol,
            "宁德",
            "宁德时代",
            "catl",
            "contemporary amperex",
        ]
    )


def _industry_terms(config: AppConfig) -> list[str]:
    return _unique_terms(
        [
            config.data.industry_name,
            *list(config.data.secondary_industry_names or []),
            "动力电池",
            "锂电池",
            "电池",
            "储能",
            "新能源车",
            "新能源汽车",
            "battery",
            "ev",
            "energy storage",
            "lithium",
        ]
    )


def _contains_any(series: pd.Series, terms: list[str]) -> pd.Series:
    if not terms:
        return pd.Series(0.0, index=series.index)
    escaped = [re.escape(term.lower()) for term in terms if str(term).strip()]
    if not escaped:
        return pd.Series(0.0, index=series.index)
    pattern = "|".join(escaped)
    return series.str.contains(pattern, regex=True, na=False).astype(float)


def _unique_terms(values: list[object]) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for value in values:
        text = str(value or "").strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        terms.append(text)
    return terms


def _news_title_fingerprint(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\d{1,4}[年./-]\d{1,2}([月./-]\d{1,2}日?)?", " ", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    text = re.sub(r"(快讯|新闻|公告|研报|研究报告|点评|点评报告|摘要)$", "", text)
    return text[:96]


def _canonical_url_key(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"[?#].*$", "", text)
    return text.rstrip("/")


def _garbled_ratio(value: object) -> float:
    text = str(value or "")
    if not text:
        return 0.0
    bad = len(re.findall(r"[\ufffd\u0400-\u04ff]", text))
    return bad / max(len(text), 1)


def _drop_near_duplicate_news(frame: pd.DataFrame, window_days: int) -> pd.DataFrame:
    if frame.empty or window_days <= 0:
        return frame
    ordered = frame.sort_values(["news_relevance_score", "news_text_chars", "date"], ascending=[False, False, False])
    kept_rows: list[int] = []
    seen_dates: dict[str, list[pd.Timestamp]] = {}
    for idx, row in ordered.iterrows():
        fingerprint = str(row.get("news_title_fingerprint", "") or "")
        if len(fingerprint) < 8:
            kept_rows.append(idx)
            continue
        date_value = pd.Timestamp(row["date"])
        dates = seen_dates.setdefault(fingerprint, [])
        if any(abs((date_value - existing).days) <= window_days for existing in dates):
            continue
        dates.append(date_value)
        kept_rows.append(idx)
    return frame.loc[kept_rows].copy()


def _drop_similar_texts(frame: pd.DataFrame, window_days: int, threshold: float) -> pd.DataFrame:
    if frame.empty or window_days <= 0 or threshold <= 0:
        return frame
    sort_columns = [col for col in ["source_quality_score", "news_relevance_score", "event_strength_score", "news_text_chars", "date"] if col in frame.columns]
    ascending = [False] * (len(sort_columns) - 1) + [False] if sort_columns else [False]
    ordered = frame.sort_values(sort_columns, ascending=ascending) if sort_columns else frame
    kept_rows: list[int] = []
    seen: list[tuple[pd.Timestamp, set[str]]] = []
    for idx, row in ordered.iterrows():
        date_value = pd.Timestamp(row["date"])
        text = f"{row.get('title', '')} {row.get('content', '')} {row.get('category', '')}"
        shingles = _text_shingles(text)
        if len(shingles) < 8:
            kept_rows.append(idx)
            continue
        is_duplicate = False
        for existing_date, existing_shingles in seen:
            if abs((date_value - existing_date).days) > window_days:
                continue
            if _jaccard(shingles, existing_shingles) >= threshold:
                is_duplicate = True
                break
        if is_duplicate:
            continue
        seen.append((date_value, shingles))
        kept_rows.append(idx)
    return frame.loc[kept_rows].copy()


def _text_shingles(value: object, size: int = 3) -> set[str]:
    text = str(value or "").lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\d{1,4}[年./-]\d{1,2}([月./-]\d{1,2}日?)?", " ", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)[:600]
    if len(text) <= size:
        return {text} if text else set()
    return {text[index : index + size] for index in range(0, len(text) - size + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left | right), 1)


def _log_news_cleaning(logger, stage: str, before: int, after: int) -> None:
    if logger is None or before <= 0:
        return
    removed = before - after
    logger.info("News quality filter [%s]: kept %s/%s rows, removed %s noisy or duplicate rows.", stage, after, before, removed)


def _log_text_cleaning(logger, stage: str, before: int, after: int) -> None:
    if logger is None or before <= 0:
        return
    removed = before - after
    logger.info("Text quality filter [%s]: kept %s/%s rows, removed %s noisy or low-signal rows.", stage, after, before, removed)


def build_text_corpus(
    config: AppConfig,
    news: pd.DataFrame | None,
    notices: pd.DataFrame | None,
    policy: pd.DataFrame | None = None,
    *,
    include_policy: bool = True,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    specs = [
        ("news", news),
        ("notice", notices),
        ("policy", policy if include_policy else None),
    ]
    for text_type, frame in specs:
        if frame is None or frame.empty:
            continue
        tmp = frame.copy()
        tmp["date"] = pd.to_datetime(tmp.get("date"), errors="coerce").dt.normalize()
        tmp = tmp.dropna(subset=["date"])
        tmp["title"] = tmp.get("title", pd.Series("", index=tmp.index)).fillna("").astype(str)
        tmp["content"] = tmp.get("content", pd.Series("", index=tmp.index)).fillna("").astype(str)
        tmp["source"] = tmp.get("source", pd.Series(text_type, index=tmp.index)).fillna(text_type).astype(str)
        tmp["publisher"] = tmp.get("publisher", pd.Series("", index=tmp.index)).fillna("").astype(str)
        tmp["content_level"] = tmp.get("content_level", pd.Series("summary", index=tmp.index)).fillna("summary").astype(str)
        tmp["url"] = tmp.get("url", pd.Series("", index=tmp.index)).fillna("").astype(str)
        tmp["category"] = tmp.get("category", pd.Series("", index=tmp.index)).fillna("").astype(str)
        tmp["text_type"] = text_type
        tmp["blob"] = (
            tmp["title"].str.strip()
            + " "
            + tmp["content"].str.strip()
            + " "
            + tmp["category"].str.strip()
        ).str.replace(r"\s+", " ", regex=True).str.strip()
        if text_type == "news":
            tmp = clean_news_corpus(config, tmp, enforce_daily_cap=True, stage="text_corpus")
            if tmp.empty:
                continue
            tmp["blob"] = (
                tmp["title"].str.strip()
                + " "
                + tmp["content"].str.strip()
                + " "
                + tmp["category"].str.strip()
            ).str.replace(r"\s+", " ", regex=True).str.strip()
        elif text_type == "notice":
            tmp = clean_notice_corpus(config, tmp, enforce_daily_cap=True, stage="text_corpus_notice")
            if tmp.empty:
                continue
            tmp["blob"] = (
                tmp["title"].str.strip()
                + " "
                + tmp["content"].str.strip()
                + " "
                + tmp["category"].str.strip()
            ).str.replace(r"\s+", " ", regex=True).str.strip()
        if "news_relevance_score" not in tmp.columns:
            tmp["news_relevance_score"] = 0.0
        if "news_title_fingerprint" not in tmp.columns:
            tmp["news_title_fingerprint"] = ""
        if "source_quality_score" not in tmp.columns:
            tmp["source_quality_score"] = _source_quality_score(config, tmp)
        if "event_strength_score" not in tmp.columns:
            tmp["event_strength_score"] = _event_strength_score(config, tmp, text_type)
        if "notice_event_class" not in tmp.columns:
            tmp["notice_event_class"] = ""
        tmp["text_quality_version"] = _TEXT_QUALITY_VERSION
        tmp = tmp.drop_duplicates(subset=["date", "title", "source", "url"]).reset_index(drop=True)
        frames.append(
            tmp[
                [
                    "date",
                    "title",
                    "content",
                    "blob",
                    "source",
                    "publisher",
                    "url",
                    "content_level",
                    "category",
                    "text_type",
                    "news_relevance_score",
                    "news_title_fingerprint",
                    "source_quality_score",
                    "event_strength_score",
                    "notice_event_class",
                    "text_quality_version",
                ]
            ]
        )
    if not frames:
        return pd.DataFrame()
    corpus = pd.concat(frames, ignore_index=True)
    corpus = corpus.loc[corpus["title"].astype(str).str.len() > 0].reset_index(drop=True)
    corpus["text_uid"] = corpus.apply(
        lambda row: _build_text_uid(
            row.get("date"),
            row.get("title"),
            row.get("source"),
            row.get("url"),
            row.get("text_type"),
        ),
        axis=1,
    )
    corpus = corpus.drop_duplicates(subset=["text_uid"]).reset_index(drop=True)
    return corpus.sort_values(["date", "source", "title"]).reset_index(drop=True)


def _build_text_uid(date_value: object, title: object, source: object, url: object, text_type: object) -> str:
    date_text = ""
    if pd.notna(date_value):
        date_text = pd.Timestamp(date_value).strftime("%Y-%m-%d")
    payload = "|".join(
        [
            date_text,
            str(text_type or "").strip(),
            str(source or "").strip(),
            str(url or "").strip(),
            str(title or "").strip(),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


class LocalFinBERTService:
    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        self.logger = get_logger()
        self.embedding_dim = int(config.features.text_embedding_dim)
        if not config.data.enable_finbert or AutoTokenizer is None or AutoModel is None or AutoModelForSequenceClassification is None:
            raise RuntimeError(
                "FinBERT is required. Set data.enable_finbert=true and install transformers with torch support."
            )
        model_path = Path(config.data.finbert_model).expanduser()
        if not model_path.is_absolute():
            model_path = (Path.cwd() / model_path).resolve()
        if not model_path.exists() or not model_path.is_dir():
            raise FileNotFoundError(f"FinBERT local model directory not found: {model_path}")
        self.model_path = model_path
        self.device = self._resolve_device(config.train.device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        self.encoder = AutoModel.from_pretrained(str(self.model_path), local_files_only=True)
        self.encoder.to(self.device)
        self.encoder.eval()
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path),
            local_files_only=True,
        )
        self.sentiment_model.to(self.device)
        self.sentiment_model.eval()
        labels = getattr(self.sentiment_model.config, "id2label", {}) or {}
        self.sentiment_labels = {int(key): str(value).lower() for key, value in labels.items()}
        self.embedding_dim = int(getattr(self.encoder.config, "hidden_size", 768))
        self.batch_size = max(1, int(config.data.finbert_batch_size))

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                tokens = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                tokens = self._tokens_to_device(tokens)
                hidden = self.encoder(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                outputs.append(pooled.cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)

    def score_sentiment(self, texts: list[str]) -> pd.DataFrame:
        columns = [
            "finbert_positive",
            "finbert_negative",
            "finbert_neutral",
            "finbert_sentiment",
        ]
        if not texts:
            return pd.DataFrame(columns=columns)
        rows: list[list[float]] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                tokens = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                tokens = self._tokens_to_device(tokens)
                logits = self.sentiment_model(**tokens).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                for prob_row in probs:
                    mapped = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
                    for idx, prob in enumerate(prob_row):
                        label = self.sentiment_labels.get(idx, str(idx)).lower()
                        if "pos" in label:
                            mapped["positive"] = float(prob)
                        elif "neg" in label:
                            mapped["negative"] = float(prob)
                        elif "neu" in label:
                            mapped["neutral"] = float(prob)
                    rows.append(
                        [
                            mapped["positive"],
                            mapped["negative"],
                            mapped["neutral"],
                            mapped["positive"] - mapped["negative"],
                        ]
                    )
        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        value = str(requested or "auto").strip().lower()
        if value == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if value.startswith("cuda"):
            return torch.device(value if torch.cuda.is_available() else "cpu")
        return torch.device(value)

    def _tokens_to_device(self, tokens: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in tokens.items()}

class TextSentimentDatasetBuilder:
    def __init__(self, config: AppConfig, *, include_embeddings: bool) -> None:
        self.cfg = config
        self.logger = get_logger()
        self.include_embeddings = include_embeddings
        self.finbert = LocalFinBERTService(config)
        self.embedding_dim = self.finbert.embedding_dim
        self.positive_words, self.negative_words = self._load_lexicon(Path(config.data.lexicon_path))
        self._positive_ascii = {word for word in self.positive_words if word.isascii()}
        self._negative_ascii = {word for word in self.negative_words if word.isascii()}
        self._positive_phrases = {word for word in self.positive_words if not word.isascii()}
        self._negative_phrases = {word for word in self.negative_words if not word.isascii()}

    def build(self, corpus: pd.DataFrame) -> TextScoredDataset:
        if corpus.empty:
            return TextScoredDataset(events=pd.DataFrame(), daily=pd.DataFrame())
        events = corpus.copy()
        events["date"] = pd.to_datetime(events["date"], errors="coerce").dt.normalize()
        events = events.dropna(subset=["date"]).reset_index(drop=True)
        events["title"] = events.get("title", pd.Series("", index=events.index)).fillna("").astype(str)
        events["blob"] = events.get("blob", pd.Series("", index=events.index)).fillna("").astype(str)
        events["source"] = events.get("source", pd.Series("", index=events.index)).fillna("").astype(str)
        events["publisher"] = events.get("publisher", pd.Series("", index=events.index)).fillna("").astype(str)
        events["url"] = events.get("url", pd.Series("", index=events.index)).fillna("").astype(str)
        events["content_level"] = events.get("content_level", pd.Series("summary", index=events.index)).fillna("summary").astype(str)
        events["text_type"] = events.get("text_type", pd.Series("news", index=events.index)).fillna("news").astype(str)
        events["text_quality_version"] = events.get(
            "text_quality_version",
            pd.Series(_TEXT_QUALITY_VERSION, index=events.index),
        ).fillna(_TEXT_QUALITY_VERSION).astype(str)
        events["text_length"] = events["blob"].astype(str).str.len().astype(float)
        events["title_length"] = events["title"].astype(str).str.len().astype(float)
        events["news_relevance_score"] = pd.to_numeric(
            events.get("news_relevance_score", pd.Series(0.0, index=events.index)),
            errors="coerce",
        ).fillna(0.0)
        events["source_quality_score"] = pd.to_numeric(
            events.get("source_quality_score", _source_quality_score(self.cfg, events)),
            errors="coerce",
        ).fillna(0.5).clip(0.0, 1.0)
        events["event_strength_score"] = pd.to_numeric(
            events.get("event_strength_score", _event_strength_score(self.cfg, events, "news")),
            errors="coerce",
        ).fillna(0.0).clip(0.0, 1.0)
        events["notice_event_class"] = events.get("notice_event_class", pd.Series("", index=events.index)).fillna("").astype(str)
        events["notice_event_code"] = events["notice_event_class"].map(
            {"performance": 1.0, "capital": 2.0, "operation": 3.0, "risk": 4.0, "routine": -1.0}
        ).fillna(0.0)
        events["lexicon_positive_hits"] = events["blob"].map(self._count_positive).astype(float)
        events["lexicon_negative_hits"] = events["blob"].map(self._count_negative).astype(float)
        events["lexicon_sentiment"] = events["lexicon_positive_hits"] - events["lexicon_negative_hits"]
        events["lexicon_sentiment_normalized"] = np.tanh(events["lexicon_sentiment"] / (1.0 + events["text_length"] / 120.0))
        sentiment = self.finbert.score_sentiment(events["blob"].tolist())
        if len(sentiment) != len(events):
            raise RuntimeError("FinBERT sentiment row count does not match the text corpus.")
        for column in sentiment.columns:
            events[column] = pd.to_numeric(sentiment[column], errors="coerce").fillna(0.0).astype(float)
        events = self._winsorize_sentiment_columns(events)
        events["hybrid_sentiment"] = 0.7 * events["finbert_sentiment"] + 0.3 * events["lexicon_sentiment_normalized"]
        events["positive_flag"] = (events["hybrid_sentiment"] > 0).astype(float)
        events["negative_flag"] = (events["hybrid_sentiment"] < 0).astype(float)
        events["source_kind"] = events["source"].map(self._classify_text_source)
        events["is_official_source"] = events.apply(
            lambda row: float(self._is_official_text_source(row.get("source"), row.get("publisher"), row.get("url"))),
            axis=1,
        )
        events["is_news_source"] = (events["source_kind"] == "news").astype(float)
        events["is_notice_source"] = (events["source_kind"] == "notice").astype(float)
        events["is_policy_source"] = (events["source_kind"] == "policy").astype(float)
        events["is_report_source"] = (events["source_kind"] == "report").astype(float)
        events["is_guba_source"] = (events["source_kind"] == "guba").astype(float)
        events["fulltext_flag"] = events["content_level"].isin({"body", "pdf_excerpt"}).astype(float)
        if self.include_embeddings:
            embeddings = self.finbert.encode(events["blob"].tolist())
            if embeddings.shape != (len(events), self.embedding_dim):
                raise RuntimeError("FinBERT embedding output shape does not match the text corpus.")
            events["embedding_vector"] = [np.asarray(vector, dtype=np.float32) for vector in embeddings]
        daily = self._build_daily(events)
        return TextScoredDataset(events=events, daily=daily)

    def _build_daily(self, events: pd.DataFrame) -> pd.DataFrame:
        daily = (
            events.groupby("date")
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
                text_sent_publisher_diversity=("publisher", lambda values: int(pd.Series([v for v in values if str(v).strip()], dtype="object").nunique())),
            )
            .reset_index()
        )
        for column in [col for col in daily.columns if col != "date"]:
            daily[column] = pd.to_numeric(daily[column], errors="coerce").fillna(0.0)
        return daily

    def _winsorize_sentiment_columns(self, events: pd.DataFrame) -> pd.DataFrame:
        lower = float(getattr(self.cfg.data, "sentiment_winsor_lower", 0.01))
        upper = float(getattr(self.cfg.data, "sentiment_winsor_upper", 0.99))
        if events.empty or lower <= 0 or upper >= 1 or lower >= upper or len(events) < 20:
            return events
        out = events.copy()
        columns = [
            "finbert_sentiment",
            "lexicon_sentiment_normalized",
            "finbert_positive",
            "finbert_negative",
            "finbert_neutral",
        ]
        for column in columns:
            values = pd.to_numeric(out[column], errors="coerce")
            lo = float(values.quantile(lower))
            hi = float(values.quantile(upper))
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                out[column] = values.clip(lo, hi)
        return out

    def _load_lexicon(self, path: Path) -> tuple[set[str], set[str]]:
        fallback_positive = {
            "beat",
            "growth",
            "upgrade",
            "improve",
            "record",
            "strong",
            "positive",
            "profit",
            "surge",
            "expansion",
            "increase",
            "gain",
            "bull",
            "buy",
            "up",
            "benefit",
            "favorable",
            "澧炴寔",
            "澧為暱",
            "鍒╁ソ",
            "鐩堝埄",
            "鍥炶喘",
            "绐佺牬",
        }
        fallback_negative = {
            "risk",
            "loss",
            "downgrade",
            "decline",
            "warning",
            "negative",
            "lawsuit",
            "default",
            "cut",
            "weak",
            "sell",
            "bear",
            "down",
            "reduce",
            "浜忔崯",
            "椋庨櫓",
            "鍑忔寔",
            "涓嬫粦",
            "闂",
            "璇夎",
        }
        if not path.exists():
            self.logger.warning("Lexicon file not found at %s, using a compact fallback lexicon.", path)
            return fallback_positive, fallback_negative
        try:
            lexicon = pd.read_csv(path)
            word_col = next((col for col in lexicon.columns if col.lower() == "word"), None)
            pos_col = next((col for col in lexicon.columns if col.lower() == "positive"), None)
            neg_col = next((col for col in lexicon.columns if col.lower() == "negative"), None)
            if not word_col or not pos_col or not neg_col:
                raise ValueError("missing Word/Positive/Negative columns")
            positive = set(lexicon.loc[lexicon[pos_col] > 0, word_col].astype(str).str.lower())
            negative = set(lexicon.loc[lexicon[neg_col] > 0, word_col].astype(str).str.lower())
            return positive | fallback_positive, negative | fallback_negative
        except Exception as exc:
            self.logger.warning("Failed to parse lexicon, using fallback keywords: %s", exc)
            return fallback_positive, fallback_negative

    def _count_positive(self, text: object) -> int:
        return self._count_terms(text, self._positive_ascii, self._positive_phrases)

    def _count_negative(self, text: object) -> int:
        return self._count_terms(text, self._negative_ascii, self._negative_phrases)

    def _count_terms(self, text: object, ascii_terms: set[str], phrase_terms: set[str]) -> int:
        lowered = str(text or "").lower()
        tokens = set(re.findall(r"[a-z][a-z'\-]+", lowered))
        token_hits = len(tokens & ascii_terms)
        phrase_hits = sum(term in lowered for term in phrase_terms)
        return int(token_hits + phrase_hits)

    def _classify_text_source(self, source: object) -> str:
        lowered = str(source or "").lower()
        if "policy" in lowered:
            return "policy"
        if "notice" in lowered or "cninfo" in lowered:
            return "notice"
        if "report" in lowered or "research" in lowered:
            return "report"
        if "guba" in lowered:
            return "guba"
        return "news"

    def _is_official_text_source(self, source: object, publisher: object, url: object) -> bool:
        lowered_source = str(source or "").lower()
        lowered_publisher = str(publisher or "").lower()
        lowered_url = str(url or "").lower()
        if "policy" in lowered_source:
            return True
        if any(domain in lowered_url for domain in self.cfg.data.policy_official_domains):
            return True
        official_tokens = [
            "gov.cn",
            "miit.gov.cn",
            "nea.gov.cn",
            "ndrc.gov.cn",
            "\u56fd\u52a1\u9662",
            "\u5de5\u4fe1\u90e8",
            "\u5de5\u4e1a\u548c\u4fe1\u606f\u5316\u90e8",
            "\u56fd\u5bb6\u80fd\u6e90\u5c40",
            "\u53d1\u6539\u59d4",
            "\u56fd\u5bb6\u53d1\u5c55\u6539\u9769\u59d4",
        ]
        return any(token.lower() in lowered_publisher for token in official_tokens)
