# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: Med-MARAG
File Name: app.py
Author: SeaOcean
Create Date: 2026-03-18
Description：Streamlit 可视化前端入口
-------------------------------------------------
"""
from __future__ import annotations

import html
import json
import re
from typing import Any
from textwrap import dedent

try:
    import pandas as pd
    import streamlit as st
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "缺少 Streamlit 运行依赖，请先安装 requirements.txt 后执行 `streamlit run app.py`。"
    ) from exc

from config import Config
from workflow.ocean_graph import PipelineConfig, run_modeller_pipeline


st.set_page_config(
    page_title="Med-MARAG",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_state() -> None:
    if "pipeline_result" not in st.session_state:
        st.session_state.pipeline_result = None
    if "requirement_input" not in st.session_state:
        st.session_state.requirement_input = ""


def build_requirement_df(result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "需求ID": item.get("id"),
                "领域": item.get("domain"),
                "主要参与者": item.get("primary_actor"),
                "触发条件": item.get("condition"),
                "核心动作": "；".join(item.get("actions", [])),
                "质量评分": item.get("validation", {}).get("score"),
            }
            for item in result.get("Requirement_Items", [])
        ]
    )


def _parse_markdown_key_value_table(content_md: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for raw_line in (content_md or "").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [part.strip() for part in re.split(r"(?<!\\)\|", stripped[1:-1])]
            cells = [cell.replace("\\|", "|") for cell in cells]
            if len(cells) < 2:
                continue
            if all(set(cell) <= {"-", ":", " "} for cell in cells[:2]):
                continue
            if cells[0] == "主要元素" and cells[1] == "说明":
                continue
            rows.append((cells[0], cells[1]))
            continue
        if rows:
            left, right = rows[-1]
            rows[-1] = (left, f"{right}\n{stripped}")
    return rows


def _format_artifact_cell(value: str) -> str:
    normalized = (
        (value or "")
        .replace("<br />", "\n")
        .replace("<br/>", "\n")
        .replace("<br>", "\n")
    )
    return html.escape(normalized).replace("\n", "<br>")


def render_artifact_table(content_md: str) -> None:
    rows = _parse_markdown_key_value_table(content_md)
    if not rows:
        st.markdown(content_md or "")
        return

    body = "\n".join(
        (
            "<tr>"
            f"<th>{_format_artifact_cell(left)}</th>"
            f"<td>{_format_artifact_cell(right)}</td>"
            "</tr>"
        )
        for left, right in rows
    )
    st.markdown(
        f"""
        <div class="artifact-table-wrap">
            <table class="artifact-table">
                <thead>
                    <tr><th>主要元素</th><th>说明</th></tr>
                </thead>
                <tbody>
                    {body}
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_rag_match(distance: Any) -> str:
    try:
        score = max(0, min(100, int(round((1 - float(distance)) * 100))))
        return f"{score}% 匹配"
    except (TypeError, ValueError):
        return "已检索"


def _clean_rag_display_text(text: Any) -> str:
    cleaned = str(text or "").replace("\u00a0", " ")
    cleaned = re.sub(r"[\uf000-\uf8ff]", " ", cleaned)
    cleaned = re.sub(r"[•●○◦▪▫■□◆◇▶▷‣⁃∙]", " ", cleaned)
    cleaned = re.sub(r"^\s*\d+(?:\.\d+){1,6}\s+(.+?)(?:\n|$)", r"\1\n", cleaned, count=1)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def render_rag_results(knowledge_context: list[dict[str, Any]]) -> None:
    if not knowledge_context:
        st.info("当前运行未返回额外知识片段。")
        return

    for index, item in enumerate(knowledge_context, start=1):
        metadata = item.get("metadata", {}) or {}
        source = str(metadata.get("source") or metadata.get("type") or "知识库片段")
        section = str(metadata.get("section") or metadata.get("domain") or "").strip()
        section_title = str(metadata.get("section_title") or "").strip()
        doc_type = str(metadata.get("type") or "").strip()
        tags = metadata.get("tags", []) or []
        match_text = _format_rag_match(item.get("distance"))
        tags_html = "".join(
            f'<span class="rag-tag">{html.escape(str(tag))}</span>'
            for tag in tags[:6]
        )
        meta_html = "".join(
            (
                f'<span class="rag-meta-chip">{html.escape(text)}</span>'
                for text in [section, section_title, doc_type]
                if text
            )
        )
        content = html.escape(_clean_rag_display_text(item.get("content", "")))
        doc_id = html.escape(str(item.get("id", f"RAG-{index}")))

        st.markdown(
            f"""
            <div class="rag-card">
                <div class="rag-card-head">
                    <div>
                        <div class="rag-card-kicker">Evidence {index:02d}</div>
                        <div class="rag-card-title">{html.escape(source)}</div>
                    </div>
                    <div class="rag-match-badge">{html.escape(match_text)}</div>
                </div>
                <div class="rag-meta-row">
                    <span class="rag-meta-chip">ID · {doc_id}</span>
                    {meta_html}
                </div>
                <div class="rag-card-body">{content}</div>
                {f'<div class="rag-tag-row">{tags_html}</div>' if tags_html else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )


def _format_review_score_label(key: str) -> str:
    mapping = {
        "overall": "总体得分",
        "diagram_consistency": "图模型一致性",
        "compliance": "规范符合度",
        "domain_knowledge_alignment": "领域知识对齐",
        "coordination_bonus": "协同加成",
        "accuracy": "准确性",
        "completeness": "完整性",
        "clarity": "清晰性",
    }
    return mapping.get(key, key.replace("_", " ").title())


def _format_review_source(source: str) -> str:
    return {
        "llm": "LLM 审查",
        "local": "本地规则审查",
    }.get((source or "").strip().lower(), source or "未知")


def _coerce_review_score(value: Any) -> int | None:
    try:
        return max(0, min(100, int(float(value))))
    except (TypeError, ValueError):
        return None


def _review_score_tone(score: int | None) -> tuple[str, str]:
    if score is None:
        return "review-score-neutral", "未评分"
    if score >= 90:
        return "review-score-strong", "表现稳定"
    if score >= 75:
        return "review-score-good", "整体良好"
    if score >= 60:
        return "review-score-watch", "需要关注"
    return "review-score-risk", "优先修正"


def _render_review_score_cards(scores: dict[str, Any]) -> None:
    if not scores:
        return
    cards = []
    for key, value in scores.items():
        score = _coerce_review_score(value)
        tone_class, caption = _review_score_tone(score)
        score_text = f"{score}" if score is not None else html.escape(str(value))
        width = score if score is not None else 0
        cards.append(
            f"""
            <div class="review-score-card {tone_class}">
                <div class="review-score-head">
                    <div class="review-score-label">{html.escape(_format_review_score_label(str(key)))}</div>
                    <div class="review-score-badge">{score_text}</div>
                </div>
                <div class="review-score-track">
                    <span style="width: {width}%;"></span>
                </div>
                <div class="review-score-caption">{html.escape(caption)}</div>
            </div>
            """
        )
    st.markdown(
        f'<div class="review-score-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def _render_review_issue_panel(title: str, items: list[str], tone: str, empty_text: str) -> None:
    tone_class = {
        "critical": "review-panel-critical",
        "warning": "review-panel-warning",
        "success": "review-panel-success",
        "neutral": "review-panel-neutral",
    }.get(tone, "review-panel-neutral")
    has_items = bool(items)
    rows = items or [empty_text]
    if has_items:
        list_html = "".join(f"<li>{html.escape(str(item))}</li>" for item in rows)
        content_html = f'<ul class="review-panel-list">{list_html}</ul>'
    else:
        content_html = f'<div class="review-panel-empty">{html.escape(empty_text)}</div>'
    st.markdown(
        f"""
        <div class="review-panel {tone_class}">
            <div class="review-panel-head">
                <div class="review-panel-title">{html.escape(title)}</div>
                <div class="review-panel-count">{len(items)}</div>
            </div>
            {content_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_review_overview(result: dict[str, Any], review_report: dict[str, Any]) -> None:
    source = _format_review_source(str(review_report.get("review_source", "local")))
    status = str(result.get("Status", "UNKNOWN"))
    iteration = int(result.get("Iteration_Count", 0) or 0)
    target = str(result.get("Review_Target", "") or "")
    blocking_count = len(result.get("Review_Blocking_Issues", []) or [])
    warning_count = len(result.get("Review_Warnings", []) or [])
    target_text = {"analyst": "Analyst Agent", "architect": "Architect Agent"}.get(target, "无")
    overall_score = _coerce_review_score((review_report.get("scores", {}) or {}).get("overall"))
    overall_tone_class, overall_caption = _review_score_tone(overall_score)
    overall_display = str(overall_score) if overall_score is not None else "N/A"
    meta_items = [
        ("审查来源", source),
        ("流程状态", status),
        ("修正轮次", str(iteration)),
        ("当前回退目标", target_text),
        ("阻断问题", str(blocking_count)),
        ("警告问题", str(warning_count)),
    ]
    meta_html = "".join(
        f"""
        <div class="review-meta-chip">
            <span class="review-meta-key">{html.escape(label)}</span>
            <span class="review-meta-value">{html.escape(value)}</span>
        </div>
        """
        for label, value in meta_items
    )
    st.markdown(
        f"""
        <div class="review-overview-card {overall_tone_class}">
            <div class="review-overview-top">
                <div>
                    <div class="review-overview-kicker">Review Snapshot</div>
                    <div class="review-overview-title">审查结果总览</div>
                </div>
                <div class="review-overall-score">
                    <div class="review-overall-value">{html.escape(overall_display)}</div>
                    <div class="review-overall-caption">{html.escape(overall_caption)}</div>
                </div>
            </div>
            <div class="review-meta-row">{meta_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_review_detail_block(name: str, detail: dict[str, Any]) -> None:
    st.markdown(
        f"""
        <div class="review-detail-summary">
            <div class="review-detail-title">{html.escape(name)}</div>
            <div class="review-detail-chip">{html.escape(_format_review_source(str(detail.get('review_source', 'local'))))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_review_score_cards(detail.get("scores", {}) or {})
    detail_cols = st.columns(2)
    with detail_cols[0]:
        _render_review_issue_panel(
            "问题列表",
            detail.get("issues", []) or [],
            "warning",
            "当前详情项未发现问题。",
        )
    with detail_cols[1]:
        _render_review_issue_panel(
            "优化建议",
            detail.get("suggestions", []) or [],
            "neutral",
            "当前详情项未生成额外建议。",
        )


def inject_custom_styles() -> None:
    st.markdown(
        dedent(
            """
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&display=swap');

                :root {
                    --app-bg: #f4f7f6;
                    --surface: rgba(255, 255, 255, 0.82);
                    --surface-strong: #ffffff;
                    --surface-soft: #eef4f3;
                    --text-main: #172126;
                    --text-muted: #5f6f73;
                    --border-soft: rgba(23, 33, 38, 0.10);
                    --border-strong: rgba(23, 33, 38, 0.16);
                    --accent: #1f8c81;
                    --accent-strong: #16695f;
                    --accent-soft: rgba(31, 140, 129, 0.12);
                    --shadow-soft: 0 20px 60px rgba(34, 69, 70, 0.10);
                    --radius-lg: 28px;
                    --radius-md: 20px;
                    --radius-sm: 14px;
                }

                .stApp {
                    background:
                        radial-gradient(circle at top left, rgba(31, 140, 129, 0.16), transparent 26%),
                        radial-gradient(circle at top right, rgba(23, 33, 38, 0.08), transparent 22%),
                        linear-gradient(180deg, #f8fbfa 0%, var(--app-bg) 58%, #eef4f3 100%);
                    color: var(--text-main);
                }

                [data-testid="stHeader"] {
                    background: transparent;
                }

                [data-testid="stAppViewContainer"] > .main {
                    background: transparent;
                }

                .block-container {
                    max-width: 1180px;
                    padding-top: 2.6rem;
                    padding-bottom: 4.5rem;
                }

                [data-testid="stSidebar"] {
                    background: rgba(255, 255, 255, 0.72);
                    border-right: 1px solid var(--border-soft);
                    backdrop-filter: blur(18px);
                }

                [data-testid="stSidebar"] > div:first-child {
                    background: transparent;
                }

                .sidebar-brand {
                    padding: 1.2rem 1.25rem;
                    border: 1px solid var(--border-soft);
                    border-radius: var(--radius-md);
                    background: linear-gradient(180deg, rgba(255,255,255,0.94), rgba(244,247,246,0.92));
                    box-shadow: var(--shadow-soft);
                    margin-bottom: 1rem;
                }

                .sidebar-brand p,
                .sidebar-brand span,
                .sidebar-brand strong,
                .sidebar-brand div {
                    color: var(--text-main);
                }

                .sidebar-kicker {
                    display: inline-flex;
                    align-items: center;
                    gap: 0.35rem;
                    font-size: 0.74rem;
                    letter-spacing: 0.12em;
                    text-transform: uppercase;
                    color: var(--accent);
                    margin-bottom: 0.75rem;
                }

                .sidebar-title {
                    font-size: 1.05rem;
                    font-weight: 700;
                    margin-bottom: 0.45rem;
                }

                .sidebar-body {
                    font-size: 0.9rem;
                    color: var(--text-muted);
                    line-height: 1.65;
                }

                .hero-shell {
                    position: relative;
                    overflow: hidden;
                    padding: 2.8rem;
                    border: 1px solid var(--border-soft);
                    border-radius: 34px;
                    background:
                        linear-gradient(135deg, rgba(255,255,255,0.96), rgba(244,247,246,0.92)),
                        linear-gradient(120deg, rgba(31,140,129,0.08), transparent 55%);
                    box-shadow: var(--shadow-soft);
                }

                .hero-shell::after {
                    content: "";
                    position: absolute;
                    top: -4rem;
                    right: -4rem;
                    width: 16rem;
                    height: 16rem;
                    border-radius: 999px;
                    background: radial-gradient(circle, rgba(31,140,129,0.16), transparent 70%);
                    pointer-events: none;
                }

                .eyebrow {
                    display: inline-flex;
                    align-items: center;
                    gap: 0.45rem;
                    padding: 0.45rem 0.8rem;
                    border-radius: 999px;
                    background: var(--accent-soft);
                    color: var(--accent-strong);
                    font-size: 0.77rem;
                    font-weight: 700;
                    letter-spacing: 0.12em;
                    text-transform: uppercase;
                }

                .hero-title {
                    margin: 1rem 0 0;
                    font-size: clamp(2.4rem, 4vw, 4.6rem);
                    line-height: 1.02;
                    letter-spacing: -0.02em;
                    color: var(--text-main);
                    font-family: "Cormorant Garamond", "Times New Roman", Georgia, serif;
                    font-weight: 700;
                    text-transform: none;
                }

                .hero-subtitle {
                    max-width: 44rem;
                    margin-top: 1.15rem;
                    font-size: 1.06rem;
                    line-height: 1.85;
                    color: var(--text-muted);
                }

                .hero-chip-row {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 0.7rem;
                    margin-top: 1.4rem;
                }

                .hero-chip {
                    padding: 0.65rem 0.9rem;
                    border: 1px solid var(--border-soft);
                    border-radius: 999px;
                    background: rgba(255, 255, 255, 0.72);
                    color: var(--text-main);
                    font-size: 0.9rem;
                }

                .feature-card {
                    height: 100%;
                    padding: 1.35rem 1.4rem;
                    border: 1px solid var(--border-soft);
                    border-radius: var(--radius-md);
                    background: rgba(255, 255, 255, 0.74);
                    box-shadow: 0 12px 36px rgba(34, 69, 70, 0.06);
                }

                .feature-label {
                    font-size: 0.78rem;
                    letter-spacing: 0.1em;
                    text-transform: uppercase;
                    color: var(--accent);
                    margin-bottom: 0.75rem;
                }

                .feature-title {
                    font-size: 1.02rem;
                    font-weight: 700;
                    color: var(--text-main);
                    margin-bottom: 0.5rem;
                }

                .feature-copy {
                    font-size: 0.94rem;
                    line-height: 1.72;
                    color: var(--text-muted);
                }

                .section-head {
                    margin-top: 2rem;
                    padding-top: 2rem;
                    border-top: 1px solid var(--border-soft);
                }

                .section-head.no-divider {
                    margin-top: 1.25rem;
                    padding-top: 0;
                    border-top: none;
                }

                .section-kicker {
                    font-size: 0.78rem;
                    letter-spacing: 0.12em;
                    text-transform: uppercase;
                    color: var(--accent);
                    margin-bottom: 0.55rem;
                }

                .section-title {
                    font-size: 1.85rem;
                    font-weight: 700;
                    letter-spacing: -0.03em;
                    color: var(--text-main);
                    margin-bottom: 0.45rem;
                }

                .section-copy {
                    max-width: 44rem;
                    font-size: 0.98rem;
                    line-height: 1.78;
                    color: var(--text-muted);
                }

                .summary-card {
                    padding: 1.2rem 1.25rem;
                    border: 1px solid var(--border-soft);
                    border-radius: var(--radius-md);
                    background: rgba(255, 255, 255, 0.72);
                    box-shadow: 0 10px 30px rgba(34, 69, 70, 0.06);
                }

                .summary-label {
                    font-size: 0.8rem;
                    letter-spacing: 0.1em;
                    text-transform: uppercase;
                    color: var(--text-muted);
                }

                .summary-value {
                    margin-top: 0.65rem;
                    font-size: 2rem;
                    line-height: 1;
                    font-weight: 700;
                    color: var(--text-main);
                }

                .summary-note {
                    margin-top: 0.55rem;
                    font-size: 0.92rem;
                    color: var(--text-muted);
                    line-height: 1.65;
                }

                .status-pill {
                    display: inline-flex;
                    align-items: center;
                    gap: 0.45rem;
                    padding: 0.55rem 0.82rem;
                    border-radius: 999px;
                    background: var(--accent-soft);
                    color: var(--accent-strong);
                    font-size: 0.82rem;
                    font-weight: 700;
                }

                h1, h2, h3, h4, h5, h6 {
                    color: var(--text-main);
                    letter-spacing: -0.02em;
                }

                p, li, label, span {
                    color: var(--text-main);
                }

                .stCaption {
                    color: var(--text-muted);
                }

                [data-testid="stMetric"] {
                    background: rgba(255, 255, 255, 0.74);
                    border: 1px solid var(--border-soft);
                    border-radius: var(--radius-md);
                    padding: 1rem 1.1rem;
                    box-shadow: 0 10px 28px rgba(34, 69, 70, 0.05);
                }

                [data-testid="stMetricLabel"],
                [data-testid="stMetricValue"] {
                    color: var(--text-main);
                }

                [data-testid="stMetricDelta"] {
                    color: var(--accent);
                }

                .stTextArea textarea,
                .stTextInput input,
                .stNumberInput input,
                div[data-baseweb="select"] > div {
                    background: rgba(255, 255, 255, 0.88);
                    color: var(--text-main);
                    border-radius: 18px;
                    border: 1px solid var(--border-strong);
                    box-shadow: 0 10px 30px rgba(34, 69, 70, 0.05);
                }

                .stTextArea textarea:focus,
                .stTextInput input:focus {
                    border-color: rgba(31, 140, 129, 0.45);
                    box-shadow: 0 0 0 0.2rem rgba(31, 140, 129, 0.12);
                }

                .stSlider [data-baseweb="slider"] {
                    padding-top: 0.5rem;
                    padding-bottom: 0.2rem;
                }

                .stButton > button,
                .stDownloadButton > button {
                    min-height: 3rem;
                    border: 1px solid transparent;
                    border-radius: 999px;
                    background: linear-gradient(135deg, var(--accent), var(--accent-strong));
                    color: #ffffff;
                    font-weight: 700;
                    box-shadow: 0 14px 30px rgba(31, 140, 129, 0.24);
                }

                .stButton > button:hover,
                .stDownloadButton > button:hover {
                    border-color: transparent;
                    background: linear-gradient(135deg, #279b8f, #175f56);
                    color: #ffffff;
                }

                .stTabs [data-baseweb="tab-list"] {
                    gap: 0.6rem;
                    background: rgba(255, 255, 255, 0.5);
                    border: 1px solid var(--border-soft);
                    border-radius: 18px;
                    padding: 0.45rem;
                }

                .stTabs [data-baseweb="tab"] {
                    height: auto;
                    padding: 0.7rem 1rem;
                    border-radius: 14px;
                    color: var(--text-muted);
                    background: transparent;
                }

                .stTabs [aria-selected="true"] {
                    background: rgba(255, 255, 255, 0.92);
                    color: var(--text-main);
                    box-shadow: 0 8px 24px rgba(34, 69, 70, 0.08);
                }

                details[data-testid="stExpander"] {
                    border: 1px solid var(--border-soft);
                    border-radius: 18px;
                    background: rgba(255, 255, 255, 0.72);
                    box-shadow: 0 10px 28px rgba(34, 69, 70, 0.05);
                }

                details[data-testid="stExpander"] summary {
                    padding-top: 0.25rem;
                    padding-bottom: 0.25rem;
                }

                div[data-testid="stDataFrame"],
                .stCodeBlock,
                pre {
                    border-radius: 18px;
                }

                div[data-testid="stDataFrame"] {
                    border: 1px solid var(--border-soft);
                    overflow: hidden;
                    box-shadow: 0 10px 30px rgba(34, 69, 70, 0.05);
                }

                .stCodeBlock,
                pre {
                    border: 1px solid rgba(23, 33, 38, 0.08);
                    background: #f8faf9;
                }

                [data-testid="stAlert"] {
                    border-radius: 18px;
                    border: 1px solid var(--border-soft);
                }

                .artifact-table-wrap {
                    overflow-x: auto;
                    border: 1px solid var(--border-soft);
                    border-radius: 18px;
                    background: rgba(255, 255, 255, 0.76);
                    box-shadow: 0 10px 30px rgba(34, 69, 70, 0.05);
                }

                .artifact-table {
                    width: 100%;
                    border-collapse: collapse;
                    table-layout: fixed;
                }

                .artifact-table thead th {
                    padding: 0.95rem 1rem;
                    background: rgba(31, 140, 129, 0.08);
                    color: var(--text-main);
                    font-weight: 700;
                    text-align: left;
                    border-bottom: 1px solid var(--border-soft);
                }

                .artifact-table tbody th,
                .artifact-table tbody td {
                    padding: 1rem;
                    vertical-align: top;
                    border-top: 1px solid var(--border-soft);
                    color: var(--text-main);
                    line-height: 1.8;
                    white-space: normal;
                    word-break: break-word;
                    overflow-wrap: anywhere;
                }

                .artifact-table tbody th {
                    width: 24%;
                    min-width: 170px;
                    background: rgba(244, 247, 246, 0.72);
                    font-weight: 700;
                }

                .artifact-table tbody td {
                    width: 76%;
                    background: rgba(255, 255, 255, 0.78);
                }

                .review-overview-card {
                    margin-bottom: 1rem;
                    padding: 1.35rem 1.4rem;
                    border: 1px solid var(--border-soft);
                    border-radius: 24px;
                    background:
                        linear-gradient(135deg, rgba(255,255,255,0.96), rgba(244,247,246,0.9)),
                        linear-gradient(120deg, rgba(31,140,129,0.08), transparent 60%);
                    box-shadow: 0 14px 36px rgba(34, 69, 70, 0.08);
                }

                .review-overview-top {
                    display: flex;
                    align-items: flex-start;
                    justify-content: space-between;
                    gap: 1rem;
                }

                .review-overview-kicker {
                    font-size: 0.78rem;
                    letter-spacing: 0.12em;
                    text-transform: uppercase;
                    color: var(--accent);
                    margin-bottom: 0.55rem;
                }

                .review-overview-title {
                    font-size: 1.3rem;
                    font-weight: 700;
                    color: var(--text-main);
                }

                .review-overall-score {
                    min-width: 112px;
                    padding: 0.95rem 1rem;
                    border-radius: 22px;
                    background: rgba(255, 255, 255, 0.72);
                    border: 1px solid var(--border-soft);
                    text-align: center;
                }

                .review-overall-value {
                    font-size: 2.3rem;
                    line-height: 1;
                    font-weight: 700;
                    color: var(--text-main);
                }

                .review-overall-caption {
                    margin-top: 0.45rem;
                    font-size: 0.82rem;
                    color: var(--text-muted);
                }

                .review-meta-row {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 0.7rem;
                    margin-top: 1rem;
                }

                .review-meta-chip {
                    display: inline-flex;
                    align-items: center;
                    gap: 0.55rem;
                    padding: 0.65rem 0.85rem;
                    border: 1px solid var(--border-soft);
                    border-radius: 999px;
                    background: rgba(255, 255, 255, 0.72);
                }

                .review-meta-key {
                    font-size: 0.82rem;
                    color: var(--text-muted);
                }

                .review-meta-value {
                    font-size: 0.9rem;
                    font-weight: 700;
                    color: var(--text-main);
                }

                .review-score-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                    gap: 0.9rem;
                    margin: 1rem 0 1.15rem;
                }

                .review-score-card {
                    padding: 1.1rem 1.15rem;
                    border: 1px solid var(--border-soft);
                    border-radius: 22px;
                    background: rgba(255, 255, 255, 0.78);
                    box-shadow: 0 10px 28px rgba(34, 69, 70, 0.05);
                }

                .review-score-head {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    gap: 0.75rem;
                }

                .review-score-label {
                    font-size: 0.82rem;
                    letter-spacing: 0.08em;
                    color: var(--text-muted);
                }

                .review-score-badge {
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    min-width: 3rem;
                    padding: 0.35rem 0.7rem;
                    border-radius: 999px;
                    background: rgba(255, 255, 255, 0.9);
                    border: 1px solid var(--border-soft);
                    font-size: 1rem;
                    font-weight: 700;
                    color: var(--text-main);
                }

                .review-score-track {
                    margin-top: 0.9rem;
                    height: 0.55rem;
                    border-radius: 999px;
                    background: rgba(23, 33, 38, 0.08);
                    overflow: hidden;
                }

                .review-score-track span {
                    display: block;
                    height: 100%;
                    border-radius: inherit;
                    background: linear-gradient(90deg, var(--accent), #39a89c);
                }

                .review-score-caption {
                    margin-top: 0.65rem;
                    font-size: 0.85rem;
                    color: var(--text-muted);
                }

                .review-score-strong {
                    background: linear-gradient(180deg, rgba(240, 251, 246, 0.95), rgba(255, 255, 255, 0.82));
                }

                .review-score-good {
                    background: linear-gradient(180deg, rgba(244, 251, 250, 0.95), rgba(255, 255, 255, 0.82));
                }

                .review-score-watch {
                    background: linear-gradient(180deg, rgba(255, 249, 239, 0.95), rgba(255, 255, 255, 0.82));
                }

                .review-score-risk {
                    background: linear-gradient(180deg, rgba(255, 243, 241, 0.95), rgba(255, 255, 255, 0.82));
                }

                .review-score-neutral {
                    background: linear-gradient(180deg, rgba(247, 250, 249, 0.95), rgba(255, 255, 255, 0.82));
                }

                .review-panel {
                    height: 100%;
                    padding: 1.15rem 1.2rem;
                    border-radius: 22px;
                    border: 1px solid var(--border-soft);
                    background: rgba(255, 255, 255, 0.76);
                    box-shadow: 0 10px 28px rgba(34, 69, 70, 0.05);
                }

                .review-panel-head {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    gap: 0.8rem;
                    margin-bottom: 0.75rem;
                }

                .review-panel-title {
                    font-size: 1rem;
                    font-weight: 700;
                    color: var(--text-main);
                }

                .review-panel-count {
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    min-width: 2rem;
                    height: 2rem;
                    padding: 0 0.55rem;
                    border-radius: 999px;
                    background: rgba(255, 255, 255, 0.85);
                    border: 1px solid var(--border-soft);
                    font-size: 0.88rem;
                    font-weight: 700;
                    color: var(--text-main);
                }

                .review-panel-list {
                    margin: 0;
                    padding-left: 1.1rem;
                }

                .review-panel-list li {
                    margin-bottom: 0.55rem;
                    color: var(--text-main);
                    line-height: 1.7;
                }

                .review-panel-empty {
                    padding: 0.2rem 0;
                    color: var(--text-muted);
                    line-height: 1.75;
                }

                .review-panel-critical {
                    background: linear-gradient(180deg, rgba(255, 244, 243, 0.96), rgba(255, 255, 255, 0.86));
                    border-color: rgba(196, 82, 70, 0.16);
                }

                .review-panel-warning {
                    background: linear-gradient(180deg, rgba(255, 249, 239, 0.96), rgba(255, 255, 255, 0.86));
                    border-color: rgba(210, 154, 61, 0.18);
                }

                .review-panel-success {
                    background: linear-gradient(180deg, rgba(241, 251, 247, 0.96), rgba(255, 255, 255, 0.86));
                    border-color: rgba(31, 140, 129, 0.18);
                }

                .review-panel-neutral {
                    background: linear-gradient(180deg, rgba(247, 250, 249, 0.96), rgba(255, 255, 255, 0.86));
                }

                .review-feedback-card {
                    margin-top: 1rem;
                    padding: 1.1rem 1.2rem;
                    border: 1px solid var(--border-soft);
                    border-radius: 24px;
                    background: linear-gradient(180deg, rgba(255, 255, 255, 0.9), rgba(245, 248, 247, 0.84));
                    box-shadow: 0 12px 28px rgba(34, 69, 70, 0.05);
                }

                .review-feedback-title {
                    font-size: 0.95rem;
                    font-weight: 700;
                    margin-bottom: 0.55rem;
                    color: var(--text-main);
                }

                .review-feedback-body {
                    color: var(--text-muted);
                    line-height: 1.8;
                    white-space: pre-wrap;
                }

                .review-detail-summary {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    gap: 0.8rem;
                    margin-bottom: 0.8rem;
                }

                .review-detail-title {
                    font-size: 1rem;
                    font-weight: 700;
                    color: var(--text-main);
                }

                .review-detail-chip {
                    display: inline-flex;
                    align-items: center;
                    padding: 0.35rem 0.7rem;
                    border-radius: 999px;
                    background: var(--accent-soft);
                    color: var(--accent-strong);
                    font-size: 0.78rem;
                    font-weight: 700;
                }

                .rag-card {
                    margin-bottom: 1rem;
                    padding: 1.15rem 1.2rem;
                    border: 1px solid var(--border-soft);
                    border-radius: 24px;
                    background:
                        linear-gradient(180deg, rgba(255, 255, 255, 0.92), rgba(244, 247, 246, 0.84));
                    box-shadow: 0 12px 28px rgba(34, 69, 70, 0.05);
                }

                .rag-card-head {
                    display: flex;
                    align-items: flex-start;
                    justify-content: space-between;
                    gap: 1rem;
                }

                .rag-card-kicker {
                    font-size: 0.75rem;
                    letter-spacing: 0.12em;
                    text-transform: uppercase;
                    color: var(--accent);
                    margin-bottom: 0.35rem;
                }

                .rag-card-title {
                    font-size: 1.02rem;
                    font-weight: 700;
                    color: var(--text-main);
                    line-height: 1.45;
                }

                .rag-match-badge {
                    display: inline-flex;
                    align-items: center;
                    padding: 0.45rem 0.75rem;
                    border-radius: 999px;
                    background: var(--accent-soft);
                    color: var(--accent-strong);
                    font-size: 0.8rem;
                    font-weight: 700;
                    white-space: nowrap;
                }

                .rag-meta-row {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 0.55rem;
                    margin-top: 0.9rem;
                }

                .rag-meta-chip,
                .rag-tag {
                    display: inline-flex;
                    align-items: center;
                    padding: 0.35rem 0.7rem;
                    border-radius: 999px;
                    border: 1px solid var(--border-soft);
                    background: rgba(255, 255, 255, 0.78);
                    color: var(--text-muted);
                    font-size: 0.78rem;
                }

                .rag-card-body {
                    margin-top: 1rem;
                    padding: 1rem 1.05rem;
                    border-radius: 18px;
                    background: rgba(248, 251, 250, 0.96);
                    color: var(--text-main);
                    line-height: 1.85;
                    white-space: pre-wrap;
                }

                .rag-tag-row {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 0.55rem;
                    margin-top: 0.9rem;
                }

                @media (max-width: 900px) {
                    .block-container {
                        padding-top: 1.5rem;
                    }

                    .hero-shell {
                        padding: 1.5rem;
                        border-radius: 24px;
                    }

                    .hero-title {
                        font-size: 2.3rem;
                    }
                }
            </style>
            """
        ),
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-title">Med-MARAG Workspace</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero_section() -> None:
    st.markdown(
        """
        <section class="hero-shell">
            <h1 class="hero-title">Med-MARAG</h1>
            <p class="hero-subtitle">
                融合大模型多智能体与检索增强生成的医疗需求工程智能化建模平台
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    feature_cards = [

    ]
    if feature_cards:
        feature_cols = st.columns(len(feature_cards))
        for col, (label, title, copy) in zip(feature_cols, feature_cards):
            with col:
                st.markdown(
                    f"""
                    <div class="feature-card">
                        <div class="feature-label">{label}</div>
                        <div class="feature-title">{title}</div>
                        <div class="feature-copy">{copy}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_section_header(kicker: str, title: str, *, show_divider: bool = True) -> None:
    st.markdown(
        f"""
        <div class="section-head{' no-divider' if not show_divider else ''}">
            <div class="section-kicker">{kicker}</div>
            <div class="section-title">{title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_overview(result: dict[str, Any]) -> None:
    summary_items = [
        (
            "Requirements",
            len(result.get("Requirement_Items", [])),
            "标准化需求条目",
        ),
        (
            "Use Cases",
            len(result.get("Use_Cases", [])),
            "生成的业务用例数量",
        ),
        (
            "Tests",
            len(result.get("Test_Cases", [])),
            "可直接查看的测试用例",
        ),
        (
            "Knowledge",
            len(result.get("Knowledge_Context", [])),
            "本次关联的 RAG 片段",
        ),
    ]
    cols = st.columns(len(summary_items))
    for col, (label, value, note) in zip(cols, summary_items):
        with col:
            st.markdown(
                f"""
                <div class="summary-card">
                    <div class="summary-label">{label}</div>
                    <div class="summary-value">{value}</div>
                    <div class="summary-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_result_tabs(result: dict[str, Any]) -> None:
    tabs = st.tabs(["EARS", "用例", "RA制品", "审查", "测试", "追溯", "RAG", "导出"])

    with tabs[0]:
        st.subheader("EARS 标准化需求")
        st.code(result.get("EARS_Requirement", ""), language="text")
        df = build_requirement_df(result)
        if not df.empty:
            st.dataframe(df, use_container_width=True)
        requirement_items = result.get("Requirement_Items", [])
        llm_items = [item for item in requirement_items if item.get("llm_summary")]
        if llm_items:
            st.markdown("**LLM 分析说明**")
            for item in llm_items:
                st.markdown(f"- `{item.get('id')}`：{item.get('llm_summary')}")
                risks = item.get("llm_risks", []) or []
                if risks:
                    st.caption("风险提示：" + "；".join(str(risk) for risk in risks))

    with tabs[1]:
        st.subheader("用例制品")
        for use_case in result.get("Use_Cases", []):
            with st.expander(f"{use_case.get('id')} · {use_case.get('name')}", expanded=True):
                st.write(f"主要参与者：{use_case.get('primary_actor')}")
                st.write(f"业务目标：{use_case.get('goal')}")
                st.write("前置条件：")
                for item in use_case.get("preconditions", []):
                    st.markdown(f"- {item}")
                st.write("主成功场景：")
                for item in use_case.get("main_flow", []):
                    st.markdown(f"- {item}")
                st.write("后置条件：")
                for item in use_case.get("postconditions", []):
                    st.markdown(f"- {item}")

    with tabs[2]:
        st.subheader("RA-1 至 RA-6 产出制品")
        ra = result.get("RA_Artifacts", {}) or {}
        uml_artifacts = result.get("UML_Artifacts", {}) or {}
        ra_valid = (result.get("RA_Validation", {}) or {}).get("valid", True)
        ra_issues = (result.get("RA_Validation", {}) or {}).get("issues", []) or []
        if ra.get("standard_sources"):
            st.caption(f"规范来源：{', '.join(ra.get('standard_sources', []))}")
        if not ra_valid:
            st.warning("RA 制品未完全通过模板校验，请优先修正以下问题：")
            for issue in ra_issues:
                st.markdown(f"- {issue}")

        ra1 = ra.get("RA-1", {}) or {}
        st.markdown("**RA-1 用例图**")
        st.caption(f"制品来源：{ra1.get('source', uml_artifacts.get('use_case_diagram_source', 'local'))}")
        st.code(ra1.get("plantuml", ""), language="plantuml")
        if ra1.get("url"):
            st.markdown(f"[查看渲染链接]({ra1.get('url')})")
        if uml_artifacts.get("llm_use_case_diagram"):
            with st.expander("查看本地版 / LLM版 用例图对比", expanded=False):
                st.markdown("**本地规则版**")
                st.code(uml_artifacts.get("local_use_case_diagram", ""), language="plantuml")
                st.markdown("**LLM 增强版**")
                st.code(uml_artifacts.get("llm_use_case_diagram", ""), language="plantuml")

        st.markdown("**RA-2 高层文本用例**")
        for item in (ra.get("RA-2", {}) or {}).get("use_cases", []) or []:
            with st.expander(f"{item.get('use_case_id','')} · 高层文本用例", expanded=False):
                render_artifact_table(item.get("content_md", ""))

        st.markdown("**RA-3 详细文本用例**")
        for item in (ra.get("RA-3", {}) or {}).get("use_cases", []) or []:
            with st.expander(f"{item.get('use_case_id','')} · 详细文本用例", expanded=False):
                render_artifact_table(item.get("content_md", ""))

        ra4 = ra.get("RA-4", {}) or {}
        st.markdown("**RA-4 概念类模型**")
        st.caption(f"制品来源：{ra4.get('source', uml_artifacts.get('class_diagram_source', 'local'))}")
        st.code(ra4.get("plantuml", ""), language="plantuml")
        if ra4.get("url"):
            st.markdown(f"[查看渲染链接]({ra4.get('url')})")
        if uml_artifacts.get("llm_class_diagram"):
            with st.expander("查看本地版 / LLM版 类图对比", expanded=False):
                st.markdown("**本地规则版**")
                st.code(uml_artifacts.get("local_class_diagram", ""), language="plantuml")
                st.markdown("**LLM 增强版**")
                st.code(uml_artifacts.get("llm_class_diagram", ""), language="plantuml")

        ra5 = ra.get("RA-5", {}) or {}
        st.markdown("**RA-5 用例序列图**")
        st.caption(f"制品来源：{ra5.get('source', uml_artifacts.get('sequence_diagram_source', 'local'))}")
        st.code(ra5.get("plantuml", ""), language="plantuml")
        if ra5.get("url"):
            st.markdown(f"[查看渲染链接]({ra5.get('url')})")
        if uml_artifacts.get("llm_sequence_diagram"):
            with st.expander("查看本地版 / LLM版 序列图对比", expanded=False):
                st.markdown("**本地规则版**")
                st.code(uml_artifacts.get("local_sequence_diagram", ""), language="plantuml")
                st.markdown("**LLM 增强版**")
                st.code(uml_artifacts.get("llm_sequence_diagram", ""), language="plantuml")

        st.markdown("**RA-6 系统操作契约**")
        for item in (ra.get("RA-6", {}) or {}).get("use_cases", []) or []:
            with st.expander(f"{item.get('use_case_id','')} · 系统操作契约", expanded=False):
                render_artifact_table(item.get("content_md", ""))

    with tabs[3]:
        st.subheader("Reviewer 审查结果")
        review_report = result.get("Review_Report", {})
        scores = review_report.get("scores", {})
        _render_review_overview(result, review_report)
        _render_review_score_cards(scores)

        blocking_issues = result.get("Review_Blocking_Issues", []) or []
        warning_issues = result.get("Review_Warnings", []) or []
        suggestion_items = review_report.get("suggestions", []) or []
        issue_cols = st.columns(2)
        with issue_cols[0]:
            _render_review_issue_panel("阻断问题", blocking_issues, "critical", "当前未发现阻断性问题。")
        with issue_cols[1]:
            _render_review_issue_panel("警告问题", warning_issues, "warning", "当前未发现警告问题。")

        suggestion_cols = st.columns(2)
        with suggestion_cols[0]:
            _render_review_issue_panel("修正建议", suggestion_items, "neutral", "当前未生成额外修正建议。")
        with suggestion_cols[1]:
            exception_report = result.get("Review_Exception_Report", {}) or {}
            if exception_report:
                intervention_items = [
                    f"触发原因：{exception_report.get('reason', '无')}",
                    f"建议动作：{exception_report.get('recommended_action', '请人工复核后继续。')}",
                ]
                for issue in exception_report.get("blocking_issues", []) or []:
                    intervention_items.append(f"保留阻断项：{issue}")
                _render_review_issue_panel("人工介入", intervention_items, "critical", "当前无需人工介入。")
            else:
                _render_review_issue_panel("人工介入", [], "success", "当前无需人工介入。")

        feedback_text = result.get("Review_Feedback", "")
        if feedback_text:
            st.markdown(
                f"""
                <div class="review-feedback-card">
                    <div class="review-feedback-title">审查反馈摘要</div>
                    <div class="review-feedback-body">{html.escape(str(feedback_text))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        details = review_report.get("details", {})
        if details:
            for name, detail in details.items():
                with st.expander(f"{name} 审查详情", expanded=False):
                    _render_review_detail_block(name, detail)

    with tabs[4]:
        st.subheader("测试用例")
        test_cases = result.get("Test_Cases", [])
        if test_cases:
            st.dataframe(pd.DataFrame(test_cases), use_container_width=True)

    with tabs[5]:
        st.subheader("追溯矩阵")
        matrix = result.get("Traceability_Matrix", [])
        if matrix:
            st.dataframe(pd.DataFrame(matrix), use_container_width=True)
        st.json(result.get("Traceability_Validation", {}))

    with tabs[6]:
        st.subheader("RAG 检索结果")
        knowledge_context = result.get("Knowledge_Context", [])
        render_rag_results(knowledge_context)

    with tabs[7]:
        st.subheader("导出")
        st.download_button(
            "导出完整 JSON 结果",
            data=json.dumps(result, ensure_ascii=False, indent=2),
            file_name="ocean_intelli_modeller_result.json",
            mime="application/json",
            use_container_width=True,
        )
        st.download_button(
            "导出 EARS 文本",
            data=result.get("EARS_Requirement", ""),
            file_name="ocean_intelli_modeller_ears.txt",
            mime="text/plain",
            use_container_width=True,
        )


def main() -> None:
    init_state()
    inject_custom_styles()

    render_sidebar_brand()
    render_hero_section()
    render_section_header(
        "Prompt",
        "输入医疗需求",
        show_divider=False,
    )

    with st.sidebar:
        profile_options = Config.get_llm_profiles()
        selected_profile: dict[str, str] | None = None
        provider = "offline"
        if profile_options:
            profile_ids = [item["id"] for item in profile_options]
            profile_map = {item["id"]: item for item in profile_options}
            default_profile_index = 0
            for index, item in enumerate(profile_options):
                if item["label"] == "DeepSeek Chat":
                    default_profile_index = index
                    break
            selected_profile_id = st.selectbox(
                "模型底座",
                profile_ids,
                index=default_profile_index,
                format_func=lambda profile_id: profile_map[profile_id]["label"],
            )
            selected_profile = profile_map[selected_profile_id]
            provider = selected_profile["provider"]
            st.caption(
                f"当前模型：`{selected_profile['model']}`"
                f" · Base URL：`{selected_profile['base_url']}`"
            )
        max_iters = st.slider("最大审查迭代次数", min_value=1, max_value=5, value=2)
        if selected_profile is None:
            st.info("当前未检测到可选模型底座，本次运行将使用离线规则模式。")
        else:
            profile_has_key = bool(selected_profile["api_key"])
            if not profile_has_key:
                st.warning("当前模型底座未检测到可用 API Key，本次运行将自动回退到离线模式。")

    info_blocks = [
        
    ]
    if info_blocks:
        info_cols = st.columns(len(info_blocks))
        for col, (title, copy) in zip(info_cols, info_blocks):
            with col:
                st.markdown(
                    f"""
                    <div class="feature-card">
                        <div class="feature-title">{title}</div>
                        <div class="feature-copy">{copy}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.text_area(
        "请在这里输入或粘贴医疗需求文本",
        key="requirement_input",
        height=180,
        placeholder="例如：当医生提交检验申请后，系统应在 5 秒内生成可追溯的检验任务，并将异常样本自动标记给质控人员。",
    )

    run_clicked = st.button("运行", type="primary", use_container_width=True)
    if run_clicked:
        requirement_input = st.session_state.get("requirement_input", "").strip()
        if not requirement_input:
            st.error("请输入至少一条需求。")
        else:
            with st.spinner("正在执行多智能体闭环流程..."):
                st.session_state.pipeline_result = run_modeller_pipeline(
                    requirement_input,
                    config=PipelineConfig(
                        provider=provider,
                        model_override=(selected_profile or {}).get("model"),
                        base_url_override=(selected_profile or {}).get("base_url"),
                        api_key_override=(selected_profile or {}).get("api_key"),
                        max_iterations=max_iters,
                    ),
                )

    result = st.session_state.get("pipeline_result")
    if result:
        render_section_header(
            "Results",
            "建模结果总览",
        )
        st.markdown(
            f"""
            <div class="status-pill">Current status · {result.get("Status", "unknown")}</div>
            """,
            unsafe_allow_html=True,
        )
        render_result_overview(result)
        cols = st.columns(3)
        cols[0].metric("实际 Provider", result.get("LLM_Provider", "offline"))
        cols[1].metric("实际 Model", result.get("LLM_Model", "") or "N/A")
        cols[2].metric("运行状态", result.get("LLM_Runtime_Status", "offline"))
        st.caption(f"当前运行模式：{'LLM 增强' if result.get('LLM_Enabled') else '本地规则'}")
        if result.get("Clarification_Questions"):
            st.warning("Analyst Agent 给出少量补充建议（流程已继续产出制品）：")
            for question in result.get("Clarification_Questions", []):
                st.markdown(f"- {question}")
        if result.get("Analyst_Thinking"):
            with st.expander("查看 Analyst 中间分析过程", expanded=False):
                for trace in result.get("Analyst_Thinking", []):
                    st.code(trace, language="xml")
        render_result_tabs(result)


if __name__ == "__main__":
    main()
