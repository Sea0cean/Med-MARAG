# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: Med-MARAG
File Name: llm/provider.py
Author: SeaOcean
Create Date: 2026-03-18
Description：LLM 运行时封装模块
-------------------------------------------------
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from config import Config

try:
    from langchain_openai import ChatOpenAI
except ModuleNotFoundError:  # pragma: no cover
    ChatOpenAI = None  # type: ignore


LLMProvider = Literal["offline", "deepseek", "openai"]


@dataclass(frozen=True)
class LLMRuntime:
    requested_provider: LLMProvider
    effective_provider: LLMProvider
    model: str
    base_url: str
    enabled: bool
    status: str
    fallback_reason: str = ""
    client: Any = None


def build_llm_runtime(
    *,
    provider: str | None = None,
    use_llm: bool | None = None,
    model_override: str | None = None,
    base_url_override: str | None = None,
    api_key_override: str | None = None,
) -> LLMRuntime:
    raw_provider = (provider if provider is not None else Config._env("LLM_PROVIDER", "")).strip().lower()
    if raw_provider and not Config.provider_is_valid(raw_provider):
        return LLMRuntime(
            requested_provider="offline",
            effective_provider="offline",
            model="",
            base_url="",
            enabled=False,
            status="fallback:invalid_provider",
            fallback_reason="invalid_provider",
        )

    requested_provider = Config.get_requested_llm_provider(provider=provider, use_llm=use_llm)

    if requested_provider == "offline":
        return LLMRuntime(
            requested_provider="offline",
            effective_provider="offline",
            model="",
            base_url="",
            enabled=False,
            status="offline",
        )

    api_key = Config.get_llm_api_key(requested_provider, override=api_key_override)
    model = Config.get_llm_model(requested_provider, override=model_override)
    base_url = Config.get_llm_base_url(requested_provider, override=base_url_override)

    if not api_key:
        return LLMRuntime(
            requested_provider=requested_provider,
            effective_provider="offline",
            model=model,
            base_url=base_url,
            enabled=False,
            status="fallback:missing_api_key",
            fallback_reason="missing_api_key",
        )

    if ChatOpenAI is None:
        return LLMRuntime(
            requested_provider=requested_provider,
            effective_provider="offline",
            model=model,
            base_url=base_url,
            enabled=False,
            status="fallback:langchain_openai_unavailable",
            fallback_reason="langchain_openai_unavailable",
        )

    try:
        client = ChatOpenAI(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=Config.LLM_REQUEST_TIMEOUT_SECONDS,
            max_retries=Config.LLM_MAX_RETRIES,
        )
    except Exception:
        return LLMRuntime(
            requested_provider=requested_provider,
            effective_provider="offline",
            model=model,
            base_url=base_url,
            enabled=False,
            status="fallback:client_init_failed",
            fallback_reason="client_init_failed",
        )

    return LLMRuntime(
        requested_provider=requested_provider,
        effective_provider=requested_provider,
        model=model,
        base_url=base_url,
        enabled=True,
        status="enabled",
        client=client,
    )


def invoke_llm_text(runtime: LLMRuntime, messages: list[tuple[str, str]]) -> str:
    if runtime.client is None:
        return ""
    try:
        result = runtime.client.invoke(messages)
    except Exception:
        return ""
    return str(result.content if hasattr(result, "content") else result).strip()


def extract_json_object(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
