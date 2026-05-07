#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: Med-MARAG
File Name: verify_system_integration.py
Author: SeaOcean
Create Date: 2026-03-13
Description：系统集成验证脚本
-------------------------------------------------
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import requests

from config import Config
from llm.provider import build_llm_runtime
from rag.knowledge_base import knowledge_base
from workflow.ocean_graph import PipelineConfig, run_modeller_pipeline


DEFAULT_REQUIREMENT = "当患者预约挂号成功时，系统应发送确认短信给患者"
DEFAULT_RAG_QUERY = "verifiable singular unambiguous requirement statement"


def print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def mask_key(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return value[:2] + "***"
    return value[:6] + "***"


def verify_provider_connection(provider: str, model_override: str | None = None) -> dict[str, Any]:
    runtime = build_llm_runtime(provider=provider, model_override=model_override)
    if not runtime.enabled:
        return {
            "status_code": None,
            "provider": runtime.effective_provider,
            "model": runtime.model,
            "content": "",
            "usage": {},
            "runtime_status": runtime.status,
        }
    payload = {
        "model": runtime.model,
        "messages": [
            {"role": "system", "content": "你是一个测试助手。"},
            {"role": "user", "content": "只回复两个字：验证"},
        ],
        "stream": False,
    }
    response = requests.post(
        f"{runtime.base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {Config.get_llm_api_key(provider)}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    return {
        "status_code": response.status_code,
        "provider": runtime.effective_provider,
        "model": body.get("model"),
        "content": body["choices"][0]["message"]["content"],
        "usage": body.get("usage", {}),
        "runtime_status": runtime.status,
    }


def call_provider(provider: str, messages: list[dict[str, str]], model_override: str | None = None) -> dict[str, Any]:
    runtime = build_llm_runtime(provider=provider, model_override=model_override)
    if not runtime.enabled:
        return {
            "status_code": None,
            "provider": runtime.effective_provider,
            "model": runtime.model,
            "content": "",
            "usage": {},
            "runtime_status": runtime.status,
        }
    payload = {
        "model": runtime.model,
        "messages": messages,
        "stream": False,
    }
    response = requests.post(
        f"{runtime.base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {Config.get_llm_api_key(provider)}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    return {
        "status_code": response.status_code,
        "provider": runtime.effective_provider,
        "model": body.get("model"),
        "content": body["choices"][0]["message"]["content"],
        "usage": body.get("usage", {}),
        "runtime_status": runtime.status,
    }


def verify_rag(query: str, n_results: int) -> list[dict[str, Any]]:
    return knowledge_base.query(query, n_results=n_results)


def summarize_rag_results(results: list[dict[str, Any]]) -> None:
    for idx, item in enumerate(results, start=1):
        metadata = item.get("metadata", {})
        source = metadata.get("source", "built-in")
        page = metadata.get("page", "-")
        print(f"[{idx}] id={item.get('id')} source={source} page={page}")
        print(item.get("content", "").replace("\n", " ")[:450])
        print()


def verify_pipeline(requirement: str, provider: str, max_iters: int, model_override: str | None = None) -> dict[str, Any]:
    return run_modeller_pipeline(
        requirement,
        config=PipelineConfig(provider=provider, model_override=model_override, max_iterations=max_iters),
    )


def verify_analyst_rag_llm(
    requirement: str,
    provider: str,
    n_results: int = 2,
    model_override: str | None = None,
) -> dict[str, Any]:
    rag_results = knowledge_base.query(requirement, n_results=n_results)
    context = "\n\n".join(
        f"[source={item.get('metadata', {}).get('source', 'built-in')} page={item.get('metadata', {}).get('page', '-')}] "
        f"{item.get('content', '')[:700]}"
        for item in rag_results
    )
    messages = [
        {
            "role": "system",
            "content": (
                "你是严格遵循 ISO/IEC/IEEE 29148:2018 标准的医疗软件需求分析专家。"
                "请先进行分步骤推理，再输出简洁、明确、可验证的中文规范化需求。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请根据输入需求和参考知识，先识别参与者与关键实体，再推演业务流程，"
                "检查是否缺少前置条件、关键约束或异常路径，最后输出规范化需求条目。"
                "若信息不足，请明确指出需要补充的内容；若信息充分，请优先采用 EARS 风格表达。"
                "\n\n"
                f"输入需求：{requirement}\n\n"
                f"参考知识：\n{context}"
            ),
        },
    ]
    result = call_provider(provider, messages, model_override=model_override)
    result["rag_results"] = rag_results
    return result


def summarize_pipeline(state: dict[str, Any]) -> None:
    print(f"Status: {state.get('Status')}")
    print("\nEARS:")
    print(state.get("EARS_Requirement", ""))

    requirement_items = state.get("Requirement_Items", [])
    if requirement_items:
        first_item = requirement_items[0]
        print("\nAnalyst Summary:")
        print(first_item.get("llm_summary") or "未返回 llm_summary")

    print("\nReview Feedback:")
    print(state.get("Review_Feedback", ""))

    print("\nClass Diagram Preview:")
    print((state.get("UML_Code", "") or "")[:1000])

    print("\nTraceability Links:", len(state.get("Traceability_Matrix", [])))
    print("Test Cases:", len(state.get("Test_Cases", [])))


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify provider + RAG + pipeline integration.")
    parser.add_argument("--requirement", default=DEFAULT_REQUIREMENT, help="Requirement text for pipeline verification.")
    parser.add_argument("--rag-query", default=DEFAULT_RAG_QUERY, help="Query text for RAG verification.")
    parser.add_argument("--rag-topk", type=int, default=3, help="Number of RAG results to print.")
    parser.add_argument(
        "--provider",
        choices=["offline", "deepseek", "openai"],
        default=Config.get_requested_llm_provider(),
        help="Provider to verify.",
    )
    parser.add_argument("--model", help="Override model name for the selected provider.")
    parser.add_argument("--skip-pipeline", action="store_true", help="Only verify provider API and RAG.")
    parser.add_argument("--json", action="store_true", help="Print final pipeline state as JSON.")
    args = parser.parse_args()

    print_section("Configuration")
    print("requested_provider:", args.provider)
    print("llm_enabled:", Config.llm_enabled(args.provider))
    print("model:", Config.get_llm_model(args.provider, override=args.model))
    print("base_url:", Config.get_llm_base_url(args.provider))
    print("api_key_prefix:", mask_key(Config.get_llm_api_key(args.provider)))

    print_section("Provider API Verification")
    api_result = verify_provider_connection(args.provider, model_override=args.model)
    print("runtime_status:", api_result["runtime_status"])
    print("status_code:", api_result["status_code"])
    print("provider:", api_result["provider"])
    print("model:", api_result["model"])
    print("content:", api_result["content"])
    print("usage:", api_result["usage"])

    print_section("RAG Verification")
    rag_results = verify_rag(args.rag_query, args.rag_topk)
    stats = knowledge_base.get_collection_stats()
    print("knowledge_base_stats:", stats)
    summarize_rag_results(rag_results)

    print_section("Analyst + RAG + Provider Verification")
    analyst_result = verify_analyst_rag_llm(
        args.requirement,
        args.provider,
        n_results=min(2, args.rag_topk),
        model_override=args.model,
    )
    print("runtime_status:", analyst_result["runtime_status"])
    print("provider:", analyst_result["provider"])
    print("model:", analyst_result["model"])
    print("content:")
    print(analyst_result["content"])
    print("usage:", analyst_result["usage"])

    if args.skip_pipeline:
        return 0

    print_section("Pipeline Verification")
    state = verify_pipeline(args.requirement, provider=args.provider, max_iters=2, model_override=args.model)
    summarize_pipeline(state)

    if args.json:
        print_section("Pipeline JSON")
        print(json.dumps(state, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
