#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: Med-MARAG
File Name: run_pipeline.py
Author: SeaOcean
Create Date: 2026-03-12
Description：命令行流水线运行入口
-------------------------------------------------
"""

from __future__ import annotations

import argparse
import json

from config import Config
from workflow.ocean_graph import PipelineConfig, run_modeller_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Med-MARAG pipeline.")
    parser.add_argument("requirement", help="Natural language requirement text")
    parser.add_argument(
        "--provider",
        choices=["offline", "deepseek", "openai"],
        default=Config.get_requested_llm_provider(),
        help="LLM provider to use. Defaults to LLM_PROVIDER or offline.",
    )
    parser.add_argument("--model", help="Override model name for the selected provider.")
    parser.add_argument("--no-llm", action="store_true", help="Legacy flag: force offline mode.")
    parser.add_argument("--max-iters", type=int, default=3, help="Max review iterations")
    parser.add_argument("--json", action="store_true", help="Print full result as JSON")
    args = parser.parse_args()

    provider = "offline" if args.no_llm else args.provider
    state = run_modeller_pipeline(
        args.requirement,
        config=PipelineConfig(provider=provider, model_override=args.model, max_iterations=args.max_iters),
    )

    if args.json:
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0

    print("Requested Provider:", provider)
    print("Effective Provider:", state.get("LLM_Provider", "offline"))
    print("Model:", state.get("LLM_Model", "") or "N/A")
    print("Runtime Status:", state.get("LLM_Runtime_Status", "offline"))
    print("Status:", state.get("Status"))
    if state.get("Clarification_Questions"):
        print("\nSuggestions (Clarification Questions):")
        for question in state.get("Clarification_Questions", []):
            print(f"- {question}")
    print("\nEARS:\n", state.get("EARS_Requirement", ""), sep="")
    print("\nUse Cases:")
    for use_case in state.get("Use_Cases", []):
        print(f"- {use_case.get('id')}: {use_case.get('name')} ({use_case.get('primary_actor')})")

    print("\nPrimary UML (Class Diagram):\n", state.get("UML_Code", ""), sep="")
    artifacts = state.get("UML_Artifacts", {})
    if artifacts.get("use_case_diagram"):
        print("\nUse Case Diagram:\n", artifacts["use_case_diagram"], sep="")
    if artifacts.get("sequence_diagram"):
        print("\nSequence Diagram:\n", artifacts["sequence_diagram"], sep="")

    feedback = state.get("Review_Feedback", "")
    if feedback:
        print("\nReview Feedback:\n", feedback, sep="")
    url = state.get("UML_Diagram_URL")
    if url:
        print("\nDiagram URL:", url)

    test_cases = state.get("Test_Cases", [])
    if test_cases:
        print("\nTest Cases:")
        for case in test_cases:
            print(f"- {case.get('id')}: {case.get('scenario')} [{case.get('priority')}]")

    matrix = state.get("Traceability_Matrix", [])
    if matrix:
        print("\nTraceability Links:", len(matrix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
