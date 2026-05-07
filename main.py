# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: Med-MARAG
File Name: main.py
Author: SeaOcean
Create Date: 2026-01-13
Description：交互式终端主入口
-------------------------------------------------
"""
from __future__ import annotations

import json

from workflow.ocean_graph import PipelineConfig, run_modeller_pipeline


def main() -> int:
    requirement = input("请输入医疗系统需求（可多行，回车后 Ctrl+D 结束）：\n").strip()
    if not requirement:
        print("未输入需求。")
        return 1

    state = run_modeller_pipeline(
        requirement,
        config=PipelineConfig(provider="offline", max_iterations=2),
    )

    print("\n========== Med-MARAG ==========")
    print("状态:", state.get("Status"))
    print("Provider:", state.get("LLM_Provider"))
    print("运行状态:", state.get("LLM_Runtime_Status"))
    if state.get("Clarification_Questions"):
        print("\n补充建议：")
        for question in state.get("Clarification_Questions", []):
            print(f"- {question}")
    print("\nEARS 需求:\n", state.get("EARS_Requirement", ""), sep="")
    print("\n类图代码:\n", state.get("UML_Code", ""), sep="")
    print("\n测试用例数量:", len(state.get("Test_Cases", [])))
    print("追溯矩阵数量:", len(state.get("Traceability_Matrix", [])))
    print("\n完整结果 JSON:")
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
