# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: Med-MARAG
File Name: test_case_generator.py
Author: SeaOcean
Create Date: 2026-03-07
Description：测试用例生成模块
-------------------------------------------------
"""
from __future__ import annotations

from typing import Any

from llm.provider import LLMRuntime, build_llm_runtime, invoke_llm_text
from rag.knowledge_base import knowledge_base
from utils.requirement_utils import RequirementUtils


class TestCaseGenerator:
    """测试用例生成器。"""

    __test__ = False

    def __init__(
        self,
        *,
        provider: str = "offline",
        use_llm: bool | None = None,
        model_override: str | None = None,
        enable_rag: bool = True,
        runtime: LLMRuntime | None = None,
    ):
        self.runtime = runtime or build_llm_runtime(
            provider=provider,
            use_llm=use_llm,
            model_override=model_override,
        )
        self.use_llm = self.runtime.enabled
        self.llm = self.runtime.client
        self.enable_rag = enable_rag

    def generate_test_cases(
        self,
        requirement: str,
        uml_model: dict[str, Any] | None = None,
        requirement_items: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        items = requirement_items or [
            RequirementUtils.analyze_requirement(text, index=index)
            for index, text in enumerate(RequirementUtils.split_requirements(requirement), start=1)
        ]
        test_cases = self._generate_local_test_cases(items)
        if self.llm is not None:
            llm_cases = self._generate_llm_test_cases(requirement)
            if llm_cases:
                test_cases = llm_cases
        return test_cases

    def _generate_local_test_cases(self, requirement_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        test_cases: list[dict[str, Any]] = []
        for index, item in enumerate(requirement_items, start=1):
            action_text = "、".join(item.get("actions", [])) or "执行业务操作"
            actor = item.get("primary_actor", "用户")
            condition = item.get("condition", "系统接收到业务请求")
            domain = item.get("domain", "通用流程")

            test_cases.append(
                {
                    "id": f"TC{index * 2 - 1:03d}",
                    "requirement_id": item.get("id", f"REQ{index:03d}"),
                    "description": f"{domain}正常流程验证",
                    "scenario": f"{actor}满足“{condition}”时，系统完成{action_text}",
                    "steps": [
                        f"准备满足条件“{condition}”的数据环境。",
                        f"{actor}发起对应业务请求。",
                        "观察系统处理日志和页面反馈。",
                    ],
                    "expected_result": f"系统应成功完成{action_text}，并输出可验证结果。",
                    "priority": "高",
                }
            )
            test_cases.append(
                {
                    "id": f"TC{index * 2:03d}",
                    "requirement_id": item.get("id", f"REQ{index:03d}"),
                    "description": f"{domain}异常流程验证",
                    "scenario": f"{actor}输入异常或上下文不满足时，系统安全拒绝或提示",
                    "steps": [
                        "构造缺失字段、越权访问或状态不一致的异常输入。",
                        f"{actor}再次发起同一业务请求。",
                        "检查系统是否记录审计信息并给出提示。",
                    ],
                    "expected_result": "系统应拒绝非法处理，返回明确错误信息，并保留审计线索。",
                    "priority": "中",
                }
            )
        return test_cases or self._generate_default_test_cases()

    def _generate_llm_test_cases(self, requirement: str) -> list[dict[str, Any]]:
        knowledge_results = knowledge_base.query(requirement, n_results=3) if self.enable_rag else []
        prompt = f"""
你是一名医疗软件测试专家，请基于需求输出 2-4 条结构化测试用例。
每条测试用例包括 ID、场景、步骤、预期结果、优先级。

需求：
{requirement}

参考：
{chr(10).join(item["content"] for item in knowledge_results)}
"""
        content = invoke_llm_text(
            self.runtime,
            [
                ("system", "你输出简洁的中文测试用例。"),
                ("user", prompt),
            ],
        )
        if not content:
            return []
        return self._parse_test_cases(content)

    def _parse_test_cases(self, result: str) -> list[dict[str, Any]]:
        test_cases: list[dict[str, Any]] = []
        if "# 测试用例" in result:
            sections = result.split("# 测试用例")
            for section in sections[1:]:
                test_case = {
                    "id": "",
                    "description": "",
                    "scenario": "",
                    "steps": [],
                    "expected_result": "",
                    "priority": "中",
                }
                if "ID:" in section:
                    test_case["id"] = section.split("ID:")[1].split("\n")[0].strip()
                if "场景:" in section:
                    scenario = section.split("场景:")[1].split("\n")[0].strip()
                    test_case["scenario"] = scenario
                    test_case["description"] = scenario
                if "步骤:" in section:
                    steps_section = section.split("步骤:")[1].split("预期结果:")[0]
                    test_case["steps"] = [line.strip("- ").strip() for line in steps_section.strip().split("\n") if line.strip()]
                if "预期结果:" in section:
                    expected_section = section.split("预期结果:")[1].split("优先级:")[0]
                    test_case["expected_result"] = expected_section.strip()
                if "优先级:" in section:
                    test_case["priority"] = section.split("优先级:")[1].split("\n")[0].strip()
                if test_case["id"]:
                    test_cases.append(test_case)
        return test_cases

    def _generate_default_test_cases(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "TC001",
                "requirement_id": "REQ001",
                "description": "正常流程测试",
                "scenario": "执行标准业务流程",
                "steps": ["执行正常操作"],
                "expected_result": "系统正常响应",
                "priority": "高",
            },
            {
                "id": "TC002",
                "requirement_id": "REQ001",
                "description": "异常流程测试",
                "scenario": "执行异常输入流程",
                "steps": ["执行异常操作"],
                "expected_result": "系统正确处理异常",
                "priority": "中",
            },
        ]

    def generate_test_report(self, test_cases: list[dict[str, Any]], results: list[str]) -> dict[str, Any]:
        report = {"total": len(test_cases), "passed": 0, "failed": 0, "skipped": 0, "details": []}
        for index, test_case in enumerate(test_cases):
            result = results[index] if index < len(results) else "未知"
            status = "通过" if result == "通过" else "失败" if result == "失败" else "跳过"
            report["details"].append(
                {
                    "id": test_case.get("id", f"TC{index + 1:03d}"),
                    "scenario": test_case.get("scenario", ""),
                    "result": result,
                    "status": status,
                }
            )
            if status == "通过":
                report["passed"] += 1
            elif status == "失败":
                report["failed"] += 1
            else:
                report["skipped"] += 1
        return report

    def validate_test_cases(self, test_cases: list[dict[str, Any]], requirement: str) -> dict[str, Any]:
        validation = {"valid": True, "coverage": 0, "issues": []}
        if len(test_cases) < 2:
            validation["valid"] = False
            validation["issues"].append("测试用例数量不足，建议至少包含正常和异常场景。")
        high_priority_count = sum(1 for case in test_cases if case.get("priority") == "高")
        if high_priority_count == 0:
            validation["issues"].append("缺少高优先级测试用例。")
        validation["coverage"] = min(100, len(test_cases) * 25)
        if "审计" in requirement and not any("审计" in case.get("expected_result", "") for case in test_cases):
            validation["issues"].append("需求涉及审计时，建议补充日志留痕验证用例。")
        return validation


test_case_generator = TestCaseGenerator(provider="offline")
