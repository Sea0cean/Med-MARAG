# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: Med-MARAG
File Name: workflow/ocean_graph.py
Author: SeaOcean
Create Date: 2026-03-12
Description：多智能体主工作流编排模块
-------------------------------------------------
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

try:
    from langgraph.graph import END, START, StateGraph
except ModuleNotFoundError:  # pragma: no cover
    END = START = None  # type: ignore
    StateGraph = None  # type: ignore

from agents.analyst_agent import AnalystAgent
from agents.architect_agent import ArchitectAgent
from agents.review_agent import ReviewAgent
from llm.provider import LLMRuntime, build_llm_runtime
from test_case_generator import TestCaseGenerator
from traceability_matrix import traceability_matrix
from config import Config
from utils.requirement_utils import RequirementUtils


class GraphState(TypedDict, total=False):
    Raw_Requirement: str
    LLM_Enabled: bool
    LLM_Provider: str
    LLM_Model: str
    LLM_Runtime_Status: str
    RAG_Enabled: bool
    Analyst_Thinking: list[str]
    Clarification_Questions: list[str]
    Requirement_Items: list[dict[str, Any]]
    Knowledge_Context: list[dict[str, Any]]
    Use_Cases: list[dict[str, Any]]
    EARS_Requirement: str
    UML_Code: str
    UML_Diagram_URL: str | None
    UML_Artifacts: dict[str, Any]
    Design_Elements: list[str]
    Review_Feedback: str
    Review_Report: dict[str, Any]
    Review_Blocking_Issues: list[str]
    Review_Warnings: list[str]
    Review_Exception_Report: dict[str, Any]
    Review_Target: str
    Test_Cases: list[dict[str, Any]]
    Traceability_Matrix: list[dict[str, Any]]
    Traceability_Validation: dict[str, Any]
    RA_Artifacts: dict[str, Any]
    RA_Validation: dict[str, Any]
    Status: str
    Iteration_Count: int


@dataclass(frozen=True)
class PipelineConfig:
    provider: Literal["offline", "deepseek", "openai"] = "offline"
    use_llm: bool | None = None
    model_override: str | None = None
    base_url_override: str | None = None
    api_key_override: str | None = None
    enable_rag: bool = True
    max_iterations: int = 2
    review_pass_threshold: int = 70


def _resolve_runtime(config: PipelineConfig) -> LLMRuntime:
    return build_llm_runtime(
        provider=config.provider,
        use_llm=config.use_llm,
        model_override=config.model_override,
        base_url_override=config.base_url_override,
        api_key_override=config.api_key_override,
    )


def _build_agents(
    config: PipelineConfig,
    runtime: LLMRuntime,
) -> tuple[AnalystAgent, ArchitectAgent, ReviewAgent]:
    return (
        AnalystAgent(runtime=runtime, enable_rag=config.enable_rag),
        ArchitectAgent(runtime=runtime, enable_rag=config.enable_rag),
        ReviewAgent(runtime=runtime, enable_rag=config.enable_rag),
    )


def _analyst_step(state: GraphState, analyst_agent: AnalystAgent) -> GraphState:
    raw = state.get("Raw_Requirement", "")
    requirement_texts = RequirementUtils.split_requirements(raw)
    if not requirement_texts:
        return {**state, "Status": "INVALID_INPUT"}

    feedback = state.get("Review_Feedback", "") if state.get("Review_Target") == "analyst" else ""
    items = [
        analyst_agent.analyze_requirement(text, index=index, feedback=feedback)
        for index, text in enumerate(requirement_texts, start=1)
    ]
    knowledge_context: dict[str, dict[str, Any]] = {}
    clarification_questions: list[str] = []
    analyst_thinking: list[str] = []
    for item in items:
        for knowledge in item.get("knowledge_results", []):
            knowledge_context[knowledge["id"]] = knowledge
        clarification_questions.extend(item.get("clarification_questions", []))
        if item.get("thinking_trace"):
            analyst_thinking.append(item["thinking_trace"])

    deduped_questions = list(dict.fromkeys(question for question in clarification_questions if question))
    if Config.MAX_CLARIFICATION_QUESTIONS > 0:
        deduped_questions = deduped_questions[: Config.MAX_CLARIFICATION_QUESTIONS]

    return {
        **state,
        "LLM_Enabled": bool(state.get("LLM_Enabled", False)),
        "Requirement_Items": items,
        "Knowledge_Context": list(knowledge_context.values()),
        "Use_Cases": [item["use_case"] for item in items],
        "EARS_Requirement": "\n\n".join(item["ears_requirement"] for item in items),
        "Analyst_Thinking": analyst_thinking,
        "Clarification_Questions": deduped_questions,
        "Review_Target": "",
        # Always continue to produce artifacts; clarification questions are treated as suggestions.
        "Status": "EARS_READY",
    }


def _architect_step(state: GraphState, architect_agent: ArchitectAgent) -> GraphState:
    requirement_items = state.get("Requirement_Items", [])
    if not requirement_items:
        return {**state, "Status": "INVALID_INPUT"}

    model = architect_agent.generate_uml(requirement_items, feedback=state.get("Review_Feedback", ""))
    return {
        **state,
        "UML_Code": model.get("plantuml_code", ""),
        "UML_Diagram_URL": model.get("diagram_url"),
        "UML_Artifacts": model.get("artifacts", {}),
        "Design_Elements": model.get("design_elements", []),
        "Status": "UML_READY",
    }


def _review_step(state: GraphState, reviewer_agent: ReviewAgent, config: PipelineConfig) -> GraphState:
    review = reviewer_agent.review_model_pack(
        state.get("UML_Artifacts", {}),
        requirement_items=state.get("Requirement_Items", []),
        ears_requirement=state.get("EARS_Requirement", ""),
    )
    issues = review.get("issues", []) or []
    overall = int(review.get("scores", {}).get("overall", 0) or 0)
    blocking_issues, warning_issues = _split_review_issues(issues)
    passed = overall >= config.review_pass_threshold and not blocking_issues
    feedback_parts = []
    if blocking_issues:
        feedback_parts.append("阻断问题：\n" + "\n".join(blocking_issues))
    if warning_issues:
        feedback_parts.append("警告问题：\n" + "\n".join(warning_issues))
    feedback = "\n\n".join(feedback_parts) if feedback_parts else "审查通过。"
    review_target = _infer_review_target(blocking_issues, warning_issues)

    if passed:
        return {
            **state,
            "Status": "REVIEW_PASSED",
            "Review_Feedback": feedback,
            "Review_Report": review,
            "Review_Blocking_Issues": [],
            "Review_Warnings": warning_issues,
            "Review_Exception_Report": {},
            "Review_Target": "",
        }

    next_iteration = int(state.get("Iteration_Count", 0) or 0) + 1
    exception_report = {}
    if next_iteration >= config.max_iterations:
        exception_report = {
            "type": "human_intervention_required",
            "reason": "审查多轮未通过，已达到自动修正阈值。",
            "iteration_count": next_iteration,
            "review_target": review_target,
            "blocking_issues": blocking_issues[:5],
            "recommended_action": "请人工检查需求规约、UML 关系以及 Reviewer 反馈后再继续。",
        }

    return {
        **state,
        "Status": "NEEDS_REQUIREMENT_REVISION" if review_target == "analyst" else "NEEDS_REVISION",
        "Review_Feedback": feedback,
        "Review_Report": review,
        "Review_Blocking_Issues": blocking_issues,
        "Review_Warnings": warning_issues,
        "Review_Exception_Report": exception_report,
        "Review_Target": review_target,
        "Iteration_Count": next_iteration,
    }


def _split_review_issues(issues: list[str]) -> tuple[list[str], list[str]]:
    blocking_markers = (
        "缺少 @startuml",
        "缺少 @enduml",
        "缺少 @startuml 或 @enduml",
        "缺失。",
        "未识别到 UML 结构元素",
        "存在未声明参与者",
        "EARS 结构不完整",
        "未体现系统行为主体",
        "动作仍然过于笼统",
        "信息不完整",
        "表达不清晰",
        "缺少参与者",
        "缺少条件",
        "缺少动作",
        "缺少对象",
        "缺少约束",
        "异常路径",
        "可能未正确闭合",
    )
    warnings: list[str] = []
    blocking: list[str] = []
    for issue in issues:
        normalized = str(issue or "")
        if any(marker in normalized for marker in blocking_markers):
            blocking.append(normalized)
        else:
            warnings.append(normalized)
    return blocking, warnings


def _infer_review_target(blocking_issues: list[str], warning_issues: list[str]) -> str:
    issues = blocking_issues + warning_issues
    requirement_markers = (
        "EARS 结构不完整",
        "未体现系统行为主体",
        "动作仍然过于笼统",
        "需求信息",
        "需求描述",
        "需求表达",
        "需求不完整",
        "需求不清晰",
        "信息不完整",
        "表达不清晰",
        "缺少参与者",
        "缺少条件",
        "缺少动作",
        "缺少对象",
        "缺少约束",
        "异常路径",
        "用例",
        "口语",
    )
    if any(any(marker in issue for marker in requirement_markers) for issue in issues):
        return "analyst"
    return "architect"


def _finalize_state(state: GraphState, runtime: LLMRuntime | None = None) -> GraphState:
    requirement_items = state.get("Requirement_Items", [])
    rag_enabled = bool(state.get("RAG_Enabled", True))
    generator = (
        TestCaseGenerator(runtime=runtime, enable_rag=rag_enabled)
        if runtime is not None
        else TestCaseGenerator(provider="offline", enable_rag=rag_enabled)
    )
    test_cases = generator.generate_test_cases(
        state.get("EARS_Requirement", ""),
        uml_model=state.get("UML_Artifacts", {}),
        requirement_items=requirement_items,
    )
    requirements = [{"id": item["id"], "description": item["original"]} for item in requirement_items]
    traceability = traceability_matrix.generate_matrix(
        requirements,
        state.get("Design_Elements", []),
        test_cases,
    )
    traceability_validation = traceability_matrix.validate_traceability()

    from utils.ra_artifacts import RAArtifactGenerator

    ra_pack = RAArtifactGenerator(
        runtime=runtime,
        provider=(runtime.effective_provider if runtime else "offline"),
        enable_rag=rag_enabled,
    ).generate(
        requirement_items=requirement_items,
        use_cases=state.get("Use_Cases", []) or [],
        uml_artifacts=state.get("UML_Artifacts", {}) or {},
    )
    return {
        **state,
        "Test_Cases": test_cases,
        "Traceability_Matrix": traceability,
        "Traceability_Validation": traceability_validation,
        "RA_Artifacts": ra_pack.artifacts,
        "RA_Validation": ra_pack.validation,
    }


def build_ocean_modeller_graph(
    *,
    analyst: AnalystAgent | None = None,
    architect: ArchitectAgent | None = None,
    reviewer: ReviewAgent | None = None,
    config: PipelineConfig | None = None,
):
    cfg = config or PipelineConfig()
    runtime = _resolve_runtime(cfg)
    analyst_agent = analyst or AnalystAgent(runtime=runtime, enable_rag=cfg.enable_rag)
    architect_agent = architect or ArchitectAgent(runtime=runtime, enable_rag=cfg.enable_rag)
    reviewer_agent = reviewer or ReviewAgent(runtime=runtime, enable_rag=cfg.enable_rag)

    if StateGraph is None:
        class SequentialApp:
            def invoke(self, initial_state: GraphState) -> GraphState:
                state = _analyst_step(initial_state, analyst_agent)
                if state.get("Status") in {"INVALID_INPUT"}:
                    return state
                while True:
                    state = _architect_step(state, architect_agent)
                    state = _review_step(state, reviewer_agent, cfg)
                    if state.get("Status") == "REVIEW_PASSED":
                        break
                    if state.get("Status") == "NEEDS_REQUIREMENT_REVISION":
                        state = _analyst_step(state, analyst_agent)
                        continue
                    if int(state.get("Iteration_Count", 0) or 0) >= cfg.max_iterations:
                        state["Status"] = "HUMAN_REVIEW_REQUIRED"
                        break
                return _finalize_state(state, runtime)

        return SequentialApp()

    workflow: StateGraph[GraphState] = StateGraph(GraphState)
    workflow.add_node("analyst", lambda state: _analyst_step(state, analyst_agent))
    workflow.add_node("architect", lambda state: _architect_step(state, architect_agent))
    workflow.add_node("reviewer", lambda state: _review_step(state, reviewer_agent, cfg))

    workflow.add_edge(START, "analyst")

    def analyst_routing_logic(state: GraphState) -> str:
        if state.get("Status") == "INVALID_INPUT":
            return "invalid_end"
        return "to_architect"

    workflow.add_conditional_edges(
        "analyst",
        analyst_routing_logic,
        {
            "invalid_end": END,
            "to_architect": "architect",
        },
    )
    workflow.add_edge("architect", "reviewer")

    def routing_logic(state: GraphState) -> str:
        if int(state.get("Iteration_Count", 0) or 0) >= cfg.max_iterations:
            return "human_intervention"
        if state.get("Status") == "REVIEW_PASSED":
            return "success_end"
        if state.get("Status") == "NEEDS_REQUIREMENT_REVISION":
            return "back_to_analyst"
        if state.get("Status") == "NEEDS_REVISION":
            return "back_to_architect"
        return "success_end"

    workflow.add_conditional_edges(
        "reviewer",
        routing_logic,
        {
            "back_to_analyst": "analyst",
            "back_to_architect": "architect",
            "human_intervention": END,
            "success_end": END,
        },
    )
    return workflow.compile()


def run_modeller_pipeline(user_input: str, *, config: PipelineConfig | None = None) -> GraphState:
    cfg = config or PipelineConfig()
    runtime = _resolve_runtime(cfg)
    if StateGraph is None:
        analyst_agent, architect_agent, reviewer_agent = _build_agents(cfg, runtime)
        state: GraphState = {
            "Raw_Requirement": user_input,
            "LLM_Enabled": runtime.enabled,
            "LLM_Provider": runtime.effective_provider,
            "LLM_Model": runtime.model,
            "LLM_Runtime_Status": runtime.status,
            "RAG_Enabled": cfg.enable_rag,
            "Analyst_Thinking": [],
            "Clarification_Questions": [],
            "Requirement_Items": [],
            "Knowledge_Context": [],
            "Use_Cases": [],
            "EARS_Requirement": "",
            "UML_Code": "",
            "UML_Diagram_URL": None,
            "UML_Artifacts": {},
            "Design_Elements": [],
            "Review_Feedback": "",
            "Review_Report": {},
            "Review_Blocking_Issues": [],
            "Review_Warnings": [],
            "Review_Exception_Report": {},
            "Test_Cases": [],
            "Traceability_Matrix": [],
            "Traceability_Validation": {},
            "RA_Artifacts": {},
            "RA_Validation": {},
            "Status": "START",
            "Iteration_Count": 0,
        }
        state = _analyst_step(state, analyst_agent)
        if state.get("Status") in {"INVALID_INPUT"}:
            return state
        while True:
            state = _architect_step(state, architect_agent)
            state = _review_step(state, reviewer_agent, cfg)
            if state.get("Status") == "REVIEW_PASSED":
                break
            if state.get("Status") == "NEEDS_REQUIREMENT_REVISION":
                state = _analyst_step(state, analyst_agent)
                continue
            if int(state.get("Iteration_Count", 0) or 0) >= cfg.max_iterations:
                state["Status"] = "HUMAN_REVIEW_REQUIRED"
                break
        return _finalize_state(state, runtime)

    app = build_ocean_modeller_graph(config=cfg)
    initial_state: GraphState = {
        "Raw_Requirement": user_input,
        "LLM_Enabled": runtime.enabled,
        "LLM_Provider": runtime.effective_provider,
        "LLM_Model": runtime.model,
        "LLM_Runtime_Status": runtime.status,
        "RAG_Enabled": cfg.enable_rag,
        "Analyst_Thinking": [],
        "Clarification_Questions": [],
        "Requirement_Items": [],
        "Knowledge_Context": [],
        "Use_Cases": [],
        "EARS_Requirement": "",
        "UML_Code": "",
        "UML_Diagram_URL": None,
        "UML_Artifacts": {},
        "Design_Elements": [],
        "Review_Feedback": "",
        "Review_Report": {},
        "Review_Blocking_Issues": [],
        "Review_Warnings": [],
        "Review_Exception_Report": {},
        "Test_Cases": [],
        "Traceability_Matrix": [],
        "Traceability_Validation": {},
        "RA_Artifacts": {},
        "RA_Validation": {},
        "Status": "START",
        "Iteration_Count": 0,
    }
    final_state: Any = app.invoke(initial_state)
    if int(final_state.get("Iteration_Count", 0) or 0) >= cfg.max_iterations and final_state.get("Status") in {
        "NEEDS_REVISION",
        "NEEDS_REQUIREMENT_REVISION",
    }:
        final_state["Status"] = "HUMAN_REVIEW_REQUIRED"
        if not final_state.get("Review_Exception_Report"):
            final_state["Review_Exception_Report"] = {
                "type": "human_intervention_required",
                "reason": "LangGraph 流程达到最大迭代次数后中止自动修正。",
                "iteration_count": int(final_state.get("Iteration_Count", 0) or 0),
                "blocking_issues": (final_state.get("Review_Report", {}) or {}).get("issues", [])[:5],
                "recommended_action": "请人工检查 Reviewer 反馈并手动调整需求或 UML 模型。",
            }
    return _finalize_state(final_state, runtime)
