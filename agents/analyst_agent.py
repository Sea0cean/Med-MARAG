# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: Med-MARAG
File Name: agents/analyst_agent.py
Author: SeaOcean
Create Date: 2026-03-19
Description：Analyst Agent 实现
-------------------------------------------------
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from llm.provider import LLMRuntime, build_llm_runtime, extract_json_object, invoke_llm_text
from rag.knowledge_base import knowledge_base
from utils.requirement_utils import RequirementUtils


class AnalystUseCaseOutput(BaseModel):
    name: str = ""
    goal: str = ""


class AnalystLLMOutput(BaseModel):
    actors: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    business_flow: list[str] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_questions: list[str] = Field(default_factory=list)
    ears_requirement: str = ""
    summary: str = ""
    risks: list[str] = Field(default_factory=list)
    use_case: AnalystUseCaseOutput = Field(default_factory=AnalystUseCaseOutput)


class AnalystAgent:
    """需求分析智能体。"""

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

    def analyze_requirement(self, requirement_text: str, index: int = 1, feedback: str = "") -> dict[str, Any]:
        analysis = RequirementUtils.analyze_requirement(requirement_text, index=index)
        retrieval_query = self._build_retrieval_query(requirement_text, analysis, feedback=feedback)
        knowledge_results = knowledge_base.query(retrieval_query, n_results=4) if self.enable_rag else []
        analysis["knowledge_results"] = knowledge_results
        analysis["compliance"] = self._local_compliance_summary(analysis, knowledge_results)
        analysis["llm_used"] = False
        analysis["thinking_trace"] = ""
        analysis["business_flow"] = self._build_local_business_flow(analysis)
        analysis["entities"] = [entity["name"] for entity in RequirementUtils.infer_entities([analysis])]

        local_questions = self._local_clarification_questions(analysis)
        analysis["clarification_questions"] = local_questions
        analysis["missing_info"] = list(local_questions)
        analysis["needs_clarification"] = bool(local_questions)

        if self.llm is not None:
            llm_result = self._llm_requirement_enhancement(requirement_text, knowledge_results, feedback=feedback)
            if llm_result:
                analysis["llm_used"] = True
                if llm_result.get("thinking_trace"):
                    analysis["thinking_trace"] = llm_result["thinking_trace"]
                if llm_result.get("actors"):
                    analysis["actors"] = llm_result["actors"]
                if llm_result.get("entities"):
                    analysis["entities"] = llm_result["entities"]
                if llm_result.get("business_flow"):
                    analysis["business_flow"] = llm_result["business_flow"]
                if llm_result.get("clarification_questions") is not None:
                    analysis["clarification_questions"] = llm_result["clarification_questions"]
                    analysis["missing_info"] = llm_result["missing_info"]
                    analysis["needs_clarification"] = bool(llm_result["needs_clarification"])
                if llm_result.get("summary"):
                    analysis["llm_summary"] = llm_result["summary"]
                if llm_result.get("risks"):
                    analysis["llm_risks"] = llm_result["risks"]
                if llm_result.get("ears_requirement"):
                    analysis["ears_requirement"] = llm_result["ears_requirement"]
                if llm_result.get("use_case"):
                    analysis["use_case"] = {**analysis["use_case"], **llm_result["use_case"]}
        return analysis

    def _llm_requirement_enhancement(
        self,
        requirement_text: str,
        knowledge_results: list[dict[str, Any]],
        feedback: str = "",
    ) -> dict[str, Any]:
        prompt = f"""
你正在 Med-MARAG 的虚拟需求分析团队中工作。该团队包含：
- 需求分析师智能体（你）：负责理解原始需求、补全缺失信息、输出规范化需求
- 系统架构师智能体：负责根据规范化需求生成 UML 模型
- 合规审查员智能体：负责检查需求与模型的一致性、完整性和规范性

你的职责是把原始医疗需求整理成可验证、可建模的规范化结果。
请先在 <thinking>...</thinking> 中完成分析，再输出严格 JSON。

团队约束：
1. 你需要与系统架构师智能体、合规审查员智能体协作
2. 合规审查员智能体可能反馈需求缺失点，你必须根据反馈决定是继续澄清，还是输出可建模的需求结果

任务：
1. 识别参与者、业务对象、触发条件、系统动作和关键约束
2. 梳理主流程，检查是否缺少异常路径、边界条件或业务规则
3. 若信息不足，生成 1~3 个最关键的澄清问题
4. 若信息充分，输出 EARS 风格需求、正式用例名称和目标

判定规则：
1. 缺少参与者、条件、动作、对象、约束或异常处理时，needs_clarification=true
2. 只描述主成功场景且缺少明显必要分支时，视为不完整
3. 不完整时，ears_requirement 必须为空字符串

规范化要求：
1. 需求应符合“条件 + 系统 + 行为 + 对象/结果”结构
2. `ears_requirement` 必须使用 IF / THEN / AND / ENDIF
3. 多个系统行为要拆分，不得写成口语化长句
4. 不得编造输入需求、审查反馈或参考知识中不存在的信息

输出要求：
1. 只能输出 <thinking>...</thinking> 和 JSON，不要附加说明
2. `summary` 不超过 120 字
3. `missing_info` 用于概括缺失点，`clarification_questions` 用于面向用户追问

JSON 结构参考如下：
{{
  "actors": ["参与者1", "参与者2"],
  "entities": ["实体1", "实体2"],
  "business_flow": ["步骤1", "步骤2"],
  "missing_info": ["缺失点1", "缺失点2"],
  "needs_clarification": false,
  "clarification_questions": ["补充问题1", "补充问题2"],
  "ears_requirement": "使用 IF / THEN / AND / ENDIF 的规范化需求",
  "summary": "不超过120字的规范化说明",
  "risks": ["风险1", "风险2"],
  "use_case": {{
    "name": "正式、简洁的用例名称，用例命名采用动词 + 宾语的动宾结构，站在用户视角，简洁无冗余",
    "goal": "正式的业务目标描述"
  }}
}}

【输入需求】
{requirement_text}

【审查反馈】
{feedback or "无"}

【参考知识】
{self._format_knowledge_results(knowledge_results)}
"""
        content = invoke_llm_text(
            self.runtime,
            [
                (
                    "system",
                    "你必须先输出<thinking>推理过程</thinking>，再输出严格、可解析的 JSON，不要输出任何额外说明。",
                ),
                ("user", prompt),
            ],
        )
        thinking_trace = self._extract_thinking_trace(content)
        parsed = self._parse_llm_output(content)
        if parsed is None:
            return {}

        ears_requirement = parsed.ears_requirement.strip()
        if ears_requirement and not self._looks_like_ears(ears_requirement):
            ears_requirement = ""

        return {
            "thinking_trace": thinking_trace,
            "actors": parsed.actors,
            "entities": parsed.entities,
            "business_flow": parsed.business_flow,
            "missing_info": parsed.missing_info,
            "needs_clarification": parsed.needs_clarification,
            "clarification_questions": self._normalize_suggestions(parsed.clarification_questions),
            "ears_requirement": ears_requirement,
            "summary": parsed.summary.strip(),
            "risks": parsed.risks,
            "use_case": parsed.use_case.model_dump(),
        }

    @staticmethod
    def _extract_thinking_trace(content: str) -> str:
        match = re.search(r"<thinking>(.*?)</thinking>", content or "", re.S)
        if not match:
            return ""
        return f"<thinking>{match.group(1).strip()}</thinking>"

    @staticmethod
    def _parse_llm_output(content: str) -> AnalystLLMOutput | None:
        parsed = extract_json_object(content)
        if not parsed:
            return None
        try:
            return AnalystLLMOutput.model_validate(parsed)
        except ValidationError:
            return None

    @staticmethod
    def _looks_like_ears(text: str) -> bool:
        return "IF " in text and "THEN " in text and "ENDIF" in text

    @staticmethod
    def _build_local_business_flow(analysis: dict[str, Any]) -> list[str]:
        actor = analysis.get("primary_actor", "用户")
        condition = analysis.get("condition", "系统接收到业务请求")
        actions = analysis.get("actions", [])
        flow = [
            f"{actor}触发业务请求。",
            f"系统识别触发条件：{condition}。",
        ]
        flow.extend(f"系统执行：{action}。" for action in actions)
        return flow

    @staticmethod
    def _normalize_suggestions(items: list[str] | None) -> list[str]:
        """Normalize questions-like strings into suggestion-style statements."""

        if not items:
            return []

        normalized: list[str] = []
        for raw in items:
            text = str(raw or "").strip()
            if not text:
                continue
            if text.startswith("建议补充："):
                normalized.append(text.rstrip("。") + "。")
                continue

            text = text.strip().strip("？?").strip()
            text = text.rstrip("。").strip()
            suggestion = text

            m = re.match(r"^请补充(.+)$", text)
            if m:
                body = m.group(1).strip("。 ").strip()
                if "支付" in body and any(k in body for k in ("失败", "超时", "取消")):
                    suggestion = "支付失败/超时/取消处理规则"
                elif "号源" in body and any(k in body for k in ("锁定", "释放", "回滚")):
                    suggestion = "号源锁定/释放/回滚等异常处理规则"
                elif "触发" in body or "前置" in body:
                    suggestion = "触发条件/前置状态"
                elif "业务动作" in body or "具体" in body or "系统需要执行" in body:
                    suggestion = "系统需要执行的具体业务动作/系统行为"
                else:
                    suggestion = body

            m = re.match(r"^请明确(.+)$", text)
            if m:
                body = m.group(1).strip("。 ").strip()
                if "主要参与" in body or "参与角色" in body:
                    suggestion = "主要参与角色（例如患者/医生/护士/管理员）"
                else:
                    suggestion = body

            suggestion = suggestion.strip().rstrip("。")
            if not suggestion:
                continue
            normalized.append(f"建议补充：{suggestion}。")

        # Preserve order while de-duping.
        deduped: list[str] = []
        for item in normalized:
            if item not in deduped:
                deduped.append(item)
        return deduped

    @staticmethod
    def _local_clarification_questions(analysis: dict[str, Any]) -> list[str]:
        original = str(analysis.get("original", "") or "")
        questions: list[str] = []
        if analysis.get("primary_actor") == "用户":
            questions.append("建议补充：主要参与角色（例如患者/医生/护士/管理员）。")
        if not analysis.get("condition"):
            questions.append("建议补充：触发条件/前置状态。")
        if not analysis.get("actions"):
            questions.append("建议补充：系统需要执行的具体业务动作/系统行为。")

        abnormal_pairs = [
            (("支付",), ("失败", "超时", "取消"), "建议补充：支付失败/超时/取消处理规则。"),
            (("号源",), ("锁定", "释放", "回滚"), "建议补充：号源锁定/释放/回滚等异常处理规则。"),
        ]
        for required_keywords, abnormal_keywords, question in abnormal_pairs:
            if any(keyword in original for keyword in required_keywords) and not any(
                keyword in original for keyword in abnormal_keywords
            ):
                questions.append(question)

        deduped: list[str] = []
        for question in questions:
            if question not in deduped:
                deduped.append(question)
        return deduped

    def _local_compliance_summary(
        self, analysis: dict[str, Any], knowledge_results: list[dict[str, Any]]
    ) -> str:
        issues = analysis.get("validation", {}).get("issues", [])
        if not issues:
            base = "需求已具备基础的条件、主体和动作表达。"
        else:
            base = "；".join(issues)
        evidence = " ".join(item["content"] for item in knowledge_results[:2])
        return f"{base} 参考依据：{evidence}"

    @staticmethod
    def _build_retrieval_query(requirement_text: str, analysis: dict[str, Any], feedback: str = "") -> str:
        actors = " ".join(actor for actor in analysis.get("actors", []) if actor and actor != "系统")
        actions = " ".join(analysis.get("actions", [])[:3])
        domain = analysis.get("domain", "")
        parts = [
            requirement_text,
            domain,
            actors,
            actions,
            "医疗软件需求分析",
            "ISO/IEC/IEEE 29148:2018",
            "EARS",
            "可验证 无歧义 单一性",
        ]
        if feedback:
            parts.extend([feedback, "需求补全 异常路径 业务规则"])
        return " ".join(part for part in parts if part)

    @staticmethod
    def _format_knowledge_results(knowledge_results: list[dict[str, Any]]) -> str:
        if not knowledge_results:
            return "无"
        lines: list[str] = []
        for index, item in enumerate(knowledge_results[:4], start=1):
            metadata = item.get("metadata", {}) or {}
            source = metadata.get("source", metadata.get("type", "知识库"))
            section = metadata.get("section", "")
            prefix = f"[依据{index}] {source}"
            if section:
                prefix += f" {section}"
            content = str(item.get("content", "") or "").replace("\n", " ").strip()
            lines.append(f"{prefix}: {content[:220]}")
        return "\n".join(lines)

    def generate_requirement_id(self, index: int | None = None, requirement_text: str = "") -> str:
        return RequirementUtils.generate_requirement_id(index=index, requirement_text=requirement_text)

    def extract_domain(self, requirement_text: str) -> str:
        return RequirementUtils.extract_domain(requirement_text)
