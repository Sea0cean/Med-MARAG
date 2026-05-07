# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: Med-MARAG
File Name: agents/review_agent.py
Author: SeaOcean
Create Date: 2026-03-07
Description：Reviewer Agent 实现
-------------------------------------------------
"""
from __future__ import annotations

import re
from typing import Any

from config import Config
from llm.provider import LLMRuntime, build_llm_runtime, extract_json_object, invoke_llm_text
from rag.knowledge_base import knowledge_base
from utils.requirement_utils import RequirementUtils


class ReviewAgent:
    """审查智能体。"""

    DIAGRAM_CONSISTENCY_WEIGHT = 0.7
    DOMAIN_KNOWLEDGE_WEIGHT = 0.3

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

    def review_requirement(self, requirement: str) -> dict[str, Any]:
        issues: list[str] = []
        if "IF " not in requirement or "THEN " not in requirement:
            issues.append("EARS 结构不完整，缺少 IF/THEN。")
        if "系统应" not in requirement:
            issues.append("未体现系统行为主体。")
        if "执行操作" in requirement:
            issues.append("动作仍然过于笼统，建议补充具体业务动作。")

        knowledge_results = knowledge_base.query(requirement, n_results=2) if self.enable_rag else []
        compliance_score = 85 if knowledge_results else 70
        completeness = 100 - len(issues) * 18
        clarity = 95 if "AND 系统应" in requirement or requirement.count("系统应") == 1 else 88
        testability = 92 if any(keyword in requirement for keyword in ("短信", "显示", "验证", "通知", "生成")) else 78
        overall = max(40, min(100, int((completeness + clarity + testability + compliance_score) / 4)))
        return {
            "scores": {
                "completeness": completeness,
                "clarity": clarity,
                "testability": testability,
                "compliance": compliance_score,
                "overall": overall,
            },
            "issues": issues,
            "suggestions": self.generate_improvement_suggestions(requirement),
        }

    def review_uml(
        self,
        plantuml_code: str,
        *,
        diagram_name: str = "",
        requirement_items: list[dict[str, Any]] | None = None,
        ears_requirement: str = "",
    ) -> dict[str, Any]:
        local_review = self._local_review_uml(
            plantuml_code,
            diagram_name=diagram_name,
            requirement_items=requirement_items,
            ears_requirement=ears_requirement,
        )
        llm_review_enabled = (
            diagram_name in {"class_diagram", "use_case_diagram", "sequence_diagram"}
            or Config.ENABLE_LLM_REVIEW_FOR_ALL_DIAGRAMS
        )
        if not llm_review_enabled:
            local_review["review_source"] = "local"
            return local_review
        if self.llm is None:
            local_review["review_source"] = "local"
            return local_review

        llm_review = self._llm_review_uml(
            plantuml_code,
            diagram_name=diagram_name,
            requirement_items=requirement_items,
            ears_requirement=ears_requirement,
        )
        llm_scores = (llm_review or {}).get("scores", {}) if isinstance(llm_review, dict) else {}
        # If the LLM output is not parseable or doesn't match expected schema, fall back to local review.
        if not llm_review or not isinstance(llm_review, dict) or "overall" not in llm_scores:
            local_review["review_source"] = "local"
            return local_review

        merged_issues = list(dict.fromkeys((llm_review.get("issues") or []) + (local_review.get("issues") or [])))
        merged_suggestions = list(
            dict.fromkeys((llm_review.get("suggestions") or []) + (local_review.get("suggestions") or []))
        )
        scores = llm_scores
        if local_review.get("issues"):
            scores["overall"] = min(int(scores.get("overall", 0) or 0), int(local_review["scores"]["overall"]))
        return {
            "scores": scores,
            "issues": merged_issues,
            "suggestions": merged_suggestions,
            "review_source": "llm",
            "llm_raw_review": llm_review.get("llm_raw_review", ""),
            "evidence": llm_review.get("evidence", []),
        }

    def _local_review_uml(
        self,
        plantuml_code: str,
        *,
        diagram_name: str = "",
        requirement_items: list[dict[str, Any]] | None = None,
        ears_requirement: str = "",
    ) -> dict[str, Any]:
        issues: list[str] = []
        if "@startuml" not in plantuml_code or "@enduml" not in plantuml_code:
            issues.append("缺少 @startuml 或 @enduml 包裹。")
        if "class " not in plantuml_code and "participant " not in plantuml_code and "usecase " not in plantuml_code:
            issues.append("未识别到 UML 结构元素。")

        declared = self._collect_declared_aliases(plantuml_code)
        used = self._collect_message_aliases(plantuml_code)
        undeclared = sorted(alias for alias in used if alias not in declared and alias != "System")
        if undeclared:
            issues.append(f"存在未声明参与者: {', '.join(undeclared)}。")

        issues.extend(self._check_uml_syntax_lines(plantuml_code))
        issues.extend(
            self._check_requirement_coverage(
                plantuml_code,
                diagram_name=diagram_name,
                requirement_items=requirement_items,
                ears_requirement=ears_requirement,
            )
        )

        completeness = max(45, 100 - len(issues) * 18)
        clarity = 95 if len(plantuml_code.splitlines()) >= 6 else 72
        compliance = 90 if any(term in plantuml_code for term in ("Appointment", "MedicalRecord", "Notification", "TriageTask")) else 76
        overall = max(40, min(100, int((completeness + clarity + compliance) / 3)))
        suggestions = []
        if "Notification" not in plantuml_code and "短信" in plantuml_code:
            suggestions.append("可在类图中补充 Notification 实体以体现通知机制。")
        if "Audit" not in plantuml_code and "病历" in plantuml_code:
            suggestions.append("病历场景建议补充审计或权限相关对象。")
        return {
            "scores": {
                "accuracy": completeness,
                "completeness": completeness,
                "clarity": clarity,
                "compliance": compliance,
                "overall": overall,
            },
            "issues": issues,
            "suggestions": suggestions,
        }

    def _llm_review_uml(
        self,
        plantuml_code: str,
        *,
        diagram_name: str = "",
        requirement_items: list[dict[str, Any]] | None = None,
        ears_requirement: str = "",
    ) -> dict[str, Any]:
        review_query = self._build_review_query(
            diagram_name=diagram_name,
            requirement_items=requirement_items,
            ears_requirement=ears_requirement,
        )
        knowledge_results = knowledge_base.query(review_query, n_results=4) if self.enable_rag else []
        evidence_lines = self._format_knowledge_results(knowledge_results)
        evidence_block = "\n".join(evidence_lines)
        prompt = f"""
你正在 Med-MARAG 的虚拟需求分析团队中工作。该团队包含：
- 需求分析智能体：负责需求澄清与 EARS 规范化
- 系统架构师智能体：负责生成 UML 模型
- 审查智能体（你）：负责检查模型质量、识别缺失信息、生成修正建议

你的任务是依据当前 EARS 需求和参考知识，对 PlantUML 和生成的文本用例进行证据驱动审查，并输出可执行反馈。

团队约束：
1. 你的上游输入来自系统架构师智能体生成的 UML 代码。
2. 你的审查结果会反馈给上游，用于下一轮需求修正或模型修正。

审查重点：
1. 是否包含完整边界：@startuml / @enduml
2. 是否具备核心类、属性、方法和关系
3. 类名、关系、多重性和方法语义是否与医疗业务一致
4. 是否缺少关键实体、关联或通知/病历/支付等场景要素

审查约束：
1. 只审查当前输入，不得编造未提供的业务事实
2. 判断必须优先参考当前 EARS 需求和参考知识
3. 问题要尽量定位到类、属性、关系或代码位置
4. 输出必须是严格 JSON，不要输出额外解释

JSON 结构：
{{
  "scores": {{
    "accuracy": 0,
    "completeness": 0,
    "clarity": 0,
    "compliance": 0,
    "overall": 0
  }},
  "issues": ["问题1", "问题2"],
  "suggestions": ["建议1", "建议2"],
 "evidence": ["依据1", "依据2"]
}}

当前 EARS 需求：
{ears_requirement or "无"}

PlantUML:
{plantuml_code}

参考知识：
{evidence_block}
"""
        content = invoke_llm_text(
            self.runtime,
            [
                ("system", "你输出严格、可解析的 JSON。"),
                ("user", prompt),
            ],
        )
        parsed = extract_json_object(content)
        if not parsed:
            return {}
        parsed["llm_raw_review"] = content
        parsed["evidence"] = parsed.get("evidence") or evidence_lines[:2]
        return parsed

    def review_model_pack(
        self,
        uml_artifacts: dict[str, Any],
        requirement_items: list[dict[str, Any]] | None = None,
        ears_requirement: str = "",
    ) -> dict[str, Any]:
        diagrams = {
            "use_case_diagram": uml_artifacts.get("use_case_diagram", ""),
            "class_diagram": uml_artifacts.get("class_diagram", ""),
            "sequence_diagram": uml_artifacts.get("sequence_diagram", ""),
        }
        issues: list[str] = []
        scores = []
        details: dict[str, Any] = {}
        sources: set[str] = set()
        for name, diagram in diagrams.items():
            if not diagram:
                issues.append(f"{name} 缺失。")
                continue
            review = self.review_uml(
                diagram,
                diagram_name=name,
                requirement_items=requirement_items,
                ears_requirement=ears_requirement,
            )
            scores.append(int((review.get("scores") or {}).get("overall", 0) or 0))
            issues.extend(f"{name}: {issue}" for issue in review.get("issues", []))
            details[name] = review
            sources.add(review.get("review_source", "local"))

        if requirement_items:
            use_case_count = len(requirement_items)
            if diagrams["use_case_diagram"].count("usecase ") < use_case_count:
                issues.append("用例图数量少于需求条目数量。")
            use_case_review = self.review_use_case_pack([item.get("use_case", {}) for item in requirement_items])
            issues.extend(use_case_review.get("issues", []))
            details["use_cases"] = use_case_review
            sources.add(use_case_review.get("review_source", "local"))

        diagram_consistency = int(sum(scores) / len(scores)) if scores else 0
        domain_knowledge = self.score_domain_knowledge_alignment(
            uml_artifacts,
            requirement_items=requirement_items,
            ears_requirement=ears_requirement,
        )
        domain_knowledge_score = int(domain_knowledge.get("score", 0) or 0)
        overall = self._compose_overall_score(
            diagram_consistency=diagram_consistency,
            domain_knowledge_alignment=domain_knowledge_score,
            issues_count=len(issues),
        )
        coordination_bonus = self._coordination_bonus(len(issues))
        details["domain_knowledge"] = domain_knowledge
        return {
            "scores": {
                "overall": overall,
                "diagram_consistency": diagram_consistency,
                "compliance": max(0, int(round(diagram_consistency * 0.8 + domain_knowledge_score * 0.2)) - 3),
                "domain_knowledge_alignment": domain_knowledge_score,
                "coordination_bonus": coordination_bonus,
            },
            "issues": issues,
            "suggestions": ["若审查未通过，可根据反馈补充实体、参与者或异常分支。"] if issues else [],
            "details": details,
            "review_source": "llm" if "llm" in sources else "local",
        }

    def score_domain_knowledge_alignment(
        self,
        uml_artifacts: dict[str, Any],
        *,
        requirement_items: list[dict[str, Any]] | None = None,
        ears_requirement: str = "",
    ) -> dict[str, Any]:
        query = self._build_domain_knowledge_query(requirement_items=requirement_items, ears_requirement=ears_requirement)
        knowledge_results = knowledge_base.query(query, n_results=5) if self.enable_rag else []
        artifact_text = "\n".join(
            str(uml_artifacts.get(name, "") or "")
            for name in ("use_case_diagram", "class_diagram", "sequence_diagram")
        )
        artifact_text_lower = artifact_text.lower()

        inferred_entities = RequirementUtils.infer_entities(requirement_items or [])
        requirement_entities = {entity["name"] for entity in inferred_entities}
        raw_knowledge_entities = self._extract_expected_entities_from_knowledge(knowledge_results)
        knowledge_entities = self._filter_knowledge_entities_by_requirement(
            raw_knowledge_entities,
            requirement_items=requirement_items,
            ears_requirement=ears_requirement,
        )
        expected_entities = requirement_entities | knowledge_entities

        matched_entities = sorted(
            entity_name
            for entity_name in expected_entities
            if f"class {entity_name}".lower() in artifact_text_lower or entity_name.lower() in artifact_text_lower
        )
        missing_entities = sorted(entity_name for entity_name in expected_entities if entity_name not in matched_entities)
        entity_coverage = len(matched_entities) / len(expected_entities) if expected_entities else 1.0
        matched_knowledge_entities = sorted(entity_name for entity_name in knowledge_entities if entity_name in matched_entities)
        missing_knowledge_entities = sorted(entity_name for entity_name in knowledge_entities if entity_name not in matched_entities)
        knowledge_entity_coverage = len(matched_knowledge_entities) / len(knowledge_entities) if knowledge_entities else 1.0

        expected_methods = self._extract_expected_methods(
            inferred_entities=inferred_entities,
            knowledge_results=knowledge_results,
            requirement_items=requirement_items,
        )
        matched_methods = sorted(
            method_name
            for method_name, indicators in expected_methods.items()
            if any(indicator in artifact_text_lower for indicator in indicators)
        )
        missing_methods = sorted(method_name for method_name in expected_methods if method_name not in matched_methods)
        method_coverage = len(matched_methods) / len(expected_methods) if expected_methods else 1.0

        expected_relationships = self._extract_expected_relationships(
            expected_entities=expected_entities,
            requirement_items=requirement_items,
        )
        matched_relationships = sorted(
            relation_name
            for relation_name, relation_spec in expected_relationships.items()
            if self._match_relationship(artifact_text_lower, relation_spec)
        )
        missing_relationships = sorted(
            relation_name for relation_name in expected_relationships if relation_name not in matched_relationships
        )
        relationship_coverage = len(matched_relationships) / len(expected_relationships) if expected_relationships else 1.0

        expected_cues = self._extract_expected_knowledge_cues(
            knowledge_results,
            requirement_items=requirement_items,
            ears_requirement=ears_requirement,
        )
        matched_cues = sorted(
            cue_name
            for cue_name, indicators in expected_cues.items()
            if any(indicator.lower() in artifact_text_lower for indicator in indicators)
        )
        missing_cues = sorted(cue_name for cue_name in expected_cues if cue_name not in matched_cues)
        cue_coverage = len(matched_cues) / len(expected_cues) if expected_cues else 1.0

        structural_grounding_bonus = 0
        if knowledge_results and (matched_knowledge_entities or matched_cues):
            structural_grounding_bonus = min(
                8,
                int(round(method_coverage * 4 + relationship_coverage * 4))
                + (2 if matched_knowledge_entities else 0),
            )

        knowledge_grounding_bonus = min(10, len(matched_knowledge_entities) * 3 + len(matched_cues))

        weighted_score = (
            26 * entity_coverage
            + 21 * knowledge_entity_coverage
            + 22 * method_coverage
            + 18 * relationship_coverage
            + 14 * cue_coverage
        )
        penalty = min(6, len(missing_knowledge_entities) * 2) + min(4, len(missing_cues))
        score = int(round(weighted_score - penalty + structural_grounding_bonus + knowledge_grounding_bonus))
        score = max(0, min(100, score))
        return {
            "score": score,
            "query": query,
            "evidence_ids": [item.get("id") for item in knowledge_results],
            "requirement_entities": sorted(requirement_entities),
            "knowledge_entities": sorted(knowledge_entities),
            "matched_entities": matched_entities,
            "missing_entities": missing_entities,
            "matched_knowledge_entities": matched_knowledge_entities,
            "missing_knowledge_entities": missing_knowledge_entities,
            "matched_methods": matched_methods,
            "missing_methods": missing_methods,
            "matched_relationships": matched_relationships,
            "missing_relationships": missing_relationships,
            "matched_cues": matched_cues,
            "missing_cues": missing_cues,
            "coverage": {
                "entity": round(entity_coverage, 4),
                "knowledge_entity": round(knowledge_entity_coverage, 4),
                "method": round(method_coverage, 4),
                "relationship": round(relationship_coverage, 4),
                "cue": round(cue_coverage, 4),
            },
            "penalty": penalty,
            "structural_grounding_bonus": structural_grounding_bonus,
            "knowledge_grounding_bonus": knowledge_grounding_bonus,
        }

    @classmethod
    def _compose_overall_score(
        cls,
        *,
        diagram_consistency: int,
        domain_knowledge_alignment: int,
        issues_count: int = 0,
    ) -> int:
        weighted = (
            diagram_consistency * cls.DIAGRAM_CONSISTENCY_WEIGHT
            + domain_knowledge_alignment * cls.DOMAIN_KNOWLEDGE_WEIGHT
        )
        return max(40, min(100, int(round(weighted + cls._coordination_bonus(issues_count)))))

    @staticmethod
    def _coordination_bonus(issues_count: int) -> int:
        if issues_count <= 2:
            return 3
        if issues_count <= 4:
            return 2
        if issues_count <= 6:
            return 1
        return 0

    @staticmethod
    def _build_review_query(
        *,
        diagram_name: str = "",
        requirement_items: list[dict[str, Any]] | None = None,
        ears_requirement: str = "",
    ) -> str:
        requirement_context = ears_requirement.strip()
        if not requirement_context and requirement_items:
            requirement_context = " ".join(
                item.get("ears_requirement", "") or item.get("original", "") for item in requirement_items
            ).strip()
        entities = ""
        if requirement_items:
            entities = " ".join(entity["name"] for entity in RequirementUtils.infer_entities(requirement_items))
        return " ".join(
            part
            for part in [
                "医疗软件",
                diagram_name,
                "UML 审查",
                requirement_context,
                entities,
                "ISO/IEC/IEEE 29148:2018",
                "EARS",
                "完整性 合规性 核心类 关系 覆盖",
            ]
            if part
        )

    @staticmethod
    def _build_domain_knowledge_query(
        *,
        requirement_items: list[dict[str, Any]] | None = None,
        ears_requirement: str = "",
    ) -> str:
        requirement_context = ears_requirement.strip()
        originals = ""
        if requirement_items:
            originals = " ".join(item.get("original", "") for item in requirement_items)
            if not requirement_context:
                requirement_context = " ".join(item.get("ears_requirement", "") or item.get("original", "") for item in requirement_items).strip()
        domain = ""
        if requirement_items:
            domain = " ".join(sorted({str(item.get("domain", "") or "") for item in requirement_items if item.get("domain")}))
        return " ".join(
            part
            for part in [
                "医疗业务领域知识 对齐",
                originals,
                requirement_context,
                domain,
                "ISO/IEC/IEEE 29148:2018",
                "医疗规范",
                "通知 权限 审计 支付 病历 分诊 挂号",
            ]
            if part
        )

    @staticmethod
    def _extract_expected_entities_from_knowledge(knowledge_results: list[dict[str, Any]]) -> set[str]:
        expected_entities: set[str] = set()
        for item in knowledge_results:
            content = str(item.get("content", "") or "")
            metadata = item.get("metadata", {}) or {}
            tags = " ".join(str(tag) for tag in metadata.get("tags", []) or [])
            combined = f"{content} {tags}"
            for entity_name, keywords in RequirementUtils.ENTITY_KEYWORD_LIBRARY.items():
                if entity_name in combined or any(keyword in combined for keyword in keywords):
                    expected_entities.add(entity_name)
        return expected_entities

    @staticmethod
    def _filter_knowledge_entities_by_requirement(
        knowledge_entities: set[str],
        *,
        requirement_items: list[dict[str, Any]] | None = None,
        ears_requirement: str = "",
    ) -> set[str]:
        if not knowledge_entities:
            return set()
        requirement_text = f"{ears_requirement} " + " ".join(
            f"{item.get('original', '')} {' '.join(item.get('actions', []))}"
            for item in (requirement_items or [])
        )
        lowered = requirement_text.lower()
        filtered = {
            entity_name
            for entity_name in knowledge_entities
            if any(keyword.lower() in lowered for keyword in RequirementUtils.ENTITY_KEYWORD_LIBRARY.get(entity_name, []))
        }
        return filtered or set(list(knowledge_entities)[:2])

    @staticmethod
    def _extract_expected_methods(
        *,
        inferred_entities: list[dict[str, Any]],
        knowledge_results: list[dict[str, Any]],
        requirement_items: list[dict[str, Any]] | None = None,
    ) -> dict[str, list[str]]:
        expected_methods: dict[str, list[str]] = {}
        knowledge_text = " ".join(
            f"{item.get('content', '')} {' '.join((item.get('metadata', {}) or {}).get('tags', []) or [])}"
            for item in knowledge_results
        )
        merged_actions = []
        if requirement_items:
            for item in requirement_items:
                merged_actions.extend(item.get("actions", []))

        for entity in inferred_entities:
            for signature in entity.get("methods", []):
                token = signature.replace("+", "").replace("()", "").lower()
                expected_methods[token] = [token]

        for entity_name in ReviewAgent._extract_expected_entities_from_knowledge(knowledge_results):
            for signature in RequirementUtils._infer_methods(entity_name, knowledge_text, merged_actions):
                token = signature.replace("+", "").replace("()", "").lower()
                expected_methods.setdefault(token, [token])

        return expected_methods

    @staticmethod
    def _extract_expected_relationships(
        *,
        expected_entities: set[str],
        requirement_items: list[dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, str]]:
        relationships = RequirementUtils.infer_relationships(sorted(expected_entities), requirement_items)
        expected: dict[str, dict[str, str]] = {}
        for relation in relationships:
            relation_head, _, relation_label = relation.partition(":")
            relation_tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", relation_head)
            if len(relation_tokens) < 2:
                continue
            source = relation_tokens[0]
            target = relation_tokens[-1]
            key = f"{source}->{target}"
            expected[key] = {
                "source": source.lower(),
                "target": target.lower(),
                "label": relation_label.strip().lower(),
            }
        return expected

    @staticmethod
    def _match_relationship(artifact_text_lower: str, relation_spec: dict[str, str]) -> bool:
        source = relation_spec.get("source", "")
        target = relation_spec.get("target", "")
        label = relation_spec.get("label", "")
        if not source or not target:
            return False
        for line in artifact_text_lower.splitlines():
            if source in line and target in line:
                if not label or label in line:
                    return True
        return False

    @staticmethod
    def _extract_expected_knowledge_cues(
        knowledge_results: list[dict[str, Any]],
        *,
        requirement_items: list[dict[str, Any]] | None = None,
        ears_requirement: str = "",
    ) -> dict[str, list[str]]:
        cue_library = {
            "通知机制": ["notification", "sendnotification", "短信", "通知", "消息"],
            "支付结算": ["billingrecord", "payment", "支付", "结算", "账单"],
            "病历管理": ["medicalrecord", "病历", "健康记录"],
            "分诊流程": ["triagetask", "分诊", "优先级", "排队"],
            "身份认证": ["authsession", "登录", "认证", "鉴权", "token"],
            "处方管理": ["prescription", "处方", "药品", "发药"],
            "预约挂号": ["appointment", "预约", "挂号", "号源"],
            "审计权限": ["audit", "审计", "权限", "访问控制"],
        }

        context = ears_requirement
        if requirement_items:
            context = f"{context} " + " ".join(item.get("original", "") for item in requirement_items)
        knowledge_text = " ".join(
            f"{item.get('content', '')} {' '.join((item.get('metadata', {}) or {}).get('tags', []) or [])}"
            for item in knowledge_results
        )
        merged = f"{context} {knowledge_text}".lower()

        expected: dict[str, list[str]] = {}
        for cue_name, indicators in cue_library.items():
            if any(indicator.lower() in merged for indicator in indicators):
                expected[cue_name] = indicators
        return expected

    @staticmethod
    def _format_knowledge_results(knowledge_results: list[dict[str, Any]]) -> list[str]:
        if not knowledge_results:
            return ["无"]
        lines: list[str] = []
        for index, item in enumerate(knowledge_results[:4], start=1):
            metadata = item.get("metadata", {}) or {}
            source = metadata.get("source", metadata.get("type", "知识库"))
            section = metadata.get("section", "")
            label = f"[审查依据{index}] {source}"
            if section:
                label += f" {section}"
            content = str(item.get("content", "") or "").replace("\n", " ").strip()
            lines.append(f"{label}: {content[:220]}")
        return lines

    @staticmethod
    def _check_uml_syntax_lines(plantuml_code: str) -> list[str]:
        issues: list[str] = []
        lines = plantuml_code.splitlines()
        for index, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("@"):
                continue
            if any(operator in stripped for operator in ("--", "->", "<|", "o--", "*--")) and ":" not in stripped:
                issues.append(f"第 {index} 行关系定义缺少说明标签或语义标注。")
            if stripped.startswith("class ") and "{" not in stripped and index < len(lines):
                block = "\n".join(lines[index - 1 : min(index + 3, len(lines))])
                if "}" not in block:
                    issues.append(f"第 {index} 行类定义可能未正确闭合。")
        return issues

    @staticmethod
    def _check_requirement_coverage(
        plantuml_code: str,
        *,
        diagram_name: str = "",
        requirement_items: list[dict[str, Any]] | None = None,
        ears_requirement: str = "",
    ) -> list[str]:
        if diagram_name != "class_diagram" or not requirement_items:
            return []

        issues: list[str] = []
        expected_entities = [entity["name"] for entity in RequirementUtils.infer_entities(requirement_items)]
        for entity_name in expected_entities:
            if f"class {entity_name}" not in plantuml_code:
                issues.append(f"类图缺少核心类 {entity_name}，与当前 EARS 需求不一致。")

        expected_pairs = [
            ("Notification", ("短信", "通知", "提醒")),
            ("BillingRecord", ("支付", "结算", "账单")),
            ("MedicalRecord", ("病历", "健康记录")),
        ]
        merged_text = f"{ears_requirement} " + " ".join(
            f"{item.get('original', '')} {' '.join(item.get('actions', []))}" for item in requirement_items
        )
        for entity_name, keywords in expected_pairs:
            if any(keyword in merged_text for keyword in keywords) and f"class {entity_name}" not in plantuml_code:
                issues.append(f"类图未覆盖与“{entity_name}”相关的业务语义，建议补充对应实体。")
        return issues

    def review_use_case_pack(self, use_cases: list[dict[str, Any]]) -> dict[str, Any]:
        issues: list[str] = []
        suggestions: list[str] = []
        colloquial_patterns = ["给患者", "给医生", "给用户", "一下", "然后", "成功了", "弄", "发短信"]

        for use_case in use_cases:
            name = str(use_case.get("name", "") or "")
            goal = str(use_case.get("goal", "") or "")
            if any(pattern in name for pattern in colloquial_patterns):
                issues.append(f"用例名称“{name}”偏口语化，建议改为正式业务短语。")
            if len(name) > 18:
                issues.append(f"用例名称“{name}”过长，建议压缩为简洁动宾结构。")
            if "完成" in goal and len(goal) > 30:
                suggestions.append(f"可进一步精炼用例目标“{goal}”。")

        return {
            "scores": {
                "overall": 100 if not issues else max(60, 100 - len(issues) * 12),
            },
            "issues": issues,
            "suggestions": suggestions,
            "review_source": "local",
        }

    @staticmethod
    def _collect_declared_aliases(plantuml_code: str) -> set[str]:
        declared = {"System"}
        for pattern in (
            r"\bactor\b.*\bas\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\bparticipant\b.*\bas\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\busecase\b.*\bas\s+([A-Za-z_][A-Za-z0-9_]*)",
        ):
            declared.update(re.findall(pattern, plantuml_code))
        return declared

    @staticmethod
    def _collect_message_aliases(plantuml_code: str) -> set[str]:
        aliases: set[str] = set()
        for line in plantuml_code.splitlines():
            if "->" not in line and "--" not in line and "<|" not in line:
                continue
            candidate = re.sub(r'"[^"]*"', " ", line.split(":", 1)[0])
            names = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", candidate)
            if len(names) >= 2:
                aliases.update({names[0], names[-1]})
        return {alias for alias in aliases if alias}

    def generate_improvement_suggestions(self, requirement: str) -> list[str]:
        suggestions = []
        if len(requirement) < 50:
            suggestions.append("建议增加更多细节，使需求更加具体。")
        if "系统应" not in requirement:
            suggestions.append("建议使用更明确的行为描述，如“系统应”或“系统必须”。")
        if "当" not in requirement and "IF " not in requirement:
            suggestions.append("建议明确需求触发条件。")
        return suggestions
