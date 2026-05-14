# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: Med-MARAG
File Name: agents/architect_agent.py
Author: SeaOcean
Create Date: 2026-03-07
Description：Architect Agent 实现
-------------------------------------------------
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from llm.provider import LLMRuntime, build_llm_runtime, extract_json_object, invoke_llm_text
from rag.knowledge_base import knowledge_base
from utils.plantuml_utils import PlantUMLUtils
from utils.requirement_utils import RequirementUtils


class ArchitectLLMOutput(BaseModel):
    plantuml_code: str = ""
    design_elements: list[str] = Field(default_factory=list)
    mapping_summary: str = ""


class ArchitectAgent:
    """系统架构师智能体。"""

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

    def generate_uml(self, requirement: str | list[dict[str, Any]], feedback: str = "") -> dict[str, Any]:
        if isinstance(requirement, list):
            requirement_items = requirement
        else:
            requirement_items = [
                RequirementUtils.analyze_requirement(item, index=index)
                for index, item in enumerate(RequirementUtils.split_requirements(requirement), start=1)
            ]

        local_artifacts = self._generate_local_artifacts(requirement_items)
        local_artifacts["local_use_case_diagram"] = local_artifacts["use_case_diagram"]
        local_artifacts["local_use_case_diagram_url"] = local_artifacts["use_case_diagram_url"]
        local_artifacts["local_class_diagram"] = local_artifacts["class_diagram"]
        local_artifacts["local_class_diagram_url"] = local_artifacts["class_diagram_url"]
        local_artifacts["local_sequence_diagram"] = local_artifacts["sequence_diagram"]
        local_artifacts["local_sequence_diagram_url"] = local_artifacts["sequence_diagram_url"]
        local_artifacts["llm_use_case_diagram"] = ""
        local_artifacts["llm_use_case_diagram_url"] = None
        local_artifacts["llm_class_diagram"] = ""
        local_artifacts["llm_class_diagram_url"] = None
        local_artifacts["llm_sequence_diagram"] = ""
        local_artifacts["llm_sequence_diagram_url"] = None
        local_artifacts["llm_sandbox_issues"] = []
        local_artifacts["use_case_diagram_source"] = "local"
        local_artifacts["class_diagram_source"] = "local"
        local_artifacts["sequence_diagram_source"] = "local"
        if self.llm is not None:
            llm_use_case_result = self._llm_use_case_diagram(requirement_items, feedback)
            use_case_candidate = llm_use_case_result.get("plantuml_code", "")
            if use_case_candidate:
                local_artifacts["llm_use_case_diagram"] = use_case_candidate
                local_artifacts["llm_use_case_diagram_url"] = PlantUMLUtils.render_plantuml(use_case_candidate)
                local_artifacts["use_case_diagram"] = use_case_candidate
                local_artifacts["use_case_diagram_url"] = local_artifacts["llm_use_case_diagram_url"]
                local_artifacts["use_case_diagram_source"] = "llm"

            llm_result = self._llm_class_diagram(requirement_items, feedback)
            class_candidate = llm_result.get("plantuml_code", "")
            if class_candidate:
                local_artifacts["llm_class_diagram"] = class_candidate
                local_artifacts["llm_class_diagram_url"] = PlantUMLUtils.render_plantuml(class_candidate)
                local_artifacts["llm_sandbox_issues"] = llm_result.get("sandbox_issues", [])
                local_artifacts["class_diagram"] = class_candidate
                local_artifacts["class_diagram_url"] = local_artifacts["llm_class_diagram_url"]
                local_artifacts["class_diagram_source"] = "llm"
                if llm_result.get("design_elements"):
                    local_artifacts["design_elements"] = llm_result["design_elements"]

            llm_sequence_result = self._llm_sequence_diagram(requirement_items, feedback)
            sequence_candidate = llm_sequence_result.get("plantuml_code", "")
            if sequence_candidate:
                local_artifacts["llm_sequence_diagram"] = sequence_candidate
                local_artifacts["llm_sequence_diagram_url"] = PlantUMLUtils.render_plantuml(sequence_candidate)
                local_artifacts["sequence_diagram"] = sequence_candidate
                local_artifacts["sequence_diagram_url"] = local_artifacts["llm_sequence_diagram_url"]
                local_artifacts["sequence_diagram_source"] = "llm"

        return {
            "plantuml_code": local_artifacts["class_diagram"],
            "diagram_url": local_artifacts["class_diagram_url"],
            "sequence_diagram": local_artifacts["sequence_diagram"],
            "sequence_url": local_artifacts["sequence_diagram_url"],
            "artifacts": local_artifacts,
            "design_elements": local_artifacts["design_elements"],
        }

    def _generate_local_artifacts(self, requirement_items: list[dict[str, Any]]) -> dict[str, Any]:
        use_case_diagram = PlantUMLUtils.generate_use_case_diagram(requirement_items)
        class_diagram = PlantUMLUtils.generate_class_diagram_from_requirements(requirement_items)
        sequence_diagram = PlantUMLUtils.generate_sequence_diagram_from_requirements(requirement_items)
        entities = RequirementUtils.infer_entities(requirement_items)
        design_elements = [entity["name"] for entity in entities] + [
            item.get("use_case", {}).get("name", item.get("id", "UseCase"))
            for item in requirement_items
        ]
        return {
            "use_case_diagram": use_case_diagram,
            "use_case_diagram_url": PlantUMLUtils.render_plantuml(use_case_diagram),
            "class_diagram": class_diagram,
            "class_diagram_url": PlantUMLUtils.render_plantuml(class_diagram),
            "sequence_diagram": sequence_diagram,
            "sequence_diagram_url": PlantUMLUtils.render_plantuml(sequence_diagram),
            "entities": entities,
            "design_elements": design_elements,
        }

    def _llm_class_diagram(self, requirement_items: list[dict[str, Any]], feedback: str) -> dict[str, Any]:
        return self._llm_diagram(
            diagram_kind="class_diagram",
            requirement_items=requirement_items,
            knowledge_results=(knowledge_base.query(self._build_retrieval_query(requirement_items, feedback), n_results=4) if self.enable_rag else []),
            feedback=feedback,
        )

    def _llm_use_case_diagram(self, requirement_items: list[dict[str, Any]], feedback: str) -> dict[str, Any]:
        return self._llm_diagram(
            diagram_kind="use_case_diagram",
            requirement_items=requirement_items,
            knowledge_results=(knowledge_base.query(self._build_retrieval_query(requirement_items, feedback), n_results=4) if self.enable_rag else []),
            feedback=feedback,
        )

    def _llm_sequence_diagram(self, requirement_items: list[dict[str, Any]], feedback: str) -> dict[str, Any]:
        return self._llm_diagram(
            diagram_kind="sequence_diagram",
            requirement_items=requirement_items,
            knowledge_results=(knowledge_base.query(self._build_retrieval_query(requirement_items, feedback), n_results=4) if self.enable_rag else []),
            feedback=feedback,
        )

    def _llm_diagram(
        self,
        *,
        diagram_kind: str,
        requirement_items: list[dict[str, Any]],
        knowledge_results: list[dict[str, Any]],
        feedback: str,
    ) -> dict[str, Any]:
        prompt = self._build_architect_prompt(
            requirement_items,
            knowledge_results,
            feedback=feedback,
            diagram_kind=diagram_kind,
        )
        content = invoke_llm_text(
            self.runtime,
            [
                ("system", "你是 UML 建模专家。你只能输出严格 JSON，不要输出 Markdown 或额外解释。"),
                ("user", prompt),
            ],
        )
        if not content:
            return {}

        parsed = self._parse_llm_output(content)
        if parsed is None or not parsed.plantuml_code.strip():
            return {}

        sandbox_result = PlantUMLUtils.sandbox_preprocess(parsed.plantuml_code)
        return {
            "plantuml_code": sandbox_result["code"],
            "sandbox_issues": sandbox_result["issues"],
            "design_elements": parsed.design_elements,
            "mapping_summary": parsed.mapping_summary,
        }

    @staticmethod
    def _parse_llm_output(content: str) -> ArchitectLLMOutput | None:
        parsed = extract_json_object(content)
        if not parsed:
            return None
        try:
            return ArchitectLLMOutput.model_validate(parsed)
        except ValidationError:
            return None

    @staticmethod
    def _build_architect_prompt(
        requirement_items: list[dict[str, Any]],
        knowledge_results: list[dict[str, Any]],
        feedback: str = "",
        diagram_kind: str = "class_diagram",
    ) -> str:
        requirements = chr(10).join(f"- {item['ears_requirement']}" for item in requirement_items)
        business_context = chr(10).join(
            f"- {item.get('use_case', {}).get('name', item.get('id', '业务用例'))}: {item.get('original', '')}"
            for item in requirement_items
        )
        references = ArchitectAgent._format_knowledge_results(knowledge_results)
        target_name = {
            "class_diagram": "PlantUML 类图",
            "use_case_diagram": "PlantUML 用例图",
            "sequence_diagram": "PlantUML 序列图",
        }.get(diagram_kind, "PlantUML 图")
        target_constraints = {
            "class_diagram": "\n".join(
                [
                    "1. 只输出类图，不得输出顺序图、用例图或解释文字",
                    "2. `plantuml_code` 必须包含 @startuml 和 @enduml",
                    "3. 属性格式为 +fieldName: Type，方法格式为 +methodName()",
                    "4. 合法关系连接符仅限 --、-->、o--、*--、<|--",
                    "5. 不得编造输入需求、审查反馈或参考知识中不存在的业务事实",
                ]
            ),
            "use_case_diagram": "\n".join(
                [
                    "1. 只输出用例图，不得输出类图、顺序图或解释文字",
                    "2. `plantuml_code` 必须包含 @startuml 和 @enduml",
                    "3. 必须声明参与者 actor，并使用 usecase 表示业务用例",
                    "4. 每个核心业务用例都应与至少一个参与者建立关联",
                    "5. 不得编造输入需求、审查反馈或参考知识中不存在的业务事实",
                ]
            ),
            "sequence_diagram": "\n".join(
                [
                    "1. 只输出序列图，不得输出类图、用例图或解释文字",
                    "2. `plantuml_code` 必须包含 @startuml 和 @enduml",
                    "3. 必须声明 actor 或 participant，并体现主流程与关键分支交互",
                    "4. 应尽量覆盖输入中的跨角色协作、异常处理或补录流程",
                    "5. 不得编造输入需求、审查反馈或参考知识中不存在的业务事实",
                ]
            ),
        }.get(diagram_kind, "1. `plantuml_code` 必须包含 @startuml 和 @enduml")
        few_shot_output = {
            "class_diagram": """{
  "plantuml_code": "@startuml\\nskinparam classAttributeIconSize 0\\nclass Patient {\\n  +patientId: String\\n}\\nclass Appointment {\\n  +appointmentId: String\\n  +createAppointment()\\n}\\nclass Notification {\\n  +notificationId: String\\n  +sendNotification()\\n}\\nPatient \\"1\\" -- \\"*\\" Appointment : creates\\nAppointment \\"1\\" --> \\"*\\" Notification : triggers\\n@enduml",
  "design_elements": ["Patient", "Appointment", "Notification"],
  "mapping_summary": "从预约与通知需求中映射出实体、方法与关联关系。"
}""",
            "use_case_diagram": """{
  "plantuml_code": "@startuml\\nleft to right direction\\nactor \\"患者\\" as Patient\\nrectangle \\"Med-MARAG\\" {\\n  usecase \\"提交预约申请\\" as UC001\\n  usecase \\"接收确认通知\\" as UC002\\n}\\nPatient --> UC001\\nPatient --> UC002\\n@enduml",
  "design_elements": ["患者", "提交预约申请", "接收确认通知"],
  "mapping_summary": "提炼出患者侧核心业务用例及其参与者关系。"
}""",
            "sequence_diagram": """{
  "plantuml_code": "@startuml\\nactor \\"患者\\" as Patient\\nparticipant \\"医疗系统\\" as System\\nparticipant \\"通知服务\\" as Notify\\nPatient -> System: 提交预约申请\\nSystem -> System: 创建预约记录\\nSystem -> Notify: 发送确认通知\\nNotify --> System: 返回发送结果\\nSystem --> Patient: 返回预约成功信息\\n@enduml",
  "design_elements": ["患者", "医疗系统", "通知服务", "预约流程"],
  "mapping_summary": "按业务时序组织预约、记录创建与通知发送交互。"
}""",
        }.get(diagram_kind, "{}")
        internal_steps = {
            "class_diagram": "\n".join(
                [
                    "1. 识别核心实体与关键属性",
                    "2. 将系统行为映射为实体方法",
                    "3. 推断实体关系与合理多重性",
                    "4. 生成完整类图，并提取核心设计元素",
                ]
            ),
            "use_case_diagram": "\n".join(
                [
                    "1. 识别主要参与者与支持参与者",
                    "2. 将业务目标归纳为规范化用例名称",
                    "3. 建立参与者与用例之间的关联",
                    "4. 生成完整用例图，并提取核心设计元素",
                ]
            ),
            "sequence_diagram": "\n".join(
                [
                    "1. 识别主参与者、系统边界与关键协作者",
                    "2. 根据需求组织主成功场景和关键异常/补偿步骤",
                    "3. 用消息调用描述跨角色交互顺序",
                    "4. 生成完整序列图，并提取核心设计元素",
                ]
            ),
        }.get(diagram_kind, "1. 生成完整 PlantUML")
        return f"""
你正在 Med-MARAG 的虚拟需求分析团队中工作。该团队包含：
- 需求分析师智能体：负责输出结构化、规范化的 EARS 需求
- 系统架构师智能体（你）：负责把规范化需求映射为 UML 模型
- 合规审查员智能体：负责检查模型质量并反馈修正意见

你的任务是依据规范化 EARS 需求，生成可解析的 {target_name}。

请在内部完成以下步骤，但不要输出推理过程：
{internal_steps}

团队约束：
1. 你的上游输入来自需求分析师智能体，下游反馈来自合规审查员智能体
2. 如果审查反馈指出问题，你必须优先修正后再生成模型

建模约束：
{target_constraints}

输出要求：
1. `design_elements` 只保留类名或核心设计元素名称
2. `design_elements` 中的元素必须在 `plantuml_code` 中出现
3. `mapping_summary` 不超过 80 字

【Few-shot 示例】
输入需求：
- IF 患者预约挂号成功 THEN 系统应创建预约记录 AND 系统应发送确认通知 ENDIF
输出：
{few_shot_output}

【当前需求】
{requirements}

【业务上下文】
{business_context}

【审查反馈】
{feedback or "无"}

【参考知识】
{references}

【输出格式】
你只能输出严格 JSON，不得输出任何额外内容，JSON结构参考如下：
{{
  "plantuml_code": "完整 {target_name} 代码",
  "design_elements": ["元素1", "元素2"],
  "mapping_summary": "一句话总结"
}}
"""

    @staticmethod
    def _build_retrieval_query(requirement_items: list[dict[str, Any]], feedback: str = "") -> str:
        ears_text = " ".join(item.get("ears_requirement", "") for item in requirement_items)
        originals = " ".join(item.get("original", "") for item in requirement_items)
        entities = " ".join(entity["name"] for entity in RequirementUtils.infer_entities(requirement_items))
        use_cases = " ".join(item.get("use_case", {}).get("name", "") for item in requirement_items)
        parts = [
            originals,
            ears_text,
            entities,
            use_cases,
            "医疗 UML 类图 建模",
            "PlantUML",
            "实体 属性 方法 关系 多重性",
        ]
        if feedback:
            parts.extend([feedback, "修正 核心类 关系 覆盖"])
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
            label = f"[建模依据{index}] {source}"
            if section:
                label += f" {section}"
            content = str(item.get("content", "") or "").replace("\n", " ").strip()
            lines.append(f"{label}: {content[:220]}")
        return "\n".join(lines)

    def generate_class_diagram(self, entities: list[dict[str, Any]]) -> dict[str, Any]:
        class_diagram = PlantUMLUtils.generate_class_diagram(entities)
        return {
            "class_diagram": class_diagram,
            "class_diagram_url": PlantUMLUtils.render_plantuml(class_diagram),
        }

    def validate_uml(self, plantuml_code: str) -> dict[str, Any]:
        sandbox_result = PlantUMLUtils.sandbox_preprocess(plantuml_code)
        normalized_code = sandbox_result["code"]
        required_elements = ["@startuml", "@enduml"]
        issues = [f"缺少 {element}" for element in required_elements if element not in normalized_code]
        issues.extend(sandbox_result["issues"])
        structural_score = 30
        if "class " in normalized_code:
            structural_score += 25
        if "actor " in normalized_code or "participant " in normalized_code:
            structural_score += 25
        if " -- " in normalized_code or "->" in normalized_code:
            structural_score += 20
        return {
            "valid": len(issues) == 0,
            "issues": list(dict.fromkeys(issues)),
            "score": min(100, structural_score),
        }
