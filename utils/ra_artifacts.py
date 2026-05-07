# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: Med-MARAG
File Name: utils/ra_artifacts.py
Author: SeaOcean
Create Date: 2026-03-21
Description：RA 制品生成与校验模块
-------------------------------------------------
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import Config
from llm.provider import LLMRuntime, build_llm_runtime, invoke_llm_text
from rag.knowledge_base import knowledge_base
from utils.plantuml_utils import PlantUMLUtils


def _md_table(rows: list[tuple[str, str]]) -> str:
    lines = ["| 主要元素 | 说明 |", "| --- | --- |"]
    for left, right in rows:
        left = (left or "").replace("|", "\\|").replace("\n", "<br>").strip()
        right = (right or "").replace("|", "\\|").replace("\n", "<br>").rstrip()
        lines.append(f"| {left} | {right} |")
    return "\n".join(lines)


def _build_process_summary(use_case: dict[str, Any]) -> str:
    main_flow = use_case.get("main_flow", []) or []
    start = main_flow[0] if main_flow else f"{use_case.get('primary_actor','主要参与者')}触发业务请求。"
    end = use_case.get("postconditions", ["系统完成业务并留下可追溯记录。"])[0]
    middle = "；".join(item.strip("。 ") for item in main_flow[1:4] if item) if len(main_flow) > 1 else "系统处理业务请求并返回结果"
    return f"用例开始于{start}\n用例执行过程：{middle}\n用例结束于{end}"


def _extract_actions_from_use_case(use_case: dict[str, Any]) -> list[str]:
    flow = use_case.get("main_flow", []) or []
    actions: list[str] = []
    for line in flow:
        if not isinstance(line, str):
            continue
        if line.startswith("系统执行："):
            actions.append(line.replace("系统执行：", "").strip("。 "))
    return actions or ["完成核心业务处理"]


def _to_operation_name(action: str) -> str:
    # Keep it readable in PlantUML even when action is Chinese.
    action = (action or "").strip().strip("。")
    action = action.replace("系统", "").replace("应", "").strip()
    if not action:
        return "executeBusiness()"
    # Heuristic: verb-object to a pseudo signature.
    compact = action.replace(" ", "")
    return f"{compact}()"


def _conceptual_class_model_from_entities(entities: list[dict[str, Any]]) -> str:
    from utils.requirement_utils import RequirementUtils

    lines = ["@startuml", "skinparam classAttributeIconSize 0"]
    entity_names: list[str] = []
    for entity in entities:
        name = entity.get("name") or "Entity"
        entity_names.append(str(name))
        attrs = entity.get("attributes", []) or []
        lines.append(f"class {name} {{")
        for attr in attrs:
            if "(" in str(attr) or ")" in str(attr):
                continue
            lines.append(f"  {attr}")
        lines.append("}")
        lines.append("")
    relationships = RequirementUtils.infer_relationships(entity_names)
    if relationships:
        lines.extend(relationships)
    lines.append("@enduml")
    return "\n".join(lines)


def _class_diagram_to_conceptual_model(class_diagram_code: str) -> str:
    cleaned = PlantUMLUtils.sanitize_plantuml(class_diagram_code)
    if not cleaned:
        return "@startuml\nskinparam classAttributeIconSize 0\n@enduml"

    lines: list[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            lines.append(line)
            continue
        if stripped.startswith("@startuml"):
            lines.append("@startuml")
            lines.append("skinparam classAttributeIconSize 0")
            continue
        if stripped.startswith("skinparam classAttributeIconSize"):
            continue
        if stripped.startswith("@enduml"):
            lines.append("@enduml")
            continue
        if stripped.startswith(("+", "-", "#")) and "(" in stripped and ")" in stripped:
            continue
        lines.append(line)

    conceptual = "\n".join(lines).strip()
    if not conceptual.endswith("@enduml"):
        conceptual += "\n@enduml"
    return conceptual


@dataclass
class RAArtifactPack:
    artifacts: dict[str, Any]
    validation: dict[str, Any]


class RAArtifactGenerator:
    """Generate RA-1..RA-6 artifacts based on the course/paper-style规范 PDFs."""

    def __init__(self, runtime: LLMRuntime | None = None, *, provider: str = "offline", enable_rag: bool = True):
        self.runtime = runtime or build_llm_runtime(provider=provider)
        self.llm = self.runtime.client
        self.enable_rag = enable_rag

    def generate(
        self,
        *,
        requirement_items: list[dict[str, Any]],
        use_cases: list[dict[str, Any]],
        uml_artifacts: dict[str, Any] | None = None,
    ) -> RAArtifactPack:
        # Always retrieve the规范 snippets and attach them to outputs for traceability.
        spec_refs = (
            knowledge_base.query("产出制品规范 RA-1 RA-2 RA-3 RA-4 RA-5 RA-6", n_results=5) if self.enable_rag else []
        )
        spec_sources = sorted({(item.get("metadata") or {}).get("source", "") for item in spec_refs if item.get("metadata")})
        uml_pack = uml_artifacts or {}

        ra1_code = (uml_pack.get("use_case_diagram", "") or "").strip() or PlantUMLUtils.generate_use_case_diagram(requirement_items)
        ra1_url = PlantUMLUtils.render_plantuml(ra1_code)

        ra2 = []
        for uc in use_cases:
            rows = [
                ("1.用例名称", str(uc.get("name", "") or "")),
                ("2.业务目标", str(uc.get("goal", "") or "")),
                ("3.用例级别", str(uc.get("level", "") or "")),
                ("4.主要参与者", str(uc.get("primary_actor", "") or "")),
                ("5.过程摘要", _build_process_summary(uc)),
            ]
            ra2.append({"use_case_id": uc.get("id", ""), "content_md": _md_table(rows)})

        # RA-3/RA-6 default to LLM rewriting when available; keep deterministic fallback.
        ra3 = []
        for uc in use_cases:
            ra3.append(
                {
                    "use_case_id": uc.get("id", ""),
                    "content_md": self._generate_ra3_detailed_use_case(uc, spec_refs),
                }
            )

        # Conceptual class model should not contain methods; reuse inferred entities from UML stage.
        entities: list[dict[str, Any]] = []
        for item in requirement_items:
            entities.extend((item.get("entities_detail") or []))  # optional hook
        if not entities:
            # fall back to local inference from requirement_items shape used in this repo
            from utils.requirement_utils import RequirementUtils

            entities = RequirementUtils.infer_entities(requirement_items)

        class_diagram_code = (uml_pack.get("class_diagram", "") or "").strip()
        if class_diagram_code:
            ra4_code = _class_diagram_to_conceptual_model(class_diagram_code)
        else:
            ra4_code = _conceptual_class_model_from_entities(entities)
        ra4_url = PlantUMLUtils.render_plantuml(ra4_code)

        ra5_code = (uml_pack.get("sequence_diagram", "") or "").strip() or PlantUMLUtils.generate_sequence_diagram_from_requirements(requirement_items)
        ra5_url = PlantUMLUtils.render_plantuml(ra5_code)

        ra6 = []
        for uc in use_cases:
            ra6.append(
                {
                    "use_case_id": uc.get("id", ""),
                    "content_md": self._generate_ra6_operation_contracts(uc, spec_refs),
                }
            )

        artifacts = {
            "standard_sources": [s for s in spec_sources if s],
            "RA-1": {
                "type": "plantuml",
                "title": "用例图",
                "plantuml": ra1_code,
                "url": ra1_url,
                "source": uml_pack.get("use_case_diagram_source", "local"),
            },
            "RA-2": {"type": "markdown", "title": "高层文本用例", "use_cases": ra2},
            "RA-3": {"type": "markdown", "title": "详细文本用例", "use_cases": ra3},
            "RA-4": {
                "type": "plantuml",
                "title": "概念类模型",
                "plantuml": ra4_code,
                "url": ra4_url,
                "source": uml_pack.get("class_diagram_source", "local"),
            },
            "RA-5": {
                "type": "plantuml",
                "title": "用例序列图",
                "plantuml": ra5_code,
                "url": ra5_url,
                "source": uml_pack.get("sequence_diagram_source", "local"),
            },
            "RA-6": {"type": "markdown", "title": "系统操作契约", "use_cases": ra6},
        }
        validation = self.validate(artifacts)
        return RAArtifactPack(artifacts=artifacts, validation=validation)

    def validate(self, artifacts: dict[str, Any]) -> dict[str, Any]:
        issues: list[str] = []

        ra1 = (artifacts.get("RA-1", {}) or {}).get("plantuml", "") or ""
        if "@startuml" not in ra1 or "@enduml" not in ra1 or "usecase" not in ra1:
            issues.append("RA-1 用例图缺少基本元素（@startuml/@enduml/usecase）。")

        ra4 = (artifacts.get("RA-4", {}) or {}).get("plantuml", "") or ""
        if "(" in ra4 or ")" in ra4:
            # conceptual class model should not contain methods.
            issues.append("RA-4 概念类模型疑似包含方法签名（出现括号）。")

        ra5 = (artifacts.get("RA-5", {}) or {})
        ra5_code = ra5.get("plantuml", "") or ""
        if not ra5_code:
            issues.append("RA-5 未生成任何用例序列图。")
        elif "@startuml" not in ra5_code or "@enduml" not in ra5_code or "participant" not in ra5_code:
            issues.append("RA-5 用例序列图缺少基本元素。")

        ra2_list = ((artifacts.get("RA-2", {}) or {}).get("use_cases", []) or [])
        for item in ra2_list:
            md = item.get("content_md", "") or ""
            if "1.用例名称" not in md or "5.过程摘要" not in md:
                issues.append(f"RA-2 用例 {item.get('use_case_id','')} 高层文本用例未按模板输出。")

        ra3_list = ((artifacts.get("RA-3", {}) or {}).get("use_cases", []) or [])
        for item in ra3_list:
            md = item.get("content_md", "") or ""
            if "8.主要过程" not in md or "9.扩展事件" not in md:
                issues.append(f"RA-3 用例 {item.get('use_case_id','')} 详细文本用例缺少关键字段。")

        ra6_list = ((artifacts.get("RA-6", {}) or {}).get("use_cases", []) or [])
        for item in ra6_list:
            md = item.get("content_md", "") or ""
            if "前置条件" not in md or "后置条件" not in md:
                issues.append(f"RA-6 用例 {item.get('use_case_id','')} 系统操作契约缺少前后置条件。")

        return {"valid": len(issues) == 0, "issues": list(dict.fromkeys(issues))}

    def _generate_ra3_detailed_use_case(self, use_case: dict[str, Any], spec_refs: list[dict[str, Any]]) -> str:
        pre = "\n".join(f"- {x}" for x in (use_case.get("preconditions", []) or [])) or "-"
        post = "\n".join(f"- {x}" for x in (use_case.get("postconditions", []) or [])) or "-"

        main_steps = []
        for idx, line in enumerate((use_case.get("main_flow", []) or []), start=1):
            if not isinstance(line, str) or not line.strip():
                continue
            main_steps.append(f"{idx}. {line.strip()}")
        main_steps_text = "\n".join(main_steps) or "1. 主要参与者触发请求。\n2. 系统处理并返回结果。"

        extensions = "E1. 触发条件：关键信息缺失或校验失败；处理过程：系统拒绝请求并提示原因。"
        nfr = "响应时间、可用性、审计与权限控制等（如需求中出现相关约束应显式列出）。"

        # Default strategy: let the LLM rewrite RA-3 when available, then fall back to the local template.
        if self.llm is not None and Config.ENABLE_LLM_RA_TEXT_ARTIFACTS:
            ref_text = "\n".join(item.get("content", "") for item in spec_refs if item.get("content"))
            prompt = f"""
你要为一个软件需求分析用例生成“详细文本用例（RA-3）”，必须严格遵循参考规范中的模板字段与写法。
只输出 Markdown 表格（两列：主要元素/说明），不要输出任何解释文字。

参考规范（节选）：
{ref_text}

用例数据：
- 用例名称：{use_case.get("name","")}
- 业务目标：{use_case.get("goal","")}
- 用例级别：{use_case.get("level","")}
- 主要参与者：{use_case.get("primary_actor","")}
- 前置条件：{pre}
- 后置条件：{post}
- 主要过程（原始步骤）：\n{main_steps_text}

必须包含字段（按编号原样出现）：
1.用例名称
2.业务目标
3.用例级别
4.主要参与者
5.过程摘要
6.前置条件
7.后置条件
8.主要过程
9.扩展事件
10.非功能性需求
11.其他
"""
            md = invoke_llm_text(self.runtime, [("system", "你只输出 Markdown 表格，不要输出任何额外文本。"), ("user", prompt)])
            if md and "| 主要元素 | 说明 |" in md and "8.主要过程" in md:
                return md.strip()

        rows = [
            ("1.用例名称", str(use_case.get("name", "") or "")),
            ("2.业务目标", str(use_case.get("goal", "") or "")),
            ("3.用例级别", str(use_case.get("level", "") or "")),
            ("4.主要参与者", str(use_case.get("primary_actor", "") or "")),
            ("5.过程摘要", _build_process_summary(use_case)),
            ("6.前置条件", pre),
            ("7.后置条件", post),
            ("8.主要过程", main_steps_text),
            ("9.扩展事件", extensions),
            ("10.非功能性需求", nfr),
            ("11.其他", "无"),
        ]
        return _md_table(rows)

    def _generate_ra6_operation_contracts(self, use_case: dict[str, Any], spec_refs: list[dict[str, Any]]) -> str:
        ops = [_to_operation_name(a) for a in _extract_actions_from_use_case(use_case)]
        pre = use_case.get("preconditions", []) or ["系统处于可服务状态"]
        post = use_case.get("postconditions", []) or ["系统状态发生可追溯变更"]

        # Default strategy: let the LLM rewrite RA-6 when available, then fall back to the local template.
        if self.llm is not None and Config.ENABLE_LLM_RA_TEXT_ARTIFACTS:
            ref_text = "\n".join(item.get("content", "") for item in spec_refs if item.get("content"))
            prompt = f"""
你要为一个用例生成“用例系统操作的契约（RA-6）”。只输出 Markdown，使用清晰小标题与表格，不要输出解释。

参考规范（节选）：
{ref_text}

用例：{use_case.get("id","")} {use_case.get("name","")}
主要参与者：{use_case.get("primary_actor","")}
候选系统操作：{", ".join(ops)}
前置条件候选：{pre}
后置条件候选：{post}

要求：
- 每个系统操作至少给出：操作签名、前置条件、后置条件、状态变化（对象创建/更新/关联变化）。
"""
            md = invoke_llm_text(self.runtime, [("system", "你只输出 Markdown，不要输出任何额外文本。"), ("user", prompt)])
            if md and "前置条件" in md and "后置条件" in md:
                return md.strip()

        sections: list[str] = []
        for op in ops[:3]:
            rows = [
                ("系统操作", op),
                ("前置条件", "\n".join(f"- {x}" for x in pre)),
                ("后置条件", "\n".join(f"- {x}" for x in post)),
                ("状态变化", "- 创建/更新核心业务对象；\n- 更新对象属性；\n- 建立或解除对象关联。"),
            ]
            sections.append(f"### {op}\n\n{_md_table(rows)}")
        return "\n\n".join(sections) if sections else "### 无可识别系统操作\n\n前置条件/后置条件无法生成。"
