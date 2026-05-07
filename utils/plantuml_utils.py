# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: Med-MARAG
File Name: utils/plantuml_utils.py
Author: SeaOcean
Create Date: 2026-03-07
Description：PlantUML 制品生成工具
-------------------------------------------------
"""
from __future__ import annotations

import re
import zlib
from typing import Any

from config import Config
from utils.requirement_utils import RequirementUtils


class PlantUMLUtils:
    """PlantUML 代码生成工具。"""

    _PLANTUML_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
    _ACTOR_ALIAS = {
        "患者": "Patient",
        "医生": "Doctor",
        "护士": "Nurse",
        "药师": "Pharmacist",
        "管理员": "Admin",
        "家属": "Family",
        "用户": "User",
    }
    _RELATION_OPERATORS = ("<|--", "<|..", "*--", "o--", "-->", "..>", "--", "<--", "<..")

    @staticmethod
    def _encode_6bit(b: int) -> str:
        return PlantUMLUtils._PLANTUML_ALPHABET[b & 0x3F]

    @staticmethod
    def _append_3bytes(b1: int, b2: int, b3: int) -> str:
        c1 = b1 >> 2
        c2 = ((b1 & 0x3) << 4) | (b2 >> 4)
        c3 = ((b2 & 0xF) << 2) | (b3 >> 6)
        c4 = b3 & 0x3F
        return (
            PlantUMLUtils._encode_6bit(c1)
            + PlantUMLUtils._encode_6bit(c2)
            + PlantUMLUtils._encode_6bit(c3)
            + PlantUMLUtils._encode_6bit(c4)
        )

    @staticmethod
    def encode_plantuml(plantuml_code: str) -> str:
        raw = plantuml_code.encode("utf-8")
        compressed = zlib.compress(raw, level=9)[2:-4]
        result: list[str] = []
        i = 0
        while i < len(compressed):
            b1 = compressed[i]
            b2 = compressed[i + 1] if i + 1 < len(compressed) else 0
            b3 = compressed[i + 2] if i + 2 < len(compressed) else 0
            result.append(PlantUMLUtils._append_3bytes(b1, b2, b3))
            i += 3
        return "".join(result)

    @staticmethod
    def sanitize_plantuml(plantuml_code: str) -> str:
        return PlantUMLUtils.sandbox_preprocess(plantuml_code)["code"]

    @staticmethod
    def sandbox_preprocess(plantuml_code: str) -> dict[str, Any]:
        text = (plantuml_code or "").strip()
        if not text:
            return {"code": "", "issues": ["PlantUML 代码为空。"]}

        fenced_match = re.search(r"```(?:plantuml|puml|uml)?\s*(.*?)```", text, re.S | re.I)
        if fenced_match:
            text = fenced_match.group(1).strip()

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        block_match = re.search(r"@startuml[\s\S]*?@enduml", text, re.I)
        if block_match:
            text = block_match.group(0).strip()

        lines: list[str] = []
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                lines.append("")
                continue
            if stripped.startswith("```"):
                continue
            if stripped.lower().startswith(("here is", "below is", "plantuml", "uml class diagram")):
                if "@startuml" not in stripped and "@enduml" not in stripped:
                    continue
            lines.append(line.rstrip())

        cleaned = "\n".join(lines).strip()
        if cleaned and not cleaned.startswith("@startuml"):
            cleaned = "@startuml\n" + cleaned
        if cleaned and not cleaned.endswith("@enduml"):
            cleaned = cleaned + "\n@enduml"

        issues: list[str] = []
        issues.extend(PlantUMLUtils._check_balance(cleaned, "{", "}"))
        issues.extend(PlantUMLUtils._check_balance(cleaned, "(", ")"))
        issues.extend(PlantUMLUtils._check_quote_balance(cleaned))
        issues.extend(PlantUMLUtils._check_relation_syntax(cleaned))
        return {"code": cleaned, "issues": list(dict.fromkeys(issues))}

    @staticmethod
    def _check_balance(text: str, left: str, right: str) -> list[str]:
        if text.count(left) != text.count(right):
            return [f"符号不平衡: {left}{right}"]
        return []

    @staticmethod
    def _check_quote_balance(text: str) -> list[str]:
        if text.count('"') % 2 != 0:
            return ['引号不平衡: "']
        return []

    @staticmethod
    def _check_relation_syntax(text: str) -> list[str]:
        issues: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("@"):
                continue
            if '"' not in line and not any(operator in line for operator in PlantUMLUtils._RELATION_OPERATORS):
                continue
            if ":" not in line and any(operator in line for operator in PlantUMLUtils._RELATION_OPERATORS):
                continue
            if any(operator in line for operator in PlantUMLUtils._RELATION_OPERATORS):
                continue
            if any(token in line for token in ("--", "->", "<|", "..")):
                issues.append(f"关系语法可能无效: {line}")
        return issues

    @staticmethod
    def _ensure_analysis(requirement: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(requirement, dict):
            return requirement
        return RequirementUtils.analyze_requirement(requirement)

    @staticmethod
    def _actor_alias(actor_name: str) -> str:
        return PlantUMLUtils._ACTOR_ALIAS.get(actor_name, actor_name.title().replace(" ", ""))

    @staticmethod
    def generate_sequence_diagram(requirement: str | dict[str, Any]) -> str:
        analysis = PlantUMLUtils._ensure_analysis(requirement)
        actors = [actor for actor in analysis.get("actors", []) if actor != "系统"]
        primary_actor = analysis.get("primary_actor", "用户")
        actions = analysis.get("actions", []) or ["执行对应业务操作"]
        condition = analysis.get("condition", "系统接收到业务请求")

        if primary_actor not in actors:
            actors.insert(0, primary_actor)

        lines = ["@startuml"]
        seen_aliases: set[str] = set()
        for actor in actors:
            alias = PlantUMLUtils._actor_alias(actor)
            if alias in seen_aliases:
                continue
            seen_aliases.add(alias)
            lines.append(f'actor "{actor}" as {alias}')
        lines.append('participant "医疗系统" as System')
        lines.append("")

        requester = PlantUMLUtils._actor_alias(primary_actor)
        lines.append(f"{requester} -> System: 触发 {condition}")
        for action in actions:
            if "医生" in action:
                alias = PlantUMLUtils._actor_alias("医生")
                if alias not in seen_aliases:
                    lines.insert(1, 'actor "医生" as Doctor')
                    seen_aliases.add(alias)
                lines.append(f"System -> {alias}: {action}")
                lines.append(f"{alias} --> System: 完成处理")
            elif any(keyword in action for keyword in ("发送", "显示", "通知", "返回", "推送")):
                lines.append(f"System --> {requester}: {action}")
            else:
                lines.append(f"System -> System: {action}")
        lines.append(f"System --> {requester}: 输出业务结果")
        lines.append("@enduml")
        return "\n".join(lines)

    @staticmethod
    def generate_sequence_diagram_from_requirements(requirement_items: list[dict[str, Any]]) -> str:
        if not requirement_items:
            return "@startuml\nparticipant \"医疗系统\" as System\n@enduml"

        actors_seen: list[str] = []
        for item in requirement_items:
            for actor in item.get("actors", []):
                if actor == "系统" or actor in actors_seen:
                    continue
                actors_seen.append(actor)
            primary_actor = item.get("primary_actor", "用户")
            if primary_actor not in {"系统"} and primary_actor not in actors_seen:
                actors_seen.append(primary_actor)

        lines = ["@startuml"]
        declared_aliases: set[str] = set()
        for actor in actors_seen:
            alias = PlantUMLUtils._actor_alias(actor)
            if alias in declared_aliases:
                continue
            declared_aliases.add(alias)
            lines.append(f'actor "{actor}" as {alias}')
        lines.append('participant "医疗系统" as System')
        lines.append("")

        for index, item in enumerate(requirement_items, start=1):
            use_case = item.get("use_case", {}) or {}
            name = str(use_case.get("name", "") or item.get("id", f"业务流程{index}"))
            condition = str(item.get("condition", "") or "系统接收到业务请求")
            primary_actor = str(item.get("primary_actor", "用户") or "用户")
            requester = PlantUMLUtils._actor_alias(primary_actor)
            supporting_actors = [
                actor
                for actor in item.get("actors", [])
                if actor not in {primary_actor, "系统"}
            ]
            actions = item.get("actions", []) or ["执行对应业务操作"]

            lines.append(f"group {item.get('id', f'REQ{index:03d}')} {name}")
            lines.append(f"{requester} -> System: 触发 {condition}")
            for action in actions:
                target_actor = next((actor for actor in supporting_actors if actor and actor in action), "")
                if target_actor:
                    alias = PlantUMLUtils._actor_alias(target_actor)
                    lines.append(f"System -> {alias}: {action}")
                    lines.append(f"{alias} --> System: 完成处理")
                    continue
                if any(keyword in action for keyword in ("发送", "显示", "通知", "返回", "推送")):
                    lines.append(f"System --> {requester}: {action}")
                    continue
                lines.append(f"System -> System: {action}")
            lines.append(f"System --> {requester}: 完成{name}")
            lines.append("end")
            lines.append("")

        lines.append("@enduml")
        return "\n".join(lines)

    @staticmethod
    def generate_use_case_diagram(requirement_items: list[dict[str, Any]]) -> str:
        lines = ["@startuml", "left to right direction"]
        actors_seen: set[str] = set()
        for item in requirement_items:
            for actor in item.get("actors", []):
                if actor == "系统":
                    continue
                alias = PlantUMLUtils._actor_alias(actor)
                if alias not in actors_seen:
                    actors_seen.add(alias)
                    lines.append(f'actor "{actor}" as {alias}')

        lines.append('rectangle "Med-MARAG" {')
        for idx, item in enumerate(requirement_items, start=1):
            use_case = item.get("use_case", {})
            alias = f"UC{idx:03d}"
            name = use_case.get("name", f"业务用例{idx}")
            lines.append(f'  usecase "{name}" as {alias}')
        lines.append("}")

        for idx, item in enumerate(requirement_items, start=1):
            alias = f"UC{idx:03d}"
            primary_actor = item.get("primary_actor", "用户")
            lines.append(f"{PlantUMLUtils._actor_alias(primary_actor)} --> {alias}")
            for actor in item.get("actors", []):
                if actor in {primary_actor, "系统"}:
                    continue
                lines.append(f"{PlantUMLUtils._actor_alias(actor)} --> {alias}")

        lines.append("@enduml")
        return "\n".join(lines)

    @staticmethod
    def generate_class_diagram(entities: list[dict[str, Any]]) -> str:
        entity_names = [entity["name"] for entity in entities]
        relationships = RequirementUtils.infer_relationships(entity_names)

        lines = ["@startuml", "skinparam classAttributeIconSize 0"]
        for entity in entities:
            lines.append(f"class {entity['name']} {{")
            for attribute in entity.get("attributes", []):
                lines.append(f"  {attribute}")
            for method in entity.get("methods", []):
                lines.append(f"  {method}")
            lines.append("}")
            lines.append("")

        lines.extend(relationships)
        lines.append("@enduml")
        return "\n".join(lines)

    @staticmethod
    def generate_class_diagram_from_requirements(requirement_items: list[dict[str, Any]]) -> str:
        entities = RequirementUtils.infer_entities(requirement_items)
        entity_names = [entity["name"] for entity in entities]
        relationships = RequirementUtils.infer_relationships(entity_names, requirement_items=requirement_items)

        lines = ["@startuml", "skinparam classAttributeIconSize 0"]
        for entity in entities:
            lines.append(f"class {entity['name']} {{")
            for attribute in entity.get("attributes", []):
                lines.append(f"  {attribute}")
            for method in entity.get("methods", []):
                lines.append(f"  {method}")
            lines.append("}")
            lines.append("")

        lines.extend(relationships)
        lines.append("@enduml")
        return "\n".join(lines)

    @staticmethod
    def render_plantuml(plantuml_code: str) -> str | None:
        cleaned = PlantUMLUtils.sandbox_preprocess(plantuml_code)["code"]
        if not cleaned:
            return None
        encoded = PlantUMLUtils.encode_plantuml(cleaned)
        return f"{Config.get_plantuml_url()}/svg/{encoded}"
