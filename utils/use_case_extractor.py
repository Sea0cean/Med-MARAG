# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: Med-MARAG
File Name: utils/use_case_extractor.py
Description：业务用例抽取与聚合工具
-------------------------------------------------
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from utils.requirement_utils import RequirementUtils


class UseCaseExtractor:
    """Extract business use cases from normalized requirement items.

    Requirement items describe verifiable system behaviors, while use cases
    describe actor goals. This extractor keeps those two layers separate by
    grouping related requirement items under a compact business goal.
    """

    @classmethod
    def extract(cls, requirement_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        order: list[str] = []

        for item in requirement_items:
            name = cls._infer_business_goal(item)
            if name not in grouped:
                grouped[name] = {
                    "name": name,
                    "items": [],
                    "actors": [],
                    "actions": [],
                    "conditions": [],
                }
                order.append(name)

            group = grouped[name]
            group["items"].append(item)
            for actor in item.get("actors", []) or []:
                if actor not in group["actors"]:
                    group["actors"].append(actor)
            for action in item.get("actions", []) or []:
                if action and action not in group["actions"]:
                    group["actions"].append(action)
            condition = item.get("condition")
            if condition and condition not in group["conditions"]:
                group["conditions"].append(condition)

        return [cls._build_use_case(index, grouped[name]) for index, name in enumerate(order, start=1)]

    @classmethod
    def _infer_business_goal(cls, item: dict[str, Any]) -> str:
        text = cls._item_text(item)
        pattern_names = [
            (("同步", "异常"), "处理同步异常"),
            (("网络超时",), "处理同步异常"),
            (("连接失败",), "处理同步异常"),
            (("运维", "日志"), "处理同步异常"),
            (("发送确认短信",), "发送确认通知"),
            (("发送确认通知",), "发送确认通知"),
            (("每天凌晨", "同步"), "同步急诊数据"),
            (("每日", "同步"), "同步急诊数据"),
            (("核心数据库", "备份"), "同步急诊数据"),
            (("处方",), "开立电子处方"),
            (("补缴",), "补缴急救费用"),
            (("抢救费",), "补缴急救费用"),
            (("临时急诊病历",), "建立临时病历"),
            (("临时急诊档案",), "建立临时病历"),
            (("挂号",), "办理急诊挂号"),
            (("缴费", "流水号"), "办理急诊挂号"),
            (("医生工作站",), "办理急诊挂号"),
            (("生命体征",), "登记分诊信息"),
            (("生理参数",), "登记分诊信息"),
            (("分诊", "分类"), "登记分诊信息"),
        ]
        for keywords, name in pattern_names:
            if all(keyword in text for keyword in keywords):
                return name

        use_case = item.get("use_case", {}) or {}
        raw_name = str(use_case.get("name", "") or item.get("id", "执行业务操作"))
        return RequirementUtils.normalize_use_case_name(
            raw_name,
            original_text=str(item.get("original", "") or ""),
            domain=str(item.get("domain", "") or ""),
            actions=item.get("actions", []) or [],
        )

    @staticmethod
    def _item_text(item: dict[str, Any]) -> str:
        parts = [
            str(item.get("original", "") or ""),
            str(item.get("domain", "") or ""),
            str(item.get("condition", "") or ""),
            " ".join(str(action) for action in item.get("actions", []) or []),
            str((item.get("use_case", {}) or {}).get("name", "") or ""),
        ]
        return " ".join(part for part in parts if part)

    @classmethod
    def _build_use_case(cls, index: int, group: dict[str, Any]) -> dict[str, Any]:
        items = group["items"]
        name = group["name"]
        primary_actor = cls._select_primary_actor(name, group["actors"], items)
        supporting_actors = [
            actor for actor in group["actors"] if actor not in {primary_actor, "系统"}
        ]
        requirement_ids = [str(item.get("id", "")) for item in items if item.get("id")]
        conditions = group["conditions"] or ["系统接收到业务请求"]
        actions = group["actions"] or [name]

        return {
            "id": f"UC{index:03d}",
            "name": name,
            "goal": f"{primary_actor}完成{name}",
            "scope": "Med-MARAG 医疗需求建模平台",
            "level": "User goal",
            "primary_actor": primary_actor,
            "supporting_actors": supporting_actors,
            "preconditions": conditions[:3],
            "main_flow": cls._build_main_flow(primary_actor, name, actions),
            "postconditions": [f"系统完成{name}并留下可追溯记录。"],
            "requirement_ids": requirement_ids,
        }

    @staticmethod
    def _select_primary_actor(name: str, actors: list[str], items: list[dict[str, Any]]) -> str:
        if name == "同步急诊数据":
            return "定时任务"
        if name == "处理同步异常":
            return "运维人员" if "运维人员" in actors else "运维"
        if name in {"登记分诊信息", "建立临时病历"} and "护士" in actors:
            return "护士"
        if name == "开立电子处方" and "医生" in actors:
            return "医生"
        if name in {"办理急诊挂号", "补缴急救费用"} and "患者" in actors:
            return "患者"

        item_actors = [
            str(item.get("primary_actor", "") or "")
            for item in items
            if item.get("primary_actor") and item.get("primary_actor") != "系统"
        ]
        if item_actors:
            return Counter(item_actors).most_common(1)[0][0]
        return next((actor for actor in actors if actor != "系统"), "用户")

    @staticmethod
    def _build_main_flow(primary_actor: str, name: str, actions: list[str]) -> list[str]:
        flow = [f"{primary_actor}发起{name}。"]
        for action in actions[:6]:
            flow.append(f"系统执行：{action}。")
        flow.append(f"系统反馈{name}结果。")
        return flow
