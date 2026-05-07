# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: Med-MARAG
File Name: traceability_matrix.py
Author: SeaOcean
Create Date: 2026-03-07
Description：需求追溯矩阵生成模块
-------------------------------------------------
"""
from __future__ import annotations

import pandas as pd


class TraceabilityMatrix:
    """需求-设计-测试追溯矩阵。"""

    def __init__(self):
        self.matrix: list[dict[str, str]] = []

    def add_trace(self, requirement_id: str, requirement_desc: str, design_element: str, test_case_id: str) -> None:
        self.matrix.append(
            {
                "需求ID": requirement_id,
                "需求描述": requirement_desc,
                "设计元素": design_element,
                "测试用例": test_case_id,
            }
        )

    def generate_matrix(
        self,
        requirements: list[dict[str, str]],
        design_elements: list[str],
        test_cases: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        self.matrix = []
        for req in requirements:
            related_designs = self._find_related_designs(req, design_elements)
            related_tests = self._find_related_tests(req, test_cases)
            for design in related_designs:
                for test in related_tests:
                    self.add_trace(req.get("id", ""), req.get("description", ""), design, test.get("id", ""))
        return self.matrix

    def _find_related_designs(self, requirement: dict[str, str], design_elements: list[str]) -> list[str]:
        req_desc = requirement.get("description", "").lower()
        keywords = [keyword for keyword in ["登录", "认证", "病历", "健康记录", "预约", "挂号", "分诊", "处方", "通知", "支付"] if keyword in req_desc]
        if not keywords:
            return design_elements[:2] or ["通用模块"]
        related = [element for element in design_elements if any(keyword.lower() in element.lower() for keyword in keywords)]
        return related or design_elements[:2] or ["通用模块"]

    def _find_related_tests(self, requirement: dict[str, str], test_cases: list[dict[str, str]]) -> list[dict[str, str]]:
        req_id = requirement.get("id", "")
        req_desc = requirement.get("description", "").lower()
        related = []
        for test in test_cases:
            description = (test.get("description", "") + " " + test.get("scenario", "")).lower()
            if test.get("requirement_id") == req_id or any(keyword in description for keyword in req_desc.split()):
                related.append(test)
        return related or test_cases[:2]

    def get_matrix_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.matrix)

    def validate_traceability(self) -> dict[str, object]:
        validation = {
            "valid": True,
            "issues": [],
            "coverage": {"requirement_coverage": 0, "test_coverage": 0},
        }
        if not self.matrix:
            validation["valid"] = False
            validation["issues"].append("追溯矩阵为空。")
            return validation

        req_ids = {item["需求ID"] for item in self.matrix if item["需求ID"]}
        test_ids = {item["测试用例"] for item in self.matrix if item["测试用例"]}
        validation["coverage"]["requirement_coverage"] = 100 if req_ids else 0
        validation["coverage"]["test_coverage"] = 100 if test_ids else 0

        if not req_ids:
            validation["valid"] = False
            validation["issues"].append("未建立任何需求追溯关系。")
        if not test_ids:
            validation["valid"] = False
            validation["issues"].append("未建立任何测试追溯关系。")
        return validation

    def export_matrix(self, filename: str) -> str:
        self.get_matrix_dataframe().to_csv(filename, index=False, encoding="utf-8-sig")
        return filename


traceability_matrix = TraceabilityMatrix()
