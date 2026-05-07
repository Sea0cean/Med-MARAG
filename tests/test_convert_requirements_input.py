# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: Med-MARAG
File Name: tests/test_convert_requirements_input.py
Author: SeaOcean
Create Date: 2026-03-22
Description：需求输入转换测试
-------------------------------------------------
"""
from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook

from scripts.convert_requirements_csv import load_rows, normalize_requirement


def test_load_rows_from_xlsx_two_columns_no_header(tmp_path: Path):
    xlsx = tmp_path / "req.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.append(["基础交互", "系统需查询并展示处方记录"])
    ws.append(["计费支付", "系统应处理支付超时并记录失败原因。"])
    wb.save(xlsx)
    wb.close()

    rows = load_rows(xlsx)
    assert len(rows) == 2
    assert rows[0].category == "基础交互"
    assert normalize_requirement(rows[0].requirement) == "系统需查询并展示处方记录。"
    assert rows[1].category == "计费支付"


def test_load_rows_from_wps_text_csv(tmp_path: Path):
    csv_path = tmp_path / "req.csv"
    csv_path.write_text(
        "\"1,基础交互,门诊医生输入卡号，系统需展示处方记录。\"\n\"2,计费支付,支付失败时系统需提示并允许重试。\"\n",
        encoding="utf-8",
    )

    rows = load_rows(csv_path)
    assert len(rows) == 2
    assert rows[0].seq == "1"
    assert rows[0].category == "基础交互"
    assert rows[1].category == "计费支付"
