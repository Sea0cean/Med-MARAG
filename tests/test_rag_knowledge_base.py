# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: Med-MARAG
File Name: tests/test_rag_knowledge_base.py
Author: SeaOcean
Create Date: 2026-03-19
Description：RAG 知识库测试
-------------------------------------------------
"""
from __future__ import annotations

import zipfile
from pathlib import Path

from config import Config
from rag.knowledge_base import KnowledgeBase
import rag.knowledge_base as knowledge_base_module


class FakeEmbeddingModel:
    def encode(self, _: str):
        class _Embedding(list):
            def tolist(self):
                return list(self)

        return _Embedding([0.1, 0.2, 0.3])


class FakeCollection:
    def __init__(self):
        self.calls: list[dict] = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        where = kwargs.get("where") or {}
        normalized_where = {}
        if "$and" in where:
            for clause in where["$and"]:
                normalized_where.update(clause)
        else:
            normalized_where = where
        if normalized_where.get("domain") == "预约挂号":
            return {
                "ids": [["PDF-APPT-001"]],
                "documents": [["挂号流程需要校验号源并生成预约记录。"]],
                "metadatas": [[{"domain": "预约挂号", "type": "领域模型"}]],
                "distances": [[0.12]],
            }
        if normalized_where.get("domain") == "电子病历" and normalized_where.get("type") == "PDF标准文档":
            return {
                "ids": [["PDF-EMR-001"]],
                "documents": [["电子病历相关标准要求需求描述应准确、完整且可验证。"]],
                "metadatas": [[{"domain": "电子病历", "type": "PDF标准文档"}]],
                "distances": [[0.16]],
            }
        return {
            "ids": [["KB001", "KB004"]],
            "documents": [["标准要求需求可验证。", "预约场景涉及患者、医生与预约记录。"]],
            "metadatas": [[{"domain": "标准", "type": "ISO标准"}, {"domain": "预约挂号", "type": "领域模型"}]],
            "distances": [[0.18, 0.28]],
        }


def build_test_kb() -> KnowledgeBase:
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.documents = []
    kb.document_ids = set()
    kb.client = None
    kb.collection = FakeCollection()
    kb.embedding_model = FakeEmbeddingModel()
    kb.ocr_engine = None
    kb.vision_runtime = None
    return kb


def test_table_rows_are_converted_to_markdown():
    markdown = KnowledgeBase._table_rows_to_markdown(
        [
            ["字段", "说明"],
            ["priority", "分诊优先级"],
            ["queue_no", "排队序号"],
        ]
    )

    assert "| 字段 | 说明 |" in markdown
    assert "| priority | 分诊优先级 |" in markdown


def test_structure_aware_chunking_preserves_clause_heads_and_overlap():
    text = (
        "5.2 Requirement quality\n"
        "The requirement shall be unambiguous and verifiable.\n\n"
        "5.2.1 Additional explanation\n"
        + "This clause provides extended guidance. " * 30
    )

    chunks = KnowledgeBase._chunk_text(text, chunk_size=140, overlap=18)

    assert len(chunks) >= 2
    assert any("5.2 Requirement quality" in chunk for chunk in chunks)
    assert any("5.2.1 Additional explanation" in chunk for chunk in chunks)
    assert chunks[1].splitlines()[0] != ""


def test_infer_metadata_filter_uses_domain_and_standard_signals():
    assert KnowledgeBase._infer_metadata_filter("急诊分诊优先级规则") == {"domain": "分诊管理"}
    assert KnowledgeBase._infer_metadata_filter("ISO 29148 requirement standard") == {
        "domain": "标准",
        "type": "PDF标准文档",
    }
    assert KnowledgeBase._infer_metadata_filter("RA-4 概念类模型 制品规范") == {
        "domain": "标准",
        "type": "软件需求标准",
    }


def test_query_prioritizes_metadata_filtered_results():
    kb = build_test_kb()

    results = kb.query("患者预约挂号流程", n_results=2)

    assert results[0]["metadata"]["domain"] == "预约挂号"
    assert kb.collection.calls[0]["where"] == {"domain": "预约挂号"}
    assert len(kb.collection.calls) >= 2


def test_collection_query_builds_and_filter_for_multi_field_metadata():
    kb = build_test_kb()

    results = kb.query("病历 EARS ISO 29148", n_results=2)

    assert results[0]["metadata"]["domain"] == "电子病历"
    assert kb.collection.calls[0]["where"] == {
        "$and": [
            {"domain": "电子病历"},
            {"type": "PDF标准文档"},
        ]
    }


def test_visual_queries_boost_pdf_image_description_metadata():
    score = KnowledgeBase._metadata_soft_score(
        {"type": "PDF图片说明", "source": "sample.pdf"},
        {},
        "预约挂号用例图中的参与者关系",
    )

    assert score >= 2.0


def test_safe_zip_members_filters_path_traversal(tmp_path: Path):
    zip_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("ok/readme.txt", "hello")
        zf.writestr("../evil.txt", "nope")
        zf.writestr("/abs.txt", "nope")
        zf.writestr("__MACOSX/._x", "nope")

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = KnowledgeBase._safe_zip_members(zf)

    assert "ok/readme.txt" in members
    assert "../evil.txt" not in members
    assert "/abs.txt" not in members


class FakeVisionRuntime:
    enabled = True


class FakePixmap:
    def save(self, path: str) -> None:
        Path(path).write_bytes(b"fake-png")


class FakePageWithImage:
    def get_images(self, full: bool = True):
        return [("xref",)]

    def get_drawings(self):
        return []

    def get_pixmap(self, matrix=None, alpha: bool = False):
        return FakePixmap()


class FakeFitz:
    class Matrix:
        def __init__(self, *_: object) -> None:
            pass


def test_pdf_image_description_builds_searchable_figure_record(tmp_path: Path, monkeypatch):
    kb = build_test_kb()
    monkeypatch.setattr(Config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(Config, "RAG_ENABLE_PDF_IMAGE_DESCRIPTIONS", True)
    monkeypatch.setattr(Config, "RAG_IMAGE_DESCRIPTION_PROVIDER", "openai")
    monkeypatch.setattr(Config, "RAG_IMAGE_DESCRIPTION_MODEL", "gpt-4.1-mini")
    monkeypatch.setattr(Config, "RAG_IMAGE_DESCRIPTION_MAX_PAGES", 5)
    monkeypatch.setattr(Config, "RAG_IMAGE_DESCRIPTION_MIN_IMAGES_PER_PAGE", 1)
    monkeypatch.setattr(knowledge_base_module, "fitz", FakeFitz)
    monkeypatch.setattr(knowledge_base_module, "build_llm_runtime", lambda **_: FakeVisionRuntime())

    def fake_vision(runtime, *, prompt: str, image_path: Path, system_prompt: str = "") -> str:
        assert runtime.enabled is True
        assert "sample.pdf" in prompt
        assert Path(image_path).exists()
        return "这是一张 UML 用例图，包含参与者患者和用例预约挂号，患者通过箭头连接到预约挂号。"

    monkeypatch.setattr(knowledge_base_module, "invoke_llm_vision", fake_vision)

    record = kb._build_page_image_description_record(
        FakePageWithImage(),
        source_name="sample.pdf",
        source_path="/tmp/sample.pdf",
        source_hash="abc123",
        page_index=0,
    )

    assert record is not None
    assert record["id"] == "PDFIMG-abc123-P001-I01"
    assert record["metadata"]["type"] == "PDF图片说明"
    assert record["metadata"]["page"] == 1
    assert "image_path" in record["metadata"]
    assert "用例图" in record["metadata"]["tags"]
    assert "[FIGURE]" in record["content"]
    assert "预约挂号" in record["content"]
