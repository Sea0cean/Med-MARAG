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


def test_chunking_keeps_adjacent_clause_sections_separate():
    text = (
        "8.2.2.3 医嘱执行与打印\n"
        "医嘱执行与打印功能应包括：\n"
        "单日诊疗执行项目；\n\n"
        "8.2.2.4 护理管理\n"
        "护理管理功能应包括：\n"
        "护理计划：提供护理计划的编辑；\n\n"
        "8.2.2.5 住院患者管理\n"
        "住院患者管理功能包括：\n"
        "入院登记：提供入院患者的登记。"
    )

    chunks = KnowledgeBase._chunk_text(text, chunk_size=1200, overlap=0)

    assert len(chunks) == 3
    assert chunks[0].startswith("8.2.2.3 医嘱执行与打印")
    assert "8.2.2.4" not in chunks[0]
    assert chunks[1].startswith("8.2.2.4 护理管理")
    assert "8.2.2.5" not in chunks[1]
    assert chunks[2].startswith("8.2.2.5 住院患者管理")


def test_pdf_paragraph_breaks_are_joined_without_flattening_structure():
    text = (
        "第四步：识别关系的种类（详见3.2.3节）\n"
        "除了一般性的关系之外，增补可能存在、重要且有意义（有意义即与当前系\n"
        "统的业务主题相关）的泛化关系，以及识别并标注重要且有意义的组合关系。\n"
        "之所以强调“重要且有意义”，原因在于软件工程的一些实践并不建议标注\n"
        "所有与当前系统的业务主题相关的关系，以免形成过于复杂的蛛网。\n"
        "——示例列表：\n"
        "重要关系；\n"
        "8.2.2.3 医嘱执行与打印\n"
        "医嘱执行与打印功能应包括："
    )

    normalized = KnowledgeBase._normalize_pdf_paragraph_breaks(text)

    assert "当前系\n统" not in normalized
    assert "当前系统的业务主题相关" in normalized
    assert "标注所有与当前系统的业务主题相关的关系" in normalized
    assert "第四步：识别关系的种类" in normalized
    assert "——示例列表：\n重要关系；" in normalized
    assert "8.2.2.3 医嘱执行与打印\n医嘱执行与打印功能应包括：" in normalized


def test_pdf_paragraph_joiner_handles_chinese_and_english_breaks():
    assert KnowledgeBase._normalize_pdf_paragraph_breaks("当前系\n统") == "当前系统"
    assert KnowledgeBase._normalize_pdf_paragraph_breaks("standards.\nAttention") == "standards. Attention"
    assert KnowledgeBase._normalize_pdf_paragraph_breaks("3)]\nThe requirements") == "3)] The requirements"
    assert KnowledgeBase._normalize_pdf_paragraph_breaks("configura-\ntion") == "configura-tion"


def test_pdf_paragraph_breaks_drop_standard_cover_noise():
    text = (
        "ICS 11.020\n"
        "C 07\n"
        "WS\n"
        "中华人民共和国卫生行业标准\n"
        "2016 - 08 - 23 发布\n"
        "基层医疗卫生信息系统基本功能规范\n"
        "系统应支持基本医疗服务。"
    )

    normalized = KnowledgeBase._normalize_pdf_paragraph_breaks(text)

    assert "ICS 11.020" not in normalized
    assert "C 07" not in normalized
    assert "2016 - 08 - 23 发布" not in normalized
    assert "基层医疗卫生信息系统基本功能规范" in normalized
    assert "系统应支持基本医疗服务" in normalized


def test_should_skip_pdf_line_filters_common_page_number_patterns():
    assert KnowledgeBase._should_skip_pdf_line("12")
    assert KnowledgeBase._should_skip_pdf_line("- 12 -")
    assert KnowledgeBase._should_skip_pdf_line("Page 12")
    assert KnowledgeBase._should_skip_pdf_line("12 / 200")
    assert KnowledgeBase._should_skip_pdf_line("xiv")
    assert not KnowledgeBase._should_skip_pdf_line("5.2 Requirement quality")


def test_pdf_line_normalization_removes_symbol_font_bullets():
    assert KnowledgeBase._normalize_pdf_line("") == ""
    assert KnowledgeBase._normalize_pdf_line(" 单日诊疗执行项目；") == "单日诊疗执行项目；"
    assert KnowledgeBase._normalize_pdf_line("• 药品单；") == "药品单；"


def test_leading_section_number_is_removed_but_title_is_kept():
    text = "8.2.2.3 医嘱执行与打印\n医嘱执行与打印功能应包括：\n单日诊疗执行项目；"

    assert KnowledgeBase._extract_section_label(text) == "8.2.2.3"
    assert KnowledgeBase._extract_section_title(text) == "医嘱执行与打印"

    cleaned = KnowledgeBase._strip_leading_section_number(text)

    assert "8.2.2.3" not in cleaned
    assert cleaned.startswith("医嘱执行与打印\n医嘱执行与打印功能应包括")
    assert "单日诊疗执行项目" in cleaned


class FakeRect:
    def __init__(self, height: float):
        self.height = height


class FakeStructuredPage:
    def __init__(self, lines: list[tuple[str, float, float]], height: float = 1000):
        self._lines = lines
        self.rect = FakeRect(height)

    def get_text(self, mode: str | None = None):
        if mode == "dict":
            return {
                "blocks": [
                    {
                        "type": 0,
                        "bbox": [0, 0, 100, self.rect.height],
                        "lines": [
                            {
                                "bbox": [0, y0, 100, y1],
                                "spans": [{"text": text}],
                            }
                            for text, y0, y1 in self._lines
                        ],
                    }
                ]
            }
        return "\n".join(text for text, _, _ in self._lines)


class FakeStructuredDoc:
    def __init__(self, pages: list[FakeStructuredPage]):
        self._pages = pages
        self.page_count = len(pages)

    def load_page(self, index: int) -> FakeStructuredPage:
        return self._pages[index]


def test_extract_clean_page_text_removes_repeated_headers_and_footers():
    pages = [
        FakeStructuredPage(
            [
                ("ISO/IEC/IEEE 29148:2018", 25, 40),
                ("5.2 Requirement quality", 180, 210),
                ("The requirement shall be unambiguous and verifiable.", 235, 270),
                ("Page 12", 955, 975),
            ]
        ),
        FakeStructuredPage(
            [
                ("ISO/IEC/IEEE 29148:2018", 25, 40),
                ("5.3 Validation", 180, 210),
                ("Validation confirms the requirement can be checked objectively.", 235, 270),
                ("Page 13", 955, 975),
            ]
        ),
    ]
    doc = FakeStructuredDoc(pages)
    repeated = KnowledgeBase._collect_repeated_margin_noise_keys(doc)

    cleaned = KnowledgeBase._extract_clean_page_text(pages[0], repeated_margin_noise_keys=repeated)

    assert "ISO/IEC/IEEE 29148:2018" not in cleaned
    assert "Page 12" not in cleaned
    assert "5.2 Requirement quality" in cleaned
    assert "unambiguous and verifiable" in cleaned


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
