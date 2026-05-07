# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: Med-MARAG
File Name: rag/knowledge_base.py
Author: SeaOcean
Create Date: 2026-03-07
Description：RAG 知识库挂载与检索模块
-------------------------------------------------
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

from config import Config

try:
    from chromadb import PersistentClient
    from chromadb.config import Settings
except ModuleNotFoundError:  # pragma: no cover
    PersistentClient = None  # type: ignore
    Settings = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

try:
    import fitz  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    fitz = None  # type: ignore

try:
    from rapidocr_onnxruntime import RapidOCR  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    RapidOCR = None  # type: ignore

try:
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    Image = None  # type: ignore


DEFAULT_KNOWLEDGE_ITEMS = [
    {
        "id": "KB001",
        "content": "ISO/IEC/IEEE 29148:2018 强调需求必须单一、无歧义、可验证，适合作为医疗软件需求审查的质量基线。",
        "metadata": {"domain": "标准", "type": "ISO标准", "tags": ["ISO", "29148", "需求", "可验证"]},
    },
    {
        "id": "KB002",
        "content": "EARS 语法建议使用条件加系统行为的模板，例如 IF 条件 THEN 系统应执行动作，用于消除自然语言二义性。",
        "metadata": {"domain": "需求工程", "type": "语法标准", "tags": ["EARS", "IF", "THEN", "AND"]},
    },
    {
        "id": "KB003",
        "content": "医疗系统对患者隐私、访问控制、审计日志和最小权限原则有较高要求，需求中应体现安全与追溯约束。",
        "metadata": {"domain": "合规", "type": "安全要求", "tags": ["隐私", "审计", "访问控制", "安全"]},
    },
    {
        "id": "KB004",
        "content": "挂号与预约场景通常包含患者、医生、号源、预约记录、通知消息等核心实体，适合抽取为概念类模型。",
        "metadata": {"domain": "预约挂号", "type": "领域模型", "tags": ["挂号", "预约", "患者", "医生"]},
    },
    {
        "id": "KB005",
        "content": "分诊场景需要体现优先级、排队顺序、护士评估以及状态流转，建模时应突出流程控制与异常处理。",
        "metadata": {"domain": "分诊管理", "type": "业务规范", "tags": ["分诊", "优先级", "护士", "状态"]},
    },
    {
        "id": "KB006",
        "content": "电子病历需求通常涉及病历查看、病历更新、权限校验和审计留痕，应同时覆盖功能性和非功能性约束。",
        "metadata": {"domain": "电子病历", "type": "最佳实践", "tags": ["病历", "权限", "审计", "记录"]},
    },
    {
        "id": "KB007",
        "content": "测试用例设计应至少覆盖主成功场景、异常场景、权限边界和数据一致性，以支持需求可追溯与验收。",
        "metadata": {"domain": "测试", "type": "最佳实践", "tags": ["测试用例", "异常场景", "验收", "追溯"]},
    },
    {
        "id": "KB008",
        "content": "多智能体闭环审查可采用 Analyst-Architect-Reviewer 模式，先标准化需求，再生成 UML，最后进行质量反馈修正。",
        "metadata": {"domain": "多智能体", "type": "架构模式", "tags": ["Analyst", "Architect", "Reviewer", "闭环"]},
    },
]

DOMAIN_FILTER_RULES = {
    "登录认证": ["登录", "认证", "身份验证", "鉴权", "账号"],
    "预约挂号": ["预约", "挂号", "号源", "就诊"],
    "分诊管理": ["分诊", "排队", "叫号", "优先级", "急诊"],
    "电子病历": ["病历", "健康记录", "电子病历", "诊断记录"],
    "处方管理": ["处方", "药品", "配药", "发药"],
    "通知提醒": ["短信", "通知", "提醒", "消息"],
    "计费支付": ["账单", "费用", "支付", "医保", "结算"],
}

STANDARD_FILTER_RULES = {
    "PDF标准文档": ["iso", "iec", "ieee", "29148", "标准", "standard", "requirements engineering"],
    "语法标准": ["ears", "if", "then", "endif"],
    "软件需求标准": ["产出制品", "制品规范", "ra-1", "ra-2", "ra-3", "ra-4", "ra-5", "ra-6", "用例图", "高层文本用例", "详细文本用例", "概念类模型", "用例序列图", "系统操作", "契约"],
}


class KnowledgeBase:
    """支持 Chroma 与本地关键字检索的轻量知识库。"""

    def __init__(self) -> None:
        self.documents = list(DEFAULT_KNOWLEDGE_ITEMS) if Config.RAG_SEED_DEFAULT_KNOWLEDGE else []
        self.document_ids = {item["id"] for item in self.documents}
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.ocr_engine = None

        if PersistentClient is not None and Settings is not None:
            try:
                self.client = PersistentClient(
                    path=Config.RAG_CHROMA_DB_PATH,
                    settings=Settings(anonymized_telemetry=False),
                )
                self.collection = self.client.get_or_create_collection(
                    name=Config.RAG_COLLECTION_NAME,
                    metadata={"description": "医疗软件需求相关知识库"},
                )
            except Exception:
                self.client = None
                self.collection = None

        if SentenceTransformer is not None and self.collection is not None:
            try:
                self.embedding_model = SentenceTransformer(
                    Config.EMBEDDING_MODEL,
                    local_files_only=True,
                )
            except TypeError:
                try:
                    self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
                except Exception:
                    self.embedding_model = None
            except Exception:
                self.embedding_model = None

        if RapidOCR is not None and np is not None and Image is not None:
            try:
                # Lazy, best-effort OCR fallback for scanned PDFs (no embedded text layer).
                self.ocr_engine = RapidOCR()
            except Exception:
                self.ocr_engine = None

        self._initialize_knowledge()
        if Config.RAG_AUTO_MOUNT_ON_STARTUP:
            self._mount_configured_pdf_sources()

    def _initialize_knowledge(self) -> None:
        if self.collection is None or self.embedding_model is None:
            return
        try:
            if self.collection.count() > 0:
                return
            for item in self.documents:
                embedding = self.embedding_model.encode(item["content"]).tolist()
                self.collection.add(
                    ids=[item["id"]],
                    documents=[item["content"]],
                    embeddings=[embedding],
                    metadatas=[item["metadata"]],
                )
        except Exception:
            self.collection = None
            self.embedding_model = None

    def _mount_configured_pdf_sources(self) -> None:
        persisted_sources = self._get_persisted_pdf_sources()
        for path in Config.RAG_SOURCE_PDF_PATHS:
            # Avoid re-OCR / re-chunking large scanned PDFs on every startup if already persisted.
            if persisted_sources and Path(path).name in persisted_sources:
                continue
            self.mount_pdf_source(path)

    @staticmethod
    def _normalize_text(text: str) -> str:
        cleaned = text.replace("\u00a0", " ")
        cleaned = re.sub(r"\s+\n", "\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        return cleaned.strip()

    @staticmethod
    def _is_clause_heading(line: str) -> bool:
        normalized = line.strip()
        return bool(
            re.match(r"^(?:\d+(?:\.\d+){0,4}|[A-Z]\.\d+)\s+\S+", normalized)
            or re.match(r"^(?:Annex|Appendix)\s+[A-Z0-9]", normalized, re.I)
        )

    @classmethod
    def _split_structured_units(cls, text: str) -> list[str]:
        lines = [line.rstrip() for line in text.splitlines()]
        units: list[str] = []
        buffer: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if buffer and buffer[-1] != "":
                    buffer.append("")
                continue

            if cls._is_clause_heading(stripped) and buffer:
                units.append("\n".join(buffer).strip())
                buffer = [stripped]
                continue

            buffer.append(stripped)

        if buffer:
            units.append("\n".join(buffer).strip())

        if units:
            return units
        return [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]

    @staticmethod
    def _split_long_unit(text: str, chunk_size: int, overlap: int) -> list[str]:
        separators = [r"\n\s*\n", r"(?<=[。！？.!?])\s+", r"(?<=[;；:：])\s+"]
        pieces = [text]

        for separator in separators:
            next_pieces: list[str] = []
            changed = False
            for piece in pieces:
                if len(piece) <= chunk_size:
                    next_pieces.append(piece)
                    continue
                segments = [item.strip() for item in re.split(separator, piece) if item.strip()]
                if len(segments) > 1:
                    next_pieces.extend(segments)
                    changed = True
                else:
                    next_pieces.append(piece)
            pieces = next_pieces
            if changed:
                break

        if all(len(piece) <= chunk_size for piece in pieces):
            return pieces

        fallback_chunks: list[str] = []
        for piece in pieces:
            if len(piece) <= chunk_size:
                fallback_chunks.append(piece)
                continue
            start = 0
            while start < len(piece):
                end = min(len(piece), start + chunk_size)
                fallback_chunks.append(piece[start:end].strip())
                if end >= len(piece):
                    break
                start = max(end - overlap, start + 1)
        return [item for item in fallback_chunks if item]

    @classmethod
    def _chunk_text(cls, text: str, chunk_size: int, overlap: int) -> list[str]:
        paragraphs = cls._split_structured_units(text)
        chunks: list[str] = []
        buffer = ""
        for paragraph in paragraphs:
            candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
            if len(candidate) <= chunk_size:
                buffer = candidate
                continue
            if buffer:
                chunks.append(buffer)
            if len(paragraph) <= chunk_size:
                buffer = paragraph
                continue
            chunks.extend(cls._split_long_unit(paragraph, chunk_size, overlap))
            buffer = ""

        if buffer:
            chunks.append(buffer)

        normalized_chunks = [chunk for chunk in chunks if chunk]
        if overlap <= 0 or len(normalized_chunks) < 2:
            return normalized_chunks

        overlapped_chunks: list[str] = []
        for index, chunk in enumerate(normalized_chunks):
            if index == 0:
                overlapped_chunks.append(chunk)
                continue
            previous = normalized_chunks[index - 1]
            overlap_text = previous[-overlap:].strip()
            if overlap_text and not chunk.startswith(overlap_text):
                chunk = f"{overlap_text}\n{chunk}".strip()
            overlapped_chunks.append(chunk)
        return overlapped_chunks

    @staticmethod
    def _should_skip_pdf_line(line: str) -> bool:
        lowered = line.lower()
        skip_markers = [
            "authorized licensed use limited to",
            "downloaded from https://",
            "restrictions apply",
            "copyright protected document",
            "all rights reserved",
        ]
        return any(marker in lowered for marker in skip_markers)

    @staticmethod
    def _escape_markdown_cell(cell: Any) -> str:
        text = str(cell or "").replace("\n", " ").strip()
        text = re.sub(r"\s{2,}", " ", text)
        return text.replace("|", "\\|")

    @classmethod
    def _table_rows_to_markdown(cls, rows: list[list[Any]]) -> str:
        cleaned_rows = [
            [cls._escape_markdown_cell(cell) for cell in row]
            for row in rows
            if any(str(cell or "").strip() for cell in row)
        ]
        if not cleaned_rows:
            return ""

        header = cleaned_rows[0]
        body = cleaned_rows[1:] or [["" for _ in header]]
        separator = ["---"] * len(header)

        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(separator) + " |",
        ]
        for row in body:
            normalized_row = row + [""] * max(0, len(header) - len(row))
            lines.append("| " + " | ".join(normalized_row[: len(header)]) + " |")
        return "\n".join(lines)

    @classmethod
    def _extract_markdown_tables(cls, page: Any) -> list[str]:
        if not hasattr(page, "find_tables"):
            return []
        try:
            found_tables = page.find_tables()
        except Exception:
            return []

        raw_tables = getattr(found_tables, "tables", found_tables) or []
        markdown_tables: list[str] = []
        for table in raw_tables:
            try:
                rows = table.extract()
            except Exception:
                continue
            markdown = cls._table_rows_to_markdown(rows)
            if markdown:
                markdown_tables.append(f"[TABLE]\n{markdown}\n[/TABLE]")
        return markdown_tables

    @staticmethod
    def _infer_domain_tag(text: str) -> str:
        normalized = text.lower()
        if any(marker in normalized for marker in ("ra-1", "ra-2", "ra-3", "ra-4", "ra-5", "ra-6")) or any(
            marker in text for marker in ("产出制品", "制品规范", "概念类模型", "用例序列图", "系统操作契约")
        ):
            return "标准"
        for domain, keywords in DOMAIN_FILTER_RULES.items():
            if any(keyword.lower() in normalized for keyword in keywords):
                return domain
        if any(keyword in normalized for keyword in ("iso", "iec", "ieee", "29148", "standard", "标准")):
            return "标准"
        return "通用医疗流程"

    @staticmethod
    def _infer_standard_type(text: str) -> str | None:
        normalized = text.lower()
        for doc_type, keywords in STANDARD_FILTER_RULES.items():
            if any(keyword.lower() in normalized for keyword in keywords):
                return doc_type
        return None

    @staticmethod
    def _infer_pdf_document_type(pdf_path: Path, text: str) -> str:
        name = pdf_path.name.lower()
        normalized = (text or "").lower()
        if "产出制品规范" in pdf_path.name or "ra-" in normalized or "用例系统操作" in text:
            return "软件需求标准"
        if any(marker in name for marker in ("iso", "iec", "ieee", "29148")) or "iso/iec/ieee" in normalized:
            return "PDF标准文档"
        return "PDF标准文档"

    @staticmethod
    def _build_pdf_tags(pdf_path: Path, chunk: str, doc_type: str) -> list[str]:
        tags: list[str] = []
        if doc_type == "软件需求标准":
            tags.extend(["软件需求标准", "产出制品规范", "RA-1", "RA-2", "RA-3", "RA-4", "RA-5", "RA-6"])
        if doc_type == "PDF标准文档":
            tags.extend(["ISO", "IEC", "IEEE", "29148", "requirements engineering", "standard"])
        tags.append(pdf_path.name)
        if "用例图" in chunk:
            tags.append("用例图")
        if "高层文本用例" in chunk:
            tags.append("高层文本用例")
        if "详细文本用例" in chunk:
            tags.append("详细文本用例")
        if "概念类模型" in chunk or "概念类图" in chunk:
            tags.append("概念类模型")
        if "用例序列图" in chunk:
            tags.append("用例序列图")
        if "契约" in chunk or "系统操作" in chunk:
            tags.append("系统操作契约")
        return list(dict.fromkeys(tags))

    def _ocr_page(self, page: Any) -> str:
        if self.ocr_engine is None or fitz is None or np is None or Image is None:
            return ""
        try:
            # 2x scale balances OCR quality and speed for typical A4 scans.
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            rgb = np.array(img)
            # RapidOCR generally expects BGR ndarray.
            bgr = rgb[:, :, ::-1].copy()
            result = self.ocr_engine(bgr)
            # rapidocr_onnxruntime returns either (res, elapsed) or res.
            ocr_items = result[0] if isinstance(result, (tuple, list)) and result else result
            if not ocr_items:
                return ""
            texts: list[str] = []
            for item in ocr_items:
                if not item:
                    continue
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    texts.append(str(item[1] or "").strip())
            return "\n".join(line for line in texts if line)
        except Exception:
            return ""

    def _extract_pdf_chunks(self, pdf_path: Path) -> list[dict[str, Any]]:
        if fitz is None or not pdf_path.exists():
            return []

        doc = fitz.open(pdf_path)
        chunk_records: list[dict[str, Any]] = []
        source_hash = hashlib.md5(str(pdf_path).encode("utf-8")).hexdigest()[:10]

        try:
            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)
                raw_text = page.get_text() or ""
                lines = [
                    line.strip()
                    for line in raw_text.splitlines()
                    if line.strip() and not self._should_skip_pdf_line(line)
                ]
                normalized = self._normalize_text("\n".join(lines))
                if not normalized:
                    ocr_text = self._normalize_text(self._ocr_page(page))
                    if ocr_text:
                        normalized = f"[OCR]\n{ocr_text}\n[/OCR]".strip()
                markdown_tables = self._extract_markdown_tables(page)
                if markdown_tables:
                    normalized = "\n\n".join([normalized, *markdown_tables]).strip()
                if not normalized:
                    continue

                page_chunks = self._chunk_text(
                    normalized,
                    chunk_size=Config.RAG_PDF_CHUNK_SIZE,
                    overlap=Config.RAG_PDF_CHUNK_OVERLAP,
                )
                for chunk_index, chunk in enumerate(page_chunks, start=1):
                    doc_type = self._infer_pdf_document_type(pdf_path, chunk)
                    tags = self._build_pdf_tags(pdf_path, chunk, doc_type)
                    chunk_records.append(
                        {
                            "id": f"PDF-{source_hash}-P{page_index + 1:03d}-C{chunk_index:02d}",
                            "content": chunk,
                            "metadata": {
                                "domain": "标准" if doc_type in {"软件需求标准", "PDF标准文档"} else self._infer_domain_tag(chunk),
                                "type": doc_type,
                                "source": pdf_path.name,
                                "source_path": str(pdf_path),
                                "page": page_index + 1,
                                "chunk": chunk_index,
                                "section": self._extract_section_label(chunk),
                                "tags": tags,
                            },
                        }
                    )
        finally:
            doc.close()

        return chunk_records

    @staticmethod
    def _extract_section_label(text: str) -> str:
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        if not first_line:
            return ""
        match = re.match(r"^((?:\d+(?:\.\d+){0,4}|Annex\s+[A-Z0-9]+))\b", first_line, re.I)
        return match.group(1) if match else ""

    def mount_pdf_source(self, pdf_path: str | Path) -> int:
        path = Path(pdf_path).expanduser()
        if not path.exists():
            return 0

        chunk_records = self._extract_pdf_chunks(path)
        mounted_count = 0
        for record in chunk_records:
            if record["id"] in self.document_ids:
                continue
            self.document_ids.add(record["id"])
            self.documents.append(record)
            mounted_count += 1

            if self.collection is not None and self.embedding_model is not None:
                try:
                    existing = self.collection.get(ids=[record["id"]], include=[])
                    if existing.get("ids"):
                        continue
                    embedding = self.embedding_model.encode(record["content"]).tolist()
                    self.collection.add(
                        ids=[record["id"]],
                        documents=[record["content"]],
                        embeddings=[embedding],
                        metadatas=[record["metadata"]],
                    )
                except Exception:
                    pass

        return mounted_count

    @staticmethod
    def _safe_zip_members(zf: zipfile.ZipFile) -> list[str]:
        safe: list[str] = []
        for info in zf.infolist():
            name = info.filename
            if not name or name.endswith("/"):
                continue
            if name.startswith("__MACOSX/") or name.endswith(".DS_Store"):
                continue
            member_path = Path(name)
            if member_path.is_absolute() or ".." in member_path.parts:
                continue
            safe.append(name)
        return safe

    @staticmethod
    def _read_text_file(path: Path) -> str:
        # Best-effort decoding for common Chinese/English sources.
        data = path.read_bytes()
        for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk", "latin-1"):
            try:
                return data.decode(enc)
            except Exception:
                continue
        return ""

    def mount_text_source(self, file_path: str | Path, *, source_label: str | None = None) -> int:
        path = Path(file_path).expanduser()
        if not path.exists() or not path.is_file():
            return 0
        raw = self._read_text_file(path)
        normalized = self._normalize_text(raw)
        if not normalized:
            return 0

        chunks = self._chunk_text(
            normalized,
            chunk_size=Config.RAG_PDF_CHUNK_SIZE,
            overlap=Config.RAG_PDF_CHUNK_OVERLAP,
        )
        source_hash = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:10]
        mounted_count = 0
        for chunk_index, chunk in enumerate(chunks, start=1):
            doc_type = "软件需求标准" if "产出制品规范" in (source_label or path.name) else "文本资料"
            doc_id = f"TXT-{source_hash}-C{chunk_index:03d}"
            record = {
                "id": doc_id,
                "content": chunk,
                "metadata": {
                    "domain": self._infer_domain_tag(chunk),
                    "type": doc_type,
                    "source": source_label or path.name,
                    "source_path": str(path),
                    "chunk": chunk_index,
                    "section": self._extract_section_label(chunk),
                    "tags": self._build_pdf_tags(Path(source_label or path.name), chunk, doc_type)
                    if doc_type == "软件需求标准"
                    else [path.suffix.lower().lstrip("."), path.name],
                },
            }
            if record["id"] in self.document_ids:
                continue
            self.document_ids.add(record["id"])
            self.documents.append(record)
            mounted_count += 1

            if self.collection is not None and self.embedding_model is not None:
                try:
                    existing = self.collection.get(ids=[record["id"]], include=[])
                    if existing.get("ids"):
                        continue
                    embedding = self.embedding_model.encode(record["content"]).tolist()
                    self.collection.add(
                        ids=[record["id"]],
                        documents=[record["content"]],
                        embeddings=[embedding],
                        metadatas=[record["metadata"]],
                    )
                except Exception:
                    pass
        return mounted_count

    def mount_office_source(self, file_path: str | Path) -> int:
        path = Path(file_path).expanduser()
        if not path.exists() or not path.is_file():
            return 0

        tmp_dir = Path(Config.DATA_DIR) / "rag_imports" / "office_cache"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        txt_path = tmp_dir / f"{hashlib.md5(str(path).encode('utf-8')).hexdigest()[:12]}.txt"

        try:
            subprocess.run(
                [
                    "/usr/bin/textutil",
                    "-convert",
                    "txt",
                    "-stdout",
                    str(path),
                ],
                check=True,
                stdout=txt_path.open("wb"),
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return 0

        mounted = self.mount_text_source(txt_path, source_label=path.name)
        try:
            txt_path.unlink(missing_ok=True)
        except Exception:
            pass
        return mounted

    def mount_zip_source(self, zip_path: str | Path) -> int:
        path = Path(zip_path).expanduser()
        if not path.exists() or not path.is_file():
            return 0

        total = 0
        try:
            with zipfile.ZipFile(path, "r") as zf:
                for name in self._safe_zip_members(zf):
                    suffix = Path(name).suffix.lower()
                    try:
                        raw = zf.read(name)
                    except Exception:
                        continue
                    if not raw:
                        continue

                    if suffix == ".pdf":
                        # Use in-memory bytes when possible; fallback to temp file on error.
                        if fitz is not None:
                            try:
                                doc = fitz.open(stream=raw, filetype="pdf")
                                try:
                                    total += self._mount_open_fitz_doc(
                                        doc,
                                        source_name=f"{path.name}:{name}",
                                        source_path=str(path),
                                    )
                                finally:
                                    doc.close()
                                continue
                            except Exception:
                                pass
                    if suffix in {".txt", ".md", ".markdown", ".rst"}:
                        tmp = Path(Config.DATA_DIR) / "rag_imports" / "zip_cache"
                        tmp.mkdir(parents=True, exist_ok=True)
                        temp_path = tmp / f"{hashlib.md5((path.name + ':' + name).encode('utf-8')).hexdigest()[:12]}{suffix}"
                        try:
                            temp_path.write_bytes(raw)
                            total += self.mount_text_source(temp_path, source_label=f"{path.name}:{name}")
                        except Exception:
                            continue
        except zipfile.BadZipFile:
            return 0
        return total

    def _mount_open_fitz_doc(self, doc: Any, *, source_name: str, source_path: str) -> int:
        # Same behavior as mount_pdf_source, but from an already-open fitz doc.
        chunk_records: list[dict[str, Any]] = []
        source_hash = hashlib.md5(source_name.encode("utf-8")).hexdigest()[:10]
        for page_index in range(getattr(doc, "page_count", 0) or 0):
            page = doc.load_page(page_index)
            raw_text = page.get_text() or ""
            lines = [
                line.strip()
                for line in raw_text.splitlines()
                if line.strip() and not self._should_skip_pdf_line(line)
            ]
            normalized = self._normalize_text("\n".join(lines))
            if not normalized:
                ocr_text = self._normalize_text(self._ocr_page(page))
                if ocr_text:
                    normalized = f"[OCR]\n{ocr_text}\n[/OCR]".strip()
            markdown_tables = self._extract_markdown_tables(page)
            if markdown_tables:
                normalized = "\n\n".join([normalized, *markdown_tables]).strip()
            if not normalized:
                continue

            page_chunks = self._chunk_text(
                normalized,
                chunk_size=Config.RAG_PDF_CHUNK_SIZE,
                overlap=Config.RAG_PDF_CHUNK_OVERLAP,
            )
            for chunk_index, chunk in enumerate(page_chunks, start=1):
                doc_type = self._infer_pdf_document_type(Path(source_name), chunk)
                tags = self._build_pdf_tags(Path(source_name), chunk, doc_type)
                chunk_records.append(
                    {
                        "id": f"PDF-{source_hash}-P{page_index + 1:03d}-C{chunk_index:02d}",
                        "content": chunk,
                        "metadata": {
                            "domain": "标准" if doc_type in {"软件需求标准", "PDF标准文档"} else self._infer_domain_tag(chunk),
                            "type": doc_type,
                            "source": source_name,
                            "source_path": source_path,
                            "page": page_index + 1,
                            "chunk": chunk_index,
                            "section": self._extract_section_label(chunk),
                            "tags": tags,
                        },
                    }
                )

        mounted_count = 0
        for record in chunk_records:
            if record["id"] in self.document_ids:
                continue
            self.document_ids.add(record["id"])
            self.documents.append(record)
            mounted_count += 1

            if self.collection is not None and self.embedding_model is not None:
                try:
                    existing = self.collection.get(ids=[record["id"]], include=[])
                    if existing.get("ids"):
                        continue
                    embedding = self.embedding_model.encode(record["content"]).tolist()
                    self.collection.add(
                        ids=[record["id"]],
                        documents=[record["content"]],
                        embeddings=[embedding],
                        metadatas=[record["metadata"]],
                    )
                except Exception:
                    pass
        return mounted_count

    def mount_source(self, path: str | Path) -> int:
        src = Path(path).expanduser()
        suffix = src.suffix.lower()
        if suffix == ".pdf":
            return self.mount_pdf_source(src)
        if suffix == ".zip":
            return self.mount_zip_source(src)
        if suffix in {".doc", ".docx"}:
            return self.mount_office_source(src)
        if suffix in {".txt", ".md", ".markdown", ".rst"}:
            return self.mount_text_source(src)
        return 0

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{1,4}", text))

    @classmethod
    def _expand_query_text(cls, query_text: str) -> str:
        query = (query_text or "").strip()
        if not query:
            return ""

        expansions: list[str] = [query]
        domain = cls._infer_domain_tag(query)
        doc_type = cls._infer_standard_type(query)
        normalized = query.lower()

        if domain and domain != "通用医疗流程":
            expansions.append(domain)
        if doc_type:
            expansions.append(doc_type)

        if any(marker in normalized for marker in ("uml", "类图", "顺序图", "用例图", "建模")):
            expansions.extend(["PlantUML", "UML 类图", "实体 方法 关系 多重性"])
        if any(marker in normalized for marker in ("review", "审查", "合规", "质量", "29148", "ears")):
            expansions.extend(["ISO/IEC/IEEE 29148:2018", "EARS", "可验证 无歧义 单一性"])
        if any(marker in normalized for marker in ("ra-", "产出制品", "系统操作契约")):
            expansions.extend(["软件需求标准", "产出制品规范"])

        return " ".join(dict.fromkeys(item for item in expansions if item))

    @classmethod
    def _infer_metadata_filter(cls, query_text: str) -> dict[str, Any]:
        query = (query_text or "").strip()
        if not query:
            return {}

        metadata_filter: dict[str, Any] = {}
        domain = cls._infer_domain_tag(query)
        if domain != "通用医疗流程":
            metadata_filter["domain"] = domain

        doc_type = cls._infer_standard_type(query)
        if doc_type:
            metadata_filter["type"] = doc_type
        return metadata_filter

    @staticmethod
    def _metadata_matches(item_metadata: dict[str, Any], metadata_filter: dict[str, Any]) -> bool:
        if not metadata_filter:
            return True
        for key, value in metadata_filter.items():
            if item_metadata.get(key) != value:
                return False
        return True

    @staticmethod
    def _extract_query_keywords(query_text: str) -> set[str]:
        raw_tokens = KnowledgeBase._tokenize(query_text)
        return {
            token
            for token in raw_tokens
            if len(token) >= 2 and token.lower() not in {"if", "then", "and", "uml", "the", "for", "with"}
        }

    @classmethod
    def _metadata_soft_score(cls, metadata: dict[str, Any], metadata_filter: dict[str, Any], query_text: str) -> float:
        score = 0.0
        if metadata_filter:
            for key, value in metadata_filter.items():
                if metadata.get(key) == value:
                    score += 2.5

        source = str(metadata.get("source", "") or "").lower()
        section = str(metadata.get("section", "") or "")
        doc_type = str(metadata.get("type", "") or "")
        normalized = (query_text or "").lower()

        if any(marker in normalized for marker in ("iso", "29148", "ears", "标准", "requirement")):
            if doc_type in {"PDF标准文档", "软件需求标准"}:
                score += 1.5
            if source and any(marker in source for marker in ("iso", "29148", "ieee", "iec")):
                score += 1.0
        if section:
            score += 0.3
        return score

    @classmethod
    def _rerank_results(
        cls,
        query_text: str,
        results: list[dict[str, Any]],
        *,
        n_results: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not results:
            return []

        metadata_filter = metadata_filter or {}
        query_keywords = cls._extract_query_keywords(query_text)
        normalized = (query_text or "").lower()
        scored: list[tuple[float, dict[str, Any]]] = []
        seen_ids: set[str] = set()

        for item in results:
            item_id = str(item.get("id", ""))
            if not item_id or item_id in seen_ids:
                continue
            seen_ids.add(item_id)

            content = str(item.get("content", "") or "")
            metadata = item.get("metadata", {}) or {}
            tags = {str(tag).lower() for tag in metadata.get("tags", []) or []}
            content_tokens = {token.lower() for token in cls._tokenize(content)}

            overlap = 0
            phrase_hits = 0
            for keyword in query_keywords:
                lowered = keyword.lower()
                if lowered in content_tokens or lowered in tags:
                    overlap += 1
                if lowered in content.lower():
                    phrase_hits += 1

            distance = float(item.get("distance", 1.0) or 1.0)
            semantic_score = max(0.0, 1.2 - distance)
            lexical_score = overlap * 1.6 + phrase_hits * 0.4
            metadata_score = cls._metadata_soft_score(metadata, metadata_filter, query_text)
            brevity_bonus = 0.4 if len(content) <= 900 else 0.0
            score = semantic_score + lexical_score + metadata_score + brevity_bonus
            scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        ranked: list[dict[str, Any]] = []
        seen_sources: dict[str, int] = {}
        seen_sections: set[tuple[str, str]] = set()
        for _, item in scored:
            metadata = item.get("metadata", {}) or {}
            source = str(metadata.get("source", "") or "")
            section = str(metadata.get("section", "") or "")
            pair_key = (source, section)

            # Encourage source/section diversity so top-k does not collapse to neighboring chunks.
            if pair_key in seen_sections and len(ranked) < n_results:
                continue
            if source and seen_sources.get(source, 0) >= 2 and len(ranked) < n_results:
                continue

            ranked.append(item)
            if source:
                seen_sources[source] = seen_sources.get(source, 0) + 1
            if source or section:
                seen_sections.add(pair_key)
            if len(ranked) >= n_results:
                break

        if len(ranked) < n_results:
            existing = {str(item.get("id", "")) for item in ranked}
            for _, item in scored:
                item_id = str(item.get("id", ""))
                if item_id in existing:
                    continue
                ranked.append(item)
                existing.add(item_id)
                if len(ranked) >= n_results:
                    break
        return ranked[:n_results]

    def _keyword_query(
        self,
        query_text: str,
        n_results: int = 3,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        metadata_filter = metadata_filter or {}
        query_tokens = self._tokenize(query_text)
        scored: list[tuple[int, dict[str, Any]]] = []
        for item in self.documents:
            metadata = item.get("metadata", {})
            tags = set(metadata.get("tags", []))
            content_tokens = self._tokenize(item["content"]) | tags
            overlap = len(query_tokens & content_tokens)
            if overlap == 0 and not any(token in item["content"] for token in query_tokens):
                continue
            score = overlap + (3 if self._metadata_matches(metadata, metadata_filter) else 0)
            scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        results = []
        for distance, item in scored[:n_results]:
            results.append(
                {
                    "id": item["id"],
                    "content": item["content"],
                    "metadata": item["metadata"],
                    "distance": max(0.0, 1.0 - distance / 10.0),
                }
            )
        if results:
            return results
        if not self.documents:
            return []
        return [
            {
                "id": self.documents[0]["id"],
                "content": self.documents[0]["content"],
                "metadata": self.documents[0]["metadata"],
                "distance": 0.9,
            }
        ]

    def _collection_query(
        self,
        query_embedding: list[float],
        n_results: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self.collection is None:
            return []
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if metadata_filter:
            kwargs["where"] = self._build_collection_where(metadata_filter)

        results = self.collection.query(**kwargs)
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append(
                {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )
        return formatted_results

    @staticmethod
    def _build_collection_where(metadata_filter: dict[str, Any]) -> dict[str, Any]:
        if not metadata_filter:
            return {}
        if len(metadata_filter) == 1:
            return metadata_filter
        return {"$and": [{key: value} for key, value in metadata_filter.items()]}

    def query(
        self,
        query_text: str,
        n_results: int = 3,
        metadata_filter: dict[str, Any] | None = None,
        use_metadata_filter: bool = True,
    ) -> list[dict[str, Any]]:
        effective_filter = metadata_filter or (self._infer_metadata_filter(query_text) if use_metadata_filter else {})
        expanded_query = self._expand_query_text(query_text)
        candidate_count = max(n_results * 4, 8)
        if self.collection is not None and self.embedding_model is not None:
            try:
                query_embedding = self.embedding_model.encode(expanded_query or query_text).tolist()
                prioritized_results = self._collection_query(
                    query_embedding,
                    n_results=candidate_count,
                    metadata_filter=effective_filter,
                )
                merged_results = list(prioritized_results)
                if len(prioritized_results) < candidate_count:
                    fallback_results = self._collection_query(
                        query_embedding,
                        n_results=candidate_count,
                        metadata_filter=None,
                    )
                    seen_ids = {item["id"] for item in merged_results}
                    for item in fallback_results:
                        if item["id"] in seen_ids:
                            continue
                        merged_results.append(item)
                        seen_ids.add(item["id"])
                        if len(merged_results) >= candidate_count:
                            break
                if merged_results:
                    reranked = self._rerank_results(
                        query_text,
                        merged_results,
                        n_results=n_results,
                        metadata_filter=effective_filter,
                    )
                    if reranked:
                        return reranked
            except Exception:
                pass
        keyword_results = self._keyword_query(expanded_query or query_text, n_results=candidate_count, metadata_filter=effective_filter)
        reranked = self._rerank_results(query_text, keyword_results, n_results=n_results, metadata_filter=effective_filter)
        return reranked or keyword_results[:n_results]

    def add_document(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        doc_id = f"KB{len(self.documents) + 1:03d}"
        document = {"id": doc_id, "content": content, "metadata": metadata or {}}
        self.documents.append(document)
        self.document_ids.add(doc_id)
        if self.collection is not None and self.embedding_model is not None:
            try:
                embedding = self.embedding_model.encode(content).tolist()
                self.collection.add(
                    ids=[doc_id],
                    documents=[content],
                    embeddings=[embedding],
                    metadatas=[metadata or {}],
                )
            except Exception:
                pass
        return doc_id

    def _get_persisted_pdf_sources(self) -> set[str]:
        if self.collection is None:
            return set()
        try:
            records = self.collection.get(include=["metadatas"])
        except Exception:
            return set()

        sources: set[str] = set()
        for metadata in records.get("metadatas", []) or []:
            if metadata and metadata.get("type") in {"PDF标准文档", "软件需求标准"} and metadata.get("source"):
                sources.add(str(metadata["source"]))
        return sources

    def get_collection_stats(self) -> dict[str, Any]:
        persisted_count = 0
        if self.collection is not None:
            try:
                persisted_count = self.collection.count()
            except Exception:
                persisted_count = 0
        local_pdf_sources = {
            item.get("metadata", {}).get("source")
            for item in self.documents
            if item.get("metadata", {}).get("type") in {"PDF标准文档", "软件需求标准"}
            and item.get("metadata", {}).get("source")
        }
        return {
            "count": max(len(self.documents), persisted_count),
            "name": Config.RAG_COLLECTION_NAME,
            "backend": "chroma" if self.collection is not None and self.embedding_model is not None else "memory",
            "pdf_sources": sorted(local_pdf_sources | self._get_persisted_pdf_sources()),
        }


knowledge_base = KnowledgeBase()
