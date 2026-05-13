# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: Med-MARAG
File Name: config.py
Author: SeaOcean
Create Date: 2026-03-07
Description：项目配置管理模块
-------------------------------------------------
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    def load_dotenv() -> bool:
        return False


load_dotenv()


class Config:
    """项目配置。"""

    LLMProvider = Literal["offline", "deepseek", "openai"]
    SUPPORTED_LLM_PROVIDERS = ("offline", "deepseek", "openai")
    DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"
    DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
    OPENAI_DEFAULT_MODEL = "gpt-4.1-mini"
    OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"

    PROJECT_ROOT = Path(
        os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent)
    ).resolve()
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "output"
    LOG_DIR = PROJECT_ROOT / "logs"

    RAG_CHROMA_DB_PATH = str(DATA_DIR / "chroma_db")
    RAG_COLLECTION_NAME = "medical_requirements"
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    RAG_AUTO_MOUNT_ON_STARTUP = os.getenv("RAG_AUTO_MOUNT_ON_STARTUP", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    RAG_SEED_DEFAULT_KNOWLEDGE = os.getenv("RAG_SEED_DEFAULT_KNOWLEDGE", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    RAG_PDF_CHUNK_SIZE = int(os.getenv("RAG_PDF_CHUNK_SIZE", "1200"))
    RAG_PDF_CHUNK_OVERLAP = int(os.getenv("RAG_PDF_CHUNK_OVERLAP", "150"))
    RAG_ENABLE_PDF_IMAGE_DESCRIPTIONS = os.getenv("RAG_ENABLE_PDF_IMAGE_DESCRIPTIONS", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    RAG_IMAGE_DESCRIPTION_PROVIDER = os.getenv("RAG_IMAGE_DESCRIPTION_PROVIDER", "openai").strip().lower()
    RAG_IMAGE_DESCRIPTION_MODEL = os.getenv("RAG_IMAGE_DESCRIPTION_MODEL", "gpt-4.1-mini").strip()
    RAG_IMAGE_DESCRIPTION_MAX_PAGES = int(os.getenv("RAG_IMAGE_DESCRIPTION_MAX_PAGES", "40"))
    RAG_IMAGE_DESCRIPTION_MIN_IMAGES_PER_PAGE = int(os.getenv("RAG_IMAGE_DESCRIPTION_MIN_IMAGES_PER_PAGE", "1"))
    MAX_CLARIFICATION_QUESTIONS = int(os.getenv("MAX_CLARIFICATION_QUESTIONS", "3"))
    LLM_REQUEST_TIMEOUT_SECONDS = int(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "60"))
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "1"))
    ENABLE_LLM_REVIEW_FOR_ALL_DIAGRAMS = os.getenv("ENABLE_LLM_REVIEW_FOR_ALL_DIAGRAMS", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    ENABLE_LLM_RA_TEXT_ARTIFACTS = os.getenv("ENABLE_LLM_RA_TEXT_ARTIFACTS", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    _DEFAULT_RAG_PDF_SOURCES = [
        DATA_DIR / "standards" / "产出制品规范_1.pdf",
        DATA_DIR / "standards" / "产出制品规范_2.pdf",
        Path(
            "/Users/seaocean/Library/Mobile Documents/com~apple~CloudDocs/论文/本科毕业论文/行业标准/260107en_29148-2018-ISOIECIEEE.pdf"
        ),
    ]
    _RAG_SOURCE_ENV = os.getenv("RAG_SOURCE_PDF_PATHS", "").strip()
    if _RAG_SOURCE_ENV:
        RAG_SOURCE_PDF_PATHS = [
            Path(path.strip()).expanduser()
            for path in _RAG_SOURCE_ENV.split(os.pathsep)
            if path.strip()
        ]
    else:
        # Keep defaults portable: only include files that exist in the current workspace.
        RAG_SOURCE_PDF_PATHS = [path for path in _DEFAULT_RAG_PDF_SOURCES if Path(path).expanduser().exists()]

    EARS_TEMPLATE = """IF {condition}
THEN {action}
{additional_actions}
ENDIF"""

    PLANTUML_SERVER_URL = os.getenv(
        "PLANTUML_SERVER_URL",
        "https://www.plantuml.com/plantuml",
    )

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = str(LOG_DIR / "app.log")

    ISO_29148_GUIDELINES = (
        "基于 ISO/IEC/IEEE 29148:2018 准则的医疗软件需求规范"
    )

    @staticmethod
    def ensure_directories() -> None:
        for path in (Config.DATA_DIR, Config.OUTPUT_DIR, Config.LOG_DIR):
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_ears_template() -> str:
        return Config.EARS_TEMPLATE

    @staticmethod
    def get_plantuml_url() -> str:
        return Config.PLANTUML_SERVER_URL

    @staticmethod
    def get_project_root() -> str:
        return str(Config.PROJECT_ROOT)

    @staticmethod
    def _env(name: str, default: str = "") -> str:
        return os.getenv(name, default).strip()

    @staticmethod
    def _is_placeholder_secret(value: str | None) -> bool:
        normalized = (value or "").strip().lower()
        return not normalized or normalized.startswith("replace_with_")

    @staticmethod
    def provider_is_valid(provider: str | None) -> bool:
        return (provider or "").strip().lower() in Config.SUPPORTED_LLM_PROVIDERS

    @staticmethod
    def get_requested_llm_provider(
        *,
        provider: str | None = None,
        use_llm: bool | None = None,
    ) -> LLMProvider:
        raw_provider = (provider or Config._env("LLM_PROVIDER", "")).strip().lower()
        if raw_provider:
            if raw_provider in Config.SUPPORTED_LLM_PROVIDERS:
                return raw_provider  # type: ignore[return-value]
            return "offline"
        if use_llm is not None:
            return "deepseek" if use_llm else "offline"
        return "offline"

    @staticmethod
    def get_llm_api_key(provider: str, override: str | None = None) -> str:
        if override and not Config._is_placeholder_secret(override):
            return override.strip()
        generic = Config._env("LLM_API_KEY")
        if generic and not Config._is_placeholder_secret(generic):
            return generic
        normalized = provider.strip().lower()
        if normalized == "deepseek":
            value = Config._env("DEEPSEEK_API_KEY")
            return "" if Config._is_placeholder_secret(value) else value
        if normalized == "openai":
            value = Config._env("OPENAI_API_KEY")
            return "" if Config._is_placeholder_secret(value) else value
        return ""

    @staticmethod
    def get_llm_model(provider: str, override: str | None = None) -> str:
        if override:
            return override.strip()
        generic = Config._env("LLM_MODEL")
        if generic:
            return generic
        normalized = provider.strip().lower()
        if normalized == "deepseek":
            return Config._env("DEEPSEEK_MODEL", Config.DEEPSEEK_DEFAULT_MODEL)
        if normalized == "openai":
            return Config._env("OPENAI_MODEL", Config.OPENAI_DEFAULT_MODEL)
        return ""

    @staticmethod
    def get_llm_base_url(provider: str, override: str | None = None) -> str:
        if override:
            return override.strip()
        generic = Config._env("LLM_BASE_URL")
        if generic:
            return generic
        normalized = provider.strip().lower()
        if normalized == "deepseek":
            return Config._env("DEEPSEEK_BASE_URL", Config.DEEPSEEK_DEFAULT_BASE_URL)
        if normalized == "openai":
            return Config._env("OPENAI_BASE_URL", Config.OPENAI_DEFAULT_BASE_URL)
        return ""

    @staticmethod
    def get_llm_profiles(provider: str | None = None) -> list[dict[str, str]]:
        pattern = re.compile(r"^LLM_PROFILE_([A-Z0-9_]+)_(LABEL|PROVIDER|MODEL|BASE_URL|API_KEY)$")
        collected: dict[str, dict[str, str]] = {}
        for key, raw_value in os.environ.items():
            match = pattern.match(key)
            if not match:
                continue
            value = raw_value.strip()
            if not value:
                continue
            profile_id = match.group(1).lower()
            field = match.group(2).lower()
            bucket = collected.setdefault(profile_id, {"id": profile_id})
            bucket[field] = value

        profiles: list[dict[str, str]] = []
        for profile_id, item in sorted(collected.items()):
            profile_provider = item.get("provider", "").strip().lower()
            if not Config.provider_is_valid(profile_provider) or profile_provider == "offline":
                continue
            if provider and profile_provider != provider.strip().lower():
                continue
            model = item.get("model", "").strip()
            if not model:
                continue
            profiles.append(
                {
                    "id": profile_id,
                    "label": item.get("label", profile_id.replace("_", " ").title()).strip(),
                    "provider": profile_provider,
                    "model": model,
                    "base_url": item.get("base_url", "").strip() or Config.get_llm_base_url(profile_provider),
                    "api_key": (
                        ""
                        if Config._is_placeholder_secret(item.get("api_key", "").strip())
                        else item.get("api_key", "").strip()
                    )
                    or Config.get_llm_api_key(profile_provider),
                }
            )

        if provider and provider.strip().lower() != "offline":
            normalized = provider.strip().lower()
            legacy_model = Config.get_llm_model(normalized)
            if legacy_model:
                legacy_base_url = Config.get_llm_base_url(normalized)
                if not any(
                    item["provider"] == normalized
                    and item["model"] == legacy_model
                    and item["base_url"] == legacy_base_url
                    for item in profiles
                ):
                    profiles.insert(
                        0,
                        {
                            "id": f"{normalized}_default",
                            "label": f"{normalized.title()} Default",
                            "provider": normalized,
                            "model": legacy_model,
                            "base_url": legacy_base_url,
                            "api_key": Config.get_llm_api_key(normalized),
                        },
                    )

        return profiles

    @staticmethod
    def llm_enabled(
        provider: str | None = None,
        *,
        use_llm: bool | None = None,
    ) -> bool:
        requested_provider = Config.get_requested_llm_provider(provider=provider, use_llm=use_llm)
        if requested_provider == "offline":
            return False
        return bool(Config.get_llm_api_key(requested_provider))


Config.ensure_directories()
