# -*- coding: utf-8 -*-
"""
Rebuild the Chroma-backed RAG knowledge base from a source manifest.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "rag_sources.txt"
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def read_manifest(path: Path) -> list[Path]:
    sources: list[Path] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        sources.append(Path(line).expanduser().resolve())
    return sources


def backup_or_clear_chroma(chroma_dir: Path, *, keep_backup: bool) -> Path | None:
    if not chroma_dir.exists():
        return None
    if keep_backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = chroma_dir.with_name(f"{chroma_dir.name}.backup_{timestamp}")
        shutil.move(str(chroma_dir), str(backup_dir))
        return backup_dir
    shutil.rmtree(chroma_dir)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild Med-MARAG RAG Chroma DB from a manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Text file containing one source path per line.",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=DEFAULT_CHROMA_DIR,
        help="Chroma persistence directory to rebuild.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Delete the old Chroma directory instead of moving it to a timestamped backup.",
    )
    parser.add_argument(
        "--verify-query",
        default="ISO 29148 requirement shall be verifiable unambiguous electronic medical record",
        help="Query used for a quick retrieval smoke test after mounting.",
    )
    parser.add_argument("--topk", type=int, default=5, help="Number of verification results to print.")
    args = parser.parse_args()

    manifest = args.manifest.expanduser().resolve()
    chroma_dir = args.chroma_dir.expanduser().resolve()
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    sources = read_manifest(manifest)
    missing = [source for source in sources if not source.exists()]
    if missing:
        print("Missing sources:")
        for source in missing:
            print(f"  {source}")
        return 2

    backup_dir = backup_or_clear_chroma(chroma_dir, keep_backup=not args.no_backup)
    if backup_dir:
        print(f"Backed up old Chroma DB: {backup_dir}")
    else:
        print("Old Chroma DB cleared without backup." if chroma_dir.exists() else "No existing Chroma DB found.")

    # Avoid Config's startup auto-mount path adding duplicate copies before the manifest is processed.
    os.environ["RAG_AUTO_MOUNT_ON_STARTUP"] = "false"

    from rag.knowledge_base import knowledge_base

    if knowledge_base.collection is None or knowledge_base.embedding_model is None:
        print("RAG backend is not using Chroma embeddings; mounted records may only live in memory.")

    total_mounted = 0
    for source in sources:
        mounted_count = knowledge_base.mount_source(source)
        total_mounted += mounted_count
        print(f"{source.name}: mounted {mounted_count} chunks")

    stats = knowledge_base.get_collection_stats()
    print(f"total_mounted: {total_mounted}")
    print(f"stats: {stats}")

    results = knowledge_base.query(args.verify_query, n_results=args.topk)
    print("verification_results:")
    for index, item in enumerate(results, start=1):
        metadata = item.get("metadata", {}) or {}
        source = metadata.get("source", "built-in")
        page = metadata.get("page", "-")
        snippet = str(item.get("content", "")).replace("\n", " ")[:240]
        print(f"[{index}] id={item.get('id')} source={source} page={page}")
        print(snippet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
