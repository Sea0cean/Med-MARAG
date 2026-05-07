# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: Med-MARAG
File Name: mount_rag_pdf.py
Author: SeaOcean
Create Date: 2026-03-18
Description：知识源挂载命令行脚本
-------------------------------------------------
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rag.knowledge_base import knowledge_base


def main() -> int:
    parser = argparse.ArgumentParser(description="Mount PDF/ZIP/TXT sources into the RAG knowledge base.")
    parser.add_argument("paths", nargs="+", help="Absolute or relative file paths to mount (.pdf/.zip/.txt/.md)")
    parser.add_argument("--query", default="ISO 29148 verifiable unambiguous requirement", help="Verification query")
    parser.add_argument("--topk", type=int, default=3, help="Number of verification results to print")
    args = parser.parse_args()

    total_mounted = 0
    for raw_path in args.paths:
        pdf_path = Path(raw_path).expanduser().resolve()
        mounted_count = knowledge_base.mount_source(pdf_path)
        total_mounted += mounted_count
        print(f"{pdf_path}: mounted {mounted_count} chunks")

    print("stats:", knowledge_base.get_collection_stats())
    print(f"total_mounted: {total_mounted}")

    results = knowledge_base.query(args.query, n_results=args.topk)
    for index, item in enumerate(results, start=1):
        metadata = item.get("metadata", {})
        print(
            f"[{index}] id={item.get('id')} "
            f"source={metadata.get('source', 'built-in')} "
            f"page={metadata.get('page', '-')}"
        )
        print(item.get("content", "").replace("\n", " ")[:400])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
