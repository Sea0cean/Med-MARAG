from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_GROUPS = [
    "Only LLM",
    "LLM with RAG",
    "LLM with MAS (max_iter=1)",
    "LLM with MAS (max_iter=3)",
    "LLM with MAS (max_iter=5)",
    "MAS + RAG (max_iter=1)",
    "MAS + RAG (max_iter=3)",
    "MAS + RAG (max_iter=5)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple experiment shard result files.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Shard JSON files to merge.")
    parser.add_argument("--output", required=True, help="Merged JSON output path.")
    return parser.parse_args()


def resolve_groups(payloads: list[dict[str, Any]]) -> list[str]:
    groups: list[str] = []
    for group_name in DEFAULT_GROUPS:
        if any(group_name in (payload.get("groups") or {}) for payload in payloads):
            groups.append(group_name)
    for payload in payloads:
        for group_name in (payload.get("meta", {}) or {}).get("groups", []):
            if group_name not in groups:
                groups.append(group_name)
        for group_name in (payload.get("groups") or {}).keys():
            if group_name not in groups:
                groups.append(group_name)
    return groups


def recompute_averages(results: dict[str, Any], groups: list[str]) -> None:
    averages: dict[str, float] = {}
    domain_knowledge_averages: dict[str, float] = {}
    for group_name in groups:
        scores = [item["score"] for item in results["groups"].get(group_name, []) if item.get("score") is not None]
        averages[group_name] = round(sum(scores) / len(scores), 2) if scores else 0.0
        domain_scores = [
            item["domain_knowledge_score"]
            for item in results["groups"].get(group_name, [])
            if item.get("domain_knowledge_score") is not None
        ]
        domain_knowledge_averages[group_name] = round(sum(domain_scores) / len(domain_scores), 2) if domain_scores else 0.0
    results["averages"] = averages
    results["domain_knowledge_averages"] = domain_knowledge_averages


def main() -> None:
    args = parse_args()
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.inputs]
    groups = resolve_groups(payloads)
    merged: dict[str, Any] = {
        "meta": {
            "merged_from": [str(Path(path).resolve()) for path in args.inputs],
            "notes": "Merged shard experiment results.",
            "groups": groups,
        },
        "cases": [],
        "groups": {group: [] for group in groups},
        "averages": {},
        "domain_knowledge_averages": {},
    }

    seen_case_ids: set[str] = set()
    for payload in payloads:
        for case in payload.get("cases", []):
            case_id = str(case.get("id"))
            if case_id in seen_case_ids:
                continue
            merged["cases"].append(case)
            seen_case_ids.add(case_id)

        for group_name in groups:
            merged["groups"][group_name].extend(payload.get("groups", {}).get(group_name, []))

    for group_name in groups:
        merged["groups"][group_name].sort(key=lambda item: str(item.get("id")))

    recompute_averages(merged, groups)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Merged averages:")
    for group_name in groups:
        print(
            f"  {group_name}: avg={merged['averages'][group_name]} "
            f"domain_knowledge={merged['domain_knowledge_averages'][group_name]}"
        )
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
