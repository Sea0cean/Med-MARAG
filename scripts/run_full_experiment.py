from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.review_agent import ReviewAgent
from config import Config
from llm.provider import build_llm_runtime, extract_json_object, invoke_llm_text
from rag.knowledge_base import knowledge_base
from utils.plantuml_utils import PlantUMLUtils
from utils.requirement_utils import RequirementUtils
from workflow.ocean_graph import PipelineConfig, run_modeller_pipeline


BASE_GROUPS = [
    "Only LLM",
    "LLM with RAG",
]

MAS_NO_RAG_GROUPS = [
    "LLM with MAS (max_iter=1)",
    "LLM with MAS (max_iter=3)",
    "LLM with MAS (max_iter=5)",
]

MAS_RAG_GROUPS = [
    "MAS + RAG (max_iter=1)",
    "MAS + RAG (max_iter=3)",
    "MAS + RAG (max_iter=5)",
]

SYNTAX_PASS_SCORE_THRESHOLD = 70


def _normalize_max_iters(raw: str | None, default: list[int]) -> list[int]:
    if not raw:
        return list(default)
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            continue
        values.append(value)
    return values or list(default)


def _format_mas_without_rag_group(max_iter: int) -> str:
    return f"LLM with MAS (max_iter={max_iter})"


def _format_mas_rag_group(max_iter: int) -> str:
    return f"MAS + RAG (max_iter={max_iter})"


def get_group_names(include_mas_without_rag: bool = False, mas_without_rag_iters: list[int] | None = None) -> list[str]:
    groups = list(BASE_GROUPS)
    if include_mas_without_rag:
        groups.extend(_format_mas_without_rag_group(max_iter) for max_iter in (mas_without_rag_iters or [1, 3, 5]))
    groups.extend(MAS_RAG_GROUPS)
    return groups


def get_mas_group_specs(
    include_mas_without_rag: bool = False,
    mas_without_rag_iters: list[int] | None = None,
) -> list[tuple[int, str, bool]]:
    specs: list[tuple[int, str, bool]] = []
    if include_mas_without_rag:
        specs.extend((max_iter, _format_mas_without_rag_group(max_iter), False) for max_iter in (mas_without_rag_iters or [1, 3, 5]))
    specs.extend(
        [
            (1, "MAS + RAG (max_iter=1)", True),
            (3, "MAS + RAG (max_iter=3)", True),
            (5, "MAS + RAG (max_iter=5)", True),
        ]
    )
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full 250-case DeepSeek experiment.")
    parser.add_argument(
        "--data-path",
        default=str(Path("output") / "requirements_input.csv"),
        help="CSV dataset path. Defaults to output/requirements_input.csv.",
    )
    parser.add_argument(
        "--output-path",
        default=str(Path("output") / "deepseek_experiment_full_250.json"),
        help="Incremental JSON result path.",
    )
    parser.add_argument(
        "--provider",
        default="deepseek",
        choices=Config.SUPPORTED_LLM_PROVIDERS,
        help="LLM provider for online generation.",
    )
    parser.add_argument(
        "--model",
        default=Config.DEEPSEEK_DEFAULT_MODEL,
        help="Model override. Defaults to deepseek-chat.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output file if present.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optional debug limit. 0 means run all rows.",
    )
    parser.add_argument(
        "--basic-count",
        type=int,
        default=0,
        help="Select the first N '基础交互' cases before sharding. 0 means no category filter.",
    )
    parser.add_argument(
        "--complex-count",
        type=int,
        default=0,
        help="Select the first N '复杂业务' cases before sharding. 0 means no category filter.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total shard count for parallel runs. Defaults to 1.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=1,
        help="1-based shard index to run when num-shards > 1.",
    )
    parser.add_argument(
        "--include-mas-without-rag",
        action="store_true",
        help="Also run the MAS ablation group without RAG retrieval.",
    )
    parser.add_argument(
        "--mas-without-rag-max-iters",
        default="1,3,5",
        help="Comma-separated max iteration values for LLM with MAS when enabled. Defaults to 1,3,5.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=20260325,
        help="Random seed used when sampling fewer than 250 cases. Defaults to 20260325.",
    )
    return parser.parse_args()


def _rows_to_cases(df: pd.DataFrame) -> list[dict[str, str]]:
    return [
        {
            "id": str(row["id"]),
            "category": str(row["category"]),
            "requirement": str(row["requirement"]),
        }
        for _, row in df.iterrows()
    ]


def _slice_for_shard(rows: list[dict[str, str]], *, num_shards: int, shard_index: int) -> list[dict[str, str]]:
    if num_shards <= 1:
        return rows
    base = len(rows) // num_shards
    remainder = len(rows) % num_shards
    start = (shard_index - 1) * base + min(shard_index - 1, remainder)
    end = start + base + (1 if shard_index <= remainder else 0)
    return rows[start:end]


def _sample_ratio_cases(df: pd.DataFrame, *, max_cases: int, sample_seed: int) -> list[dict[str, str]]:
    if max_cases <= 0 or max_cases >= len(df):
        return _rows_to_cases(df)

    basic_df = df[df["category"] == "基础交互"]
    complex_df = df[df["category"] == "复杂业务"]
    basic_target = int(math.floor(max_cases * 2 / 5))
    complex_target = max_cases - basic_target

    basic_take = min(len(basic_df), basic_target)
    complex_take = min(len(complex_df), complex_target)
    remainder = max_cases - basic_take - complex_take

    if remainder > 0:
        basic_room = max(0, len(basic_df) - basic_take)
        extra_basic = min(remainder, basic_room)
        basic_take += extra_basic
        remainder -= extra_basic
    if remainder > 0:
        complex_room = max(0, len(complex_df) - complex_take)
        extra_complex = min(remainder, complex_room)
        complex_take += extra_complex
        remainder -= extra_complex

    sampled_frames = []
    if basic_take > 0:
        sampled_frames.append(basic_df.sample(n=basic_take, random_state=sample_seed))
    if complex_take > 0:
        sampled_frames.append(complex_df.sample(n=complex_take, random_state=sample_seed + 1))

    if not sampled_frames:
        return []

    sampled_df = pd.concat(sampled_frames, ignore_index=False)
    sampled_df = sampled_df.sample(frac=1.0, random_state=sample_seed + 2).reset_index(drop=True)
    return _rows_to_cases(sampled_df)


def load_cases(
    csv_path: Path,
    *,
    max_cases: int = 0,
    basic_count: int = 0,
    complex_count: int = 0,
    num_shards: int = 1,
    shard_index: int = 1,
    sample_seed: int = 20260325,
) -> list[dict[str, str]]:
    df = pd.read_csv(csv_path)
    if basic_count > 0 or complex_count > 0:
        basic_rows = _rows_to_cases(df[df["category"] == "基础交互"].head(basic_count or len(df)))
        complex_rows = _rows_to_cases(df[df["category"] == "复杂业务"].head(complex_count or len(df)))
        rows = _slice_for_shard(basic_rows, num_shards=num_shards, shard_index=shard_index) + _slice_for_shard(
            complex_rows,
            num_shards=num_shards,
            shard_index=shard_index,
        )
    else:
        rows = _sample_ratio_cases(df, max_cases=max_cases, sample_seed=sample_seed)
        rows = _slice_for_shard(rows, num_shards=num_shards, shard_index=shard_index)

    return rows[:max_cases] if max_cases > 0 and (basic_count > 0 or complex_count > 0) else rows


def load_results(output_path: Path, cases: list[dict[str, str]], args: argparse.Namespace) -> dict[str, Any]:
    mas_without_rag_iters = _normalize_max_iters(args.mas_without_rag_max_iters, [1, 3, 5])
    groups = get_group_names(args.include_mas_without_rag, mas_without_rag_iters)
    if args.resume and output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        payload.setdefault("groups", {})
        for group in groups:
            payload["groups"].setdefault(group, [])
        payload.setdefault("meta", {})
        payload["meta"]["groups"] = groups
        payload["meta"]["include_mas_without_rag"] = bool(args.include_mas_without_rag)
        payload["meta"]["mas_without_rag_iters"] = mas_without_rag_iters
        payload["meta"]["score_weights"] = {
            "diagram_consistency": ReviewAgent.DIAGRAM_CONSISTENCY_WEIGHT,
            "domain_knowledge_alignment": ReviewAgent.DOMAIN_KNOWLEDGE_WEIGHT,
        }
        payload["meta"]["scoring_rule"] = (
            "All experiment groups are rescored by ReviewAgent(provider='offline').review_model_pack. "
            "Overall score = 0.7 * diagram_consistency + 0.3 * domain_knowledge_alignment."
        )
        payload.setdefault("syntax_pass_rates", {})
        payload.setdefault("domain_knowledge_averages", {})
        return payload

    return {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_source": str(Path(args.data_path).resolve()),
            "note": "If reqirement_input.xlsx is absent, the experiment uses output/requirements_input.csv.",
            "provider": args.provider,
            "model": args.model,
            "mode": "full_path_for_mas_groups",
            "scoring_rule": (
                "All experiment groups are rescored by ReviewAgent(provider='offline').review_model_pack. "
                "Overall score = 0.7 * diagram_consistency + 0.3 * domain_knowledge_alignment."
            ),
            "syntax_pass_rule": (
                "Syntax pass requires both diagram-level structural checks to pass and overall score >= "
                f"{SYNTAX_PASS_SCORE_THRESHOLD}."
            ),
            "score_weights": {
                "diagram_consistency": ReviewAgent.DIAGRAM_CONSISTENCY_WEIGHT,
                "domain_knowledge_alignment": ReviewAgent.DOMAIN_KNOWLEDGE_WEIGHT,
            },
            "resume_enabled": bool(args.resume),
            "max_cases": int(args.max_cases or 0),
            "basic_count": int(args.basic_count or 0),
            "complex_count": int(args.complex_count or 0),
            "num_shards": int(args.num_shards or 1),
            "shard_index": int(args.shard_index or 1),
            "include_mas_without_rag": bool(args.include_mas_without_rag),
            "mas_without_rag_iters": mas_without_rag_iters,
            "sample_seed": int(args.sample_seed),
            "sampling_rule": "When max_cases < 250 and no explicit category counts are provided, randomly sample cases with 基础交互:复杂业务 = 2:3.",
            "groups": groups,
        },
        "cases": cases,
        "groups": {group: [] for group in groups},
        "averages": {},
        "syntax_pass_rates": {},
        "domain_knowledge_averages": {},
    }


def save_results(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_with_retries(label: str, func: Any, *, attempts: int = 4, pause_seconds: int = 5) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:  # pragma: no cover - long-running online batch guard
            last_error = exc
            print(f"    {label} failed on attempt {attempt}/{attempts}: {exc}", flush=True)
            if attempt < attempts:
                time.sleep(pause_seconds)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{label} failed without an explicit exception.")


def completed_case_ids(results: dict[str, Any], group_name: str) -> set[str]:
    return {str(item.get("id")) for item in results.get("groups", {}).get(group_name, [])}


def build_one_shot_prompt(requirement: str, context: str = "") -> str:
    context_block = f"\n参考知识：\n{context}\n" if context.strip() else ""
    return f"""
你是医疗软件 UML 建模专家。
你的任务是：根据给定需求直接生成完整的 UML 模型制品。

要求：
1. 只输出严格 JSON，不要输出解释文字
2. 必须同时输出 use_case_diagram、class_diagram、sequence_diagram
3. 每段 PlantUML 代码都必须以 @startuml 开头、以 @enduml 结束
4. 不得编造输入需求中不存在的关键业务事实
5. 类图应覆盖核心实体、属性、方法和关系
6. 用例图应体现主要参与者与核心用例
7. 顺序图应体现主成功流程

输入需求：
{requirement}
{context_block}
输出 JSON：
{{
  "use_case_diagram": "完整 PlantUML 用例图代码",
  "class_diagram": "完整 PlantUML 类图代码",
  "sequence_diagram": "完整 PlantUML 顺序图代码"
}}
""".strip()


def sanitize_artifacts(artifacts: dict[str, Any]) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for key in ("use_case_diagram", "class_diagram", "sequence_diagram"):
        cleaned[key] = PlantUMLUtils.sandbox_preprocess(str(artifacts.get(key, "") or ""))["code"]
    return cleaned


def _strict_diagram_issues(diagram_name: str, plantuml_code: str) -> dict[str, list[str]]:
    blocking: list[str] = []
    advisory: list[str] = []
    lines = [line.strip() for line in (plantuml_code or "").splitlines() if line.strip() and not line.strip().startswith("@")]

    if diagram_name == "use_case_diagram":
        actor_count = len(re.findall(r'^\s*actor\b', plantuml_code, re.M))
        use_case_count = len(re.findall(r'^\s*usecase\b', plantuml_code, re.M)) + len(
            re.findall(r'\([^)]+\)', plantuml_code)
        )
        relation_count = sum(1 for line in lines if "-->" in line or "..>" in line)
        if actor_count == 0:
            blocking.append("用例图缺少参与者。")
        if use_case_count == 0:
            blocking.append("用例图缺少用例定义。")
        if relation_count == 0:
            blocking.append("用例图缺少参与者与用例之间的关系。")
        if "rectangle" not in plantuml_code.lower():
            advisory.append("用例图缺少系统边界。")
        return {"blocking": blocking, "advisory": advisory}

    if diagram_name == "class_diagram":
        class_count = len(
            re.findall(r'^\s*(?:class|entity|interface|enum)\s+(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)', plantuml_code, re.M)
        )
        relationship_lines = [
            line
            for line in lines
            if any(operator in line for operator in ("--", "-->", "<|--", "o--", "*--", "..>"))
        ]
        method_count = len(re.findall(r'^\s*\+\w+\(.*\)', plantuml_code, re.M))
        if class_count == 0:
            blocking.append("类图核心类数量不足。")
        elif class_count < 2:
            advisory.append("类图核心类数量不足。")
        if not relationship_lines:
            advisory.append("类图缺少类间关系。")
        if class_count >= 3 and len(relationship_lines) < 2:
            advisory.append("类图关系数量不足以支撑多实体场景。")
        if method_count == 0:
            advisory.append("类图缺少可操作方法定义。")
        for line in relationship_lines:
            if ":" not in line:
                advisory.append(f"类图关系缺少语义标签: {line}")
        return {"blocking": blocking, "advisory": advisory}

    if diagram_name == "sequence_diagram":
        actor_count = len(re.findall(r'^\s*actor\b', plantuml_code, re.M))
        participant_count = len(re.findall(r'^\s*participant\b', plantuml_code, re.M))
        message_lines = [line for line in lines if any(op in line for op in ("->", "-->"))]
        self_calls = 0
        for line in message_lines:
            left, _, right = line.partition(":")[0].partition("->")
            if left.strip().replace("-", "") == right.strip().replace("-", ""):
                self_calls += 1
        if actor_count == 0:
            blocking.append("顺序图缺少参与者。")
        if participant_count == 0:
            blocking.append("顺序图缺少系统参与对象。")
        if len(message_lines) < 2:
            blocking.append("顺序图交互步骤过少。")
        elif len(message_lines) < 3:
            advisory.append("顺序图交互步骤较少。")
        if message_lines and self_calls / len(message_lines) > 0.6:
            blocking.append("顺序图自调用比例过高。")
        elif message_lines and self_calls / len(message_lines) > 0.35:
            advisory.append("顺序图自调用比例偏高。")
        if "-->" not in plantuml_code:
            advisory.append("顺序图缺少明确返回消息。")
        return {"blocking": blocking, "advisory": advisory}

    return {"blocking": blocking, "advisory": advisory}


def compute_syntax_pass(artifacts: dict[str, Any]) -> dict[str, Any]:
    cleaned = sanitize_artifacts(artifacts)
    details: dict[str, Any] = {}
    diagram_names = ("use_case_diagram", "class_diagram", "sequence_diagram")
    all_passed = True
    for key in diagram_names:
        result = PlantUMLUtils.sandbox_preprocess(str(cleaned.get(key, "") or ""))
        strict_issues = _strict_diagram_issues(key, result["code"])
        combined_issues = list(
            dict.fromkeys((result["issues"] or []) + strict_issues["blocking"] + strict_issues["advisory"])
        )
        passed = bool(result["code"]) and not (result["issues"] or []) and not strict_issues["blocking"]
        details[key] = {
            "passed": passed,
            "issues": combined_issues,
        }
        all_passed = all_passed and passed
    return {
        "passed": all_passed,
        "details": details,
    }


def apply_syntax_pass_threshold(
    syntax_result: dict[str, Any],
    *,
    overall_score: int,
    score_threshold: int = SYNTAX_PASS_SCORE_THRESHOLD,
) -> dict[str, Any]:
    passed = bool(syntax_result.get("passed")) and int(overall_score) >= score_threshold
    details = dict(syntax_result.get("details", {}))
    if syntax_result.get("passed") and int(overall_score) < score_threshold:
        details["score_gate"] = {
            "passed": False,
            "issues": [f"综合分低于 UML 编译通过阈值: {overall_score} < {score_threshold}"],
        }
    return {
        "passed": passed,
        "details": details,
        "score_threshold": score_threshold,
    }


def run_one_shot_group(runtime: Any, requirement: str, *, use_rag: bool) -> dict[str, Any]:
    knowledge_items = knowledge_base.query(requirement, n_results=3) if use_rag else []
    context = "\n".join(item["content"] for item in knowledge_items)
    content = invoke_llm_text(
        runtime,
        [
            ("system", "你只能输出严格 JSON。"),
            ("user", build_one_shot_prompt(requirement, context)),
        ],
    )
    parsed = extract_json_object(content)
    artifacts = sanitize_artifacts(parsed if isinstance(parsed, dict) else {})
    return {
        "artifacts": artifacts,
        "knowledge_hits": len(knowledge_items),
        "used_fallback": not bool(content),
    }


def score_artifacts(
    scorer: ReviewAgent,
    artifacts: dict[str, Any],
    *,
    requirement_text: str,
    requirement_items: list[dict[str, Any]] | None = None,
    ears_requirement: str = "",
) -> dict[str, Any]:
    items = requirement_items or [RequirementUtils.analyze_requirement(requirement_text, index=1)]
    ears = ears_requirement or "\n\n".join(item.get("ears_requirement", "") for item in items)
    review = scorer.review_model_pack(
        sanitize_artifacts(artifacts),
        requirement_items=items,
        ears_requirement=ears,
    )
    return {
        "score": int(((review.get("scores") or {}).get("overall", 0) or 0)),
        "domain_knowledge_score": int(((review.get("scores") or {}).get("domain_knowledge_alignment", 0) or 0)),
        "issues_count": len(review.get("issues", []) or []),
        "review_source": review.get("review_source", "local"),
        "review_scores": review.get("scores", {}),
        "domain_knowledge_detail": ((review.get("details") or {}).get("domain_knowledge", {}) or {}),
    }


def append_result(results: dict[str, Any], group_name: str, record: dict[str, Any]) -> None:
    results.setdefault("groups", {}).setdefault(group_name, []).append(record)


def recompute_averages(results: dict[str, Any], groups: list[str]) -> None:
    averages: dict[str, float] = {}
    syntax_pass_rates: dict[str, float] = {}
    domain_knowledge_averages: dict[str, float] = {}
    for group_name in groups:
        scores = [item["score"] for item in results["groups"].get(group_name, []) if item.get("score") is not None]
        averages[group_name] = round(sum(scores) / len(scores), 2) if scores else 0.0
        syntax_flags = [bool(item.get("syntax_pass")) for item in results["groups"].get(group_name, [])]
        syntax_pass_rates[group_name] = round((sum(syntax_flags) / len(syntax_flags)) * 100, 2) if syntax_flags else 0.0
        domain_scores = [
            item["domain_knowledge_score"]
            for item in results["groups"].get(group_name, [])
            if item.get("domain_knowledge_score") is not None
        ]
        domain_knowledge_averages[group_name] = round(sum(domain_scores) / len(domain_scores), 2) if domain_scores else 0.0
    results["averages"] = averages
    results["syntax_pass_rates"] = syntax_pass_rates
    results["domain_knowledge_averages"] = domain_knowledge_averages


def main() -> None:
    args = parse_args()
    csv_path = Path(args.data_path).resolve()
    output_path = Path(args.output_path).resolve()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1.")
    if not 1 <= args.shard_index <= args.num_shards:
        raise ValueError("--shard-index must be between 1 and --num-shards.")

    mas_without_rag_iters = _normalize_max_iters(args.mas_without_rag_max_iters, [1, 3, 5])
    groups = get_group_names(args.include_mas_without_rag, mas_without_rag_iters)
    mas_group_specs = get_mas_group_specs(args.include_mas_without_rag, mas_without_rag_iters)
    cases = load_cases(
        csv_path,
        max_cases=args.max_cases,
        basic_count=args.basic_count,
        complex_count=args.complex_count,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        sample_seed=args.sample_seed,
    )
    results = load_results(output_path, cases, args)

    runtime = build_llm_runtime(provider=args.provider, model_override=args.model)
    scorer = ReviewAgent(provider="offline")

    print(f"Loaded {len(cases)} cases from {csv_path}")
    print(
        "Runtime:",
        runtime.requested_provider,
        runtime.effective_provider,
        runtime.model,
        runtime.status,
        flush=True,
    )

    for index, case in enumerate(cases, start=1):
        case_id = case["id"]
        requirement = case["requirement"]
        base_item = RequirementUtils.analyze_requirement(requirement, index=1)

        print(f"\n[{index}/{len(cases)}] {case_id} | {case['category']}", flush=True)

        if case_id not in completed_case_ids(results, "Only LLM"):
            baseline = run_with_retries(
                f"{case_id} Only LLM",
                lambda: run_one_shot_group(runtime, requirement, use_rag=False),
            )
            score = score_artifacts(
                scorer,
                baseline["artifacts"],
                requirement_text=requirement,
                requirement_items=[base_item],
                ears_requirement=base_item.get("ears_requirement", ""),
            )
            syntax = apply_syntax_pass_threshold(
                compute_syntax_pass(baseline["artifacts"]),
                overall_score=score["score"],
            )
            append_result(
                results,
                "Only LLM",
                {
                    "id": case_id,
                    "category": case["category"],
                    "score": score["score"],
                    "issues_count": score["issues_count"],
                    "domain_knowledge_score": score["domain_knowledge_score"],
                    "domain_knowledge_detail": score["domain_knowledge_detail"],
                    "used_fallback": baseline["used_fallback"],
                    "syntax_pass": syntax["passed"],
                    "syntax_details": syntax["details"],
                },
            )
            recompute_averages(results, groups)
            save_results(output_path, results)
            print("  Only LLM:", score["score"], flush=True)

        if case_id not in completed_case_ids(results, "LLM with RAG"):
            rag_result = run_with_retries(
                f"{case_id} LLM with RAG",
                lambda: run_one_shot_group(runtime, requirement, use_rag=True),
            )
            score = score_artifacts(
                scorer,
                rag_result["artifacts"],
                requirement_text=requirement,
                requirement_items=[base_item],
                ears_requirement=base_item.get("ears_requirement", ""),
            )
            syntax = apply_syntax_pass_threshold(
                compute_syntax_pass(rag_result["artifacts"]),
                overall_score=score["score"],
            )
            append_result(
                results,
                "LLM with RAG",
                {
                    "id": case_id,
                    "category": case["category"],
                    "score": score["score"],
                    "issues_count": score["issues_count"],
                    "domain_knowledge_score": score["domain_knowledge_score"],
                    "domain_knowledge_detail": score["domain_knowledge_detail"],
                    "knowledge_hits": rag_result["knowledge_hits"],
                    "used_fallback": rag_result["used_fallback"],
                    "syntax_pass": syntax["passed"],
                    "syntax_details": syntax["details"],
                },
            )
            recompute_averages(results, groups)
            save_results(output_path, results)
            print("  LLM with RAG:", score["score"], flush=True)

        for max_iter, group_name, enable_rag in mas_group_specs:
            if case_id in completed_case_ids(results, group_name):
                continue

            final_state = run_with_retries(
                f"{case_id} {group_name}",
                lambda: run_modeller_pipeline(
                    requirement,
                    config=PipelineConfig(
                        provider=args.provider,
                        model_override=args.model,
                        enable_rag=enable_rag,
                        max_iterations=max_iter,
                    ),
                ),
            )
            artifacts = final_state.get("UML_Artifacts", {}) or {}
            score = score_artifacts(
                scorer,
                artifacts,
                requirement_text=requirement,
                requirement_items=final_state.get("Requirement_Items") or [base_item],
                ears_requirement=final_state.get("EARS_Requirement", ""),
            )
            syntax = apply_syntax_pass_threshold(
                compute_syntax_pass(artifacts),
                overall_score=score["score"],
            )
            append_result(
                results,
                group_name,
                {
                    "id": case_id,
                    "category": case["category"],
                    "score": score["score"],
                    "issues_count": score["issues_count"],
                    "domain_knowledge_score": score["domain_knowledge_score"],
                    "domain_knowledge_detail": score["domain_knowledge_detail"],
                    "status": final_state.get("Status"),
                    "iteration_count": int(final_state.get("Iteration_Count", 0) or 0),
                    "runtime_provider": final_state.get("LLM_Provider"),
                    "runtime_status": final_state.get("LLM_Runtime_Status"),
                    "runtime_model": final_state.get("LLM_Model"),
                    "rag_enabled": bool(final_state.get("RAG_Enabled", enable_rag)),
                    "syntax_pass": syntax["passed"],
                    "syntax_details": syntax["details"],
                    "pipeline_review_score": int(
                        (((final_state.get("Review_Report") or {}).get("scores") or {}).get("overall", 0) or 0)
                    ),
                },
            )
            recompute_averages(results, groups)
            save_results(output_path, results)
            print(f"  {group_name}: {score['score']}", flush=True)

    recompute_averages(results, groups)
    save_results(output_path, results)
    print("\nFinal averages:")
    for group_name in groups:
        print(
            f"  {group_name}: avg={results['averages'][group_name]} "
            f"domain_knowledge={results['domain_knowledge_averages'][group_name]} "
            f"syntax_pass={results['syntax_pass_rates'][group_name]}%",
        )
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
