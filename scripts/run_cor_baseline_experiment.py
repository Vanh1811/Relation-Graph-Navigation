#!/usr/bin/env python3
"""Run the first, unchanged-CoR CWQ baseline experiment.

The input files are inference-safe structural subsets.  Gold answers are used
only after inference for evaluation; gold structure metadata is never passed
to the CoR agent.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import ssl
import sys
import time
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Iterable

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import certifi

from _bootstrap import PROJECT_ROOT  # noqa: E402


class _RawOpenAIClient:
    """Small Chat Completions-compatible client used by this experiment only.

    The repository's pinned OpenAI SDK takes an unusually long time to import
    on this machine.  This adapter keeps the existing LLMAPI/CoRAgent contract
    while using the official HTTPS endpoint directly.
    """

    def __init__(self, api_key: str, base_url: str | None = None, **_: Any) -> None:
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        payload = dict(kwargs)
        timeout = float(payload.pop("timeout", 30))
        payload.pop("stream", None)
        payload.pop("frequency_penalty", None)
        payload.pop("presence_penalty", None)

        # Reasoning models use max_completion_tokens.  The existing LLMAPI
        # calls this field max_tokens for compatibility with older models.
        if "max_tokens" in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens")
        if str(payload.get("model", "")).startswith("gpt-5"):
            # The configured Luna model is a reasoning model and does not use
            # sampling temperature in this Chat Completions request shape.
            payload.pop("temperature", None)

        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        context = ssl.create_default_context(cafile=certifi.where())
        try:
            with urlopen(request, timeout=timeout, context=context) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read(800).decode("utf-8", "replace").replace("\n", " ")
            raise RuntimeError(f"OpenAI HTTP {exc.code}: {body[:500]}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"OpenAI network error: {exc}") from exc

        choices = data.get("choices") or []
        usage = data.get("usage") or {}
        message = (choices[0].get("message") if choices else {}) or {}
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=message.get("content")))],
            usage=SimpleNamespace(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
        )


def _install_openai_transport() -> None:
    module = ModuleType("openai")
    module.OpenAI = _RawOpenAIClient
    sys.modules["openai"] = module


_install_openai_transport()

from chain_of_relations.eval.accuracy import eval_f1, eval_hit  # noqa: E402
from chain_of_relations.kg_backend import get_backend  # noqa: E402
from chain_of_relations.methods.cor.agent import CoRAgent  # noqa: E402
from chain_of_relations.schema import Entity  # noqa: E402


DEFAULT_SEED = 20261013
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "cor_baseline"
SUBSET_ROOT = PROJECT_ROOT / "outputs" / "subsets"

SUBSETS = {
    "cwq_linear": {"path": SUBSET_ROOT / "cwq_linear.jsonl", "sample_size": 50},
    "cwq_branch_2": {"path": SUBSET_ROOT / "cwq_branch_2.jsonl", "sample_size": 50},
    "cwq_branch_3_plus": {"path": SUBSET_ROOT / "cwq_branch_3_plus.jsonl", "sample_size": None},
}


def load_safe_project_env() -> None:
    """Load only single-line project settings from .env.

    The local .env also contains a multiline credential unrelated to this
    experiment.  It must not be shell-sourced, so only the five named
    single-line settings used here are parsed.
    """
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    allowed = {
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "MODEL_NAME",
        "FREEBASE_SPARQL_ENDPOINT",
        "WIKIDATA_SPARQL_ENDPOINT",
    }
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("export ") or "=" not in line:
            continue
        key, value = line[7:].split("=", 1)
        key = key.strip()
        if key not in allowed or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
            value = value[1:-1]
        os.environ[key] = value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def select_rows(rows: list[dict[str, Any]], sample_size: int | None, seed: int) -> list[dict[str, Any]]:
    if sample_size is None or sample_size >= len(rows):
        return list(rows)
    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size}")
    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(rows)), sample_size))
    return [rows[index] for index in selected_indices]


def normalize_topic_entities(row: dict[str, Any]) -> list[Entity]:
    raw_ids = row.get("topic_entities", [])
    raw_names = row.get("topic_entity_names", [])
    if not isinstance(raw_ids, list) or not isinstance(raw_names, list):
        return []
    return [
        Entity(id=str(entity_id), name=str(entity_name))
        for entity_id, entity_name in zip(raw_ids, raw_names)
        if str(entity_id).strip() and str(entity_name).strip()
    ]


def prediction_names(result: dict[str, Any]) -> list[str]:
    values = result.get("results", [])
    if not isinstance(values, list):
        return []
    names: list[str] = []
    for value in values:
        if isinstance(value, dict) and str(value.get("name", "")).strip():
            names.append(str(value["name"]).strip())
        elif isinstance(value, str) and value.strip():
            names.append(value.strip())
    return names


def relation_chain_from_trace(result: dict[str, Any]) -> list[str]:
    """Extract the final relation path without exposing gold structure."""
    reasoning_chains = result.get("reasoning_chains", "")
    if isinstance(reasoning_chains, list):
        reasoning_chains = "\n".join(str(item) for item in reasoning_chains)
    if not isinstance(reasoning_chains, str):
        return []

    relations: list[str] = []
    for line in reasoning_chains.splitlines():
        match = re.search(r"\)\s+(\S+)\s+e\d+\s+\.", line)
        if match:
            relations.append(match.group(1))
        else:
            match = re.search(r"e\d+\s+(\S+)\s+e\d+\s+\.", line)
            if match:
                relations.append(match.group(1))
    return relations


def integer_usage(prompt_history: Iterable[dict[str, Any]], field: str) -> int:
    total = 0
    for event in prompt_history:
        value = event.get(field, 0)
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            total += value
    return total


def evaluate(prediction: list[str], gold_answers: list[str]) -> tuple[int, float]:
    prediction_for_eval = prediction or ["None"]
    prediction_text = " ".join(prediction_for_eval)
    hit = eval_hit(prediction_text, gold_answers) if gold_answers else 0
    f1, _, _ = eval_f1(prediction_for_eval, gold_answers) if gold_answers else (0.0, 0.0, 0.0)
    return int(hit), float(f1)


def run_one(agent: CoRAgent, row: dict[str, Any], detail_dir: Path) -> dict[str, Any]:
    question = str(row.get("question", ""))
    example_id = str(row.get("id", ""))
    gold_answers = [str(value) for value in row.get("answers", []) if str(value).strip()]
    topic_entities = normalize_topic_entities(row)
    prompt_history: list[dict[str, Any]] = []
    sparql_history: list[dict[str, Any]] = []
    started = time.perf_counter()
    error: str | None = None

    try:
        if not topic_entities:
            raise ValueError("inference-safe subset has no valid topic entities")
        result = agent.answer(
            question=question,
            topic_entities=topic_entities,
            prompt_history=prompt_history,
            sparql_history=sparql_history,
        )
    except Exception as exc:  # preserve one row and continue the experiment
        result = {"action": "ERROR", "results": [], "reasoning_chains": ""}
        error = f"{type(exc).__name__}: {exc}"
        logging.exception("Baseline failed for %s", example_id)

    latency_ms = (time.perf_counter() - started) * 1000.0
    prediction = prediction_names(result)
    hit, f1 = evaluate(prediction, gold_answers)
    action = str(result.get("action", ""))
    if action.upper() == "ERROR" and error is None:
        error = str(result.get("reasoning_chains", "LLM/agent error"))

    input_tokens = integer_usage(prompt_history, "input_tokens")
    output_tokens = integer_usage(prompt_history, "output_tokens")
    token_usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    row_result = {
        "id": example_id,
        "question": question,
        "gold_answers": gold_answers,
        "prediction": prediction,
        "hits_at_1": hit,
        "f1": f1,
        "relation_chain": relation_chain_from_trace(result),
        "llm_calls": len(prompt_history),
        "kg_queries": len(sparql_history),
        "token_usage": token_usage,
        "latency_ms": latency_ms,
        "error": error,
    }

    detail = {
        **row_result,
        "action": action,
        "reasoning_chains": result.get("reasoning_chains", ""),
        "prompt_history": prompt_history,
        "sparql_history": sparql_history,
        "step_history": result.get("step_history", []),
    }
    detail_dir.mkdir(parents=True, exist_ok=True)
    with (detail_dir / f"{example_id}.json").open("w", encoding="utf-8") as handle:
        json.dump(detail, handle, ensure_ascii=False, indent=2)
    return row_result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    errors = sum(1 for row in rows if row.get("error"))
    total_tokens = sum(row["token_usage"]["total_tokens"] for row in rows)
    total_llm_calls = sum(int(row.get("llm_calls", 0)) for row in rows)
    total_kg_queries = sum(int(row.get("kg_queries", 0)) for row in rows)
    return {
        "n": count,
        "hits_at_1": sum(row["hits_at_1"] for row in rows) / count if count else 0.0,
        "f1": sum(row["f1"] for row in rows) / count if count else 0.0,
        "error_rate": errors / count if count else 0.0,
        "average_llm_calls": total_llm_calls / count if count else 0.0,
        "average_kg_queries": total_kg_queries / count if count else 0.0,
        "average_tokens": total_tokens / count if count else 0.0,
        "average_latency_ms": sum(row["latency_ms"] for row in rows) / count if count else 0.0,
        "total_tokens": total_tokens,
        "total_llm_calls": total_llm_calls,
        "total_kg_queries": total_kg_queries,
        "errors": errors,
    }


def write_failure_analysis(all_rows: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    labels = Counter()
    sections: list[str] = ["# CoR baseline failure analysis", ""]
    for subset, rows in all_rows.items():
        for row in rows:
            if row.get("hits_at_1"):
                continue
            relation_chain = row.get("relation_chain", [])
            error = row.get("error")
            if error:
                label = "KG_EXECUTION_FAILURE" if "SPARQL" in error.upper() else "OTHER"
            elif not relation_chain:
                label = "EARLY_STOP"
            else:
                label = "WRONG_RELATION"
            labels[label] += 1
            sections.extend(
                [
                    f"## {subset} — {row['id']}",
                    f"- Question: {row['question']}",
                    f"- Relation chain: `{relation_chain}`",
                    f"- Prediction: `{row['prediction']}`",
                    f"- Gold answer: `{row['gold_answers']}`",
                    f"- Failure label: `{label}`",
                    f"- Explanation: baseline prediction did not match the gold answer under the shared evaluator.",
                    "",
                ]
            )
    sections[1:1] = ["## Label counts", ""] + [f"- `{key}`: {value}" for key, value in sorted(labels.items())] + [""]
    output_path.write_text("\n".join(sections), encoding="utf-8")


def write_summary(summaries: dict[str, dict[str, Any]], output_path: Path) -> None:
    lines = [
        "# CoR baseline summary",
        "",
        "This is the unchanged CoR baseline. No gold structural metadata was passed to the agent.",
        "",
        "| Subset | n | Hits@1 | F1 | Error rate | Avg LLM calls | Avg KG queries | Avg tokens | Avg latency (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for subset, summary in summaries.items():
        lines.append(
            "| {subset} | {n} | {hit:.2%} | {f1:.4f} | {error:.2%} | {calls:.2f} | {kg:.2f} | {tokens:.1f} | {latency:.1f} |".format(
                subset=subset,
                n=summary["n"],
                hit=summary["hits_at_1"],
                f1=summary["f1"],
                error=summary["error_rate"],
                calls=summary["average_llm_calls"],
                kg=summary["average_kg_queries"],
                tokens=summary["average_tokens"],
                latency=summary["average_latency_ms"],
            )
        )
    lines.extend(["", "No comparative claims are made until the branch-aware prototype is run on the identical IDs.", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_config(args: argparse.Namespace, selected_ids: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "experiment": "cor_baseline",
        "branch": "experiment/cor-baseline",
        "method": "cor",
        "model": os.getenv("MODEL_NAME", ""),
        "openai_base_url": os.getenv("OPENAI_BASE_URL", ""),
        "freebase_sparql_endpoint": os.getenv("FREEBASE_SPARQL_ENDPOINT", "http://127.0.0.1:8890/sparql"),
        "sparql_timeout": os.getenv("SPARQL_TIMEOUT", "30"),
        "temperature_exploration": args.temperature_exploration,
        "temperature_reasoning": args.temperature_reasoning,
        "max_tokens": args.max_token,
        "max_reasoning_depth": args.depth,
        "relation_width": args.relation_width,
        "sample_relation_threshold": args.sample_relation_threshold,
        "entity_width": "not used by CoR",
        "sample_entity_threshold": "not used by CoR",
        "remove_unnecessary_rel": args.remove_unnecessary_rel,
        "prompt_configuration": "chain_of_relations/methods/cor/prompt.yml",
        "retry_settings": {
            "max_retries": int(os.getenv("OPENAI_MAX_RETRIES", "3")),
            "retry_interval": float(os.getenv("OPENAI_RETRY_INTERVAL", "1")),
            "timeout": float(os.getenv("OPENAI_TIMEOUT", "30")),
        },
        "random_seed": args.seed,
        "subset_files": {name: str(spec["path"]) for name, spec in SUBSETS.items()},
        "selected_ids": selected_ids,
        "sampling": {
            "cwq_linear": 50,
            "cwq_branch_2": 50,
            "cwq_branch_3_plus": "all available",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--relation-width", type=int, default=3)
    parser.add_argument("--sample-relation-threshold", type=int, default=1000)
    parser.add_argument("--temperature-exploration", type=float, default=0.3)
    parser.add_argument("--temperature-reasoning", type=float, default=0.1)
    parser.add_argument("--max-token", type=int, default=512)
    parser.add_argument("--remove-unnecessary-rel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_safe_project_env()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if not os.getenv("MODEL_NAME"):
        raise SystemExit("MODEL_NAME is required")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    selected_rows: dict[str, list[dict[str, Any]]] = {}
    selected_ids: dict[str, list[str]] = {}
    for subset_name, spec in SUBSETS.items():
        path = Path(spec["path"])
        if not path.exists():
            raise SystemExit(f"missing inference-safe subset: {path}")
        rows = select_rows(read_jsonl(path), spec["sample_size"], args.seed)
        selected_rows[subset_name] = rows
        selected_ids[subset_name] = [str(row.get("id", "")) for row in rows]

    (output_root / "baseline_config.json").write_text(
        json.dumps(build_config(args, selected_ids), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "sample_ids.json").write_text(
        json.dumps(selected_ids, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logging.info("Initializing Freebase backend and unchanged CoR agent")
    agent = CoRAgent(
        model_name=os.environ["MODEL_NAME"],
        relation_width=args.relation_width,
        depth=args.depth,
        sample_relation_threshold=args.sample_relation_threshold,
        remove_unnecessary_rel=args.remove_unnecessary_rel,
        temperature_exploration=args.temperature_exploration,
        temperature_reasoning=args.temperature_reasoning,
        max_token=args.max_token,
        backend=get_backend("freebase"),
    )

    all_results: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for subset_name, rows in selected_rows.items():
        results_path = output_root / f"{subset_name}_results.jsonl"
        detail_dir = output_root / "details" / subset_name
        if results_path.exists():
            results_path.unlink()
        results: list[dict[str, Any]] = []
        logging.info("Running %s: %d examples", subset_name, len(rows))
        for index, row in enumerate(rows, start=1):
            logging.info("[%s] %d/%d id=%s", subset_name, index, len(rows), row.get("id"))
            result = run_one(agent, row, detail_dir)
            results.append(result)
            with results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        all_results[subset_name] = results
        summaries[subset_name] = summarize(results)

    (output_root / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(summaries, output_root / "summary.md")
    write_failure_analysis(all_results, output_root / "failure_analysis.md")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
