import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import mlflow
import pandas as pd
from anthropic import AnthropicVertex
from dotenv import load_dotenv

from src.rag import RAGPipeline

load_dotenv()

JUDGE_MODEL = "claude-haiku-4-5@20251001"
VERTEX_PROJECT = os.getenv("VERTEX_PROJECT")
VERTEX_REGION = os.getenv("VERTEX_LOCATION", "global")

DATA_PATH = Path(__file__).parent.parent / "data" / "eval_dataset.json"
_default_mlflow_uri = "sqlite:///" + str(Path(__file__).parent / "mlflow.db")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", _default_mlflow_uri)
EXPERIMENT = "matsuri-rag-eval"

PROMPT_V2 = (
    "You are an expert on Japanese matsuri (festivals).\n\n"
    "<context>\n{context}\n</context>\n\n"
    "<question>\n{query}\n</question>\n\n"
    "Using ONLY the information in the context above, provide a concise and accurate answer. "
    "Structure your response as:\n"
    "- Direct answer (1-2 sentences)\n"
    "- Supporting detail from the context\n"
    "If the context does not fully answer the question, state what is missing.\n\n"
    "<answer>"
)

RELEVANCE_PROMPT = """\
You are an evaluation judge for a Japanese matsuri chatbot.

Question: {question}
Answer: {answer}

Score the answer on RELEVANCE — does it directly address the question?
  5 - fully relevant: directly and accurately answers the question
  3 - partially relevant: addresses the question but misses key aspects or adds noise
  1 - non relevant: does not answer the question or is off-topic

Respond with valid JSON only:
{{"score": <1, 3, or 5>, "reasoning": "<one sentence>"}}""".strip()

FAITHFULNESS_PROMPT = """\
You are an evaluation judge for a Japanese matsuri chatbot.

Question: {question}
Ground Truth: {ground_truth}
Answer: {answer}

Score the answer on FAITHFULNESS — is it grounded in and consistent with the ground truth?
  5 - fully faithful: all claims are supported by the ground truth
  3 - partially faithful: mostly accurate but contains unsupported or embellished claims
  1 - non faithful: contradicts the ground truth or introduces hallucinated facts

Respond with valid JSON only:
{{"score": <1, 3, or 5>, "reasoning": "<one sentence>"}}""".strip()


class RAGPipelineV2(RAGPipeline):
    def _build_prompt(self, query: str, context: str) -> str:
        return PROMPT_V2.format(context=context, query=query)


def get_judge_client() -> AnthropicVertex:
    if not VERTEX_PROJECT:
        raise RuntimeError("VERTEX_PROJECT env var is required for the judge.")
    return AnthropicVertex(region=VERTEX_REGION, project_id=VERTEX_PROJECT)


def _call_judge(prompt: str) -> dict:
    client = get_judge_client()
    message = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = (message.content[0].text if message.content else "").strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    parsed = json.loads(raw)
    score = int(parsed.get("score", 1))
    if score not in (1, 3, 5):
        score = min([1, 3, 5], key=lambda x: abs(x - score))
    return {"score": score, "reasoning": parsed.get("reasoning", "")}


def judge_item(item: dict) -> dict:
    question = item["q_star"]
    answer = item["answer"]
    ground_truth = item["ground_truth_text"]

    try:
        rel = _call_judge(RELEVANCE_PROMPT.format(question=question, answer=answer))
    except Exception as exc:
        rel = {"score": 1, "reasoning": f"judge_failed: {exc}"}

    try:
        faith = _call_judge(FAITHFULNESS_PROMPT.format(
            question=question, answer=answer, ground_truth=ground_truth,
        ))
    except Exception as exc:
        faith = {"score": 1, "reasoning": f"judge_failed: {exc}"}

    return {
        "id": item["id"],
        "matsuri_id": item["matsuri_id"],
        "q_star": question,
        "answer": answer,
        "relevance_score": rel["score"],
        "relevance_reasoning": rel["reasoning"],
        "faithfulness_score": faith["score"],
        "faithfulness_reasoning": faith["reasoning"],
    }


def run_variant(
    run_name: str,
    variant: str,
    records: list[dict],
    workers: int,
) -> None:
    print(f"\n── {run_name} │ {len(records)} items ──")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "variant": variant,
            "judge_model": JUDGE_MODEL,
            "n_eval_items": len(records),
        })

        results: list[dict] = []
        done = 0

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(judge_item, rec): rec["id"] for rec in records}
            for future in as_completed(futures):
                results.append(future.result())
                done += 1
                print(f"\r  judging: {done}/{len(records)}", end="", flush=True)
        print()

        rel_scores = [r["relevance_score"] for r in results]
        faith_scores = [r["faithfulness_score"] for r in results]
        n = len(results)

        mlflow.log_metric("avg_relevance", sum(rel_scores) / n)
        mlflow.log_metric("avg_faithfulness", sum(faith_scores) / n)

        for label, scores in [("relevance", rel_scores), ("faithfulness", faith_scores)]:
            for bucket in (1, 3, 5):
                pct = sum(1 for s in scores if s == bucket) / n * 100
                mlflow.log_metric(f"{label}_pct{bucket}", pct)

        slug = run_name.replace("-", "_")
        jsonl_path = Path(__file__).parent.parent / "data" / f"eval_rag_{slug}.jsonl"
        jsonl_path.write_text("\n".join(json.dumps(r) for r in results))
        mlflow.log_artifact(str(jsonl_path))

        df = pd.DataFrame(results)
        dataset = mlflow.data.from_pandas(df, name=run_name, targets="relevance_score")
        mlflow.log_input(dataset, context="evaluation")

        print(
            f"  avg_relevance={sum(rel_scores)/n:.3f}  "
            f"avg_faithfulness={sum(faith_scores)/n:.3f}"
        )


def generate_v2_answers(records: list[dict], workers: int) -> dict[str, str]:
    rag = RAGPipelineV2()
    print("Pre-warming models for v2 generation...")
    rag.retrieve("warm up")

    answers: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(rag.answer, rec["q_star"]): rec["q_star"] for rec in records}
        done = 0
        for future in as_completed(futures):
            question = futures[future]
            try:
                answers[question] = future.result()["answer"]
            except Exception:
                answers[question] = ""
            done += 1
            print(f"\r  generating v2: {done}/{len(records)}", end="", flush=True)
    print()
    return answers


def build_ground_truth_text(record: dict) -> str:
    gt = record.get("ground_truth", {})
    if isinstance(gt, dict):
        return "\n".join(f"{k}: {v}" for k, v in gt.items())
    return str(gt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    # synchronous trace writes to prevent SQLite QueuePool overflow
    os.environ["MLFLOW_ENABLE_ASYNC_LOGGING"] = "false"

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    raw: list[dict] = json.loads(DATA_PATH.read_text())
    if args.limit:
        raw = raw[: args.limit]
    print(f"Loaded {len(raw)} eval items")

    ground_truth_texts = {rec["id"]: build_ground_truth_text(rec) for rec in raw}

    # v1: use pre-computed a_prime from eval dataset
    v1_records = [
        {**rec, "answer": rec["a_prime"], "ground_truth_text": ground_truth_texts[rec["id"]]}
        for rec in raw
    ]

    # v2: generate answers with structured prompt
    v2_answers = generate_v2_answers(raw, workers=args.workers)
    v2_records = [
        {**rec, "answer": v2_answers.get(rec["q_star"], ""), "ground_truth_text": ground_truth_texts[rec["id"]]}
        for rec in raw
    ]

    run_variant("v1_default", "v1", v1_records, workers=args.workers)
    run_variant("v2_structured", "v2", v2_records, workers=args.workers)


if __name__ == "__main__":
    main()
