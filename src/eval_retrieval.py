from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import mlflow
from dotenv import load_dotenv

from src.rag import RAGPipeline

load_dotenv()

DATA_PATH = Path(__file__).parent.parent / "data" / "eval_dataset.json"
K = 5
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///" + str(Path(__file__).parent / "mlflow.db"))
EXPERIMENT = "matsuri-retrieval-eval"

# hybrid retrieved_ids are pre-computed in the eval dataset
# dense and lexical are re-run here for a fair comparison
MODES = ["hybrid", "dense", "lexical"]


def hit_rate(items: list[dict], k: int = K) -> float:
    hits = sum(1 for item in items if item["matsuri_id"] in item["retrieved_ids"][:k])
    return hits / len(items)


def mrr(items: list[dict], k: int = K) -> float:
    rrs: list[float] = []
    for item in items:
        ids = item["retrieved_ids"][:k]
        try:
            rrs.append(1.0 / (ids.index(item["matsuri_id"]) + 1))
        except ValueError:
            rrs.append(0.0)
    return sum(rrs) / len(rrs)


def re_run_search(mode: str, eval_items: list[dict]) -> list[dict]:
    rag = RAGPipeline(search_mode=mode)
    rag.retrieve("warm up")

    enriched = [dict(item) for item in eval_items]
    total = len(enriched)
    done = 0

    def search_one(idx: int, q: str) -> tuple[int, list[str]]:
        return idx, [r.id for r in rag.retrieve(q)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(search_one, i, item["q_star"]): i for i, item in enumerate(enriched)}
        for future in as_completed(futures):
            idx, ids = future.result()
            enriched[idx]["retrieved_ids"] = ids
            done += 1
            print(f"\r  {mode}: {done}/{total}", end="", flush=True)
    print()
    return enriched


def log_run(mode: str, items: list[dict]) -> dict:
    hr = hit_rate(items)
    mr = mrr(items)
    with mlflow.start_run(run_name=mode):
        mlflow.log_param("approach", mode)
        mlflow.log_param("k", K)
        mlflow.log_param("n_queries", len(items))
        mlflow.log_metric("hit_rate_at_5", hr)
        mlflow.log_metric("mrr_at_5", mr)
    return {"approach": mode, "hit_rate_at_5": hr, "mrr_at_5": mr}


def main() -> None:
    eval_items = json.loads(DATA_PATH.read_text())
    print(f"Loaded {len(eval_items)} eval items\n")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    results = []
    for mode in MODES:
        print(f"Evaluating {mode}...")
        items = eval_items if mode == "hybrid" else re_run_search(mode, eval_items)
        results.append(log_run(mode, items))

    print(f"\n{'Approach':<12} {'Hit Rate@5':>12} {'MRR@5':>10}")
    print("-" * 36)
    for r in results:
        print(f"{r['approach']:<12} {r['hit_rate_at_5']:>12.4f} {r['mrr_at_5']:>10.4f}")

    best = max(results, key=lambda r: r["mrr_at_5"])
    print(f"\nWinner: {best['approach']}  (MRR@5 = {best['mrr_at_5']:.4f})")


if __name__ == "__main__":
    main()
