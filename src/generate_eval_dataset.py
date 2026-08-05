"""
Generate evaluation dataset from matsuris.json.

Flow per record:
  MatsuriRecord (A) -> Gemini generates 3 Q* questions -> RAG answers each Q* (A`)

Outputs data/eval_dataset.json — pre-computed so reviewers don't re-run LLM calls.

Usage:
  uv run python -m src.generate_eval_dataset              # all 100 records
  uv run python -m src.generate_eval_dataset --limit 10  # smoke test
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from src.models import MatsuriRecord
from src.rag import RAGPipeline

load_dotenv()

DATA_PATH = Path(__file__).parent.parent / "data" / "matsuris.json"
OUT_PATH = Path(__file__).parent.parent / "data" / "eval_dataset.json"

QUESTIONS_PER_RECORD = 3
GEN_MODEL = "gemini-3.1-flash-lite"


def build_gemini_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        max_retries=3,
    )


class Questions(BaseModel):
    questions: list[str]


def generate_questions(client: OpenAI, record: MatsuriRecord) -> list[str]:
    context = (
        f"Name: {record.name}\n"
        f"City: {record.City}, Prefecture: {record.Prefecture}\n"
        f"Date: {record.Date}\n"
        f"Description: {record.Description}\n"
        f"Annual Turnout: {record.AnnualTurnOut}\n"
        f"Relevance: {record.RelevanceRating}"
    )
    prompt = (
        f"You are building a retrieval evaluation dataset for a Japanese matsuri chatbot.\n\n"
        f"Given the following matsuri record, generate exactly {QUESTIONS_PER_RECORD} distinct, "
        f"natural questions a user might ask. Each must be answerable from this record alone. "
        f"Answer in English. Vary the angle: location, timing, traditions, crowd size, cultural significance, etc.\n\n"
        f"Matsuri record:\n{context}"
    )

    response = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "Questions",
                "schema": Questions.model_json_schema(),
            },
        },
    )
    content = response.choices[0].message.content or "{}"
    parsed = Questions(**json.loads(content))
    return parsed.questions[:QUESTIONS_PER_RECORD]


def process_record(
    record: MatsuriRecord,
    index: int,
    total: int,
    gemini: OpenAI,
    rag: RAGPipeline,
    print_lock: Lock,
) -> list[dict]:
    with print_lock:
        print(f"[{index}/{total}] {record.id}")

    try:
        questions = generate_questions(gemini, record)
    except Exception as e:
        with print_lock:
            print(f"  [{record.id}] Q* generation failed: {e}")
        return []

    items = []
    for j, q_star in enumerate(questions):
        try:
            result = rag.answer(q_star)
            retrieved_ids = [r["id"] for r in result["retrieved"]]
            a_prime = result["answer"]
        except Exception as e:
            with print_lock:
                print(f"  [{record.id}] RAG failed for q{j}: {e}")
            retrieved_ids = []
            a_prime = ""

        items.append({
            "id": f"{record.id}-q{j}",
            "matsuri_id": record.id,
            "q_star": q_star,
            "a_prime": a_prime,
            "retrieved_ids": retrieved_ids,
            "ground_truth": record.model_dump(),
        })
        with print_lock:
            print(f"  q{j}: {q_star[:70]}...")
            print(f"       retrieved: {retrieved_ids}")

    return items


def main(limit: int | None = None, workers: int = 5) -> None:
    records_raw: list[dict] = json.loads(DATA_PATH.read_text())
    if limit:
        records_raw = records_raw[:limit]
    records = [MatsuriRecord(**r) for r in records_raw]
    total = len(records)
    print(f"Loaded {total} records — generating {total * QUESTIONS_PER_RECORD} eval items ({workers} workers)")

    gemini = build_gemini_client()
    rag = RAGPipeline(search_mode="hybrid")

    # pre-warm ONNX models before threads start to avoid a race on lazy init
    print("Pre-warming embedding models...")
    rag.retrieve("warm up")

    print_lock = Lock()
    eval_items: list[dict] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_record, record, i + 1, total, gemini, rag, print_lock): record
            for i, record in enumerate(records)
        }
        for future in as_completed(futures):
            eval_items.extend(future.result())

    # restore original record order
    id_order = {r.id: i for i, r in enumerate(records)}
    eval_items.sort(key=lambda x: (id_order.get(x["matsuri_id"], 0), x["id"]))

    OUT_PATH.write_text(json.dumps(eval_items, ensure_ascii=False, indent=2))
    print(f"\nSaved {len(eval_items)} eval items -> {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Cap records processed (for testing)")
    parser.add_argument("--workers", type=int, default=5, help="Concurrent threads (default: 5)")
    args = parser.parse_args()
    main(limit=args.limit, workers=args.workers)
