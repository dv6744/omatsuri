import os
import time
import uuid
from contextlib import asynccontextmanager

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.rag import RAGPipeline

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://omatsuri:omatsuri@localhost:5432/omatsuri",
)

_rag: RAGPipeline | None = None


def get_db():
    return psycopg2.connect(DATABASE_URL)


def init_db() -> None:
    conn = get_db()
    with conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id          SERIAL PRIMARY KEY,
                session_id  TEXT,
                query       TEXT,
                search_mode TEXT,
                answer_len  INTEGER,
                n_retrieved INTEGER,
                latency_ms  FLOAT,
                created_at  TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id          SERIAL PRIMARY KEY,
                session_id  TEXT,
                query       TEXT,
                answer      TEXT,
                search_mode TEXT,
                rating      INTEGER,
                created_at  TIMESTAMPTZ DEFAULT NOW()
            )
        """)
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag
    try:
        init_db()
    except Exception as e:
        print(f"[warn] DB not available at startup ({e}); feedback logging disabled until Postgres is up")
    _rag = RAGPipeline(search_mode="hybrid")
    _rag.retrieve("warm up")
    yield


app = FastAPI(title="Omatsuri API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str
    search_mode: str = "hybrid"
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    search_mode: str
    retrieved: list[dict]
    latency_ms: float


class FeedbackRequest(BaseModel):
    session_id: str
    query: str
    answer: str
    search_mode: str
    rating: int  # 1 = thumbs up, -1 = thumbs down


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    if _rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not ready")

    if _rag.search_mode != req.search_mode:
        pipeline = RAGPipeline(search_mode=req.search_mode)
    else:
        pipeline = _rag

    t0 = time.perf_counter()
    result = pipeline.answer(req.query)
    latency_ms = (time.perf_counter() - t0) * 1000

    try:
        conn = get_db()
        init_db()  # no-op if tables exist; catches first-use when Postgres starts late
        with conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO queries (session_id, query, search_mode, answer_len, n_retrieved, latency_ms) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    session_id,
                    req.query,
                    req.search_mode,
                    len(result["answer"]),
                    len(result["retrieved"]),
                    latency_ms,
                ),
            )
        conn.close()
    except Exception:
        pass  # don't let DB errors break chat

    return ChatResponse(
        session_id=session_id,
        query=req.query,
        answer=result["answer"],
        search_mode=req.search_mode,
        retrieved=result["retrieved"],
        latency_ms=round(latency_ms, 1),
    )


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if req.rating not in (1, -1):
        raise HTTPException(status_code=422, detail="rating must be 1 or -1")

    try:
        init_db()
        conn = get_db()
        with conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (session_id, query, answer, search_mode, rating) "
                "VALUES (%s, %s, %s, %s, %s)",
                (req.session_id, req.query, req.answer, req.search_mode, req.rating),
            )
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {e}")
    return {"status": "recorded"}
