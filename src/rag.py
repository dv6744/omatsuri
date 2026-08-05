import os
from typing import List

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding, TextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

from src.models import MatsuriRecord

load_dotenv()

MODEL = "gemini-2.5-flash"
TOP_K = 5
COLLECTION = "matsuris"


class RAGPipeline:
    def __init__(
        self,
        model: str = MODEL,
        top_k: int = TOP_K,
        search_mode: str = "hybrid",
        qdrant_url: str | None = None,
    ):
        self.model = model
        self.top_k = top_k
        self.search_mode = search_mode

        self._llm = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            max_retries=5,
        )
        self._qdrant = QdrantClient(url=qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333"))

        # loaded once, reused across all calls
        self._dense = TextEmbedding("nomic-ai/nomic-embed-text-v1.5")
        self._sparse = SparseTextEmbedding(model_name="Qdrant/bm25")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _embed_query(self, query: str) -> tuple[list[float], SparseVector]:
        dense = next(self._dense.embed([f"search_query: {query}"])).tolist()
        sp = next(self._sparse.embed([query]))
        sparse = SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist())
        return dense, sparse

    def hybrid_search(self, query: str) -> List[MatsuriRecord]:
        dense, sparse = self._embed_query(query)
        results = self._qdrant.query_points(
            collection_name=COLLECTION,
            prefetch=[
                Prefetch(query=dense, using="text-dense", limit=self.top_k * 2),
                Prefetch(query=sparse, using="text-sparse", limit=self.top_k * 2),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=self.top_k,
            with_payload=True,
        )
        return self._to_records(results)

    def dense_search(self, query: str) -> List[MatsuriRecord]:
        dense, _ = self._embed_query(query)
        results = self._qdrant.query_points(
            collection_name=COLLECTION,
            query=dense,
            using="text-dense",
            limit=self.top_k,
            with_payload=True,
        )
        return self._to_records(results)

    def lexical_search(self, query: str) -> List[MatsuriRecord]:
        _, sparse = self._embed_query(query)
        results = self._qdrant.query_points(
            collection_name=COLLECTION,
            query=sparse,
            using="text-sparse",
            limit=self.top_k,
            with_payload=True,
        )
        return self._to_records(results)

    def retrieve(self, query: str) -> List[MatsuriRecord]:
        if self.search_mode == "hybrid":
            return self.hybrid_search(query)
        if self.search_mode == "lexical":
            return self.lexical_search(query)
        return self.dense_search(query)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _build_context(self, records: List[MatsuriRecord]) -> str:
        sections = [f"[{i + 1}] {self._record_to_text(r)}" for i, r in enumerate(records)]
        return "\n\n".join(sections)

    def _build_prompt(self, query: str, context: str) -> str:
        return (
            "You are a knowledgeable assistant about Japanese matsuri (festivals).\n"
            "Use ONLY the context below to answer the question. "
            "If the answer is not in the context, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

    def answer(self, query: str) -> dict:
        retrieved = self.retrieve(query)
        context = self._build_context(retrieved)
        prompt = self._build_prompt(query, context)

        response = self._llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        return {
            "query": query,
            "answer": response.choices[0].message.content,
            "retrieved": [r.model_dump() for r in retrieved],
            "search_mode": self.search_mode,
            "usage": response.usage,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _record_to_text(record: MatsuriRecord) -> str:
        return (
            f"Name: {record.name}\n"
            f"City: {record.City}, Prefecture: {record.Prefecture}\n"
            f"Date: {record.Date}\n"
            f"Description: {record.Description}\n"
            f"Annual Turnout: {record.AnnualTurnOut}\n"
            f"Relevance: {record.RelevanceRating}"
        )

    @staticmethod
    def _to_records(results) -> List[MatsuriRecord]:
        return [MatsuriRecord(**p.payload) for p in results.points]


if __name__ == "__main__":
    rag = RAGPipeline(search_mode="hybrid")
    query = "What are famous summer matsuri in Tohoku?"
    result = rag.answer(query)
    print(result["answer"])
