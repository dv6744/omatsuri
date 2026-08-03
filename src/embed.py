import json
import os

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

load_dotenv()

COLLECTION = "matsuris"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "matsuris.json")


def build_text(record: dict) -> str:
    return f"{record['name']} {record['City']} {record['Prefecture']} {record['Date']} {record['Description']}"


def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data["matsuris"]
    print(f"Loaded {len(records)} records")

    texts = [build_text(r) for r in records]

    print("Generating dense embeddings...")
    dense_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    # nomic requires a task prefix when encoding documents for indexing
    prefixed_texts = ["search_document: " + t for t in texts]
    dense_vectors = dense_model.encode(prefixed_texts, show_progress_bar=True, batch_size=32)
    print(f"Dense embeddings shape: {dense_vectors.shape}")

    print("Generating BM25 sparse embeddings...")
    bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")
    sparse_embeddings = list(bm25.embed(texts))
    print(f"Sparse embeddings generated: {len(sparse_embeddings)}")

    client = QdrantClient(url=QDRANT_URL)

    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config={
            "text-dense": models.VectorParams(size=768, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "text-sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )
    print(f"Collection '{COLLECTION}' recreated")

    points = []
    for i, record in enumerate(records):
        sparse = sparse_embeddings[i]
        point = models.PointStruct(
            id=i,
            vector={
                "text-dense": dense_vectors[i].tolist(),
                "text-sparse": models.SparseVector(
                    indices=sparse.indices.tolist(),
                    values=sparse.values.tolist(),
                ),
            },
            payload=record,
        )
        points.append(point)

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Uploaded {len(points)} points to collection '{COLLECTION}'")


if __name__ == "__main__":
    main()
