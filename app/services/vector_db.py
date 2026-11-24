import hashlib
import math
import os
import re
import uuid

from litellm import embedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

QDRANT_VECTOR_SIZE = 768


def _generate_fallback_embedding(text: str, size: int = QDRANT_VECTOR_SIZE) -> list[float]:
    """Deterministic local embedding used when the embedding provider is unavailable."""
    vector = [0.0] * size
    for token in re.findall(r"[a-z0-9]+", (text or "").lower()):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        vector[int(digest, 16) % size] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


def _cosine(left: list[float], right: list[float]) -> float:
    length = min(len(left), len(right))
    if not length:
        return 0.0
    return sum(left[index] * right[index] for index in range(length))


class VectorStoreService:
    def __init__(self):
        self.collection_name = "compliance_documents"
        self._local_chunks: list[dict] = []
        self._local_mode = False
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        try:
            self.client = QdrantClient(host=qdrant_host, port=6333, timeout=3.0)
            self._ensure_collection()
        except Exception:
            self.client = None
            self._local_mode = True

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=QDRANT_VECTOR_SIZE, distance=Distance.COSINE
                ),
            )

    def _embed_with_retry(self, texts: list[str], api_key: str | None = None) -> list[list[float]]:
        for emb_model in ("gemini/gemini-embedding-001", "gemini/text-embedding-004"):
            try:
                kwargs = {"model": emb_model, "input": texts}
                if api_key:
                    kwargs["api_key"] = api_key
                response = embedding(**kwargs)
                return [record["embedding"] for record in response["data"]]
            except Exception:
                continue
        return [_generate_fallback_embedding(text) for text in texts]

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        from app.agents.nodes import get_gemini_api_key

        return self._embed_with_retry(texts, get_gemini_api_key())

    def index_chunks(self, document_id: str, chunks: list[dict]):
        valid_chunks = [
            item for item in (chunks or []) if str(item.get("child_text") or "").strip()
        ]
        if not valid_chunks:
            return

        texts = [str(item["child_text"]) for item in valid_chunks]
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), 50):
            embeddings.extend(self._get_embeddings(texts[start : start + 50]))
        if len(embeddings) < len(valid_chunks):
            embeddings.extend(
                _generate_fallback_embedding(text)
                for text in texts[len(embeddings) :]
            )

        points = []
        for idx, item in enumerate(valid_chunks):
            vector = list(embeddings[idx][:QDRANT_VECTOR_SIZE])
            if len(vector) < QDRANT_VECTOR_SIZE:
                vector.extend([0.0] * (QDRANT_VECTOR_SIZE - len(vector)))
            payload = {
                "document_id": document_id,
                "child_text": item["child_text"],
                "parent_text": item.get("parent_text", ""),
                "page_number": item.get("page", 1),
            }
            points.append(
                {"id": str(uuid.uuid4()), "vector": vector, "payload": payload}
            )

        if self.client is not None and not self._local_mode:
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        PointStruct(id=point["id"], vector=point["vector"], payload=point["payload"])
                        for point in points
                    ],
                )
                return
            except Exception:
                self._local_mode = True
                self.client = None

        self._local_chunks.extend(points)

    def _local_search(self, vector: list[float], document_id: str | None, limit: int) -> list[dict]:
        candidates = [
            point
            for point in getattr(self, "_local_chunks", [])
            if document_id is None
            or point["payload"].get("document_id") == document_id
        ]
        ranked = sorted(
            candidates,
            key=lambda point: _cosine(vector, point["vector"]),
            reverse=True,
        )
        return ranked[:limit]

    def query_relevant_context(
        self,
        query_text: str,
        document_id: str | None = None,
        limit: int = 3,
        top_k: int | None = None,
    ) -> str:
        limit = int(top_k or limit or 3)
        embeddings = self._embed_with_retry([query_text], None)
        vector = embeddings[0] if embeddings else _generate_fallback_embedding(query_text)
        vector = list(vector[:QDRANT_VECTOR_SIZE])
        if len(vector) < QDRANT_VECTOR_SIZE:
            vector.extend([0.0] * (QDRANT_VECTOR_SIZE - len(vector)))

        hits = []
        if self.client is not None and not getattr(self, "_local_mode", False):
            try:
                kwargs = {
                    "collection_name": self.collection_name,
                    "query_vector": vector,
                    "limit": limit,
                }
                if document_id:
                    kwargs["query_filter"] = {
                        "must": [
                            {"key": "document_id", "match": {"value": document_id}}
                        ]
                    }
                hits = self.client.search(**kwargs)
            except Exception:
                hits = []

        if not hits:
            hits = self._local_search(vector, document_id, limit)

        seen_parents = set()
        context_blocks = []
        for hit in hits:
            if isinstance(hit, dict):
                payload = hit.get("payload") or {}
            else:
                payload = getattr(hit, "payload", None) or {}
            parent_text = str(payload.get("parent_text") or "")
            if parent_text and parent_text not in seen_parents:
                seen_parents.add(parent_text)
                context_blocks.append(
                    f"[Page {payload.get('page_number')} | Retrieved contract context]: "
                    f"{parent_text}"
                )
        return "\n\n---\n\n".join(context_blocks)
