import hashlib
import math
import os
import re
import time
import uuid
from litellm import embedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

EMBEDDING_MODEL = "gemini/gemini-embedding-001"
QDRANT_VECTOR_SIZE = 3072


def _generate_fallback_embedding(text: str, dim: int = QDRANT_VECTOR_SIZE) -> list[float]:
    """Generates a deterministic, normalized pseudo-semantic vector when remote embedding quotas are exhausted."""
    vec = [0.0] * dim
    words = re.findall(r"\w+", text.lower())
    for w in words:
        h = int(hashlib.sha256(w.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0
    # L2 normalize
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    else:
        vec[0] = 1.0
    return vec


class VectorStoreService:
    def __init__(self):
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.collection_name = "compliance_documents"
        client_connected = False
        for target in [qdrant_host, "localhost", ":memory:"]:
            try:
                if target == ":memory:":
                    self.client = QdrantClient(":memory:")
                else:
                    self.client = QdrantClient(host=target, port=6333, timeout=2.0)
                self._ensure_collection()
                client_connected = True
                break
            except Exception:
                continue
        if not client_connected:
            self.client = QdrantClient(":memory:")
            self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if exists:
            try:
                info = self.client.get_collection(self.collection_name)
                current_size = getattr(info.config.params.vectors, "size", None)
                if current_size != QDRANT_VECTOR_SIZE:
                    self.client.delete_collection(self.collection_name)
                    exists = False
            except Exception:
                pass
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=QDRANT_VECTOR_SIZE, distance=Distance.COSINE),
            )

    def _embed_with_retry(self, texts: list[str], api_key: str, max_retries: int = 3) -> list[list[float]]:
        if not api_key:
            return [_generate_fallback_embedding(t) for t in texts]

        for attempt in range(max_retries):
            try:
                response = embedding(model=EMBEDDING_MODEL, api_key=api_key, input=texts, timeout=10)
                return [record["embedding"] for record in response["data"]]
            except Exception as exc:
                err_str = str(exc)
                is_rate_limit = (
                    "429" in err_str
                    or "RESOURCE_EXHAUSTED" in err_str
                    or "RateLimit" in type(exc).__name__
                    or "quota" in err_str.lower()
                )
                if is_rate_limit and attempt < max_retries - 1:
                    delay = 4.0 * (attempt + 1)
                    time.sleep(delay)
                    continue
                # If remote quota exhausted on Vertex AI, fallback gracefully
                if is_rate_limit:
                    return [_generate_fallback_embedding(t) for t in texts]
                raise

        return [_generate_fallback_embedding(t) for t in texts]

    def index_chunks(self, document_id: str, chunks: list[dict]):
        if not chunks:
            return
        texts = [item["child_text"] for item in chunks]
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        api_key = api_key.strip().strip(chr(34)).strip(chr(39))

        # Use conservative batch size with pacing to respect Vertex AI quotas
        batch_size = 5
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = self._embed_with_retry(batch_texts, api_key)
            embeddings.extend(batch_embeddings)
            if i + batch_size < len(texts):
                time.sleep(0.3)

        points = []
        for idx, item in enumerate(chunks):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embeddings[idx],
                    payload={
                        "document_id": document_id,
                        "child_text": item["child_text"],
                        "parent_text": item["parent_text"],
                        "page_number": item["page"],
                    },
                )
            )

        upsert_batch_size = 100
        for i in range(0, len(points), upsert_batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i : i + upsert_batch_size],
            )

    def query_relevant_context(
        self,
        query_text: str,
        document_id: str | None = None,
        limit: int = 3,
        *,
        top_k: int | None = None,
    ) -> str:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        api_key = api_key.strip().strip(chr(34)).strip(chr(39))
        query_vectors = self._embed_with_retry([query_text], api_key)
        query_vector = query_vectors[0]

        search_params = {
            "collection_name": self.collection_name,
            "query_vector": query_vector,
            "limit": top_k if top_k is not None else limit,
        }
        if document_id is not None:
            search_params["query_filter"] = {
                "must": [
                    {"key": "document_id", "match": {"value": document_id}}
                ]
            }
        search_results = self.client.search(**search_params)

        seen_contexts = set()
        context_blocks = []
        separator = chr(10) * 2 + "---" + chr(10) * 2

        for hit in search_results:
            child_text = hit.payload.get("child_text", "")
            parent_text = hit.payload.get("parent_text", "")
            context_text = parent_text or child_text
            if context_text and context_text not in seen_contexts:
                seen_contexts.add(context_text)
                page = hit.payload.get("page_number")
                context_blocks.append(f"[Page {page} | Retrieved contract context]: {context_text}")

        return separator.join(context_blocks)
