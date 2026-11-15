import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from litellm import embedding

class VectorStoreService:
    def __init__(self):
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.client = QdrantClient(host=qdrant_host, port=6333)
        self.collection_name = "compliance_documents"
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

    def _get_embeddings(self, texts: list[str]):
        for emb_model in ["gemini/gemini-embedding-001", "gemini/text-embedding-004"]:
            try:
                res = embedding(model=emb_model, input=texts)
                return [record["embedding"] for record in res["data"]]
            except Exception:
                continue
        return None

    def index_chunks(self, document_id: str, chunks: list[dict]):
        texts = [item["child_text"] for item in chunks]
        embeddings = self._get_embeddings(texts)
        if not embeddings:
            return
        points = []
        for idx, item in enumerate(chunks):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embeddings[idx][:768],
                    payload={
                        "document_id": document_id,
                        "child_text": item["child_text"],
                        "parent_text": item["parent_text"],
                        "page_number": item["page"]
                    }
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points)

    def query_relevant_context(self, query_text: str, document_id: str, limit: int = 3) -> str:
        embeddings = self._get_embeddings([query_text])
        if not embeddings:
            return ""
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embeddings[0][:768],
            query_filter={"must": [{"key": "document_id", "match": {"value": document_id}}]},
            limit=limit
        )
        seen_parents = set()
        context_blocks = []
        for hit in search_results:
            p_text = hit.payload.get("parent_text", "")
            if p_text and p_text not in seen_parents:
                seen_parents.add(p_text)
                context_blocks.append(f"[Page {hit.payload.get('page_number')}]: {p_text}")
        return "\n\n---\n\n".join(context_blocks)
