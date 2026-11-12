from pypdf import PdfReader
import io

class DocumentParserService:
    @staticmethod
    def extract_pdf_text(file_bytes: bytes) -> str:
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()

    @staticmethod
    def extract_and_chunk_pdf(file_bytes: bytes, child_size: int = 400, parent_size: int = 2000) -> list[dict]:
        reader = PdfReader(io.BytesIO(file_bytes))
        processed_chunks = []

        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if not page_text.strip():
                continue
                
            for i in range(0, len(page_text), parent_size - 400):
                parent_chunk = page_text[i : i + parent_size]
                for j in range(0, len(parent_chunk), child_size):
                    child_chunk = parent_chunk[j : j + child_size]
                    if len(child_chunk.strip()) > 50:
                        processed_chunks.append({
                            "child_text": child_chunk.strip(),
                            "parent_text": parent_chunk.strip(),
                            "page": page_idx + 1
                        })
        return processed_chunks
