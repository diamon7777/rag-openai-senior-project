# %%
# %% [markdown]
# # 03 — RAG Pipeline
#
# Transforms extracted documents into chunks and embedding records, builds a persistent Chroma index with BGE-M3, retrieves and reranks relevant passages, and generates answers grounded in selected context and source URLs.

# %% [markdown]
# ## 1. Document Chunking

# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inline Chunking Script (no pipeline functions)
- Same behavior as the original: same heuristics, same IDs, same output fields
- Reads DOCS_JSONL -> writes CHUNKS_JSONL
"""

from __future__ import annotations
import os, sys, json, hashlib, re
from typing import Dict, Any, List

# ------------------------ Config ------------------------
DOCS_JSONL   = os.getenv("DOCS_JSONL",   "output/stage/documents.jsonl")
CHUNKS_JSONL = os.getenv("CHUNKS_JSONL", "output/stage/chunks.jsonl")

CHUNK_SIZE_STRUCT     = int(os.getenv("CHUNK_SIZE_STRUCT", "512"))     # tokens
CHUNK_OVERLAP_STRUCT  = float(os.getenv("CHUNK_OVERLAP_STRUCT", "0.12"))  # 12%

CHUNK_SIZE_OCR        = int(os.getenv("CHUNK_SIZE_OCR", "256"))        # tokens
CHUNK_OVERLAP_OCR     = float(os.getenv("CHUNK_OVERLAP_OCR", "0.20"))     # 20%

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# ------------------------ Tokenization utils ------------------------

import tiktoken  # type: ignore
_ENC = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))

def trim_to_tokens(text: str, max_tokens: int) -> str:
        ids = _ENC.encode(text)
        return text if len(ids) <= max_tokens else _ENC.decode(ids[:max_tokens])

TOKEN_MODE = "tiktoken"

# ------------------------ Structure heuristics ------------------------
MD_TABLE_HDR = re.compile(r"^\s*\|.*\|\s*$", re.M)
MD_TABLE_SEP = re.compile(r"^\s*\|\s*:?-{3,}.*\|\s*$", re.M)
HTML_TABLE   = re.compile(r"<\s*table[\s>].*?>", re.I | re.S)
MD_HEADER    = re.compile(r"^\s*#{1,6}\s+.+", re.M)

def looks_structured(text: str) -> bool:
    if not text:
        return False
    if MD_HEADER.search(text) or HTML_TABLE.search(text):
        return True
    if MD_TABLE_HDR.search(text) and MD_TABLE_SEP.search(text):
        return True
    nl = text.count("\n")
    return (nl >= 3 and (nl / max(1, len(text))) > 0.002)

# ------------------------ Minimal helpers for splitting ------------------------
def _hard_chunk(text: str, size: int, overlap: int) -> List[str]:
    if size <= 0:
        return [text]
    chunks: List[str] = []
        ids = _ENC.encode(text)  # type: ignore  # only if tiktoken available
        n = len(ids)
        step = size - overlap if size > overlap else size
        i = 0
        while i < n:
            end = min(n, i + size)
            chunk = _ENC.decode(ids[i:end])  # type: ignore
            chunks.append(chunk)
            if end >= n:
                break
            i += max(1, step)
    return chunks

def _chunk_once_at_sep(text: str, size: int, overlap: int, sep: str) -> List[str] | None:
    if sep == "":
        return _hard_chunk(text, size, overlap)
    if sep not in text:
        return None
    parts = text.split(sep)
    chunks: List[str] = []
    buf = ""
    for part in parts:
        piece = part if not buf else (sep + part)
        if count_tokens((buf + piece) if buf else part) <= size:
            buf = (buf + piece) if buf else part
            continue
        if buf:
            chunks.append(buf)
            if overlap > 0:
                approx_chars = overlap * (4 if TOKEN_MODE == "approx_chars" else 3)
                tail = buf[-int(approx_chars):]
                buf = tail + part
            else:
                buf = part
            if count_tokens(buf) > size:
                sub = _hard_chunk(buf, size, overlap)
                if sub:
                    chunks.extend(sub[:-1])
                    buf = sub[-1]
                else:
                    buf = ""
        else:
            sub = _hard_chunk(part, size, overlap)
            chunks.extend(sub)
            buf = ""
    if buf:
        chunks.append(buf)
    return chunks

# ------------------------ Main (inline pipeline) ------------------------
print(f"[INFO] Reading:  {DOCS_JSONL}")
print(f"[INFO] Writing:  {CHUNKS_JSONL}")
print(f"[INFO] Profiles: structured(size={CHUNK_SIZE_STRUCT}, overlap={CHUNK_OVERLAP_STRUCT}), "
      f"ocr_raw(size={CHUNK_SIZE_OCR}, overlap={CHUNK_OVERLAP_OCR})")

if not os.path.exists(DOCS_JSONL):
    raise FileNotFoundError(f"Input not found: {DOCS_JSONL}")

os.makedirs(os.path.dirname(CHUNKS_JSONL) or ".", exist_ok=True)

total_docs   = 0
total_chunks = 0
seen_ids: set[str] = set()

with open(DOCS_JSONL, "r", encoding="utf-8") as fin, \
     open(CHUNKS_JSONL, "w", encoding="utf-8") as fout:

    for raw in fin:
        line = raw.strip()
        if not line:
            continue
        try:
            rec: Dict[str, Any] = json.loads(line)
        except Exception:
            print(f"[WARN] Bad JSONL line skipped: {line[:200]}...", file=sys.stderr)
            continue

        total_docs += 1

        # ---------- Derive parent_id (same as original behavior)
        content_hash = rec.get("content_hash")
        if content_hash:
            parent_id = content_hash
        else:
            src_url = rec.get("source_url") or ""
            parent_id = hashlib.sha1(src_url.encode("utf-8", errors="ignore")).hexdigest()

        # ---------- Text field
        text = (rec.get("text_full") or rec.get("text") or "")
        text = text.strip()
        if not text:
            continue

        # ---------- Choose profile (same heuristics)
        st  = (rec.get("source_type") or "").lower()
        fmt = (rec.get("format") or "").lower()

        if st == "image":
            profile = "ocr_raw"
        elif st in {"pdf_md", "html", "text"}:
            if fmt == "markdown" and looks_structured(text):
                profile = "structured"
            else:
                profile = "structured" if looks_structured(text) else "ocr_raw"
        else:
            profile = "ocr_raw"

        if profile == "structured":
            size = CHUNK_SIZE_STRUCT
            ov   = int(round(size * CHUNK_OVERLAP_STRUCT))
        else:
            size = CHUNK_SIZE_OCR
            ov   = int(round(size * CHUNK_OVERLAP_OCR))

        # ---------- Chunking (iterative, same as original)
        if count_tokens(text) <= size:
            chunks = [text]
        else:
            stack: List[str] = [text]
            for sep in SEPARATORS:
                next_stack: List[str] = []
                for seg in stack:
                    if count_tokens(seg) <= size:
                        next_stack.append(seg)
                        continue
                    out = _chunk_once_at_sep(seg, size, ov, sep)
                    if out is None:
                        next_stack.append(seg)
                    else:
                        next_stack.extend(out)
                stack = next_stack

            # finalize: any leftover > size => hard chunk
            final: List[str] = []
            for seg in stack:
                if count_tokens(seg) <= size:
                    final.append(seg)
                else:
                    final.extend(_hard_chunk(seg, size, ov))
            chunks = final

        # ---------- Write out chunks
        total = len(chunks)
        for idx, ch in enumerate(chunks):
            cid = f"{parent_id}:{idx}"
            if cid in seen_ids:
                # preserve original behavior: skip duplicates silently
                continue
            seen_ids.add(cid)

            out_rec = {
                "id": cid,
                "parent_id": parent_id,
                "source_url": rec.get("source_url"),
                "source_type": rec.get("source_type"),
                "profile": profile,
                "chunk_index": idx,
                "chunk_total": total,
                "n_tokens": count_tokens(ch),
                "text": ch,
                "metadata": None,  # unchanged
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            total_chunks += 1

print(f"[DONE] Chunked {total_docs} documents into {total_chunks} chunks → {CHUNKS_JSONL}")

# %% [markdown]
# ## 2. Embeddable Record Formatting

# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 — Prepare embeddable chunks (normalize fields → id,text,metadata)

ENV (defaults):
  IN_JSONL=output/stage/chunks.jsonl
  OUT_JSONL=output/stage/embeddable.jsonl
  TEXT_KEY=           # ถ้าไม่ระบุ จะลอง text|content|page_content ตามลำดับ
  KEEP_META=1         # 1=เก็บ metadata, 0=ไม่เก็บ
  DEDUP=none          # none|sha1_text  (dedup ด้วย hash ของข้อความ)
  TOKEN_MODE=approx_chars  # approx_chars|tiktoken
  CHAR_PER_TOKEN=3.6
  MIN_TOKENS=0
  MIN_CHARS=0
"""
from __future__ import annotations
import os, json, sys, hashlib

IN_JSONL  = os.getenv("IN_JSONL",  "output/stage/chunks.jsonl")
OUT_JSONL = os.getenv("OUT_JSONL", "output/stage/embeddable.jsonl")
TEXT_KEY  = os.getenv("TEXT_KEY",  "")
KEEP_META = os.getenv("KEEP_META", "1")
DEDUP     = os.getenv("DEDUP",     "none")
TOKEN_MODE= os.getenv("TOKEN_MODE","approx_chars")
CHAR_PER_TOKEN = float(os.getenv("CHAR_PER_TOKEN","3.6"))
MIN_TOKENS = int(os.getenv("MIN_TOKENS","0"))
MIN_CHARS  = int(os.getenv("MIN_CHARS","0"))

def approx_tokens(s: str) -> int:
    return int(len(s) / CHAR_PER_TOKEN)

def token_count(s: str) -> int:
    if TOKEN_MODE == "approx_chars":
        return approx_tokens(s)
    else:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(s))
        except Exception:
            return approx_tokens(s)

def choose_text(rec: dict) -> str:
    if TEXT_KEY:
        return rec.get(TEXT_KEY, "")
    for k in ("text", "content", "page_content"):
        v = rec.get(k)
        if isinstance(v, str):
            return v
    return ""

if __name__ == "__main__":
    seen = set()
    total = kept = deduped = filtered = 0

    with open(IN_JSONL, "r", encoding="utf-8") as fi, \
         open(OUT_JSONL, "w", encoding="utf-8") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue

            text = choose_text(rec)
            if not isinstance(text, str) or not text.strip():
                filtered += 1; continue
            if MIN_CHARS and len(text) < MIN_CHARS:
                filtered += 1; continue
            if MIN_TOKENS and token_count(text) < MIN_TOKENS:
                filtered += 1; continue

            if DEDUP == "sha1_text":
                h = hashlib.sha1(text.encode("utf-8")).hexdigest()
                if h in seen:
                    deduped += 1; continue
                seen.add(h)

            url = (rec.get("metadata") or {}).get("source_url") or rec.get("source_url")
            out = {
                "id": str(rec.get("id", total)),
                "text": text,
                "metadata": {"url": url}  # บังคับให้มีคีย์ url ทุกชิ้น}
            }
            fo.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[DONE] in={total} kept={kept} filtered={filtered} deduped={deduped} -> {OUT_JSONL}")


# %% [markdown]
# ## 3. Embedding and Vector Indexing

# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 — Embed → Chroma (with metadata:url)
- IN_JSONL: แต่ละบรรทัดต้องมี id, text, metadata.url
"""

import os, json
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer
import numpy as np

IN_JSONL        = os.getenv("IN_JSONL",        "output/stage/embeddable.jsonl")
CHROMA_PATH     = os.getenv("CHROMA_PATH",     "chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_chunks")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
HNSW_SPACE      = os.getenv("HNSW_SPACE",      "cosine")
BATCH_SIZE      = int(os.getenv("BATCH_SIZE",  "256"))

class BGEM3EmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    def __call__(self, docs: Documents) -> Embeddings:
        embs = self.model.encode(
            docs, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        return [e.tolist() for e in embs]

def main():
    print(f"[INFO] input={IN_JSONL}")
    print(f"[INFO] chroma={CHROMA_PATH} | collection={COLLECTION_NAME} | space={HNSW_SPACE}")
    print(f"[INFO] model={EMBEDDING_MODEL} | batch={BATCH_SIZE}")

    ef = BGEM3EmbeddingFunction(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": HNSW_SPACE},
    )

    ids, docs, metas = [], [], []
    added = 0  # ← ประกาศตัวนับก่อนใช้

    with open(IN_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ids.append(str(rec["id"]))
            docs.append(rec["text"])
            metas.append(rec["metadata"])   # ต้องมี {"url": ...}

            if len(ids) >= BATCH_SIZE:
                n = len(ids)
                col.add(ids=ids, documents=docs, metadatas=metas)
                added += n
                print(f"[ADD] +{n} (total={added})")
                ids, docs, metas = [], [], []

    if ids:
        n = len(ids)
        col.add(ids=ids, documents=docs, metadatas=metas)
        added += n
        print(f"[ADD] +{n} (total={added})")

    print(f"[DONE] collection.count() = {col.count()}")

if __name__ == "__main__":
    main()

# %%
ef = BGEM3EmbeddingFunction(EMBEDDING_MODEL)
client = chromadb.PersistentClient(path=CHROMA_PATH)
col = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": HNSW_SPACE},
    )

# %% [markdown]
# ## 4. Vector Deletion by Source URL

# %%
url_to_delete = "https://www.it.chula.ac.th/wp-content/uploads/2020/01/Path_Academic_2551-2555.pdf"

# ลองดึง id มาก่อนเพื่อนับ
hits = col.get(where={"url": url_to_delete}, include=[])  # ขอเฉพาะ id
ids = hits.get("ids", [])
if not ids:
    print("no url match")
else:
    col.delete(where={"url": url_to_delete})
    print(f"deleted {len(ids)} vectors with url = {url_to_delete}")

# %% [markdown]
# ## 5. Retrieval, Reranking, and Answer Generation

# %%
OPENAI_API_KEY =  "[REDACTED_OPENAI_API_KEY]"

from __future__ import annotations
import os
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
from openai import OpenAI
import numpy as np

# Define the embedding function class (reuse from cell 5)
class BGEM3EmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    def __call__(self, docs: Documents) -> Embeddings:
        embs = self.model.encode(
            docs, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        return [e.tolist() for e in embs]

# ---------------- Config (reuse same ENV names as earlier) ----------------
CHROMA_PATH     = os.getenv("CHROMA_PATH", "chroma")  # Match the default from cell 5
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_chunks")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
HNSW_SPACE      = os.getenv("HNSW_SPACE", "cosine")
TOP_K           = int(os.getenv("TOP_K", "5"))             # final K for plain, or for display

# Rerank knobs
RETRIEVAL_MODE  = os.getenv("RETRIEVAL_MODE", "auto")      # "plain" | "rerank" | "auto"
INITIAL_K       = int(os.getenv("INITIAL_K", "50"))        # used if rerank
FINAL_K         = int(os.getenv("FINAL_K",   "10"))         # top-K after rerank
RERANKER_MODEL  = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# LLM
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-5-mini")
client_llm      = OpenAI(api_key=OPENAI_API_KEY)  # Use the variable directly

# ---------------- Open collection (same embedding fn as ingest) -----------
ef = BGEM3EmbeddingFunction(EMBEDDING_MODEL)
client = chromadb.PersistentClient(path=CHROMA_PATH)
col = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=ef,
    metadata={"hnsw:space": HNSW_SPACE},
)

# %%
# === Retrieval (no rerank) ====================================================
# Use the same test query as in the next cell
query = "ขั้นตอนลงทะเบียนเรียน จุฬาฯ ทำอย่างไร"
print(f"[QUERY] {query}")

# --- plain retrieval (show scores) ---
res   = col.query(query_texts=[query], n_results=TOP_K, include=["documents","metadatas","distances"])
ids   = res["ids"][0]
docs  = res["documents"][0]
metas = res["metadatas"][0]
dists = res["distances"][0]

for i, (doc_id, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), 1):
    cos_sim = 1.0 - dist
    score01 = (cos_sim + 1.0) / 2.0
    print(f"\n#{i} id={doc_id} score={score01:.4f} url={meta['url']}")
    print(doc)

contexts = [f"URL: {m['url']}\n{d}" for m, d in zip(metas, docs)]


# %%
# --- Rerank path (with URL) ---------------------------------------------------
query = "ขั้นตอนลงทะเบียนเรียน จุฬาฯ ทำอย่างไร"
res    = col.query(query_texts=[query], n_results=INITIAL_K, include=["documents", "metadatas"])
ids    = res["ids"][0]
docs   = res["documents"][0]
metas  = res["metadatas"][0]     # ต้องมี {"url": ...} จากขั้น embed
print(f"[QUERY] {query}")

reranker = FlagReranker(RERANKER_MODEL, use_fp16=True)
scores   = reranker.compute_score([[query, d] for d in docs], normalize=True)

# sort ตามคะแนน reranker แล้วตัดเหลือ FINAL_K
ranked = sorted(zip(ids, docs, metas, scores), key=lambda x: x[3], reverse=True)[:FINAL_K]

for i, (doc_id, doc_text, meta, score) in enumerate(ranked, start=1):
    url = meta["url"]                         # ใช้คีย์ 'url' ตามที่เตรียมไว้ใน metadata
    preview = (doc_text[:280] + "…") if len(doc_text) > 280 else doc_text
    print(f"\n#{i} id={doc_id}  rerank_score={score:.4f}  url={url}")
    print(preview)

# contexts สำหรับส่งเข้า LLM: แนบ URL ไว้กับแต่ละชิ้น
contexts = [f"URL: {meta['url']}\n{doc}" for (_, doc, meta, _) in ranked]

# %% [markdown]
# ## 6. Retrieval and Answer Examples

# %%
# contexts สำหรับส่งเข้า LLM: แนบ URL ไว้กับแต่ละชิ้น
contexts = [f"URL: {meta['url']}\n{doc}" for (_, doc, meta, _) in ranked]

context_blob = "\n\n---\n\n".join(contexts)
prompt = (
    "คุณเป็นผู้ช่วยตอบคำถามจากเอกสาร (RAG). "
    "ตอบโดยอ้างอิงเฉพาะจาก Context ด้านล่างเท่านั้น หากไม่พบข้อมูลให้บอกว่าไม่พบ.\n\n"
    f"Context:\n{context_blob}\n\n"
    f"คำถาม: {query}\n"
    "รูปแบบคำตอบ: กระชับ ชัดเจน (ตอบตามภาษาที่ผู้ใช้ถาม)ให้อ้างอิง URL ที่นำมาตอบด้วย"
)

resp = client_llm.chat.completions.create(
    model=OPENAI_MODEL,
    messages=[{"role": "user", "content": prompt}],
   
)
print("\n[LLM ANSWER]\n" + resp.choices[0].message.content)

# %%
# ---------------- LLM smoke test ------------------------------------------
query = "ขั้นตอนถอนรายวิชาต้องทำอย่างไร"
print(f"[QUERY] {query}")

res    = col.query(query_texts=[query], n_results=INITIAL_K, include=["documents", "metadatas"])
ids    = res["ids"][0]
docs   = res["documents"][0]
metas  = res["metadatas"][0]     # ต้องมี {"url": ...} จากขั้น embed

reranker = FlagReranker(RERANKER_MODEL, use_fp16=True)
scores   = reranker.compute_score([[query, d] for d in docs], normalize=True)

# sort ตามคะแนน reranker แล้วตัดเหลือ FINAL_K
ranked = sorted(zip(ids, docs, metas, scores), key=lambda x: x[3], reverse=True)[:FINAL_K]

# contexts สำหรับส่งเข้า LLM: แนบ URL ไว้กับแต่ละชิ้น
contexts = [f"URL: {meta['url']}\n{doc}" for (_, doc, meta, _) in ranked]


context_blob = "\n\n---\n\n".join(contexts)
prompt = (
    "คุณเป็นผู้ช่วยตอบคำถามจากเอกสาร (RAG). "
    "ตอบโดยอ้างอิงเฉพาะจาก Context ด้านล่างเท่านั้น หากไม่พบข้อมูลให้บอกว่าไม่พบ.\n\n"
    f"Context:\n{context_blob}\n\n"
    f"คำถาม: {query}\n"
    "รูปแบบคำตอบ: กระชับ ชัดเจน (ตอบตามภาษาที่ผู้ใช้ถาม)ให้อ้างอิง URL ที่นำมาตอบด้วย"
)

resp = client_llm.chat.completions.create(
    model=OPENAI_MODEL,
    messages=[{"role": "user", "content": prompt}],
   
)
print("\n[LLM ANSWER]\n" + resp.choices[0].message.content)

# %%
# ---------------- LLM smoke test ------------------------------------------
query = "how to withdraw course?"
print(f"[QUERY] {query}")

res    = col.query(query_texts=[query], n_results=INITIAL_K, include=["documents", "metadatas"])
ids    = res["ids"][0]
docs   = res["documents"][0]
metas  = res["metadatas"][0]     # ต้องมี {"url": ...} จากขั้น embed

reranker = FlagReranker(RERANKER_MODEL, use_fp16=True)
scores   = reranker.compute_score([[query, d] for d in docs], normalize=True)

# sort ตามคะแนน reranker แล้วตัดเหลือ FINAL_K
ranked = sorted(zip(ids, docs, metas, scores), key=lambda x: x[3], reverse=True)[:FINAL_K]

# contexts สำหรับส่งเข้า LLM: แนบ URL ไว้กับแต่ละชิ้น
contexts = [f"URL: {meta['url']}\n{doc}" for (_, doc, meta, _) in ranked]


context_blob = "\n\n---\n\n".join(contexts)
prompt = (
    "คุณเป็นผู้ช่วยตอบคำถามจากเอกสาร (RAG). "
    "ตอบโดยอ้างอิงเฉพาะจาก Context ด้านล่างเท่านั้น หากไม่พบข้อมูลให้บอกว่าไม่พบ.\n\n"
    f"Context:\n{context_blob}\n\n"
    f"คำถาม: {query}\n"
    "รูปแบบคำตอบ: กระชับ ชัดเจน (ตอบตามภาษาที่ผู้ใช้ถาม)ให้อ้างอิง URL ที่นำมาตอบด้วย"
)

resp = client_llm.chat.completions.create(
    model=OPENAI_MODEL,
    messages=[{"role": "user", "content": prompt}],
   
)
print("\n[LLM ANSWER]\n" + resp.choices[0].message.content)

# %%
# ---------------- LLM smoke test ------------------------------------------
query = "วันสอบปลายภาคเรียนปีการศึกษา 2570 คือวันไหน"
print(f"[QUERY] {query}")

res    = col.query(query_texts=[query], n_results=INITIAL_K, include=["documents", "metadatas"])
ids    = res["ids"][0]
docs   = res["documents"][0]
metas  = res["metadatas"][0]     # ต้องมี {"url": ...} จากขั้น embed

reranker = FlagReranker(RERANKER_MODEL, use_fp16=True)
scores   = reranker.compute_score([[query, d] for d in docs], normalize=True)

# sort ตามคะแนน reranker แล้วตัดเหลือ FINAL_K
ranked = sorted(zip(ids, docs, metas, scores), key=lambda x: x[3], reverse=True)[:FINAL_K]

# contexts สำหรับส่งเข้า LLM: แนบ URL ไว้กับแต่ละชิ้น
contexts = [f"URL: {meta['url']}\n{doc}" for (_, doc, meta, _) in ranked]


context_blob = "\n\n---\n\n".join(contexts)
prompt = (
    "คุณเป็นผู้ช่วยตอบคำถามจากเอกสาร (RAG). "
    "ตอบโดยอ้างอิงเฉพาะจาก Context ด้านล่างเท่านั้น หากไม่พบข้อมูลให้บอกว่าไม่พบ.\n\n"
    f"Context:\n{context_blob}\n\n"
    f"คำถาม: {query}\n"
    "รูปแบบคำตอบ: กระชับ ชัดเจน (ตอบตามภาษาที่ผู้ใช้ถาม)ให้อ้างอิง URL ที่นำมาตอบด้วย"
)

resp = client_llm.chat.completions.create(
    model=OPENAI_MODEL,
    messages=[{"role": "user", "content": prompt}],
   
)
print("\n[LLM ANSWER]\n" + resp.choices[0].message.content)

# %%
# ---------------- LLM smoke test ------------------------------------------
query = "ใช้งานอีเมลจุฬายังไง"
print(f"[QUERY] {query}")

res    = col.query(query_texts=[query], n_results=INITIAL_K, include=["documents", "metadatas"])
ids    = res["ids"][0]
docs   = res["documents"][0]
metas  = res["metadatas"][0]     # ต้องมี {"url": ...} จากขั้น embed

reranker = FlagReranker(RERANKER_MODEL, use_fp16=True)
scores   = reranker.compute_score([[query, d] for d in docs], normalize=True)

# sort ตามคะแนน reranker แล้วตัดเหลือ FINAL_K
ranked = sorted(zip(ids, docs, metas, scores), key=lambda x: x[3], reverse=True)[:FINAL_K]

# contexts สำหรับส่งเข้า LLM: แนบ URL ไว้กับแต่ละชิ้น
contexts = [f"URL: {meta['url']}\n{doc}" for (_, doc, meta, _) in ranked]


context_blob = "\n\n---\n\n".join(contexts)
prompt = (
    "คุณเป็นผู้ช่วยตอบคำถามจากเอกสาร (RAG). "
    "ตอบโดยอ้างอิงเฉพาะจาก Context ด้านล่างเท่านั้น หากไม่พบข้อมูลให้บอกว่าไม่พบ.\n\n"
    f"Context:\n{context_blob}\n\n"
    f"คำถาม: {query}\n"
    "รูปแบบคำตอบ: กระชับ ชัดเจน (ตอบตามภาษาที่ผู้ใช้ถาม)ให้อ้างอิง URL ที่นำมาตอบด้วย"
)

resp = client_llm.chat.completions.create(
    model=OPENAI_MODEL,
    messages=[{"role": "user", "content": prompt}],
   
)
print("\n[LLM ANSWER]\n" + resp.choices[0].message.content)

# %%
# ---------------- LLM smoke test ------------------------------------------
query = "เปลี่ยนรหัสผ่านอีเมลจุฬายังไง"
print(f"[QUERY] {query}")

res    = col.query(query_texts=[query], n_results=INITIAL_K, include=["documents", "metadatas"])
ids    = res["ids"][0]
docs   = res["documents"][0]
metas  = res["metadatas"][0]     # ต้องมี {"url": ...} จากขั้น embed

reranker = FlagReranker(RERANKER_MODEL, use_fp16=True)
scores   = reranker.compute_score([[query, d] for d in docs], normalize=True)

# sort ตามคะแนน reranker แล้วตัดเหลือ FINAL_K
ranked = sorted(zip(ids, docs, metas, scores), key=lambda x: x[3], reverse=True)[:FINAL_K]

# contexts สำหรับส่งเข้า LLM: แนบ URL ไว้กับแต่ละชิ้น
contexts = [f"URL: {meta['url']}\n{doc}" for (_, doc, meta, _) in ranked]


context_blob = "\n\n---\n\n".join(contexts)
prompt = (
    "คุณเป็นผู้ช่วยตอบคำถามจากเอกสาร (RAG). "
    "ตอบโดยอ้างอิงเฉพาะจาก Context ด้านล่างเท่านั้น หากไม่พบข้อมูลให้บอกว่าไม่พบ.\n\n"
    f"Context:\n{context_blob}\n\n"
    f"คำถาม: {query}\n"
    "รูปแบบคำตอบ: กระชับ ชัดเจน (ตอบตามภาษาที่ผู้ใช้ถาม)ให้อ้างอิง URL ที่นำมาตอบด้วย"
)

resp = client_llm.chat.completions.create(
    model=OPENAI_MODEL,
    messages=[{"role": "user", "content": prompt}],
   
)
print("\n[LLM ANSWER]\n" + resp.choices[0].message.content)

# %%
# ---------------- LLM smoke test ------------------------------------------
query = "อยากขอทุนการศึกษาต้องทำยังไง"
print(f"[QUERY] {query}")

res    = col.query(query_texts=[query], n_results=INITIAL_K, include=["documents", "metadatas"])
ids    = res["ids"][0]
docs   = res["documents"][0]
metas  = res["metadatas"][0]     # ต้องมี {"url": ...} จากขั้น embed

reranker = FlagReranker(RERANKER_MODEL, use_fp16=True)
scores   = reranker.compute_score([[query, d] for d in docs], normalize=True)

# sort ตามคะแนน reranker แล้วตัดเหลือ FINAL_K
ranked = sorted(zip(ids, docs, metas, scores), key=lambda x: x[3], reverse=True)[:FINAL_K]

# contexts สำหรับส่งเข้า LLM: แนบ URL ไว้กับแต่ละชิ้น
contexts = [f"URL: {meta['url']}\n{doc}" for (_, doc, meta, _) in ranked]


context_blob = "\n\n---\n\n".join(contexts)
prompt = (
    "คุณเป็นผู้ช่วยตอบคำถามจากเอกสาร (RAG). "
    "ตอบโดยอ้างอิงเฉพาะจาก Context ด้านล่างเท่านั้น หากไม่พบข้อมูลให้บอกว่าไม่พบ.\n\n"
    f"Context:\n{context_blob}\n\n"
    f"คำถาม: {query}\n"
    "รูปแบบคำตอบ: กระชับ ชัดเจน (ตอบตามภาษาที่ผู้ใช้ถาม)ให้อ้างอิง URL ที่นำมาตอบด้วย"
)

resp = client_llm.chat.completions.create(
    model=OPENAI_MODEL,
    messages=[{"role": "user", "content": prompt}],
   
)
print("\n[LLM ANSWER]\n" + resp.choices[0].message.content)

# %%
# ---------------- LLM smoke test ------------------------------------------
query = "เงื่อนไขในการได้ทุนการศึกษาคืออะไร"
print(f"[QUERY] {query}")

res    = col.query(query_texts=[query], n_results=INITIAL_K, include=["documents", "metadatas"])
ids    = res["ids"][0]
docs   = res["documents"][0]
metas  = res["metadatas"][0]     # ต้องมี {"url": ...} จากขั้น embed

reranker = FlagReranker(RERANKER_MODEL, use_fp16=True)
scores   = reranker.compute_score([[query, d] for d in docs], normalize=True)

# sort ตามคะแนน reranker แล้วตัดเหลือ FINAL_K
ranked = sorted(zip(ids, docs, metas, scores), key=lambda x: x[3], reverse=True)[:FINAL_K]

# contexts สำหรับส่งเข้า LLM: แนบ URL ไว้กับแต่ละชิ้น
contexts = [f"URL: {meta['url']}\n{doc}" for (_, doc, meta, _) in ranked]


context_blob = "\n\n---\n\n".join(contexts)
prompt = (
    "คุณเป็นผู้ช่วยตอบคำถามจากเอกสาร (RAG). "
    "ตอบโดยอ้างอิงเฉพาะจาก Context ด้านล่างเท่านั้น หากไม่พบข้อมูลให้บอกว่าไม่พบ.\n\n"
    f"Context:\n{context_blob}\n\n"
    f"คำถาม: {query}\n"
    "รูปแบบคำตอบ: กระชับ ชัดเจน (ตอบตามภาษาที่ผู้ใช้ถาม)ให้อ้างอิง URL ที่นำมาตอบด้วย"
)

resp = client_llm.chat.completions.create(
    model=OPENAI_MODEL,
    messages=[{"role": "user", "content": prompt}],
   
)
print("\n[LLM ANSWER]\n" + resp.choices[0].message.content)

# %%
# ---------------- LLM smoke test ------------------------------------------
query = "ขอวิธีสำเร็จการศึกษาหน่อยสิ แบบละเอียดรายขั้นตอนเลย"
print(f"[QUERY] {query}")

res    = col.query(query_texts=[query], n_results=INITIAL_K, include=["documents", "metadatas"])
ids    = res["ids"][0]
docs   = res["documents"][0]
metas  = res["metadatas"][0]     # ต้องมี {"url": ...} จากขั้น embed

reranker = FlagReranker(RERANKER_MODEL, use_fp16=True)
scores   = reranker.compute_score([[query, d] for d in docs], normalize=True)

# sort ตามคะแนน reranker แล้วตัดเหลือ FINAL_K
ranked = sorted(zip(ids, docs, metas, scores), key=lambda x: x[3], reverse=True)[:FINAL_K]

# contexts สำหรับส่งเข้า LLM: แนบ URL ไว้กับแต่ละชิ้น
contexts = [f"URL: {meta['url']}\n{doc}" for (_, doc, meta, _) in ranked]


context_blob = "\n\n---\n\n".join(contexts)
prompt = (
    "คุณเป็นผู้ช่วยตอบคำถามจากเอกสาร (RAG). "
    "ตอบโดยอ้างอิงเฉพาะจาก Context ด้านล่างเท่านั้น หากไม่พบข้อมูลให้บอกว่าไม่พบ.\n\n"
    f"Context:\n{context_blob}\n\n"
    f"คำถาม: {query}\n"
    "รูปแบบคำตอบ: กระชับ ชัดเจน (ตอบตามภาษาที่ผู้ใช้ถาม)ให้อ้างอิง URL ที่นำมาตอบด้วย"
)

resp = client_llm.chat.completions.create(
    model=OPENAI_MODEL,
    messages=[{"role": "user", "content": prompt}],
   
)
print("\n[LLM ANSWER]\n" + resp.choices[0].message.content)

