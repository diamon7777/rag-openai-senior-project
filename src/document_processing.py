# - Config (CFG): MANIFEST, DOCS, POPPLER_PATH, OCR_LANGS
# - Helpers: ensure_dirs, hydrate_seen_sha1, content_sha1, sha1_text,
#            append_jsonl, now, log, get_easyocr_reader
# - Globals: SEEN_SHA1, LOG_LEVEL

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, Iterable, Optional, Set, Union
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os
import sys
import easyocr
# ------------------------------- CONFIG ------------------------------------

@dataclass
class Config:
    OUT_HTML: str = os.environ.get("OUT_HTML", "output/raw/html")
    OUT_PDF: str  = os.environ.get("OUT_PDF",  "output/raw/pdfs")
    OUT_IMG: str  = os.environ.get("OUT_IMG",  "output/raw/images")
    OUT_STAGE: str= os.environ.get("OUT_STAGE","output/stage")
    MANIFEST: str = os.environ.get("MANIFEST", "output/stage/manifest.jsonl")
    DOCS: str     = os.environ.get("DOCS",     "output/stage/documents.jsonl")
    POPPLER_PATH: Optional[Path] = None
    OCR_LANGS: tuple[str, ...] = ("en", "th")



# ค่าตั้งต้น ปรับตามโปรเจกต์ได้
CFG = Config(
    MANIFEST=Path("data/manifest.jsonl"),
    DOCS=Path("output/documents.jsonl"),
    POPPLER_PATH=Path(os.getenv("POPPLER_PATH", r"C:\Users\gpaki\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin")).resolve() if os.getenv("POPPLER_PATH") else None,
    OCR_LANGS=tuple([s.strip() for s in os.getenv("OCR_LANGS", "en,th").split(",") if s.strip()]),
)

# ------------------------------- LOGGING -----------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}
_TZ = ZoneInfo(os.getenv("TZ", "Asia/Bangkok"))

def now() -> str:
    """คืนค่าเวลาปัจจุบันในเขตเวลา Asia/Bangkok เป็น ISO8601 (ตัด milliseconds)"""
    return datetime.now(_TZ).replace(microsecond=0).isoformat()

def log(level: str, message: str, **fields: Any) -> None:
    lvl = level.upper()
    if _LEVELS.get(lvl, 999) < _LEVELS.get(LOG_LEVEL, 20):
        return
    rec = {"ts": now(), "level": lvl, "msg": message}
    if fields:
        rec.update(fields)
    print(json.dumps(rec, ensure_ascii=False), file=sys.stderr)

# ------------------------------- I/O & PATHS -------------------------------

def ensure_dirs(*paths: Path) -> None:
    targets = list(paths) or [CFG.DOCS.parent]
    for p in targets:
        p = Path(p)
        if p.suffix:  # ถ้าเป็นไฟล์ ให้ใช้โฟลเดอร์แม่
            p = p.parent
        p.mkdir(parents=True, exist_ok=True)

def append_jsonl(path: Union[str, Path], record: Dict[str, Any]) -> None:
    """
    เขียน 1 ระเบียนลงไฟล์ .jsonl (append) — ใช้ทุกสเต็ปเวลาบันทึกเอกสาร
    """
    path = Path(path)
    ensure_dirs(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ------------------------------- HASH/ID -----------------------------------

def _to_bytes(x: Union[str, bytes]) -> bytes:
    if isinstance(x, bytes):
        return x
    return x.encode("utf-8", "ignore")

def content_sha1(content: Union[str, bytes]) -> str:
    return hashlib.sha1(_to_bytes(content)).hexdigest()

def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()

# ------------------------------- DEDUP STATE -------------------------------

SEEN_SHA1: Set[str] = set()

def hydrate_seen_sha1(docs_path: Union[str, Path] = None) -> Set[str]:
    docs = Path(docs_path) if docs_path else CFG.DOCS
    if not docs.exists():
        SEEN_SHA1.clear()
        return SEEN_SHA1

    loaded: Set[str] = set()
    with docs.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("sha1", "content_sha1", "hash"):
                val = obj.get(key)
                if isinstance(val, str) and len(val) >= 8:
                    loaded.add(val)
                    break
    SEEN_SHA1.clear()
    SEEN_SHA1.update(loaded)
    log("DEBUG", "hydrate_seen_sha1", count=len(SEEN_SHA1), path=str(docs))
    return SEEN_SHA1

# ------------------------------- OCR READER --------------------------------

def get_easyocr_reader():
    # ตรวจ GPU อัตโนมัติ (ถ้าไม่มี torch ให้ทำงานแบบ CPU)
    use_gpu = False
    try:
        import torch  # type: ignore
        use_gpu = bool(torch.cuda.is_available())
    except Exception:
        use_gpu = False

    langs = list(CFG.OCR_LANGS) if CFG.OCR_LANGS else ["en"]
    reader = easyocr.Reader(langs, gpu=use_gpu)
    log("INFO", "easyocr.Reader ready", langs=langs, gpu=use_gpu)
    return reader

import hashlib
import json
import trafilatura
from bs4 import BeautifulSoup
# ===== HTML extraction (Trafilatura) =====
# No Extractor needed, use trafilatura.extract directly

def extract_meta_min(html: str, default_url: Optional[str] = None) -> Dict[str, Any]:
    try:
        meta_json = trafilatura.extract(html, output_format="json", with_metadata=True)
        if not meta_json: return {}
        m = json.loads(meta_json)
        return {"title": m.get("title"), "published_at": m.get("date"), "updated_at": m.get("modified") or m.get("date_modified")}
    except Exception as e:
        log("WARNING","trafilatura_meta_failed", err=str(e)); return {}
def html_to_text(html: str) -> str:
    if trafilatura is not None:
        txt = trafilatura.extract(
            html,
            include_tables=True,
            include_formatting=True,
            favor_recall=True,
            deduplicate=True,)
        if txt:
            return txt
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text("\n", strip=True)









# ===== Step 2: แปลง HTML → ข้อความ แล้วเขียนลง documents.jsonl =====
ensure_dirs(); hydrate_seen_sha1(); added = 0
if not os.path.exists(CFG.MANIFEST):
    log("ERROR","manifest_missing", path=CFG.MANIFEST)
else:
    with open(CFG.MANIFEST, encoding="utf-8") as f:
        for line in f:
            try: m = json.loads(line)
            except Exception: continue
            if m.get("record") != "page":
                continue
            html_path = m.get("html_path"); src = m.get("canonical")
            if (not html_path) or (not os.path.exists(html_path)) or (not src):
                log("ERROR","manifest_row_invalid"); continue
            with open(html_path, encoding="utf-8") as hf:
                html = hf.read()
            clean = html_to_text(html)
            if not clean: continue
            sha1_val = content_sha1(clean)
            if sha1_val in SEEN_SHA1: continue
            SEEN_SHA1.add(sha1_val)
            rec = {"id": sha1_text(src), "source_url": src, "source_type": "html", "text_full": clean, "content_hash": sha1_val, "captured_at": now()}
            meta = extract_meta_min(html, src)
            if meta: rec.update(meta)
            append_jsonl(CFG.DOCS, rec); added += 1
log("INFO","step2_done", html_docs=added)
import cv2
import numpy as np
import easyocr

def _poly_area(poly: List[List[float]]) -> float:
    x = np.array([p[0] for p in poly], dtype=float); y = np.array([p[1] for p in poly], dtype=float)
    return 0.5 * abs(np.dot(x, np.roll(y,1)) - np.dot(y, np.roll(x,1)))
def is_infographic(img_np: np.ndarray, alt_text: str = "", src_hint: str = "") -> bool:
    h, w = img_np.shape[:2]
    if min(h, w) < 320: return False
    long_side = max(h, w); scale = 800/long_side
    img_small = cv2.resize(img_np, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else img_np
    reader = get_easyocr_reader(); res = reader.readtext(img_small, detail=1, paragraph=False)
    if not res:
        at = f"{alt_text} {src_hint}".lower();
        for k in ("กำหนดการ","ประกาศ","ขั้นตอน","รายชื่อ","ลงทะเบียน","ชำระเงิน","reg","chula","office of the registrar","student journey"):
            if k in at: return True
        return False
    img_area = img_small.shape[0]*img_small.shape[1]
    boxes_area = sum(_poly_area(r[0]) for r in res)
    text_area_ratio = boxes_area / max(1, img_area)
    confs = [float(r[2]) for r in res if isinstance(r[2], (float,int))]
    avg_conf = (sum(confs)/len(confs)) if confs else 0.0
    texts_join = " ".join([r[1] for r in res if isinstance(r[1], str)]).lower()
    if len(res) >= 6: return True
    if text_area_ratio >= 0.08 and avg_conf >= 0.5: return True
    for k in ("กำหนดการ","ประกาศ","ขั้นตอน","รายชื่อ","ลงทะเบียน","ชำระเงิน","reg","chula","office of the registrar","student journey"):
        if k in texts_join: return True
    return False


def ocr_image_highres(img_np: np.ndarray) -> str:
    H, W = img_np.shape[:2]; scale = 1.8 if max(H, W) < 1600 else 1.2
    img_big = cv2.resize(img_np, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_CUBIC)
    reader = get_easyocr_reader(); out = reader.readtext(img_big, detail=0)
    return "".join(out).strip()

# ===== Step 3: อ่านรูปจาก manifest → OCR เฉพาะ infographic =====
import aiohttp as _aiohttp
async def _http_get_bytes(url: str, referer: str | None = None, timeout: int = 30) -> bytes | None:
    headers = {"User-Agent":"RAG-Ingest/1.0"};
    if referer: headers["Referer"] = referer
    try:
        async with _aiohttp.ClientSession(headers=headers) as s:
            async with s.get(url, timeout=timeout) as r:
                if r.status == 200: return await r.read()
    except Exception as e:
        log("WARNING","http_get_bytes_failed", url=url, err=str(e))
    return None

# ===== Run Step 3 (inline, local-only) =====
ensure_dirs(); hydrate_seen_sha1(); added = 0

if not os.path.exists(CFG.MANIFEST):
    log("ERROR","manifest_missing", path=CFG.MANIFEST)
else:
    
    with open(CFG.MANIFEST, encoding="utf-8") as f:
        for line in f:
            try: m = json.loads(line)
            except Exception: continue
            if m.get("record") != "asset" or m.get("asset_type") != "image":
                continue
            img_path = m.get("path"); iu = m.get("url"); alt = m.get("alt", ""); canon = m.get("page")
            if (not img_path) or (not os.path.exists(img_path)) or (not iu):
                log("ERROR","asset_image_missing", path=img_path, url=iu); 
                continue
            try:
                with open(img_path, "rb") as rf:
                    data = rf.read()
                buf = np.frombuffer(data, np.uint8); img_np = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img_np is None:
                    log("ERROR","asset_image_decode_failed", path=img_path); 
                    continue
            except Exception as e:
                log("ERROR","asset_image_read_failed", path=img_path, err=str(e));
                continue

            if not is_infographic(img_np, alt_text=alt, src_hint=m.get("src_hint","")):
                continue

            # High-res OCR + QC metrics (เก็บเฉพาะ 2 ค่า)
            H, W = img_np.shape[:2]
            scale = 1.8 if max(H, W) < 1600 else 1.2
            img_big = cv2.resize(img_np, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_CUBIC)

            reader = get_easyocr_reader()
            res = reader.readtext(img_big, detail=1, paragraph=False)

            text_img = " ".join([r[1] for r in res]).strip()
            if not text_img:
                continue

            confs = [float(r[2]) for r in res if isinstance(r[2], (int, float))]
            ocr_avg_conf = (sum(confs)/len(confs)) if confs else 0.0

            img_area = img_big.shape[0] * img_big.shape[1]
            boxes_area = sum(_poly_area(r[0]) for r in res) if res else 0.0
            ocr_text_area_ratio = float(boxes_area / img_area) if img_area else 0.0

            sha1_val = content_sha1(text_img)
            if sha1_val in SEEN_SHA1:
                continue
            SEEN_SHA1.add(sha1_val)

            rec = {
                "id": sha1_val,
                "source_url": iu,
                "source_type": "image",
                "is_infographic": True,
                "text_full": text_img,
                "alt": alt,
                "content_hash": sha1_val,
                "captured_at": now(),
                "page": canon,
                "ocr_avg_conf": round(ocr_avg_conf, 3),
                "ocr_text_area_ratio": round(ocr_text_area_ratio, 4),
            }
            append_jsonl(CFG.DOCS, rec); added += 1

log("INFO","step3_done", image_docs=added)
import fitz  # ถ้าใช้ร่วม
import pymupdf4llm
import hashlib
import numpy as np
import cv2
import subprocess, shutil, unicodedata, re, numpy as np, cv2, fitz, pymupdf4llm

# ใช้ pattern เดิม (อนุโลม TH/EN + punctuation ทั่วไป)
_ALLOWED_RE = re.compile(r"[A-Za-z0-9\u0E00-\u0E7F\s\.,;:'\"!?()\[\]{}\-\_/\\@#%&*+|=<>~`°–—…·•]")

from langid.langid import LanguageIdentifier, model
_LANGID_ID = LanguageIdentifier.from_modelstring(model, norm_probs=True)

def md_has_table(md: str) -> bool:
    if not md:
        return False
    low = md.lower()
    if "<table" in low:
        return True
    lines = [ln.strip() for ln in md.splitlines()]
    for i in range(len(lines) - 1):
        a, b = lines[i], lines[i + 1]
        if a.startswith("|") and a.endswith("|") and "|" in a:
            parts = [p.strip() for p in b.split("|") if p.strip()]
            if parts and all(set(p).issubset(set("-: ")) and len(p.replace(":", "").replace(" ", "")) >= 3 for p in parts):
                return True
    return False








def is_usable_th_en(text: str, min_len: int = 150, min_prob: float = 0.70) -> bool:
    if not text or len(text) < min_len:
        return False
    try:
        lang, prob = _LANGID_ID.classify(text)
        return (lang in ("th", "en")) and (prob >= min_prob)
    except Exception:
        return False

def _qc_ok(t: str, min_chars: int) -> bool:
    if not t or len(t) < min_chars:
        return False
    if "�" in t:                         # replacement char -> treat as failed
        return False
    total = len(t)
    bad = sum(1 for ch in t if not _ALLOWED_RE.match(ch))
    bad_ratio = (bad / total) if total else 1.0
    if bad_ratio >= 0.05:                # เข้มขึ้นเพื่อลด noise
        return False
    return True

def _pdftotext_whole(pdf_path: str) -> str:
    exe = "pdftotext.exe" if os.name == "nt" else "pdftotext"
    cmd = [exe, "-layout", "-enc", "UTF-8", pdf_path, "-"]
    env = os.environ.copy()
    if getattr(CFG, "POPPLER_PATH", None):
        env["PATH"] = str(CFG.POPPLER_PATH) + os.pathsep + env.get("PATH","")
    out = subprocess.run(cmd, env=env, capture_output=True, check=True)
    return out.stdout.decode("utf-8", errors="strict").strip()

def _pdftotext_page(pdf_path: str, pno: int) -> str:
    exe = "pdftotext.exe" if os.name == "nt" else "pdftotext"
    cmd = [exe, "-layout", "-enc", "UTF-8", "-f", str(pno), "-l", str(pno), pdf_path, "-"]
    env = os.environ.copy()
    if getattr(CFG, "POPPLER_PATH", None):
        env["PATH"] = str(CFG.POPPLER_PATH) + os.pathsep + env.get("PATH","")
    out = subprocess.run(cmd, env=env, capture_output=True, check=True)
    return out.stdout.decode("utf-8", errors="strict").strip()

def pdf_text_per_page_with_langdetect_fallback(pdf_path: str, ocr_dpi: int = 220) -> str:
    # เรียบง่าย: ต่อหน้า → PyMuPDF text → ถ้าไม่โอเคลอง pdftotext หน้าเดียว → ถ้ายังว่าง/เสีย → OCR เฉพาะหน้านั้น
    out_pages = []
    reader = get_easyocr_reader()
    with fitz.open(pdf_path) as doc:
        for p in doc:
            t = (p.get_text("text", sort=True) or "").strip()
            if t and _qc_ok(t, min_chars=30) and is_usable_th_en(t, min_len=30, min_prob=0.55):
                out_pages.append(t); continue
            # ลอง Poppler เฉพาะหน้านั้นก่อนแทน OCR
            try:
                t2 = _pdftotext_page(pdf_path, p.number+1)  # pdftotext หน้าฐาน 1
            except Exception:
                t2 = ""
            if t2 and _qc_ok(t2, min_chars=30) and is_usable_th_en(t2, min_len=30, min_prob=0.55):
                out_pages.append(t2); continue
            # สุดท้ายค่อย OCR เฉพาะหน้านี้
            pix = p.get_pixmap(dpi=ocr_dpi)
            img_bytes = pix.tobytes("png")
            arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                out_pages.append("")
            else:
                ocr_lines = reader.readtext(img, detail=0)
                out_pages.append(" ".join(ocr_lines).strip())
    return "".join([x for x in out_pages if x]).strip()

def pdf_to_markdown_with_ocr_fallback(pdf_path: str, min_chars: int = 150, ocr_dpi: int = 220) -> str:
    # ลำดับพยายาม: PyMuPDF4LLM → pdftotext ทั้งไฟล์ → per-page salvage (เฉพาะหน้า) + OCR หน้า
    # 1) PyMuPDF4LLM
    md = ""
    try:
        md = (pymupdf4llm.to_markdown(pdf_path) or "").strip()
    except Exception as e:
        log("WARNING", "pymupdf4llm_failed", path=pdf_path, err=str(e)); md = ""
    if md and _qc_ok(md, min_chars) and (is_usable_th_en(md, min_len=min_chars, min_prob=0.65) or len(md) >= (min_chars*2)):
        return md

    # 2) Poppler (ทั้งไฟล์) – ยัง “ไม่ต้อง OCR”
    try:
        txt = _pdftotext_whole(pdf_path)
    except Exception as e:
        txt = ""
        log("WARNING", "pdftotext_whole_failed", path=pdf_path, err=str(e))
    if txt and _qc_ok(txt, min_chars) and (is_usable_th_en(txt, min_len=min_chars, min_prob=0.65) or len(txt) >= (min_chars*2)):
        return txt

    # 3) Per-page salvage + OCR เฉพาะหน้า
    txt2 = pdf_text_per_page_with_langdetect_fallback(pdf_path, ocr_dpi=ocr_dpi)
    if not txt2 or not txt2.strip():
        raise RuntimeError("pdf_markdown_and_ocr_fallback_empty")
    return txt2

# ===== Step 4: ดาวน์โหลด/แปลง PDF จาก manifest → documents.jsonl =====
import aiohttp as _aiohttp
# ===== Run Step 4 (inline, local-only) =====
ensure_dirs(); hydrate_seen_sha1(); added = 0

if not os.path.exists(CFG.MANIFEST):
    log("ERROR","manifest_missing", path=CFG.MANIFEST)
else:
    # Read only pdf assets from manifest (local-only, no network)
    with open(CFG.MANIFEST, encoding="utf-8") as f:
        for line in f:
            try: m = json.loads(line)
            except Exception: continue
            if m.get("record") != "asset" or m.get("asset_type") != "pdf":
                continue
            pdf_path = m.get("path"); pu = m.get("url"); canon = m.get("page")
            if (not pdf_path) or (not os.path.exists(pdf_path)) or (not pu):
                log("ERROR","asset_pdf_missing", path=pdf_path, url=pu); 
                continue
            try:
                md = pdf_to_markdown_with_ocr_fallback(pdf_path)
                has_tbl = md_has_table(md)
            except Exception as e:
                log("ERROR","pdf_to_markdown_failed", path=pdf_path, err=str(e));
                continue
            if not md:
                continue
            sha1_val = content_sha1(md)
            if sha1_val in SEEN_SHA1:
                continue
            SEEN_SHA1.add(sha1_val)
            rec = {"id": sha1_val, "source_url": pu, "source_type":"pdf_md", "format":"markdown", "has_tables": has_tbl, "text_full": md, "content_hash": sha1_val, "captured_at": now(), "page": canon}
            append_jsonl(CFG.DOCS, rec); added += 1
log("INFO","step4_done", pdf_docs=added)

