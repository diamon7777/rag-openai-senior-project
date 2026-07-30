from __future__ import annotations
import asyncio, hashlib, io, json, os, re, sys, time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit

import aiohttp
from bs4 import BeautifulSoup
import numpy as np
import cv2

import trafilatura
from trafilatura.settings import Extractor

import fitz  # PyMuPDF
import pymupdf4llm  # PDF -> Markdown
from pdf2image import convert_from_path
import easyocr
from langid.langid import LanguageIdentifier, model as LANGID_MODEL
_LANGID_ID = LanguageIdentifier.from_modelstring(LANGID_MODEL, norm_probs=True)

from playwright.async_api import async_playwright
from w3lib.url import canonicalize_url as w3_canonicalize_url, url_query_cleaner
# ===== Logging =====
LOG_LVL = 20
LEVELS = {"DEBUG":10, "INFO":20, "WARNING":30, "ERROR":40, "CRITICAL":50}

def now() -> str: return datetime.now(timezone.utc).isoformat()


def log(level: str, msg: str, **kw):
    if LEVELS.get(level, 100) < LOG_LVL: return
    extra = " ".join(f"{k}={kw[k]}" for k in kw)
    print(f"[{now()}] {level} {msg} {extra}")

# ===== Config =====
class CFG:
    ALLOWED_HOSTS = (
        "it.chula.ac.th",
        "reg.chula.ac.th", "www.reg.chula.ac.th",
        "sa.chula.ac.th",
    )
    START_URLS = (
        "https://www.it.chula.ac.th/en/",
        "https://www.it.chula.ac.th/th/",
        "https://www.reg.chula.ac.th/",
    )
    OUT_HTML = "output/raw/html"
    OUT_PDF = "output/raw/pdfs"
    OUT_IMG = "output/raw/images"
    OUT_STAGE = "output/stage"
    MANIFEST = "output/stage/manifest.jsonl"
    DOCS = "output/stage/documents.jsonl"
    MAX_PAGES = 1000000
    MAX_DEPTH = 10
    PER_HOST_RPS = 2.0
    HEADLESS = True
    NAV_TIMEOUT_MS = 30000
    WAIT_SELECTOR = "body"
    POPPLER_PATH = r"C:\Users\gpaki\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"
    OCR_LANGS = ("th","en")

# ===== Shared helpers =====
SEEN_SHA1: set[str] = set()

def ensure_dirs():
    for p in (CFG.OUT_HTML, CFG.OUT_PDF, CFG.OUT_IMG, CFG.OUT_STAGE):
        os.makedirs(p, exist_ok=True)

def sha1_text(s: str) -> str: return hashlib.sha1(s.encode("utf-8")).hexdigest()

def host_allowed(u: str) -> bool:
    try:
        host = urlparse(u).hostname or ""
        return any(host == h or host.endswith("."+h) for h in CFG.ALLOWED_HOSTS)
    except Exception:
        return False

def safe_urljoin(base: str, href: Optional[str]) -> Optional[str]:
    if not href: return None
    href = href.strip()
    if not href or href.startswith(("#","mailto:","tel:","javascript:","data:")):
        return None
    try:
        u = urljoin(base, href)
        urlparse(u)
        return u
    except Exception:
        return None

_TRACKING_PARAMS = {"utm_source","utm_medium","utm_campaign","utm_term","utm_content","utm_id","fbclid","gclid","_gl","ref"}

def canonicalize_url(url: str, keep_params: set[str] | None = None, drop_params: set[str] | None = None, force_https: bool = True) -> str:
    if not url: return url
    try: u = urlsplit(url)
    except Exception: return url
    scheme = (u.scheme or "").lower()
    if force_https and scheme in ("http","https"):
        url = urlunsplit(("https", u.netloc, u.path, u.query, ""))
    url2 = w3_canonicalize_url(url, keep_fragments=False)
    if keep_params:
        cleaned = url_query_cleaner(url2, parameterlist=list(keep_params), remove=False)
    else:
        drops = _TRACKING_PARAMS | set(drop_params or [])
        cleaned = url_query_cleaner(url2, parameterlist=list(drops), remove=True)
    try:
        s = urlsplit(cleaned)
        path = s.path or "/"
        if path.endswith(("/index.html","/index.htm")):
            path = path.rsplit("/",1)[0] + "/"
        cleaned = urlunsplit((s.scheme, s.netloc, path, s.query, ""))
    except Exception: pass
    return cleaned

def canonical_from_html(html: str, base: str) -> Optional[str]:
    soup = BeautifulSoup(html, "lxml")
    link = soup.find("link", attrs={"rel": re.compile("^canonical$", re.I)})
    if link and link.get("href"): return safe_urljoin(base, link.get("href"))
    return None

def append_jsonl(path: str, obj: dict):
    import os, json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

class RateLimiter:
    def __init__(self, rps: float):
        self.delay = 1.0 / max(0.1, rps)
        self._last: dict[str,float] = {}
        self._lock = asyncio.Lock()
    async def wait(self, host: str):
        async with self._lock:
            now_t = time.monotonic(); t = self._last.get(host, 0.0)
            delta = now_t - t
            if delta < self.delay: await asyncio.sleep(self.delay - delta)
            self._last[host] = time.monotonic()

RATE = RateLimiter(CFG.PER_HOST_RPS)

async def fetch_html_playwright(page, url: str) -> Optional[str]:
    try:
        await RATE.wait(urlparse(url).hostname or "")
        resp = await page.goto(url, wait_until="domcontentloaded", timeout=CFG.NAV_TIMEOUT_MS)
        if not resp or resp.status >= 400:
            log("WARNING","page_non200", url=url, status=(resp.status if resp else None)); return None
        raw_ct = await resp.header_value("content-type")
        ct = (raw_ct or "").split(";",1)[0].strip().lower()
        if (not ct) or (ct in {"text/html","application/xhtml+xml","text/plain"}):
            try:
                await page.wait_for_selector(CFG.WAIT_SELECTOR, state="attached", timeout=CFG.NAV_TIMEOUT_MS)
            except Exception as e:
                log("WARNING","wait_selector_timeout", url=url, selector=CFG.WAIT_SELECTOR, err=str(e)); return None
            return await page.content()
        log("INFO","skip_non_html", url=url, content_type=(ct or "<none>")); return None
    except Exception as e:
        log("ERROR","goto_error", url=url, err=str(e)); return None

async def fetch_text(session: aiohttp.ClientSession, url: str, timeout: int = 20) -> Optional[str]:
    try:
        async with session.get(url, timeout=timeout) as r:
            if r.status == 200: return await r.text()
            return None
    except Exception: return None

async def discover_sitemaps(session: aiohttp.ClientSession, origin: str) -> List[str]:
    robots_url = urljoin(origin, "/robots.txt")
    text = await fetch_text(session, robots_url)
    maps: set[str] = set()
    if text:
        for line in text.splitlines():
            if line.lower().startswith("sitemap:"):
                maps.add(line.split(":",1)[1].strip())
    for path in ("/sitemap.xml","/sitemap_index.xml","/wp-sitemap.xml"):
        maps.add(urljoin(origin, path))
    return list(maps)

async def parse_sitemap(session: aiohttp.ClientSession, sm_url: str) -> List[str]:
    xml = await fetch_text(session, sm_url)
    if not xml or "<" not in xml: return []
    urls: List[str] = []
    for m in re.finditer(r"<loc>(.*?)</loc>", xml, re.I | re.S):
        u = m.group(1).strip()
        if host_allowed(u): urls.append(u)
    return urls[:20000]

@dataclass
class PageItem:
    url: str
    depth: int

async def gather_seeds(origins: List[str]) -> List[str]:
    urls: List[str] = []
    async with aiohttp.ClientSession(headers={"User-Agent":"RAG-Ingest/1.0"}) as s:
        for origin in origins:
            maps = await discover_sitemaps(s, origin)
            for sm in maps:
                urls += await parse_sitemap(s, sm)
    urls += list(CFG.START_URLS)
    out, seen = [], set()
    for u in urls:
        cu = canonicalize_url(u)
        if host_allowed(cu) and cu not in seen:
            seen.add(cu); out.append(cu)
    return out

# ---- link/pdf/image discovery (เนเธเนเนเธ Step 1 เน€เธเธทเนเธญเธเธขเธฒเธขเธเธดเธงเธฅเธดเธเธเน)
PDF_URL_PAT = re.compile(r"\.pdf(?:[?#].*)?$", re.I)
IMG_URL_PAT = re.compile(r"\.(?:png|jpe?g|webp|gif|bmp|tiff?)(?:[?#].*)?$", re.I)

def extract_links_pdfs_imgs(html: str, base: str) -> Tuple[List[str], List[str], List[Tuple[str, Dict[str,str]]]]:
    soup = BeautifulSoup(html, "lxml")
    links: List[str] = []; pdfs: List[str] = []; images: List[Tuple[str, Dict[str,str]]] = []
    def _add(u: Optional[str]) -> Optional[str]:
        if u and host_allowed(u): return u
        return None
    for a in soup.find_all("a"):
        u = safe_urljoin(base, a.get("href"));
        if not u: continue
        if host_allowed(u):
            links.append(u)
            if PDF_URL_PAT.search(u): pdfs.append(u)
    for img in soup.find_all("img"):
        alt = (img.get("alt") or "").strip()
        src = _add(safe_urljoin(base, img.get("src")))
        if src and IMG_URL_PAT.search(src):
            images.append((src, {"alt": alt, "src_hint": img.get("src") or ""}))
        ss = img.get("srcset")
        if ss:
            for part in ss.split(","):
                cand = part.strip().split(" ")[0]
                u2 = _add(safe_urljoin(base, cand))
                if u2 and IMG_URL_PAT.search(u2):
                    images.append((u2, {"alt": alt, "src_hint": cand}))
    for s in soup.select("picture source[srcset]"):
        ss = s.get("srcset") or ""
        for part in ss.split(","):
            cand = part.strip().split(" ")[0]
            u3 = _add(safe_urljoin(base, cand))
            if u3 and IMG_URL_PAT.search(u3): images.append((u3, {"alt":"","src_hint":cand}))
    for el in soup.select("[style*='background-image']"):
        st = el.get("style") or ""; m = re.search(r"url\(([^\)]+)\)", st)
        if m:
            cand = m.group(1).strip(" ' \""); u4 = _add(safe_urljoin(base, cand))
            if u4 and IMG_URL_PAT.search(u4): images.append((u4, {"alt":"","src_hint":cand}))
    for el in soup.find_all(True):
        for a in ("data-href","data-src"):
            u = safe_urljoin(base, el.get(a))
            if u and host_allowed(u) and PDF_URL_PAT.search(u): pdfs.append(u)
    def _strip_frag(u: str) -> str:
        try: p = urlparse(u); return p._replace(fragment="").geturl()
        except Exception: return u
    links = list(dict.fromkeys(_strip_frag(u) for u in links))
    pdfs = list(dict.fromkeys(pdfs))
    seen: set[str] = set(); imgs: List[Tuple[str, Dict[str,str]]] = []
    for u, meta in images:
        u2 = _strip_frag(u)
        if u2 in seen: continue
        seen.add(u2); imgs.append((u2, meta))
    return links, pdfs, imgs

# ====== Step 1: เธ”เธถเธ HTML + เธเธฑเธเธ—เธถเธ manifest ======
MAX_PAGES = None  # เน€เธเนเธ 200
MAX_DEPTH = None  # เน€เธเนเธ 2

async def step1_fetch_html(max_pages: int | None = None, max_depth: int | None = None):
    ensure_dirs()
    origins = sorted({f"https://{h}" for h in CFG.ALLOWED_HOSTS})
    seeds = await gather_seeds(origins)
    log("INFO","seeds_ready", count=len(seeds))

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=CFG.HEADLESS)
        ctx = await browser.new_context(accept_downloads=True, ignore_https_errors=True)
        page = await ctx.new_page()
        q: asyncio.Queue[PageItem] = asyncio.Queue()
        seen: set[str] = set()
        assets_seen: set[str] = set()
        # Seed queue (expand XML sitemaps into concrete page URLs first)
        async with aiohttp.ClientSession(headers={"User-Agent":"RAG-Ingest/1.0"}) as sx:
            for u in seeds:
                cu = canonicalize_url(u)
                if cu.lower().endswith(".xml"):
                    try:
                        pending = [cu]
                        visited_xml: set[str] = set()
                        while pending:
                            x = pending.pop()
                            if x in visited_xml:
                                continue
                            visited_xml.add(x)
                            try:
                                child_urls = await parse_sitemap(sx, x)
                            except Exception:
                                child_urls = []
                            for c in child_urls:
                                cc = canonicalize_url(c)
                                if cc.lower().endswith(".xml"):
                                    pending.append(cc)
                                elif host_allowed(cc):
                                    await q.put(PageItem(url=cc, depth=0))
                    except Exception as e:
                        log("WARNING","seed_xml_expand_failed", url=cu, err=str(e))
                else:
                    await q.put(PageItem(url=cu, depth=0))
        pages_done = 0
        depth_limit = CFG.MAX_DEPTH if max_depth is None else max_depth
        page_limit  = CFG.MAX_PAGES if max_pages is None else max_pages
        while not q.empty() and pages_done < page_limit:
            item = await q.get()
            if item.url in seen or not host_allowed(item.url):
                continue
            seen.add(item.url)
            html = await fetch_html_playwright(page, item.url)
            if not html:
                continue
            canon = canonicalize_url(canonical_from_html(html, item.url) or item.url)
            html_path = os.path.join(CFG.OUT_HTML, f"{sha1_text(canon)}.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
            append_jsonl(CFG.MANIFEST, {"record": "page", "url": item.url, "canonical": canon, "html_path": html_path, "html_path_abs": os.path.abspath(html_path), "ts": now()})
            # extract links and assets from this page
            links, pdfs, images = extract_links_pdfs_imgs(html, canon)

            # download & index assets immediately (RAW-first design)
            async with aiohttp.ClientSession(headers={"User-Agent":"RAG-Ingest/1.0"}) as dl:
                # PDFs
                for pu in pdfs:
                    pu_c = canonicalize_url(pu)
                    if pu_c in assets_seen:
                        continue
                    try:
                        await RATE.wait(urlparse(pu_c).hostname or "")
                        async with dl.get(pu_c, headers={"Referer": canon}, timeout=30) as r:
                            if r.status != 200:
                                log("WARNING","pdf_download_non200", url=pu_c, status=r.status)
                                continue
                            data = await r.read()
                    except Exception as e:
                        log("ERROR","pdf_download_failed", url=pu_c, err=str(e))
                        continue
                    pdf_path = os.path.join(CFG.OUT_PDF, f"{sha1_text(pu_c)}.pdf")
                    with open(pdf_path, "wb") as pf:
                        pf.write(data)
                    append_jsonl(CFG.MANIFEST, {"record":"asset","asset_type":"pdf","url":pu_c,"path":pdf_path,"page":canon,"ts":now()})
                    assets_seen.add(pu_c)

                # Images
                for iu, meta in images:
                    iu_c = canonicalize_url(iu)
                    if iu_c in assets_seen:
                        continue
                    try:
                        await RATE.wait(urlparse(iu_c).hostname or "")
                        async with dl.get(iu_c, headers={"Referer": canon}, timeout=30) as r:
                            if r.status != 200:
                                log("WARNING","img_download_non200", url=iu_c, status=r.status)
                                continue
                            data = await r.read()
                    except Exception as e:
                        log("ERROR","img_download_failed", url=iu_c, err=str(e))
                        continue
                    ext = os.path.splitext(urlparse(iu_c).path)[1].lower() or ".bin"
                    img_path = os.path.join(CFG.OUT_IMG, f"{sha1_text(iu_c)}{ext}")
                    with open(img_path, "wb") as fimg:
                        fimg.write(data)
                    append_jsonl(CFG.MANIFEST, {"record":"asset","asset_type":"image","url":iu_c,"path":img_path,"alt":meta.get("alt",""),"page":canon,"ts":now()})
                    assets_seen.add(iu_c)

            # Enqueue new links if within depth limit
            if item.depth < depth_limit:
                for lk in links:
                    lk_c = canonicalize_url(lk)
                    if lk_c not in seen and host_allowed(lk_c):
                        await q.put(PageItem(url=lk_c, depth=item.depth+1))
            pages_done += 1
            if pages_done % 10 == 0:
                log("INFO","progress", pages=pages_done, queue=q.qsize())
        await browser.close()
        log("INFO","step1_done", pages=pages_done)

# ===== Run Step 1 =====
import asyncio as _asyncio
_asyncio.run(step1_fetch_html(max_pages=MAX_PAGES, max_depth=MAX_DEPTH))
