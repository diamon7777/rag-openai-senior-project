# 68B21a - Retrieval-Augmented Generation for Chulalongkorn University Information

Senior Project in Information Technology for Business, Chulalongkorn University.

This project implements a retrieval-augmented generation (RAG) workflow for collecting public university information, extracting Thai and English text from HTML pages, images, and PDF files, indexing the resulting documents, and generating answers with links to the original sources.

## Project objectives

- Collect information from approved university websites.
- Support HTML pages, infographic images, text-based PDFs, and scanned PDFs.
- Preserve source URLs throughout the processing pipeline.
- Improve search relevance through vector retrieval followed by reranking.
- Generate answers grounded in retrieved evidence with source citations.

## Repository structure

```text
.
|-- README.md
|-- requirements.txt
|-- .env.example
|-- .gitignore
|-- notebooks/
|   |-- 01_web_crawling.ipynb
|   |-- 02_document_processing.ipynb
|   `-- 03_rag_pipeline.ipynb
|-- src/
|   |-- web_crawling.py
|   |-- document_processing.py
|   `-- rag_pipeline.py
|-- sample_data/
|   `-- RESULTS.md
`-- report/
    `-- senior_project_68B21A.pdf
```

The notebooks provide the complete project workflow with explanations for each stage. Matching files under `src/` provide a Python view for readers who prefer source files or cannot render Jupyter notebooks on GitHub.

## Workflow

### 1. Web crawling

[`01_web_crawling.ipynb`](notebooks/01_web_crawling.ipynb) collects raw information from configured university domains.

The crawler:

- discovers page URLs through seed URLs and XML sitemaps;
- renders JavaScript-based pages with Playwright;
- canonicalizes URLs and restricts crawling to allowed hosts;
- applies a per-host request-rate limiter;
- extracts links to internal pages, PDF files, and images;
- stores raw HTML, PDF, and image files; and
- records page and asset metadata in `manifest.jsonl`.

### 2. Document processing

[`02_document_processing.ipynb`](notebooks/02_document_processing.ipynb) converts the crawl output into normalized records in `documents.jsonl`.

The processing stage contains four parts:

1. shared configuration, file helpers, content hashing, and EasyOCR setup;
2. HTML extraction using Trafilatura and Beautiful Soup;
3. infographic filtering and Thai-English image OCR using OpenCV and EasyOCR; and
4. PDF extraction using PyMuPDF4LLM, PyMuPDF, Poppler, and page-level OCR fallbacks.

Each accepted document retains its source URL, source type, extracted content, content hash, capture time, and relevant format metadata.

### 3. RAG pipeline

[`03_rag_pipeline.ipynb`](notebooks/03_rag_pipeline.ipynb) prepares the extracted documents for semantic search and answer generation.

The RAG stage performs:

1. profile-based token chunking for structured and OCR-derived text;
2. normalization into `embeddable.jsonl`;
3. BGE-M3 embedding and storage in ChromaDB;
4. vector deletion and update by source URL;
5. vector retrieval;
6. reranking with `BAAI/bge-reranker-v2-m3`; and
7. grounded answer generation with `gpt-5-mini` and source URLs.

## Data flow

```text
Approved university websites
            |
            v
Web crawler -> raw HTML / PDF / images -> manifest.jsonl
            |
            v
Extraction and OCR -> documents.jsonl
            |
            v
Token chunking -> chunks.jsonl -> embeddable.jsonl
            |
            v
BGE-M3 -> ChromaDB -> vector retrieval -> BGE reranking
            |
            v
Grounded answer with source URLs
```

## Project results

The full project run collected 3,367 manifest page records and produced 2,473 documents containing more than 35.5 million characters. The documents generated 108,893 chunks, all of which were retained for embedding and stored in ChromaDB.

| Result | Value |
|---|---:|
| Manifest page records | 3,367 |
| Extracted documents | 2,473 |
| Generated chunks | 108,893 |
| Embeddable records | 108,893 |
| Chroma collection size | 108,893 |
| Best vector retrieval score | 0.8810 |
| Best reranker score | 0.9964 |

Reranking promoted official registration manuals and regulations above repeated landing-page passages. Detailed distributions and evaluation examples are available in [`sample_data/RESULTS.md`](sample_data/RESULTS.md), and the complete senior-project report is included at [`report/senior_project_68B21A.pdf`](report/senior_project_68B21A.pdf).

## Technology stack

| Area | Libraries / models |
|---|---|
| Web crawling | Playwright, aiohttp, Beautiful Soup |
| HTML extraction | Trafilatura |
| PDF processing | PyMuPDF4LLM, PyMuPDF, Poppler |
| OCR | EasyOCR, OpenCV, Thai and English recognition |
| Chunking | tiktoken |
| Embeddings | BAAI/bge-m3, Sentence Transformers |
| Vector database | ChromaDB |
| Reranking | BAAI/bge-reranker-v2-m3 |
| Answer generation | OpenAI `gpt-5-mini` |

## Setup

Python 3.12 is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
pip install -r requirements.txt
playwright install chromium
```

Poppler is required for the PDF fallback path. Set `POPPLER_PATH` to its `bin` directory when needed.

Review the configuration paths in each notebook before execution. Configuration settings for document processing and the RAG stages are listed in `.env.example` and can be supplied through the shell or notebook environment where supported by the corresponding stage.

## Running the project

Open the notebooks in numerical order:

1. `notebooks/01_web_crawling.ipynb`
2. `notebooks/02_document_processing.ipynb`
3. `notebooks/03_rag_pipeline.ipynb`

The equivalent Python source is available under `src/`.

Running the complete workflow can trigger a large live crawl, OCR processing, model downloads, GPU or CPU workloads, ChromaDB writes, and OpenAI API usage. Review the configured domains, paths, and resource requirements before execution.

Generated crawl assets, intermediate JSONL files, model files, and the ChromaDB index are runtime data and are not stored in this repository. A new crawl may produce different totals as university websites change over time.

## Security

No API credential is included in the repository. Supply a valid OpenAI key through a secure local runtime mechanism before executing answer-generation cells, and never commit a real key to a notebook or source file.
