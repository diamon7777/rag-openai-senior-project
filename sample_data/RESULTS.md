# Project Results

This page summarizes the data collection, document processing, vector indexing, retrieval, and answer-generation results from the senior project. The complete methodology and evaluation are available in [`../report/senior_project_68B21A.pdf`](../report/senior_project_68B21A.pdf).

## Data collection

The collection run was conducted on 4 October 2025 from 00:15 to 02:54.

- Manifest page records: **3,367**
- Extracted documents: **2,473**
- Total characters: **35,502,176**
- Approximate tokens: **8,876,487**
- Average characters per document: **14,355.9**
- Approximate tokens per document: **3,589.4**

The manifest and document totals measure different stages: the manifest records collected pages and assets, while the document total contains records accepted after extraction and filtering.

## Documents by source type

| Type | Documents | Share |
|---|---:|---:|
| HTML | 1,087 | 43.95% |
| PDF / Markdown | 623 | 25.19% |
| Image OCR | 763 | 30.86% |
| **Total** | **2,473** | **100%** |

## Language and content distribution

- Thai documents: **1,797 (72.66%)**
- English documents: **676 (27.34%)**
- Documents with detected tables: **174 (7.04%)**
- PDFs with detected tables: **27.93%**

## Documents by domain

| Domain | Documents | Share |
|---|---:|---:|
| `it.chula.ac.th` | 937 | 37.89% |
| `sa.chula.ac.th` | 840 | 33.97% |
| `reg.chula.ac.th` | 690 | 27.90% |
| Other domains | 6 | 0.24% |
| **Total** | **2,473** | **100%** |

## Chunking and vector indexing

| Metric | Result |
|---|---:|
| Input documents | 2,473 |
| Generated chunks | 108,893 |
| Embeddable records retained | 108,893 |
| Chroma collection size | 108,893 |

### Chunks by processing profile

| Profile | Chunks | Share |
|---|---:|---:|
| OCR / unstructured | 91,744 | 84.25% |
| Structured | 17,149 | 15.75% |
| **Total** | **108,893** | **100%** |

### Chunks by source type

| Source type | Chunks | Share |
|---|---:|---:|
| HTML | 91,678 | 84.19% |
| PDF / Markdown | 15,733 | 14.45% |
| Image OCR | 1,482 | 1.36% |
| **Total** | **108,893** | **100%** |

### Chunks by language

| Language | Chunks | Share |
|---|---:|---:|
| Thai | 105,142 | 96.56% |
| English | 3,751 | 3.44% |
| **Total** | **108,893** | **100%** |

## Retrieval and reranking

The evaluation used a Thai question about the Chulalongkorn University course-registration procedure.

### Vector retrieval

- Highest vector score: **0.8810**
- The highest-ranked candidates included Registrar information pages relevant to course registration.

### BGE reranking

- Highest reranker score: **0.9964**
- Highest-ranked source: `CR99_2566_T_S.pdf`
- Following recorded scores included **0.9907**, **0.9871**, and **0.9807**.

The cross-encoder reranker promoted official registration manuals and regulations above repeated landing-page passages, producing more focused evidence for answer generation.

## Answer-generation evaluation

The RAG pipeline was evaluated with Thai and English questions covering:

- course registration procedures;
- course withdrawal procedures;
- examination schedule availability;
- Chula email use;
- password reset procedures;
- scholarship applications and eligibility; and
- graduation procedures.

The answer-generation prompt required the model to use only retrieved context, state when information was unavailable, and include the source URLs supporting the response.

## Vector deletion and updates

- A source-level deletion test removed **5 vectors** associated with one IT Chula PDF URL.
- An update-processing case generated **1,280 chunks** for refreshed content.

These operations demonstrate maintenance of the vector collection when a source is removed or updated.

## Data availability

The repository contains the project notebooks, Python source views, dependency list, result summary, and senior-project report. Generated website files, processed JSONL datasets, downloaded model files, and the persistent ChromaDB index are not distributed with the source repository.

Because the input websites are live external sources, a new crawl may produce different document and chunk totals as their content changes.
