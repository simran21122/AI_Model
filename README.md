# VectorDB Studio — A Vector Database + RAG Pipeline Built From Scratch in Python

A fully working **vector database** written from scratch in Python (Flask + NumPy), with a single-file web UI.
It implements three nearest-neighbour search algorithms — **HNSW**, **KD-Tree** and **Brute Force** — three distance
metrics, a **PCA scatter-plot visualizer**, and a complete **RAG (Retrieval-Augmented Generation)** pipeline that works
with either the **Google Gemini API** (free tier, works on cloud hosts) or a **local Ollama** install (works offline on a laptop).

> An educational project that shows how production vector databases such as Pinecone, Weaviate, Chroma and Milvus work under the hood.

![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![Flask](https://img.shields.io/badge/flask-3.x-lightgrey) ![License](https://img.shields.io/badge/license-MIT-green)

---

## Table of Contents

1. [Features](#1-features)
2. [How It Works](#2-how-it-works)
3. [Project Structure](#3-project-structure)
4. [Prerequisites](#4-prerequisites)
5. [Quick Start](#5-quick-start)
6. [Configuration (Environment Variables)](#6-configuration-environment-variables)
7. [Choosing an AI Provider: Gemini vs Ollama](#7-choosing-an-ai-provider-gemini-vs-ollama)
8. [Using the Application](#8-using-the-application)
9. [Permanent Knowledge Base (`knowledge/`)](#9-permanent-knowledge-base-knowledge)
10. [REST API Reference](#10-rest-api-reference)
11. [Architecture Deep Dive](#11-architecture-deep-dive)
12. [Algorithm Deep Dive](#12-algorithm-deep-dive)
13. [Deploying to the Cloud (Render)](#13-deploying-to-the-cloud-render)
14. [Security & Safety Notes](#14-security--safety-notes)
15. [Limitations & Design Trade-offs](#15-limitations--design-trade-offs)
16. [Troubleshooting](#16-troubleshooting)
17. [Ideas for Extending the Project](#17-ideas-for-extending-the-project)
18. [License](#18-license)

---

## 1. Features

| Feature | Description |
|---|---|
| **3 search algorithms** | HNSW (approximate, production-grade), KD-Tree (exact for Euclidean), Brute Force (exact baseline) — run them side-by-side |
| **3 distance metrics** | Cosine distance, Euclidean (L2), Manhattan (L1) |
| **16-D demo index** | 20 pre-loaded semantic vectors across 4 categories: Computer Science, Math, Food, Sports |
| **2-D PCA scatter plot** | Live visualisation of the semantic space, computed in the browser — watch the clusters form |
| **Benchmark mode** | One click runs all three algorithms on the same query and compares latency in microseconds |
| **HNSW inspector** | `/hnsw-info` exposes the layered graph: top layer, nodes and edges per layer, each node's max layer |
| **Real document embeddings** | Paste any text → it is chunked, embedded (768-D) and stored in a separate HNSW index |
| **RAG pipeline** | Ask questions → question embedded → top-k chunks retrieved → LLM answers using those chunks |
| **Two AI back-ends** | Google Gemini API (`gemini-embedding-001` + `gemini-flash-latest`) or local Ollama (`nomic-embed-text` + `llama3.2`) |
| **Permanent knowledge base** | Every `.txt` / `.md` file in `knowledge/` is embedded automatically at startup |
| **Public-demo safeguards** | `READ_ONLY` mode, per-IP rate limiting, request size cap, chunk-count cap |
| **Full REST API** | Insert, delete, search, benchmark, graph inspection, document CRUD, RAG |
| **Zero build step** | Pure Python backend + one `index.html` file. No Node, no bundler, no compiler |

---

## 2. How It Works

```
                       ┌─────────────────────────────────────────────┐
                       │                 Browser UI                  │
                       │  index.html  (Search · Documents · Ask AI)  │
                       │  PCA scatter plot, benchmark, chat, graph   │
                       └───────────────┬─────────────────────────────┘
                                       │  fetch() — same origin
                                       ▼
┌──────────────────────────────  Flask server (main.py)  ──────────────────────────────┐
│                                                                                      │
│  16-D DEMO PATH                               DOCUMENT / RAG PATH                    │
│  ───────────────                              ───────────────────                    │
│  VectorDB                                     DocumentDB                             │
│   ├─ BruteForce  (exact)                       ├─ HNSW  (cosine)                     │
│   ├─ KDTree      (exact, L2)                   └─ BruteForce (used when < 10 chunks) │
│   └─ HNSW        (approximate)                                                       │
│                                                       ▲                              │
│                                                       │ embeddings (768-D)           │
│                                               ┌───────┴────────┐                     │
│                                               │   AIClient     │                     │
│                                               └───────┬────────┘                     │
└───────────────────────────────────────────────────────┼──────────────────────────────┘
                                                        │
                          GEMINI_API_KEY set? ──yes──►  Google Gemini API  (embed + generate)
                                              └──no───► Local Ollama       (embed + generate)
```

### The RAG flow, step by step

```
1.  Your text      ─►  split into overlapping chunks (250 words, 30-word overlap)
2.  Each chunk     ─►  embedding model  ─►  768-D vector  ─►  stored in the HNSW index
3.  Your question  ─►  embedding model  ─►  768-D query vector
4.  HNSW search    ─►  top-k nearest chunks (cosine distance ≤ 0.7)
5.  Prompt         ─►  "Context: [1] title: text … Question: …"
6.  LLM            ─►  natural-language answer + the chunks it was shown
```

**HNSW (Hierarchical Navigable Small World)** is the algorithm behind most production vector databases. It builds a
multi-layer graph: upper layers are sparse "express lanes" that jump quickly to the right neighbourhood, and the bottom
layer contains every point for the final precise search — giving roughly **O(log N)** search time instead of **O(N)**.

---

## 3. Project Structure

```
VectorDB/
├── main.py                  ← Entire backend: algorithms, DBs, AI client, REST API, knowledge loader
├── index.html               ← Entire frontend: UI, PCA, scatter plot, benchmark, chat (no build step)
├── requirements.txt         ← flask, numpy, requests, gunicorn
├── knowledge/
│   └── vector-databases-intro.txt   ← Auto-embedded at startup (permanent knowledge)
├── .env                     ← LOCAL secrets/config (git-ignored — never commit)
├── .gitignore
├── LICENSE                  ← MIT
└── README.md                ← This file
```

### Inside `main.py`

| Component | Role | Complexity / Notes |
|---|---|---|
| `euclidean`, `cosine`, `manhattan` | Distance functions (`get_dist_fn` selects by name; unknown names fall back to Euclidean) | Cosine returns `1 − cos θ`; zero vectors return `1.0` |
| `VectorItem` | Dataclass: `id`, `metadata`, `category`, `emb` | |
| `BruteForce` | Compares the query to every vector | `O(N·d)` + sort — exact baseline |
| `KDNode` / `KDTree` | Axis-aligned binary space partition, iterative insert | Exact for Euclidean; rebuilt on delete |
| `HNSW` | Multi-layer small-world graph (`M=16`, `M0=32`, `ef_build=200`, query `ef=50`) | Approximate, ≈ `O(log N)` |
| `VectorDB` | Unified, thread-safe wrapper over all three indexes (16-D demo data) | Single `threading.Lock` |
| `chunk_text` | 250-word windows with 30-word overlap | Short texts stay as one chunk |
| `AIClient` | Embeddings + generation via Gemini **or** Ollama; retry/back-off; query-embedding cache | See [section 7](#7-choosing-an-ai-provider-gemini-vs-ollama) |
| `DocItem` / `DocumentDB` | Stores real document chunks; HNSW for ≥ 10 chunks, brute force below that | Distance cut-off `max_dist = 0.7` |
| `DEMO` / `load_demo` | The 20 hand-crafted 16-D vectors | dims 0-3 CS · 4-7 Math · 8-11 Food · 12-15 Sports |
| Flask routes | REST API + serves `index.html` | CORS enabled, 512 KB body cap |
| `ingest_knowledge` | Background thread that embeds `knowledge/*.txt|md` at startup | Throttled for the Gemini free tier |

---

## 4. Prerequisites

| Requirement | Version | Needed for |
|---|---|---|
| **Python** | 3.9 or newer | Everything |
| **pip** | bundled with Python | Installing dependencies |
| **Git** | any recent | Cloning the repo (optional if you download the ZIP) |
| **A Gemini API key** *(option A)* | free at <https://aistudio.google.com/apikey> | RAG with the Gemini back-end |
| **Ollama** *(option B)* | <https://ollama.com> | RAG fully offline, no API key |

You only need **one** of the two AI options. The 16-D Search tab works with **neither** — it needs no AI at all.

---

## 5. Quick Start

### 5.1 Get the code

```bash
git clone https://github.com/simran21122/AI_Model.git
cd AI_Model          # or the VectorDB folder if you unzipped the archive
```

### 5.2 Create a virtual environment (recommended)

**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 5.3 Install dependencies

```bash
pip install -r requirements.txt
```

Installs **Flask** (HTTP server), **NumPy** (vector maths), **Requests** (talks to Gemini/Ollama) and **Gunicorn** (production
server, used only for deployment). Takes ~30 seconds, no compile step.
If `pip` is not found, use `python -m pip install -r requirements.txt`.

### 5.4 Configure an AI provider

**Option A — Gemini (recommended; also what you need for cloud hosting)**

Create a file named `.env` next to `main.py`:

```ini
GEMINI_API_KEY=your_key_here
# Optional overrides:
# GEN_MODEL=gemini-flash-latest
# EMBED_MODEL=gemini-embedding-001
```

**Option B — Ollama (offline)**

```bash
ollama pull nomic-embed-text      # ~274 MB, embedding model
ollama pull llama3.2              # ~2 GB, language model
ollama list                       # verify both are installed
```
Leave `GEMINI_API_KEY` unset. Ollama must be running (`ollama serve`, or the tray app on Windows/macOS).

**Option C — No AI**
Skip this step. The Search tab and the whole 16-D demo work; Documents and Ask AI will report the provider as unavailable.

### 5.5 Run

```bash
python main.py
```

Expected output (Gemini example):

```
=== VectorDB Engine (Python) ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
AI provider: gemini - ONLINE
  embed model: gemini-embedding-001  gen model: gemini-flash-latest
[knowledge] loaded 1 chunks from knowledge/
```

Open **<http://localhost:8080>** in your browser.

---

## 6. Configuration (Environment Variables)

Variables are read from the real environment **and** from a local `.env` file. The built-in `.env` loader only sets a
variable if it is **not already set**, so real environment variables (e.g. on Render) always win. Quotes around values are stripped; lines starting with `#` are ignored.

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(unset)* | If set, the **Gemini** provider is used. If empty/unset, **Ollama** is used |
| `GEN_MODEL` | `gemini-flash-latest` (Gemini) / `llama3.2` (Ollama) | Language model for answers |
| `EMBED_MODEL` | `gemini-embedding-001` (Gemini) / `nomic-embed-text` (Ollama) | Embedding model |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Where the Ollama server listens |
| `GEMINI_BASE` | `https://generativelanguage.googleapis.com/v1beta/models` | Gemini REST base URL |
| `PORT` | `8080` | HTTP port (cloud hosts inject this automatically) |
| `READ_ONLY` | `0` | `1` = visitors cannot insert or delete anything (public-demo mode) |
| `RATE_LIMIT_PER_MIN` | `30` | Max AI-backed requests per IP per minute |

Hard-coded limits in `main.py`: 16 demo dimensions, `MAX_CHUNKS_PER_INSERT = 40` (~9,000 words), request body ≤ 512 KB,
question ≤ 2,000 characters, title ≤ 200 characters, document `k` clamped to 1–10.

> **Never commit `.env`.** It is listed in `.gitignore`. On a cloud host, enter these values in the host's *Environment* tab instead.

---

## 7. Choosing an AI Provider: Gemini vs Ollama

| | **Gemini** | **Ollama** |
|---|---|---|
| Setup | API key only | Install app + pull ~2.3 GB of models |
| Works on cloud hosts (Render, etc.) | Yes | No — needs a machine running Ollama |
| Needs internet | Yes | No |
| Cost | Free tier (rate-limited) | Free (your hardware) |
| Embedding dimensions | 768 (requested via `outputDimensionality` to save RAM) | 768 (`nomic-embed-text`) |
| Query/document task types | Uses `RETRIEVAL_QUERY` for questions and `RETRIEVAL_DOCUMENT` for chunks | Single embedding mode |
| Resilience | Auto-retries HTTP 429/500/502/503/504 with 3 s, 6 s, 9 s back-off | Fails fast with an install hint |
| Speed | ~1–3 s per answer | 10–30 s per answer on a laptop CPU |

Other behaviour worth knowing:

* **Query-embedding cache** — `/doc/search` and `/doc/ask` embed the same question; the second call is served from a 256-entry cache, halving API usage.
* **Gemini availability check** does not spend quota: `is_available()` simply returns `true` when a key is configured. Real errors (bad key, quota exhausted, safety block) surface in the UI as the actual API message.
* **Do not mix embedding models.** Vectors from different models are not comparable. Because storage is in-memory, restarting after changing `EMBED_MODEL` or the provider rebuilds everything cleanly.
* **Faster Ollama answers:** `ollama pull llama3.2:1b` and set `GEN_MODEL=llama3.2:1b`.

---

## 8. Using the Application

The UI has three tabs and a status badge (`GEMINI ●` or `OLLAMA ●`, green when ready, red on error).

### Tab 1 — SEARCH (16-D demo vectors)

1. Type a concept: `binary tree`, `sushi`, `basketball`, `calculus`, `probability`.
2. Pick an **algorithm**: `HNSW`, `KD-TREE` or `BRUTE`.
3. Pick a **metric**: Cosine, Euclidean or Manhattan.
4. Click **SEARCH** — nearest results appear with their distances, and the matching points glow on the scatter plot.
5. Click the **benchmark** button to run all three algorithms on the same query and compare latencies.
6. Use **+ INSERT** to add your own vector to the 16-D index; the ✕ button deletes a vector.

**How a typed phrase becomes a 16-D vector:** the Search tab does *not* call an AI model. `index.html` contains a
small keyword dictionary for the four categories (CS, Math, Food, Sports). Matching words raise the score of that category's
four dimensions (plus a little random jitter), and everything else stays at a 0.08 baseline. This is a deliberate teaching
device so the 16-D index works with zero setup — and it means words outside the dictionary produce a flat, uninformative vector.

**The scatter plot** projects all vectors from 16-D to 2-D with a hand-written **PCA** (power iteration, in JavaScript). The four
semantic categories form distinct clusters — that is what "semantic similarity" looks like.

### Tab 2 — DOCUMENTS (real embeddings)

1. Enter a title (e.g. `Operating Systems Notes`).
2. Paste any text — lecture notes, articles, documentation.
3. Click **EMBED & INSERT**.

Text longer than 250 words is split into overlapping chunks (30-word overlap) named `Title [1/4]`, `Title [2/4]`, … Each
chunk gets its own 768-D embedding and is inserted into the HNSW index. The response reports `chunks` and `dims`. Inserts above
40 chunks (~9,000 words) are rejected — split the text and insert in parts.

### Tab 3 — ASK AI (RAG)

1. Make sure some documents exist (your own, or the auto-loaded `knowledge/` files).
2. Ask a question and click **ASK AI**.
3. The answer appears together with **context chips** showing which chunks were retrieved and their cosine distances; click one to read the exact passage the model saw.

If no stored chunk is close enough (cosine distance > 0.7), the prompt contains no context and the model answers from its
general knowledge. The prompt also instructs the model not to say things like "the context doesn't mention…" and to describe
itself as an AI assistant that answers questions about the user's documents.

---

## 9. Permanent Knowledge Base (`knowledge/`)

Drop `.txt` or `.md` files into the `knowledge/` folder and they are embedded **automatically every time the server starts** —
so a freshly deployed server can answer questions immediately, even though document storage is in-memory.

* Runs in a **background thread**, so the web server is up instantly (also under Gunicorn).
* The chunk title is the filename with `_` and `-` turned into spaces (`vector-databases-intro.txt` → `vector databases intro`).
* With Gemini, a **0.7 s pause** between chunks keeps you under the free-tier requests-per-minute limit.
* If the provider is unavailable or an embedding fails, ingestion stops and prints a `[knowledge]` message to the console.
* Included example: `knowledge/vector-databases-intro.txt` — a short primer on embeddings, HNSW, KD-Trees, brute force and RAG. Try asking *"What is HNSW?"*

---

## 10. REST API Reference

Base URL: `http://localhost:8080`. All responses are JSON. CORS is open (`Access-Control-Allow-Origin: *`).
Validation errors return `{"error": "..."}` — note that several return **HTTP 200** with an `error` field, so check the body.

### 10.1 Demo vector endpoints (16-D)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,…,f16&k=5&metric=cosine&algo=hnsw` | k-NN search |
| `POST` | `/insert` | Insert a 16-D vector |
| `DELETE` | `/delete/<id>` | Delete a vector |
| `GET` | `/items` | List all vectors |
| `GET` | `/benchmark?v=…&k=5&metric=cosine` | Time all three algorithms |
| `GET` | `/hnsw-info` | HNSW graph structure |
| `GET` | `/stats` | Counts, dimensions, supported algorithms/metrics |

**`GET /search`** — parameters: `v` (exactly 16 comma-separated floats), `k` (default 5), `metric` = `cosine` \| `euclidean` \| `manhattan`
(unknown → Euclidean), `algo` = `hnsw` \| `kdtree` \| `bruteforce` (unknown → HNSW).

```json
{
  "results": [
    { "id": 1, "metadata": "Linked List: nodes connected by pointers", "category": "cs",
      "distance": 0.001234, "embedding": [0.9, 0.85, 0.72, "…16 values"] }
  ],
  "latencyUs": 182,
  "algo": "hnsw",
  "metric": "cosine"
}
```

**`POST /insert`**
```json
{ "metadata": "Merge Sort: divide and conquer", "category": "cs", "embedding": [/* exactly 16 floats */] }
```
→ `{ "id": 21 }`. Returns `{"error":"invalid body"}` if `metadata` is empty or the vector is not 16-D.

**`GET /benchmark`** → `{ "bruteforceUs": 95, "kdtreeUs": 140, "hnswUs": 210, "itemCount": 20 }`

**`GET /hnsw-info`** → `{ "topLayer", "nodeCount", "nodesPerLayer": [...], "edgesPerLayer": [...], "nodes": [{id, metadata, category, maxLyr}], "edges": [{src, dst, lyr}] }`

### 10.2 Document & RAG endpoints

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/doc/insert` | `{"title":"…","text":"…"}` | Chunk, embed and store a document |
| `GET` | `/doc/list` | — | List stored chunks (id, title, 120-char preview, word count) |
| `DELETE` | `/doc/delete/<id>` | — | Delete one chunk |
| `POST` | `/doc/search` | `{"question":"…","k":3}` | Retrieval only (no LLM) |
| `POST` | `/doc/ask` | `{"question":"…","k":3}` | Full RAG: retrieve + generate |
| `GET` | `/status` | — | Provider, models, counts, read-only flag |

**`POST /doc/insert`** → `{ "ids": [1, 2, 3], "chunks": 3, "dims": 768 }`

**`POST /doc/search`** → `{ "contexts": [ { "id": 2, "title": "Notes [2/3]", "distance": 0.2143 } ] }`

**`POST /doc/ask`**
```json
{
  "answer": "Dynamic programming solves problems by …",
  "model": "gemini-flash-latest",
  "contexts": [ { "id": 2, "title": "Notes [2/3]", "text": "…", "distance": 0.2143 } ],
  "docCount": 12
}
```

**`GET /status`**
```json
{ "ollamaAvailable": true, "provider": "gemini", "readOnly": false,
  "embedModel": "gemini-embedding-001", "genModel": "gemini-flash-latest",
  "docCount": 1, "docDims": 768, "demoDims": 16, "demoCount": 20 }
```
> The field is still called `ollamaAvailable` for backward compatibility with the UI; it reflects whichever provider is active.

### 10.3 Status codes

| Code | Meaning |
|---|---|
| `200` | Success **or** a validation error described in the `error` field |
| `403` | `READ_ONLY=1` and the request tried to insert/delete |
| `404` | `index.html` missing next to `main.py` |
| `413` | Request body larger than 512 KB |
| `429` | Per-IP rate limit exceeded (applies to `/doc/insert`, `/doc/search`, `/doc/ask`) |

### 10.4 Examples

**macOS / Linux**
```bash
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"

curl -X POST http://localhost:8080/doc/insert \
  -H "Content-Type: application/json" \
  -d '{"title":"DP Notes","text":"Dynamic programming stores results of overlapping subproblems..."}'

curl -X POST http://localhost:8080/doc/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is dynamic programming?","k":3}'
```

**Windows PowerShell** — `curl` is an alias for `Invoke-WebRequest`; use `curl.exe`:
```powershell
curl.exe "http://localhost:8080/status"
curl.exe -X POST http://localhost:8080/doc/ask -H "Content-Type: application/json" -d "{\"question\":\"What is HNSW?\",\"k\":3}"
```

---

## 11. Architecture Deep Dive

### 11.1 Two separate databases

| | `VectorDB` (demo) | `DocumentDB` (documents) |
|---|---|---|
| Dimensions | 16 (fixed) | Whatever the embedding model returns (768) |
| Algorithms | Brute Force + KD-Tree + HNSW, all kept in sync | HNSW (+ brute force kept for small sets) |
| Metric | Chosen per query | Cosine only |
| Data | 20 built-in vectors + user inserts | Chunks from the UI and `knowledge/` |
| Purpose | Learning / comparing algorithms | Real semantic search and RAG |

### 11.2 Concurrency

Flask runs with `threaded=True`. Each database owns a `threading.Lock`, so inserts, deletes and searches are serialised per database.
The rate limiter and query cache have their own locks. This is safe **within one process**; see
[Limitations](#15-limitations--design-trade-offs) for the multi-worker caveat.

### 11.3 Rate limiting

A tiny in-memory sliding window per client IP (`X-Forwarded-For` first hop, else `remote_addr`): at most `RATE_LIMIT_PER_MIN`
requests per 60 seconds. It protects your free-tier API quota from strangers. The table is cleared if it grows beyond 5,000 IPs.

### 11.4 Document search strategy

`DocumentDB.search` uses **brute force when there are fewer than 10 chunks** (exact, and faster than walking a tiny graph) and
**HNSW from 10 chunks up**. Results farther than cosine distance **0.7** are discarded so irrelevant chunks never reach the LLM.

### 11.5 Frontend

A single `index.html` (no framework): vanilla JS, canvas-drawn scatter plot, in-browser PCA, Fira Code font from Google Fonts and
Lucide icons from unpkg. It talks to the backend through `fetch(location.origin + path)`, so it works on any host/port with no configuration.

---

## 12. Algorithm Deep Dive

### 12.1 Distance metrics

| Metric | Formula | Notes |
|---|---|---|
| Cosine distance | `1 − (a·b) / (‖a‖‖b‖)` | Direction only, ignores magnitude — the standard for text embeddings |
| Euclidean | `√Σ(aᵢ − bᵢ)²` | Straight-line distance |
| Manhattan | `Σ|aᵢ − bᵢ|` | Grid / taxicab distance |

### 12.2 Brute Force — `O(N·d)`

Compute the distance to every stored vector, sort, return the top *k*. Always exact; the baseline everything else is compared against.

### 12.3 KD-Tree

A binary tree that splits space along one dimension per level, cycling through dimensions. Search descends to the closest
leaf, then backtracks, **pruning** any subtree whose splitting plane is farther than the current k-th best distance.

* Excellent in low dimensions; degrades toward brute force as dimensionality grows (curse of dimensionality).
* The pruning bound `|q[axis] − split|` is a valid lower bound **only for Euclidean distance**. With cosine or Manhattan the tree is
  still traversed, but results may differ slightly from brute force. Use Euclidean for exact KD-Tree results.
* Deletion triggers a full rebuild.

### 12.4 HNSW — Hierarchical Navigable Small World

Parameters used here: **`M = 16`** (max neighbours on upper layers), **`M0 = 32`** (max on layer 0), **`ef_construction = 200`**,
**`ef_search = 50`**, level multiplier `mL = 1 / ln(M)`, deterministic RNG seed `42`.

* **Layer assignment:** each node draws `level = ⌊−ln(U) · mL⌋`, so higher layers are exponentially rarer.
* **Insert:** greedy-descend from the entry point to the node's level; then on every layer from `min(top, level)` down to 0, run a beam search of width `ef_construction`, connect to the closest `M`/`M0` nodes **bidirectionally**, and prune any neighbour list that exceeds its cap to its closest members.
* **Search:** greedy-descend from the top layer with `ef = 1`; on layer 0 run a beam search of width `max(ef, k)` using a priority queue and return the best *k*.
* **Delete:** removes the node from every neighbour list and, if it was the entry point, promotes the remaining node with the highest level.

**Why HNSW wins at high dimensions:** KD-Tree pruning depends on axis-aligned bounds, which stop eliminating subtrees once there are
hundreds of dimensions. HNSW navigates a proximity *graph* and has no such dependence, so it stays fast on 768-D embeddings.

**Approximate by design:** HNSW trades a small amount of recall for large speed gains. The demo index is built with cosine
distance, so querying it with another metric works but is less principled than querying with cosine.

---

## 13. Deploying to the Cloud (Render)

The project is ready for any Python host that provides `$PORT` (Render, Railway, Fly.io, Heroku-style platforms). Ollama cannot run on
free cloud tiers, so **use Gemini** there.

1. Push the repo to GitHub (make sure `.env` is **not** committed).
2. On Render create a **Web Service** from the repo.
3. **Build command:** `pip install -r requirements.txt`
4. **Start command:**
   ```bash
   gunicorn main:app --workers 1 --threads 4 --bind 0.0.0.0:$PORT --timeout 120
   ```
5. In the **Environment** tab add:
   * `GEMINI_API_KEY` = your key
   * `READ_ONLY` = `1` *(recommended for a public demo)*
   * `RATE_LIMIT_PER_MIN` = `30` *(or lower)*
6. Deploy and open the service URL.

**Why `--workers 1`?** All data lives in process memory. With several Gunicorn workers each worker would hold its own separate
copy of the database (and its own knowledge-ingestion thread and API calls), so inserts would appear and vanish at random.
Scale with `--threads`, not `--workers`.

**Cold starts:** free instances sleep when idle. On wake-up the server restarts, in-memory documents are lost, and `knowledge/` is
re-embedded in the background (a few seconds per chunk on the Gemini free tier).

---

## 14. Security & Safety Notes

* **Keep your API key secret.** `.env` is in `.gitignore`, but it is easy to leak it by zipping the project folder or sharing a screenshot. If a key ever leaves your machine, **revoke it and create a new one** in Google AI Studio.
* **Public demos:** set `READ_ONLY=1` so strangers cannot add or delete data, and keep `RATE_LIMIT_PER_MIN` low so nobody can burn your free quota.
* **Rate limiting trusts `X-Forwarded-For`.** That is correct behind a proxy such as Render's, but a client can spoof the header if the app is exposed directly to the internet. Do not rely on it as real abuse protection.
* **No authentication, no HTTPS in the app itself, wide-open CORS.** Fine for a local or demo project; put it behind a gateway and add auth before anything serious.
* **Prompt injection:** retrieved text is pasted straight into the LLM prompt. In a multi-user setting, treat user-supplied documents as untrusted.
* **Your documents are sent to the provider** (Google) when using Gemini. Use Ollama if the text is private.
* `app.run(...)` is Flask's development server — use Gunicorn for anything public.

---

## 15. Limitations & Design Trade-offs

| Limitation | Detail |
|---|---|
| **In-memory only** | Documents and inserted vectors are lost on restart (only `knowledge/` is restored). There is no disk persistence |
| **Single process** | Locks protect threads, not multiple processes — run one worker |
| **HNSW delete is simple** | Deleting scans every node's neighbour lists (`O(N)`) and does not re-link orphaned neighbours; heavy deletion can reduce recall |
| **KD-Tree delete** | Rebuilds the whole tree |
| **Pure-Python speed** | The interpreter dominates on tiny datasets — at 20 vectors, Brute Force often beats HNSW. HNSW's advantage appears with thousands of vectors and high dimensions |
| **Demo embeddings are heuristic** | The 16-D Search tab uses a keyword lookup, not a language model |
| **Single embedding model per run** | Never mix vectors from different models in one index |
| **No update operation** | To change a chunk, delete it and insert again |
| **Validation errors use HTTP 200** | Check the `error` field in JSON responses |
| **Free-tier quotas** | Heavy use of Gemini can hit rate limits; the client retries a few times, then reports the API message |

---

## 16. Troubleshooting

| Problem | Fix |
|---|---|
| Badge shows `OLLAMA ●` red / `Ollama unavailable` | Start Ollama (`ollama serve`) and make sure both models are pulled — or set `GEMINI_API_KEY` |
| Badge shows `GEMINI ●` but requests fail | Read the error text: `API key not valid` → wrong key; `429` → quota/rate limit, wait a minute; `Network error` → check internet/proxy |
| `ModuleNotFoundError: flask` | Activate your venv and run `pip install -r requirements.txt` |
| `python: command not found` | Reinstall Python and tick **Add python.exe to PATH**; on macOS/Linux use `python3` |
| `pip: command not found` | Use `python -m pip install -r requirements.txt` |
| Port 8080 already in use | Set another port (`PORT=9000`) or free it: `netstat -ano \| findstr 8080` then `taskkill /PID <pid> /F` |
| `.env` seems ignored | It must sit **next to `main.py`**, use `KEY=value` lines (no spaces around `=` needed), and a variable already set in your shell takes precedence |
| Knowledge file not answering | Check the console for `[knowledge] …` messages; with Gemini ingestion takes ~0.7 s per chunk after startup |
| `Text too long (N chunks)` | Max 40 chunks (~9,000 words) per insert — split the text |
| `Too many requests` (429) | You hit `RATE_LIMIT_PER_MIN`; wait a minute or raise the limit |
| `This demo is read-only` (403) | `READ_ONLY=1` is set — unset it to allow inserts/deletes |
| First Ollama embedding takes minutes | The model is loading/downloading on first use |
| Answers are slow with Ollama | Normal on CPU; switch to `llama3.2:1b` |
| Documents disappeared after restart | Expected — storage is in-memory. Put permanent text into `knowledge/` |
| Search results look "off" for unusual words | The 16-D tab only knows its keyword dictionary; use the Documents/Ask AI tabs for real semantics |

---

## 17. Ideas for Extending the Project

* **Persistence** — save/load the HNSW graph and chunk store to disk (`pickle`, SQLite, or `numpy.savez`).
* **File upload** — accept PDFs/DOCX in the Documents tab and extract text server-side.
* **Streaming answers** — use Gemini/Ollama streaming and Server-Sent Events for a true typewriter effect.
* **Better HNSW** — implement the paper's neighbour-selection heuristic and proper re-linking on delete.
* **Metadata filtering** — filter by category/title before or during the search.
* **Hybrid search** — combine BM25 keyword scores with vector similarity.
* **Recall measurement** — compare HNSW results with Brute Force to compute recall@k at different `ef` values.
* **Vectorised distances** — batch NumPy matrix operations to speed up Brute Force dramatically.
* **Tests** — unit tests for each index (exactness of KD-Tree/Brute Force, recall of HNSW).
* **Auth + multi-tenant collections** — separate indexes per user.

---

## 18. License

Released under the **MIT License** — Copyright © 2026 D Simran Patra. See [`LICENSE`](LICENSE) for the full text.

---

*Built to learn how vector databases actually work. If it helped you, give the repo a ⭐.*