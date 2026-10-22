"""
VectorDB in Python — HNSW + KD-Tree + Brute Force + RAG via Ollama.
Drop-in replacement for the C++ backend: same REST API, same index.html.

Run:  python main.py     ->  http://localhost:8080
"""
import heapq
import math
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from flask import Flask, Response, jsonify, request

# load .env for local runs (no extra package needed)
_env = Path(__file__).parent / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

DIMS = 16
PORT = int(os.environ.get("PORT", 8080))      # cloud hosts give you the port via $PORT
READ_ONLY = os.environ.get("READ_ONLY", "0") == "1"   # 1 = visitors cannot add/delete data
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MIN", "30"))
MAX_CHUNKS_PER_INSERT = 40
BASE_DIR = Path(__file__).parent

# =====================================================================
#  DISTANCE METRICS
# =====================================================================

def euclidean(a, b):
    return float(np.linalg.norm(a - b))

def cosine(a, b):
    na, nb = float(a @ a), float(b @ b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - float(a @ b) / (math.sqrt(na) * math.sqrt(nb))

def manhattan(a, b):
    return float(np.abs(a - b).sum())

def get_dist_fn(name):
    return {"cosine": cosine, "manhattan": manhattan}.get(name, euclidean)

# =====================================================================
#  DATA TYPE
# =====================================================================

@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: np.ndarray

# =====================================================================
#  BRUTE FORCE  — exact, O(N·d)
# =====================================================================

class BruteForce:
    def __init__(self):
        self.items = []

    def insert(self, v):
        self.items.append(v)

    def knn(self, q, k, dist):
        r = sorted((dist(q, v.emb), v.id) for v in self.items)
        return r[:k]

    def remove(self, id_):
        self.items = [v for v in self.items if v.id != id_]

# =====================================================================
#  KD-TREE  — exact (for Euclidean), axis-aligned partitioning
# =====================================================================

class KDNode:
    __slots__ = ("item", "left", "right")
    def __init__(self, item):
        self.item, self.left, self.right = item, None, None

class KDTree:
    def __init__(self, dims):
        self.dims = dims
        self.root = None

    def insert(self, v):
        # iterative insert (no recursion-depth problems)
        node = KDNode(v)
        if self.root is None:
            self.root = node
            return
        cur, depth = self.root, 0
        while True:
            ax = depth % self.dims
            side = "left" if v.emb[ax] < cur.item.emb[ax] else "right"
            nxt = getattr(cur, side)
            if nxt is None:
                setattr(cur, side, node)
                return
            cur, depth = nxt, depth + 1

    def knn(self, q, k, dist):
        heap = []  # max-heap via negated distance: (-dist, id)

        def visit(n, depth):
            if n is None:
                return
            dn = dist(q, n.item.emb)
            if len(heap) < k or dn < -heap[0][0]:
                heapq.heappush(heap, (-dn, n.item.id))
                if len(heap) > k:
                    heapq.heappop(heap)
            ax = depth % self.dims
            diff = q[ax] - n.item.emb[ax]
            closer, farther = (n.left, n.right) if diff < 0 else (n.right, n.left)
            visit(closer, depth + 1)
            if len(heap) < k or abs(diff) < -heap[0][0]:
                visit(farther, depth + 1)

        visit(self.root, 0)
        return sorted((-d, i) for d, i in heap)

    def rebuild(self, items):
        self.root = None
        for v in items:
            self.insert(v)

# =====================================================================
#  HNSW — Hierarchical Navigable Small World
# =====================================================================

class HNSW:
    def __init__(self, M=16, ef_build=200, seed=42):
        self.M, self.M0, self.ef_build = M, 2 * M, ef_build
        self.mL = 1.0 / math.log(M)
        self.G = {}            # id -> {"item", "max_lyr", "nbrs": [[ids] per layer]}
        self.top_layer = -1
        self.entry = -1
        self.rng = random.Random(seed)

    def _rand_level(self):
        return int(math.floor(-math.log(1.0 - self.rng.random()) * self.mL))

    def _search_layer(self, q, ep, ef, lyr, dist):
        G = self.G
        d0 = dist(q, G[ep]["item"].emb)
        visited = {ep}
        cands = [(d0, ep)]        # min-heap
        found = [(-d0, ep)]       # max-heap (negated)

        while cands:
            cd, cid = heapq.heappop(cands)
            if len(found) >= ef and cd > -found[0][0]:
                break
            nbrs = G[cid]["nbrs"]
            if lyr >= len(nbrs):
                continue
            for nid in nbrs[lyr]:
                if nid in visited or nid not in G:
                    continue
                visited.add(nid)
                nd = dist(q, G[nid]["item"].emb)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)
        return sorted((-d, i) for d, i in found)

    def insert(self, item, dist):
        G = self.G
        id_ = item.id
        lvl = self._rand_level()
        G[id_] = {"item": item, "max_lyr": lvl, "nbrs": [[] for _ in range(lvl + 1)]}

        if self.entry == -1:
            self.entry, self.top_layer = id_, lvl
            return

        ep = self.entry
        # greedy descent through layers above the node's level
        for lc in range(self.top_layer, lvl, -1):
            if lc < len(G[ep]["nbrs"]):
                W = self._search_layer(item.emb, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]

        # connect on every layer from min(top, lvl) down to 0
        for lc in range(min(self.top_layer, lvl), -1, -1):
            W = self._search_layer(item.emb, ep, self.ef_build, lc, dist)
            max_m = self.M0 if lc == 0 else self.M
            sel = [i for _, i in W[:max_m] if i != id_]
            G[id_]["nbrs"][lc] = sel

            for nid in sel:
                nb = G[nid]["nbrs"]
                while len(nb) <= lc:
                    nb.append([])
                conn = nb[lc]
                conn.append(id_)
                if len(conn) > max_m:   # prune to the closest max_m
                    ds = sorted((dist(G[nid]["item"].emb, G[c]["item"].emb), c)
                                for c in conn if c in G)
                    nb[lc] = [c for _, c in ds[:max_m]]
            if W:
                ep = W[0][1]

        if lvl > self.top_layer:
            self.top_layer, self.entry = lvl, id_

    def knn(self, q, k, ef, dist):
        if self.entry == -1:
            return []
        ep = self.entry
        for lc in range(self.top_layer, 0, -1):
            if lc < len(self.G[ep]["nbrs"]):
                W = self._search_layer(q, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]
        return self._search_layer(q, ep, max(ef, k), 0, dist)[:k]

    def remove(self, id_):
        if id_ not in self.G:
            return
        for nd in self.G.values():
            for layer in nd["nbrs"]:
                if id_ in layer:
                    layer.remove(id_)
        del self.G[id_]
        if self.entry == id_:  # pick the remaining node with the highest level
            if self.G:
                self.entry = max(self.G, key=lambda i: self.G[i]["max_lyr"])
                self.top_layer = self.G[self.entry]["max_lyr"]
            else:
                self.entry, self.top_layer = -1, -1

    def info(self):
        max_l = max(self.top_layer + 1, 1)
        nodes_per, edges_per = [0] * max_l, [0] * max_l
        nodes, edges = [], []
        for id_, nd in self.G.items():
            it = nd["item"]
            nodes.append({"id": id_, "metadata": it.metadata,
                          "category": it.category, "maxLyr": nd["max_lyr"]})
            for lc in range(min(nd["max_lyr"], max_l - 1) + 1):
                nodes_per[lc] += 1
                if lc < len(nd["nbrs"]):
                    for nid in nd["nbrs"][lc]:
                        if id_ < nid:
                            edges_per[lc] += 1
                            edges.append({"src": id_, "dst": nid, "lyr": lc})
        return {"topLayer": self.top_layer, "nodeCount": len(self.G),
                "nodesPerLayer": nodes_per, "edgesPerLayer": edges_per,
                "nodes": nodes, "edges": edges}

# =====================================================================
#  VECTOR DATABASE (16-D demo index; all three algorithms)
# =====================================================================

class VectorDB:
    def __init__(self, dims):
        self.dims = dims
        self.store = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(16, 200)
        self.lock = threading.Lock()
        self.next_id = 1

    def insert(self, meta, cat, emb, dist=cosine):
        with self.lock:
            v = VectorItem(self.next_id, meta, cat, np.asarray(emb, dtype=np.float32))
            self.next_id += 1
            self.store[v.id] = v
            self.bf.insert(v)
            self.kdt.insert(v)
            self.hnsw.insert(v, dist)
            return v.id

    def remove(self, id_):
        with self.lock:
            if id_ not in self.store:
                return False
            del self.store[id_]
            self.bf.remove(id_)
            self.hnsw.remove(id_)
            self.kdt.rebuild(list(self.store.values()))
            return True

    def _run(self, algo, q, k, dfn):
        if algo == "bruteforce":
            return self.bf.knn(q, k, dfn)
        if algo == "kdtree":
            return self.kdt.knn(q, k, dfn)
        return self.hnsw.knn(q, k, 50, dfn)

    def search(self, q, k, metric, algo):
        with self.lock:
            dfn = get_dist_fn(metric)
            t0 = time.perf_counter()
            raw = self._run(algo, q, k, dfn)
            us = int((time.perf_counter() - t0) * 1e6)
            hits = [(d, self.store[i]) for d, i in raw if i in self.store]
            return hits, us

    def benchmark(self, q, k, metric):
        with self.lock:
            dfn = get_dist_fn(metric)
            def timed(algo):
                t0 = time.perf_counter()
                self._run(algo, q, k, dfn)
                return int((time.perf_counter() - t0) * 1e6)
            return {"bruteforceUs": timed("bruteforce"), "kdtreeUs": timed("kdtree"),
                    "hnswUs": timed("hnsw"), "itemCount": len(self.store)}

    def all(self):
        with self.lock:
            return list(self.store.values())

    def hnsw_info(self):
        with self.lock:
            return self.hnsw.info()

    def size(self):
        with self.lock:
            return len(self.store)

# =====================================================================
#  TEXT CHUNKER
# =====================================================================

def chunk_text(text, chunk_words=250, overlap=30):
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]
    chunks, step = [], chunk_words - overlap
    for i in range(0, len(words), step):
        end = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
    return chunks

# =====================================================================
#  AI CLIENT  (embeddings + answer generation)
#    - GEMINI_API_KEY set  -> Google Gemini API  (free tier, works on cloud hosts)
#    - otherwise           -> local Ollama       (laptop only)
# =====================================================================

GEMINI_BASE = os.environ.get("GEMINI_BASE", "https://generativelanguage.googleapis.com/v1beta/models")
EMBED_DIMS_GEMINI = 768   # gemini-embedding-001 can return 768 numbers (smaller = less RAM)

class AIClient:
    def __init__(self):
        self.gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self.provider = "gemini" if self.gemini_key else "ollama"
        self.base = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
        if self.provider == "gemini":
            self.embed_model = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
            self.gen_model = os.environ.get("GEN_MODEL", "gemini-flash-latest")
        else:
            self.embed_model = os.environ.get("EMBED_MODEL", "nomic-embed-text")
            self.gen_model = os.environ.get("GEN_MODEL", "llama3.2")
        self.last_error = ""
        self._qcache = {}          # question -> embedding (/doc/search and /doc/ask embed the same text)
        self._qlock = threading.Lock()

    # ---------- availability ----------
    def is_available(self):
        if self.provider == "gemini":
            return True            # a key is configured; we don't spend quota just to ping
        try:
            return requests.get(self.base + "/api/tags", timeout=2).status_code == 200
        except requests.RequestException:
            return False

    def unavailable_msg(self):
        if self.provider == "gemini":
            return self.last_error or "Gemini API request failed"
        return ("Ollama unavailable. Install from https://ollama.com then run: "
                "ollama pull nomic-embed-text && ollama pull llama3.2")

    # ---------- Gemini helper ----------
    def _gemini_post(self, model, method, payload, timeout):
        url = f"{GEMINI_BASE}/{model}:{method}"
        headers = {"x-goog-api-key": self.gemini_key, "Content-Type": "application/json"}
        for attempt in range(4):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            except requests.RequestException as e:
                self.last_error = f"Network error calling Gemini ({e.__class__.__name__})"
                return None
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 3:   # busy/rate limited: wait and retry
                time.sleep(3 * (attempt + 1))
                continue
            if r.status_code != 200:
                try:
                    msg = r.json().get("error", {}).get("message", "")
                except ValueError:
                    msg = ""
                self.last_error = f"Gemini API error {r.status_code}: {msg[:200]}"
                return None
            self.last_error = ""
            return r.json()
        return None

    # ---------- embeddings ----------
    def embed(self, text, query=False):
        """Returns [] on failure (reason in self.last_error)."""
        if query:
            with self._qlock:
                if text in self._qcache:
                    return self._qcache[text]
        if self.provider == "gemini":
            data = self._gemini_post(self.embed_model, "embedContent", {
                "model": f"models/{self.embed_model}",
                "content": {"parts": [{"text": text}]},
                "taskType": "RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT",
                "outputDimensionality": EMBED_DIMS_GEMINI,
            }, timeout=30)
            vec = (data or {}).get("embedding", {}).get("values", [])
        else:
            try:
                r = requests.post(self.base + "/api/embeddings",
                                  json={"model": self.embed_model, "prompt": text}, timeout=60)
                vec = r.json().get("embedding", []) if r.status_code == 200 else []
            except (requests.RequestException, ValueError):
                vec = []
        if query and vec:
            with self._qlock:
                if len(self._qcache) > 256:
                    self._qcache.clear()
                self._qcache[text] = vec
        return vec

    # ---------- generation ----------
    def generate(self, prompt):
        if self.provider == "gemini":
            data = self._gemini_post(self.gen_model, "generateContent",
                                     {"contents": [{"parts": [{"text": prompt}]}]}, timeout=90)
            if not data:
                return "ERROR: " + self.last_error
            try:
                parts = data["candidates"][0]["content"]["parts"]
                return "".join(p.get("text", "") for p in parts).strip() or "ERROR: empty answer from Gemini"
            except (KeyError, IndexError, TypeError):
                return "ERROR: Gemini returned no answer (possibly blocked by a safety filter)"
        try:
            r = requests.post(self.base + "/api/generate",
                              json={"model": self.gen_model, "prompt": prompt, "stream": False},
                              timeout=180)
            if r.status_code != 200:
                return "ERROR: Ollama unavailable. Run: ollama serve"
            return r.json().get("response", "")
        except requests.RequestException:
            return "ERROR: Ollama unavailable. Run: ollama serve"

# =====================================================================
#  DOCUMENT DATABASE — HNSW over real Ollama embeddings
# =====================================================================

@dataclass
class DocItem:
    id: int
    title: str
    text: str
    emb: np.ndarray

class DocumentDB:
    def __init__(self):
        self.store = {}
        self.hnsw = HNSW(16, 200)
        self.bf = BruteForce()
        self.lock = threading.Lock()
        self.next_id = 1
        self.dims = 0

    def insert(self, title, text, emb):
        with self.lock:
            emb = np.asarray(emb, dtype=np.float32)
            if self.dims == 0:
                self.dims = len(emb)
            item = DocItem(self.next_id, title, text, emb)
            self.next_id += 1
            self.store[item.id] = item
            vi = VectorItem(item.id, title, "doc", emb)
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)
            return item.id

    def search(self, q, k, max_dist=0.7):
        with self.lock:
            if not self.store:
                return []
            q = np.asarray(q, dtype=np.float32)
            raw = self.bf.knn(q, k, cosine) if len(self.store) < 10 \
                else self.hnsw.knn(q, k, 50, cosine)
            return [(d, self.store[i]) for d, i in raw
                    if i in self.store and d <= max_dist]

    def remove(self, id_):
        with self.lock:
            if id_ not in self.store:
                return False
            del self.store[id_]
            self.hnsw.remove(id_)
            self.bf.remove(id_)
            return True

    def all(self):
        with self.lock:
            return list(self.store.values())

    def size(self):
        with self.lock:
            return len(self.store)

# =====================================================================
#  DEMO DATA (16-D: dims 0-3 CS | 4-7 Math | 8-11 Food | 12-15 Sports)
# =====================================================================

DEMO = [
 ("Linked List: nodes connected by pointers","cs",[.90,.85,.72,.68,.12,.08,.15,.10,.05,.08,.06,.09,.07,.11,.08,.06]),
 ("Binary Search Tree: O(log n) search and insert","cs",[.88,.82,.78,.74,.15,.10,.08,.12,.06,.07,.08,.05,.09,.06,.07,.10]),
 ("Dynamic Programming: memoization overlapping subproblems","cs",[.82,.76,.88,.80,.20,.18,.12,.09,.07,.06,.08,.07,.08,.09,.06,.07]),
 ("Graph BFS and DFS: breadth and depth first traversal","cs",[.85,.80,.75,.82,.18,.14,.10,.08,.06,.09,.07,.06,.10,.08,.09,.07]),
 ("Hash Table: O(1) lookup with collision chaining","cs",[.87,.78,.70,.76,.13,.11,.09,.14,.08,.07,.06,.08,.07,.10,.08,.09]),
 ("Calculus: derivatives integrals and limits","math",[.12,.15,.18,.10,.91,.86,.78,.72,.08,.06,.07,.09,.07,.08,.06,.10]),
 ("Linear Algebra: matrices eigenvalues eigenvectors","math",[.20,.18,.15,.12,.88,.90,.82,.76,.09,.07,.08,.06,.10,.07,.08,.09]),
 ("Probability: distributions random variables Bayes theorem","math",[.15,.12,.20,.18,.84,.80,.88,.82,.07,.08,.06,.10,.09,.06,.09,.08]),
 ("Number Theory: primes modular arithmetic RSA cryptography","math",[.22,.16,.14,.20,.80,.85,.76,.90,.08,.09,.07,.06,.08,.10,.07,.06]),
 ("Combinatorics: permutations combinations generating functions","math",[.18,.20,.16,.14,.86,.78,.84,.80,.06,.07,.09,.08,.06,.09,.10,.07]),
 ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes","food",[.08,.06,.09,.07,.07,.08,.06,.09,.90,.86,.78,.72,.08,.06,.09,.07]),
 ("Sushi: vinegared rice raw fish and nori rolls","food",[.06,.08,.07,.09,.09,.06,.08,.07,.86,.90,.82,.76,.07,.09,.06,.08]),
 ("Ramen: noodle soup with chashu pork and soft-boiled eggs","food",[.09,.07,.06,.08,.08,.09,.07,.06,.82,.78,.90,.84,.09,.07,.08,.06]),
 ("Tacos: corn tortillas with carnitas salsa and cilantro","food",[.07,.09,.08,.06,.06,.07,.09,.08,.78,.82,.86,.90,.06,.08,.07,.09]),
 ("Croissant: laminated pastry with buttery flaky layers","food",[.06,.07,.10,.09,.10,.06,.07,.10,.85,.80,.76,.82,.09,.07,.10,.06]),
 ("Basketball: fast-paced shooting dribbling slam dunks","sports",[.09,.07,.08,.10,.08,.09,.07,.06,.08,.07,.09,.06,.91,.85,.78,.72]),
 ("Football: tackles touchdowns field goals and strategy","sports",[.07,.09,.06,.08,.09,.07,.10,.08,.07,.09,.08,.07,.87,.89,.82,.76]),
 ("Tennis: racket volleys groundstrokes and Wimbledon serves","sports",[.08,.06,.09,.07,.07,.08,.06,.09,.09,.06,.07,.08,.83,.80,.88,.82]),
 ("Chess: openings endgames tactics strategic board game","sports",[.25,.20,.22,.18,.22,.18,.20,.15,.06,.08,.07,.09,.80,.84,.78,.90]),
 ("Swimming: butterfly freestyle backstroke Olympic competition","sports",[.06,.08,.07,.09,.08,.06,.09,.07,.10,.08,.06,.07,.85,.82,.86,.80]),
]

def load_demo(db):
    for meta, cat, emb in DEMO:
        db.insert(meta, cat, emb)

# =====================================================================
#  HTTP SERVER
# =====================================================================

app = Flask(__name__)
db = VectorDB(DIMS)
doc_db = DocumentDB()
ai = AIClient()
load_demo(db)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024   # reject giant uploads

@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

@app.route("/<path:_>", methods=["OPTIONS"])
def preflight(_):
    return Response(status=204)

_rate, _rate_lock = {}, threading.Lock()

def rate_ok():
    """Tiny per-IP limiter so strangers can't burn your free API quota."""
    ip = (request.headers.get("X-Forwarded-For") or request.remote_addr or "?").split(",")[0].strip()
    now = time.time()
    with _rate_lock:
        hits = [t for t in _rate.get(ip, []) if now - t < 60]
        if len(hits) >= RATE_LIMIT:
            _rate[ip] = hits
            return False
        hits.append(now)
        _rate[ip] = hits
        if len(_rate) > 5000:
            _rate.clear()
    return True

def too_many():
    return jsonify(error="Too many requests - please wait a minute and try again."), 429

def read_only_block():
    return jsonify(error="This demo is read-only: visitors cannot add or delete data."), 403

def safe_k(b, default=3):
    try:
        return max(1, min(10, int(b.get("k", default))))
    except (TypeError, ValueError):
        return default

def parse_vec(s):
    out = []
    for t in (s or "").split(","):
        try:
            out.append(float(t))
        except ValueError:
            pass
    return np.array(out, dtype=np.float32)

def int_arg(name, default):
    try:
        return int(request.args.get(name, default))
    except (TypeError, ValueError):
        return default

# ── Demo vector endpoints ────────────────────────────────────────────

@app.get("/search")
def search():
    q = parse_vec(request.args.get("v", ""))
    if len(q) != DIMS:
        return jsonify(error=f"need {DIMS}D vector")
    k = int_arg("k", 5)
    metric = request.args.get("metric") or "cosine"
    algo = request.args.get("algo") or "hnsw"
    hits, us = db.search(q, k, metric, algo)
    return jsonify(
        results=[{"id": v.id, "metadata": v.metadata, "category": v.category,
                  "distance": round(d, 6), "embedding": [round(float(x), 4) for x in v.emb]}
                 for d, v in hits],
        latencyUs=us, algo=algo, metric=metric)

@app.post("/insert")
def insert():
    if READ_ONLY:
        return read_only_block()
    b = request.get_json(silent=True) or {}
    meta, cat, emb = b.get("metadata", ""), b.get("category", ""), b.get("embedding", [])
    if not meta or len(emb) != DIMS:
        return jsonify(error="invalid body")
    return jsonify(id=db.insert(meta, cat, emb))

@app.delete("/delete/<int:id_>")
def delete(id_):
    if READ_ONLY:
        return read_only_block()
    return jsonify(ok=db.remove(id_))

@app.get("/items")
def items():
    return jsonify([{"id": v.id, "metadata": v.metadata, "category": v.category,
                     "embedding": [round(float(x), 4) for x in v.emb]} for v in db.all()])

@app.get("/benchmark")
def benchmark():
    q = parse_vec(request.args.get("v", ""))
    if len(q) != DIMS:
        return jsonify(error=f"need {DIMS}D vector")
    return jsonify(db.benchmark(q, int_arg("k", 5), request.args.get("metric") or "cosine"))

@app.get("/hnsw-info")
def hnsw_info():
    return jsonify(db.hnsw_info())

# ── Document + RAG endpoints ─────────────────────────────────────────

@app.post("/doc/insert")
def doc_insert():
    if READ_ONLY:
        return read_only_block()
    if not rate_ok():
        return too_many()
    b = request.get_json(silent=True) or {}
    title, text = str(b.get("title", ""))[:200], str(b.get("text", ""))
    if not title or not text:
        return jsonify(error="need title and text")
    chunks = chunk_text(text)
    if len(chunks) > MAX_CHUNKS_PER_INSERT:
        return jsonify(error=f"Text too long ({len(chunks)} chunks). Max {MAX_CHUNKS_PER_INSERT} chunks (~9,000 words) per insert.")
    ids = []
    for i, ch in enumerate(chunks):
        emb = ai.embed(ch)
        if not emb:
            return jsonify(error=ai.unavailable_msg())
        name = f"{title} [{i+1}/{len(chunks)}]" if len(chunks) > 1 else title
        ids.append(doc_db.insert(name, ch, emb))
    return jsonify(ids=ids, chunks=len(chunks), dims=doc_db.dims)

@app.delete("/doc/delete/<int:id_>")
def doc_delete(id_):
    if READ_ONLY:
        return read_only_block()
    return jsonify(ok=doc_db.remove(id_))

@app.get("/doc/list")
def doc_list():
    out = []
    for d in doc_db.all():
        preview = d.text[:120] + ("…" if len(d.text) > 120 else "")
        out.append({"id": d.id, "title": d.title, "preview": preview,
                    "words": len(d.text.split())})
    return jsonify(out)

@app.post("/doc/search")
def doc_search():
    b = request.get_json(silent=True) or {}
    question, k = str(b.get("question", ""))[:2000], safe_k(b)
    if not question:
        return jsonify(error="need question")
    if not rate_ok():
        return too_many()
    q = ai.embed(question, query=True)
    if not q:
        return jsonify(error=ai.unavailable_msg())
    hits = doc_db.search(q, k)
    return jsonify(contexts=[{"id": d.id, "title": d.title, "distance": round(dist, 4)}
                             for dist, d in hits])

@app.post("/doc/ask")
def doc_ask():
    b = request.get_json(silent=True) or {}
    question, k = str(b.get("question", ""))[:2000], safe_k(b)
    if not question:
        return jsonify(error="need question")
    if not rate_ok():
        return too_many()

    q = ai.embed(question, query=True)               # 1. embed question
    if not q:
        return jsonify(error=ai.unavailable_msg())
    hits = doc_db.search(q, k)                       # 2. retrieve

    ctx = "".join(f"[{i+1}] {d.title}:\n{d.text}\n\n" for i, (_, d) in enumerate(hits))
    prompt = (                                       # 3. build prompt
        "You are an AI assistant that answers questions using the documents provided below. "
        "Use them when they are relevant. If they are not relevant, use your own general knowledge. "
        "You are an AI assistant, not a person. If asked about yourself (your name, who you are), "
        "say you are an AI assistant that answers questions about the user's documents. "
        "IMPORTANT: Do NOT mention the 'context' or 'provided text', and do not say things like "
        "'the context doesn't mention'. Just answer naturally.\n\n"
        f"Context:\n{ctx}Question: {question}\n\nAnswer:")
    answer = ai.generate(prompt)                     # 4. generate
    if answer.startswith("ERROR:"):
        return jsonify(error=answer[6:].strip())

    return jsonify(answer=answer, model=ai.gen_model,
                   contexts=[{"id": d.id, "title": d.title, "text": d.text,
                              "distance": round(dist, 4)} for dist, d in hits],
                   docCount=doc_db.size())

@app.get("/status")
def status():
    return jsonify(ollamaAvailable=ai.is_available(), provider=ai.provider, readOnly=READ_ONLY,
                   embedModel=ai.embed_model, genModel=ai.gen_model,
                   docCount=doc_db.size(), docDims=doc_db.dims,
                   demoDims=DIMS, demoCount=db.size())

@app.get("/stats")
def stats():
    return jsonify(count=db.size(), dims=DIMS,
                   algorithms=["bruteforce", "kdtree", "hnsw"],
                   metrics=["euclidean", "cosine", "manhattan"])

# -- Permanent knowledge: every .txt/.md file in ./knowledge is embedded at startup --

KNOWLEDGE_DIR = BASE_DIR / "knowledge"

def ingest_knowledge():
    files = (sorted(list(KNOWLEDGE_DIR.glob("*.txt")) + list(KNOWLEDGE_DIR.glob("*.md")))
             if KNOWLEDGE_DIR.exists() else [])
    if not files:
        return
    if not ai.is_available():
        print("[knowledge] AI provider not available - skipping")
        return
    for f in files:
        try:
            chunks = chunk_text(f.read_text(encoding="utf-8"))
        except OSError:
            continue
        for i, ch in enumerate(chunks):
            emb = ai.embed(ch)
            if not emb:
                print(f"[knowledge] embedding failed on {f.name}: {ai.unavailable_msg()}")
                return
            name = f.stem.replace("_", " ").replace("-", " ")
            doc_db.insert(f"{name} [{i+1}/{len(chunks)}]" if len(chunks) > 1 else name, ch, emb)
            if ai.provider == "gemini":
                time.sleep(0.7)          # stay under the free-tier requests-per-minute limit
    print(f"[knowledge] loaded {doc_db.size()} chunks from {KNOWLEDGE_DIR.name}/")

# runs in the background so the web server starts instantly (also under gunicorn)
threading.Thread(target=ingest_knowledge, daemon=True).start()

@app.get("/")
def index():
    f = BASE_DIR / "index.html"
    if not f.exists():
        return Response("index.html not found next to main.py", status=404)
    return Response(f.read_text(encoding="utf-8"), mimetype="text/html")

if __name__ == "__main__":
    up = ai.is_available()
    print("=== VectorDB Engine (Python) ===")
    print(f"http://localhost:{PORT}")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(f"AI provider: {ai.provider}", "- ONLINE" if up else "- OFFLINE (install Ollama or set GEMINI_API_KEY)")
    if up:
        print(f"  embed model: {ai.embed_model}  gen model: {ai.gen_model}")
    app.run(host="0.0.0.0", port=PORT, threaded=True)