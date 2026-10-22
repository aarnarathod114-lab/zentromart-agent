"""Hybrid retriever: multilingual embeddings + BM25 over KB sections and the product catalog,
plus direct SKU / pincode lookups. Supports several queries (e.g. original + English rewrite)."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATA = Path("dataset/Capstone_Project_Dataset_FTTT_2026")
EMBED_MODEL = "intfloat/multilingual-e5-small"
SECTION_RE = re.compile(r"^##\s*\[([A-Za-z0-9.\-]+)\]\s*(.*)$", re.M)

# Mild boost following KB-00.1 precedence: policy > agent guide > help > FAQ > marketing
PRIOR = {"policy": 1.06, "agent_guide": 1.03, "help": 1.00, "faq": 0.97,
         "marketing": 0.85, "meta": 0.97, "catalog": 1.00}

def tokenize(text):
    return re.findall(r"\w+", str(text).lower())


def row_text(row):
    return "; ".join(f"{k}: {v}" for k, v in row.items() if str(v).strip())


def parse_doc(path):
    """One chunk per '## [ID] Heading' section, keeping the document header fields."""
    text = path.read_text(encoding="utf-8")
    meta, body = {}, text
    if text.startswith("---"):
        _, head, body = text.split("---", 2)
        for line in head.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    base = {
        "title": meta.get("title", path.stem),
        "doc_type": meta.get("doc_type", ""),
        "status": meta.get("status", ""),
        "effective_from": meta.get("effective_from", ""),
        "file": path.name,
    }
    matches = list(SECTION_RE.finditer(body))
    if not matches:
        return [{**base, "id": meta.get("doc_id", path.stem), "heading": "", "text": body.strip()}]
    chunks = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        chunks.append({**base, "id": m.group(1), "heading": m.group(2).strip(),
                       "text": body[m.end():end].strip()})
    return chunks


class Retriever:
    def __init__(self):
        self.products = pd.read_csv(DATA / "data" / "products.csv", dtype=str).fillna("")
        self.pincodes = pd.read_csv(DATA / "data" / "pincodes.csv", dtype=str).fillna("")

        self.chunks = [c for p in sorted((DATA / "kb").glob("*.md")) for c in parse_doc(p)]
        for _, r in self.products.iterrows():
            self.chunks.append({"id": f"products.csv:{r['sku']}", "title": "Product catalog",
                                "heading": "", "doc_type": "catalog", "status": "current",
                                "effective_from": "", "file": "products.csv",
                                "text": row_text(r)})

        texts = [f"{c['title']} - {c['heading']}\n{c['text']}" for c in self.chunks]
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        self.model = SentenceTransformer(EMBED_MODEL)
        self.emb = self.model.encode(["passage: " + t for t in texts],
                                     normalize_embeddings=True, batch_size=32)

    def _fused(self, queries):
        """Reciprocal rank fusion of dense + BM25 rankings across all queries, with precedence prior."""
        fused = {}
        for query in queries:
            q = self.model.encode(["query: " + query], normalize_embeddings=True)[0]
            dense = np.argsort(-(self.emb @ q))[:25]
            sparse = np.argsort(-self.bm25.get_scores(tokenize(query)))[:25]
            for ranking in (dense, sparse):
                for rank, idx in enumerate(ranking):
                    fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (60 + rank)
        for idx in fused:
            fused[idx] *= PRIOR.get(self.chunks[idx]["doc_type"], 1.0)
        return sorted(fused, key=fused.get, reverse=True)

    def search(self, queries, k=6, k_cat=2):
        """Top-k document sections plus at most k_cat catalog rows (separate slots)."""
        docs, cats = [], []
        for idx in self._fused(queries):
            c = self.chunks[idx]
            if c["doc_type"] == "catalog":
                if len(cats) < k_cat:
                    cats.append(c)
            elif len(docs) < k:
                docs.append(c)
            if len(docs) >= k and len(cats) >= k_cat:
                break
        return docs + cats

    def lookups(self, query):
        """Exact catalog / pincode rows for SKUs and 6-digit pincodes in the question."""
        out = []
        for sku in sorted(set(re.findall(r"\b[A-Za-z]{3}-\d{4}\b", query))):
            for _, r in self.products[self.products["sku"].str.upper() == sku.upper()].iterrows():
                out.append({"id": f"products.csv:{r['sku']}", "title": "Product catalog",
                            "doc_type": "catalog", "status": "current", "effective_from": "",
                            "text": row_text(r)})
        for pin in sorted(set(re.findall(r"(?<!\d)\d{6}(?!\d)", query))):
            for _, r in self.pincodes[self.pincodes["pincode"] == pin].iterrows():
                out.append({"id": f"pincodes.csv:{pin}", "title": "Delivery pincodes",
                            "doc_type": "catalog", "status": "current", "effective_from": "",
                            "text": row_text(r)})
        return out

    def context(self, query, extra_queries=(), k=6):
        """Exact lookups first, then search results; fewer catalog slots if a SKU was found."""
        exact = self.lookups(query)
        k_cat = 1 if any(c["id"].startswith("products.csv") for c in exact) else 2
        seen, items = set(), []
        for c in exact + self.search([query, *extra_queries], k=k, k_cat=k_cat):
            if c["id"] not in seen:
                seen.add(c["id"])
                items.append(c)
        return items


def format_context(items):
    blocks = []
    for c in items:
        meta = f"{c['title']} | type: {c['doc_type']} | status: {c['status']}"
        if c.get("effective_from"):
            meta += f" | effective_from: {c['effective_from']}"
        blocks.append(f"[{c['id']}] {meta}\n{c['text']}")
    return "\n\n".join(blocks)