"""
Knowledge-base indexing and retrieval.

Chunking strategy (per the KB README):
  - Split each markdown file on standalone '---' horizontal rules.
  - Track heading hierarchy (#, ##, ###...) so every chunk knows the
    section path it came from, even though the heading itself may live
    in a previous chunk.
  - Tables are left intact inside whichever chunk they fall in, so an
    error-code table stays queryable as one atomic unit.

Retrieval: a dependency-free BM25 implementation (stdlib + math only).
This keeps the project runnable with zero extra installs and is a fair
baseline for a KB of this size; swapping in embeddings later only means
replacing KBIndex.search's internals -- callers don't change.
"""
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List

TOKEN_RE = re.compile(r"[a-z0-9]+")
HR_RE = re.compile(r"^\s*-{3,}\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class Chunk:
    doc_path: str          # path relative to kb_root, e.g. products/databridge-pro.md
    heading_path: str      # e.g. "products/databridge-pro.md > Error Codes"
    text: str
    tokens: List[str] = field(default_factory=list)


class KBIndex:
    def __init__(self, kb_root: str):
        self.kb_root = kb_root
        self.chunks: List[Chunk] = []
        self._df: Dict[str, int] = {}
        self._avgdl: float = 0.0
        self._build()

    # ---------- indexing ----------

    def _build(self) -> None:
        if not os.path.isdir(self.kb_root):
            raise FileNotFoundError(f"Knowledge base folder not found: {self.kb_root}")
        for dirpath, _dirnames, filenames in os.walk(self.kb_root):
            for fn in sorted(filenames):
                if fn.lower().endswith(".md"):
                    full_path = os.path.join(dirpath, fn)
                    rel_path = os.path.relpath(full_path, self.kb_root).replace(os.sep, "/")
                    self._index_file(full_path, rel_path)
        self._compute_stats()

    def _index_file(self, full_path: str, rel_path: str) -> None:
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()

        heading_stack: Dict[int, str] = {}
        current_chunk_lines: List[str] = []
        current_heading_path = rel_path

        def flush():
            text = "\n".join(current_chunk_lines).strip()
            if text:
                self.chunks.append(
                    Chunk(
                        doc_path=rel_path,
                        heading_path=current_heading_path,
                        text=text,
                        tokens=tokenize(text),
                    )
                )

        for line in content.splitlines():
            if HR_RE.match(line):
                flush()
                current_chunk_lines = []
                continue

            heading_match = HEADING_RE.match(line)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                for existing_level in [l for l in heading_stack if l >= level]:
                    del heading_stack[existing_level]
                heading_stack[level] = title
                current_heading_path = rel_path + " > " + " > ".join(
                    heading_stack[l] for l in sorted(heading_stack)
                )

            current_chunk_lines.append(line)

        flush()

    def _compute_stats(self) -> None:
        total_len = 0
        for chunk in self.chunks:
            total_len += len(chunk.tokens)
            for term in set(chunk.tokens):
                self._df[term] = self._df.get(term, 0) + 1
        self._avgdl = total_len / len(self.chunks) if self.chunks else 0.0

    # ---------- retrieval ----------

    def search(self, query: str, top_k: int = 3, k1: float = 1.5, b: float = 0.75) -> List[dict]:
        """Return up to top_k chunks as plain dicts, ranked by BM25 score desc."""
        q_tokens = tokenize(query)
        if not q_tokens or not self.chunks:
            return []

        n_docs = len(self.chunks)
        scored: List[tuple] = []

        for chunk in self.chunks:
            dl = len(chunk.tokens) or 1
            tf_counts: Dict[str, int] = {}
            for t in chunk.tokens:
                tf_counts[t] = tf_counts.get(t, 0) + 1

            score = 0.0
            for term in q_tokens:
                df = self._df.get(term, 0)
                if df == 0:
                    continue
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                tf = tf_counts.get(term, 0)
                denom = tf + k1 * (1 - b + b * dl / max(self._avgdl, 1e-6))
                score += idf * (tf * (k1 + 1)) / max(denom, 1e-6)

            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        results = []
        for score, chunk in scored[:top_k]:
            snippet = chunk.text if len(chunk.text) <= 700 else chunk.text[:700] + "..."
            results.append(
                {
                    "doc_path": chunk.doc_path,
                    "heading_path": chunk.heading_path,
                    "snippet": snippet,
                    "score": round(score, 4),
                }
            )
        return results

    def known_product_names(self) -> List[str]:
        """Filenames under products/ -> human-readable product names, for prompt hints."""
        products_dir = os.path.join(self.kb_root, "products")
        if not os.path.isdir(products_dir):
            return []
        names = []
        for fn in os.listdir(products_dir):
            if fn.lower().endswith(".md"):
                stem = os.path.splitext(fn)[0]
                names.append(stem.replace("-", " ").replace("_", " ").title())
        return names
