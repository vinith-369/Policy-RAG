"""
Qdrant in-memory client with hybrid search.

Dense:  sentence-transformers/all-MiniLM-L6-v2 (384-dim, semantic)
Sparse: rank-bm25 (pure Python BM25 — no ONNX, works on Python 3.14)
Fusion: Reciprocal Rank Fusion (RRF) via separate dense+sparse queries and score merge
"""
import math
import re
import uuid
from collections import Counter
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    NamedVector,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from app.config import settings

# ─── Singletons ───────────────────────────────────────────────────────────────

_qdrant_client: Optional[QdrantClient] = None
_dense_model:   Optional[SentenceTransformer] = None

# BM25 corpus state (maintained in-memory alongside Qdrant)
_corpus_tokens: list[list[str]] = []   # tokenized docs
_corpus_texts:  list[str]       = []   # raw texts (parallel to Qdrant points)
_bm25_index = None                     # BM25Okapi instance





def get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(":memory:")
    return _qdrant_client


def get_dense_model() -> SentenceTransformer:
    global _dense_model
    if _dense_model is None:
        _dense_model = SentenceTransformer(settings.dense_model, device="cpu")
    return _dense_model


# ─── Simple tokenizer ────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer — strips punctuation."""
    return re.findall(r"\b[a-z]{2,}\b", text.lower())


# ─── Collection management ────────────────────────────────────────────────────

def ensure_collection() -> None:
    client = get_qdrant()
    if not client.collection_exists(settings.collection_name):
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=settings.dense_vector_size,
                    distance=Distance.COSINE,
                )
            },
        )


# ─── BM25 helpers ─────────────────────────────────────────────────────────────

def _rebuild_bm25() -> None:
    """Rebuild the BM25 index from current corpus."""
    global _bm25_index
    if not _corpus_tokens:
        _bm25_index = None
        return
    from rank_bm25 import BM25Okapi
    _bm25_index = BM25Okapi(_corpus_tokens)


def _bm25_scores(query: str) -> list[float]:
    """Return BM25 scores for the query over the current corpus."""
    if _bm25_index is None:
        return []
    tokens = _tokenize(query)
    if not tokens:
        return [0.0] * len(_corpus_texts)
    return _bm25_index.get_scores(tokens).tolist()


# ─── Embedding helpers ────────────────────────────────────────────────────────

def embed_dense(texts: list[str]) -> list[list[float]]:
    return get_dense_model().encode(texts, show_progress_bar=False).tolist()


# ─── Upsert ───────────────────────────────────────────────────────────────────

def upsert_chunks(chunks: list[dict]) -> int:
    """
    Upsert chunk dicts into Qdrant (dense vectors) and update BM25 corpus.
    Each dict: { text, source_file, chunk_index, category }
    Returns number of points upserted.
    """
    global _corpus_texts, _corpus_tokens

    ensure_collection()
    client = get_qdrant()

    texts      = [c["text"] for c in chunks]
    dense_vecs = embed_dense(texts)

    points = []
    for chunk, dense_vec in zip(chunks, dense_vecs):
        point_id = str(uuid.uuid4())
        points.append(
            PointStruct(
                id=point_id,
                vector={"dense": dense_vec},
                payload={
                    "text":        chunk["text"],
                    "source_file": chunk.get("source_file", "unknown"),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "category":    chunk.get("category", "general"),
                    "corpus_idx":  len(_corpus_texts) + len(points),
                },
            )
        )

    client.upsert(collection_name=settings.collection_name, points=points)

    # Update BM25 corpus
    _corpus_texts.extend(texts)
    _corpus_tokens.extend([_tokenize(t) for t in texts])
    _rebuild_bm25()

    return len(points)


# ─── Hybrid search ────────────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    top_k: int = 5,
    filter_source: Optional[str] = None,
) -> list[dict]:
    """
    Hybrid search using RRF fusion of dense + BM25 results.

    Strategy:
    1. Dense search → top 20 candidates from Qdrant
    2. BM25 scores over full corpus
    3. RRF merge on overlapping + BM25 top candidates
    4. Return top_k
    """
    ensure_collection()
    client = get_qdrant()

    # Filter builder
    query_filter = None
    if filter_source:
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        query_filter = Filter(
            must=[FieldCondition(key="source_file", match=MatchValue(value=filter_source))]
        )

    # ── Dense retrieval ───────────────────────────────────────────────────────
    dense_vec = embed_dense([query])[0]
    dense_results = client.query_points(
        collection_name=settings.collection_name,
        query=dense_vec,
        using="dense",
        limit=min(top_k * 4, 20),
        with_payload=True,
        query_filter=query_filter,
    ).points

    # ── BM25 retrieval ────────────────────────────────────────────────────────
    bm25_scores_all = _bm25_scores(query)

    # Build corpus_idx → bm25_score for all corpus entries
    bm25_ranked: dict[int, float] = {}
    if bm25_scores_all:
        for i, score in enumerate(bm25_scores_all):
            bm25_ranked[i] = score

    # ── Build per-point score maps ────────────────────────────────────────────
    # Map: point_id -> { text, source_file, ... , dense_rank, bm25_score }
    point_map: dict[str, dict] = {}

    for rank, r in enumerate(dense_results, start=1):
        pid = str(r.id)
        point_map[pid] = {
            "text":        r.payload["text"],
            "source_file": r.payload.get("source_file", "unknown"),
            "chunk_index": r.payload.get("chunk_index", 0),
            "dense_rank":  rank,
            "bm25_score":  bm25_ranked.get(r.payload.get("corpus_idx", -1), 0.0),
            "dense_score": r.score,
        }

    # Also rank BM25 top candidates that may not be in dense results
    # Get top BM25 points from Qdrant storage as well
    if bm25_scores_all:
        top_bm25_indices = sorted(
            range(len(bm25_scores_all)),
            key=lambda i: bm25_scores_all[i],
            reverse=True,
        )[:top_k * 2]

        # Fetch those points by corpus_idx from Qdrant
        scroll_results, _ = client.scroll(
            collection_name=settings.collection_name,
            limit=500,
            with_payload=True,
            with_vectors=False,
        )
        corpus_idx_map: dict[int, any] = {}
        for pt in scroll_results:
            cidx = pt.payload.get("corpus_idx", -1)
            corpus_idx_map[cidx] = pt

        for rank, cidx in enumerate(top_bm25_indices, start=1):
            pt = corpus_idx_map.get(cidx)
            if pt is None:
                continue
            pid = str(pt.id)
            if pid not in point_map:
                point_map[pid] = {
                    "text":        pt.payload["text"],
                    "source_file": pt.payload.get("source_file", "unknown"),
                    "chunk_index": pt.payload.get("chunk_index", 0),
                    "dense_rank":  len(dense_results) + rank,  # penalized rank
                    "bm25_score":  bm25_scores_all[cidx],
                    "dense_score": 0.0,
                }
            else:
                # Already in map — update bm25_score
                point_map[pid]["bm25_score"] = bm25_scores_all[cidx]

    # ── RRF fusion ────────────────────────────────────────────────────────────
    # Rank BM25 candidates by score
    bm25_rank_map: dict[str, int] = {}
    bm25_sorted = sorted(
        point_map.items(), key=lambda x: x[1]["bm25_score"], reverse=True
    )
    for rank, (pid, _) in enumerate(bm25_sorted, start=1):
        bm25_rank_map[pid] = rank

    K = 60  # RRF constant
    rrf_scores: dict[str, float] = {}
    for pid, data in point_map.items():
        dense_r = data["dense_rank"]
        bm25_r  = bm25_rank_map.get(pid, len(point_map))
        rrf_scores[pid] = 1.0 / (K + dense_r) + 1.0 / (K + bm25_r)

    # Sort by RRF score
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for pid, score in ranked[:top_k]:
        data = point_map[pid]
        results.append({
            "text":        data["text"],
            "source_file": data["source_file"],
            "chunk_index": data["chunk_index"],
            "score":       score,
        })

    return results


# ─── Utility ──────────────────────────────────────────────────────────────────

def collection_count() -> int:
    """Return number of points currently in the collection (0 if not exists)."""
    client = get_qdrant()
    if not client.collection_exists(settings.collection_name):
        return 0
    return client.count(collection_name=settings.collection_name).count
