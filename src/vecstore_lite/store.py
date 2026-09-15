from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .embedding import hash_embed, random_projection
from .filters import match_metadata
from .mmr import mmr_rerank

EmbedFn = Callable[[str], np.ndarray]


@dataclass(frozen=True)
class SearchHit:
    id: str
    score: float
    metadata: dict[str, Any]
    collection: str = "default"
    text: str | None = None


class VecStore:
    """嵌入式向量库：SQLite 存储 + NumPy 精确检索。

    适合本地 RAG 原型、单测与中小规模语料（通常 < 10 万向量，视内存而定）。
    """

    def __init__(self, path: str | Path, *, default_collection: str = "default"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.default_collection = default_collection
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "VecStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS collections (
              name TEXT PRIMARY KEY,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS vectors (
              collection TEXT NOT NULL DEFAULT 'default',
              id TEXT NOT NULL,
              dim INTEGER NOT NULL,
              embedding BLOB NOT NULL,
              metadata TEXT NOT NULL DEFAULT '{}',
              text TEXT,
              created_at TEXT NOT NULL,
              PRIMARY KEY (collection, id),
              FOREIGN KEY (collection) REFERENCES collections(name)
            );

            CREATE INDEX IF NOT EXISTS idx_vectors_collection
              ON vectors(collection);

            CREATE VIRTUAL TABLE IF NOT EXISTS vectors_fts USING fts5(
              collection,
              id,
              text,
              metadata,
              content='',
              tokenize='unicode61'
            );
            """
        )
        # 确保默认集合存在
        now = _utcnow()
        self.conn.execute(
            "INSERT OR IGNORE INTO collections(name, created_at) VALUES (?, ?)",
            ("default", now),
        )
        self.conn.commit()

    @staticmethod
    def _pack(vector: np.ndarray) -> bytes:
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        return arr.tobytes(order="C")

    @staticmethod
    def _unpack(blob: bytes, dim: int) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32, count=dim).copy()

    def ensure_collection(self, name: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO collections(name, created_at) VALUES (?, ?)",
            (name, _utcnow()),
        )
        self.conn.commit()

    def list_collections(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT c.name, c.created_at, COUNT(v.id) AS count
            FROM collections c
            LEFT JOIN vectors v ON v.collection = c.name
            GROUP BY c.name
            ORDER BY c.name
            """
        ).fetchall()
        return [
            {"name": r["name"], "created_at": r["created_at"], "count": int(r["count"])}
            for r in rows
        ]

    def upsert(
        self,
        id: str,
        vector: Iterable[float] | np.ndarray,
        metadata: dict[str, Any] | None = None,
        *,
        collection: str | None = None,
        text: str | None = None,
        project_to: int | None = None,
        projection_seed: int = 42,
    ) -> None:
        coll = collection or self.default_collection
        self.ensure_collection(coll)
        arr = np.asarray(list(vector) if not isinstance(vector, np.ndarray) else vector, dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("vector must be a non-empty 1-D array")
        if project_to is not None:
            arr = random_projection(arr, project_to, seed=projection_seed)
        meta = metadata or {}
        now = _utcnow()
        self.conn.execute(
            """
            INSERT INTO vectors(collection, id, dim, embedding, metadata, text, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(collection, id) DO UPDATE SET
              dim=excluded.dim,
              embedding=excluded.embedding,
              metadata=excluded.metadata,
              text=excluded.text,
              created_at=excluded.created_at
            """,
            (
                coll,
                id,
                int(arr.size),
                self._pack(arr),
                json.dumps(meta, ensure_ascii=False),
                text,
                now,
            ),
        )
        self._sync_fts(coll, id, text, meta)
        self.conn.commit()

    def upsert_many(
        self,
        rows: Iterable[dict[str, Any]],
        *,
        collection: str | None = None,
        project_to: int | None = None,
        projection_seed: int = 42,
    ) -> int:
        coll = collection or self.default_collection
        self.ensure_collection(coll)
        count = 0
        now = _utcnow()
        try:
            for row in rows:
                rid = str(row["id"])
                arr = np.asarray(row["vector"], dtype=np.float32)
                if project_to is not None:
                    arr = random_projection(arr, project_to, seed=projection_seed)
                meta = row.get("metadata") or {}
                text = row.get("text")
                self.conn.execute(
                    """
                    INSERT INTO vectors(collection, id, dim, embedding, metadata, text, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(collection, id) DO UPDATE SET
                      dim=excluded.dim,
                      embedding=excluded.embedding,
                      metadata=excluded.metadata,
                      text=excluded.text,
                      created_at=excluded.created_at
                    """,
                    (
                        coll,
                        rid,
                        int(arr.size),
                        self._pack(arr),
                        json.dumps(meta, ensure_ascii=False),
                        text,
                        now,
                    ),
                )
                self._sync_fts(coll, rid, text, meta)
                count += 1
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return count

    def _sync_fts(self, collection: str, id: str, text: str | None, meta: dict[str, Any]) -> None:
        self.conn.execute(
            "DELETE FROM vectors_fts WHERE collection = ? AND id = ?",
            (collection, id),
        )
        blob_text = text or ""
        meta_text = " ".join(f"{k} {v}" for k, v in meta.items())
        if blob_text or meta_text:
            self.conn.execute(
                "INSERT INTO vectors_fts(collection, id, text, metadata) VALUES (?, ?, ?, ?)",
                (collection, id, blob_text, meta_text),
            )

    def ingest_jsonl(
        self,
        path: str | Path,
        *,
        collection: str | None = None,
        project_to: int | None = None,
    ) -> int:
        rows: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"第 {line_no} 行 JSON 无效") from exc
                if "id" not in row or "vector" not in row:
                    raise ValueError(f"第 {line_no} 行必须包含 id 与 vector")
                rows.append(row)
        return self.upsert_many(rows, collection=collection, project_to=project_to)

    def ingest_texts(
        self,
        items: Iterable[dict[str, Any]],
        *,
        collection: str | None = None,
        dim: int = 64,
        embed_fn: EmbedFn | None = None,
    ) -> int:
        """对文本条目做离线嵌入后写入。每项至少含 id、text。"""
        embed = embed_fn or (lambda t: hash_embed(t, dim=dim))
        rows = []
        for item in items:
            text = str(item["text"])
            rows.append(
                {
                    "id": str(item["id"]),
                    "vector": embed(text),
                    "metadata": item.get("metadata") or {},
                    "text": text,
                }
            )
        return self.upsert_many(rows, collection=collection)

    def get(self, id: str, *, collection: str | None = None) -> dict[str, Any] | None:
        coll = collection or self.default_collection
        row = self.conn.execute(
            "SELECT * FROM vectors WHERE collection = ? AND id = ?",
            (coll, id),
        ).fetchone()
        if row is None:
            return None
        return {
            "collection": row["collection"],
            "id": row["id"],
            "vector": self._unpack(row["embedding"], row["dim"]).tolist(),
            "metadata": json.loads(row["metadata"]),
            "text": row["text"],
            "created_at": row["created_at"],
        }

    def delete(self, id: str, *, collection: str | None = None) -> bool:
        coll = collection or self.default_collection
        cur = self.conn.execute(
            "DELETE FROM vectors WHERE collection = ? AND id = ?",
            (coll, id),
        )
        self.conn.execute(
            "DELETE FROM vectors_fts WHERE collection = ? AND id = ?",
            (coll, id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def count(self, *, collection: str | None = None) -> int:
        if collection:
            row = self.conn.execute(
                "SELECT COUNT(*) AS c FROM vectors WHERE collection = ?",
                (collection,),
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) AS c FROM vectors").fetchone()
        return int(row["c"])

    def stats(self) -> dict[str, Any]:
        dim_rows = self.conn.execute(
            "SELECT collection, dim, COUNT(*) AS c FROM vectors GROUP BY collection, dim"
        ).fetchall()
        dims: dict[str, dict[str, int]] = {}
        for row in dim_rows:
            dims.setdefault(row["collection"], {})[str(row["dim"])] = int(row["c"])
        return {
            "path": str(self.path),
            "count": self.count(),
            "collections": self.list_collections(),
            "dims": dims,
        }

    def export_jsonl(self, path: str | Path, *, collection: str | None = None) -> int:
        coll = collection or self.default_collection
        rows = self.conn.execute(
            "SELECT id, dim, embedding, metadata, text FROM vectors WHERE collection = ?",
            (coll,),
        ).fetchall()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            for row in rows:
                payload = {
                    "id": row["id"],
                    "vector": self._unpack(row["embedding"], row["dim"]).tolist(),
                    "metadata": json.loads(row["metadata"]),
                }
                if row["text"] is not None:
                    payload["text"] = row["text"]
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return len(rows)

    def search(
        self,
        query: Iterable[float] | np.ndarray,
        *,
        top_k: int = 5,
        metric: str = "cosine",
        metadata_filter: dict[str, Any] | None = None,
        collection: str | None = None,
        mmr: bool = False,
        mmr_lambda: float = 0.7,
        candidate_multiplier: int = 4,
    ) -> list[SearchHit]:
        if top_k <= 0:
            return []
        q = np.asarray(list(query) if not isinstance(query, np.ndarray) else query, dtype=np.float32)
        if q.ndim != 1:
            raise ValueError("query must be 1-D")

        coll = collection or self.default_collection
        rows = self.conn.execute(
            "SELECT id, dim, embedding, metadata, text, collection FROM vectors WHERE collection = ?",
            (coll,),
        ).fetchall()
        if not rows:
            return []

        ids: list[str] = []
        metas: list[dict[str, Any]] = []
        texts: list[str | None] = []
        collections: list[str] = []
        vectors: list[np.ndarray] = []
        for row in rows:
            meta = json.loads(row["metadata"])
            if not match_metadata(meta, metadata_filter):
                continue
            if int(row["dim"]) != int(q.size):
                continue
            ids.append(row["id"])
            metas.append(meta)
            texts.append(row["text"])
            collections.append(row["collection"])
            vectors.append(self._unpack(row["embedding"], row["dim"]))

        if not vectors:
            return []

        mat = np.vstack(vectors)
        scores = _similarity(mat, q, metric)
        fetch_k = min(len(ids), top_k * candidate_multiplier if mmr else top_k)
        idx = np.argpartition(-scores, kth=fetch_k - 1)[:fetch_k]
        idx = idx[np.argsort(-scores[idx])]

        if mmr:
            cand_vecs = mat[idx]
            cand_scores = scores[idx]
            chosen_local = mmr_rerank(
                [ids[i] for i in idx],
                cand_scores,
                cand_vecs,
                top_k=top_k,
                lambda_mult=mmr_lambda,
            )
            idx = np.array([idx[i] for i in chosen_local], dtype=int)

        idx = idx[:top_k]
        return [
            SearchHit(
                id=ids[i],
                score=float(scores[i]),
                metadata=metas[i],
                collection=collections[i],
                text=texts[i],
            )
            for i in idx
        ]

    def hybrid_search(
        self,
        *,
        query_text: str,
        query_vector: Iterable[float] | np.ndarray | None = None,
        top_k: int = 5,
        alpha: float = 0.6,
        metric: str = "cosine",
        metadata_filter: dict[str, Any] | None = None,
        collection: str | None = None,
        embed_fn: EmbedFn | None = None,
        dim: int = 64,
        mmr: bool = False,
    ) -> list[SearchHit]:
        """混合检索：向量相似度 + FTS5 关键词，线性融合分数。

        ``alpha`` 为向量分权重，``1-alpha`` 为关键词分权重。
        """
        coll = collection or self.default_collection
        if query_vector is None:
            embed = embed_fn or (lambda t: hash_embed(t, dim=dim))
            q = embed(query_text)
        else:
            q = np.asarray(list(query_vector) if not isinstance(query_vector, np.ndarray) else query_vector, dtype=np.float32)

        # FTS 候选
        fts_scores: dict[str, float] = {}
        try:
            fts_rows = self.conn.execute(
                """
                SELECT id, bm25(vectors_fts) AS rank
                FROM vectors_fts
                WHERE vectors_fts MATCH ? AND collection = ?
                ORDER BY rank
                LIMIT ?
                """,
                (_fts_query(query_text), coll, max(top_k * 10, 20)),
            ).fetchall()
            if fts_rows:
                raw = np.array([-float(r["rank"]) for r in fts_rows], dtype=np.float32)
                # bm25 越小越好，上面取负；再 min-max
                normed = _minmax(raw)
                for row, score in zip(fts_rows, normed):
                    fts_scores[row["id"]] = float(score)
        except sqlite3.OperationalError:
            # 查询语法不合法时忽略关键词通道
            fts_scores = {}

        vector_hits = self.search(
            q,
            top_k=max(top_k * 5, 20),
            metric=metric,
            metadata_filter=metadata_filter,
            collection=coll,
            mmr=False,
        )
        if not vector_hits and not fts_scores:
            return []

        vec_scores = {h.id: h.score for h in vector_hits}
        if vec_scores:
            vals = np.array(list(vec_scores.values()), dtype=np.float32)
            normed = _minmax(vals)
            vec_scores = {k: float(v) for k, v in zip(vec_scores.keys(), normed)}

        all_ids = list(dict.fromkeys([*vec_scores.keys(), *fts_scores.keys()]))
        meta_map = {h.id: h for h in vector_hits}
        # 补齐仅 FTS 命中的元数据
        for rid in all_ids:
            if rid not in meta_map:
                row = self.get(rid, collection=coll)
                if row is None:
                    continue
                if not match_metadata(row["metadata"], metadata_filter):
                    continue
                meta_map[rid] = SearchHit(
                    id=rid,
                    score=0.0,
                    metadata=row["metadata"],
                    collection=coll,
                    text=row.get("text"),
                )

        fused: list[SearchHit] = []
        for rid in all_ids:
            if rid not in meta_map:
                continue
            score = alpha * vec_scores.get(rid, 0.0) + (1.0 - alpha) * fts_scores.get(rid, 0.0)
            base = meta_map[rid]
            fused.append(
                SearchHit(
                    id=rid,
                    score=float(score),
                    metadata=base.metadata,
                    collection=base.collection,
                    text=base.text,
                )
            )
        fused.sort(key=lambda h: h.score, reverse=True)
        fused = fused[: max(top_k * 4, top_k)]

        if mmr and fused:
            vectors = []
            for h in fused:
                row = self.get(h.id, collection=coll)
                assert row is not None
                vectors.append(np.asarray(row["vector"], dtype=np.float32))
            mat = np.vstack(vectors)
            scores = np.array([h.score for h in fused], dtype=np.float32)
            chosen = mmr_rerank(
                [h.id for h in fused],
                scores,
                mat,
                top_k=top_k,
            )
            return [fused[i] for i in chosen]
        return fused[:top_k]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _similarity(mat: np.ndarray, q: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        qn = q / (np.linalg.norm(q) + 1e-12)
        mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
        return mn @ qn
    if metric == "l2":
        return -np.linalg.norm(mat - q, axis=1)
    if metric == "dot":
        return mat @ q
    raise ValueError("metric 必须是 cosine / l2 / dot")


def _minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if abs(hi - lo) < 1e-12:
        return np.ones_like(values)
    return (values - lo) / (hi - lo)


def _fts_query(text: str) -> str:
    # 简单分词后 OR，避免用户输入破坏 FTS 语法
    tokens = [t for t in text.replace('"', " ").split() if t]
    if not tokens:
        # 中文：按字
        tokens = [ch for ch in text if not ch.isspace()][:8]
    cleaned = []
    for token in tokens[:12]:
        token = "".join(ch for ch in token if ch.isalnum() or ord(ch) > 127)
        if token:
            cleaned.append(f'"{token}"')
    return " OR ".join(cleaned) if cleaned else '""'
