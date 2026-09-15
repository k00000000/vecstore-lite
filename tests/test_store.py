from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vecstore_lite import VecStore, hash_embed
from vecstore_lite.cli import main
from vecstore_lite.filters import match_metadata


def test_upsert_search_filter(tmp_path: Path) -> None:
    db = tmp_path / "demo.db"
    with VecStore(db) as store:
        store.upsert("a", [1.0, 0.0, 0.0], {"tag": "red", "year": 2024}, text="alpha")
        store.upsert("b", [0.9, 0.1, 0.0], {"tag": "red", "year": 2025}, text="beta")
        store.upsert("c", [0.0, 1.0, 0.0], {"tag": "blue", "year": 2023}, text="gamma")
        hits = store.search([1.0, 0.0, 0.0], top_k=2, metadata_filter={"tag": "red"})
        assert [h.id for h in hits] == ["a", "b"]
        hits2 = store.search(
            [1.0, 0.0, 0.0],
            top_k=5,
            metadata_filter={"tag": {"$in": ["red"]}, "year": {"$gte": 2025}},
        )
        assert [h.id for h in hits2] == ["b"]
        assert store.count() == 3
        assert store.get("a")["metadata"]["tag"] == "red"
        assert store.delete("c") is True
        assert store.count() == 2


def test_collections_and_batch(tmp_path: Path) -> None:
    with VecStore(tmp_path / "c.db") as store:
        n = store.upsert_many(
            [
                {"id": "1", "vector": [1, 0], "metadata": {"k": 1}, "text": "one"},
                {"id": "2", "vector": [0, 1], "metadata": {"k": 2}, "text": "two"},
            ],
            collection="ns1",
        )
        assert n == 2
        assert store.count(collection="ns1") == 2
        assert store.count(collection="default") == 0
        names = {c["name"] for c in store.list_collections()}
        assert "ns1" in names


def test_hybrid_and_mmr(tmp_path: Path) -> None:
    with VecStore(tmp_path / "h.db") as store:
        store.ingest_texts(
            [
                {"id": "a", "text": "无人机巡检影像检索", "metadata": {"domain": "uav"}},
                {"id": "b", "text": "本地混合检索与向量库", "metadata": {"domain": "rag"}},
                {"id": "c", "text": "电力线路缺陷检测", "metadata": {"domain": "uav"}},
            ],
            dim=32,
        )
        hits = store.hybrid_search(query_text="巡检影像", top_k=2, dim=32, alpha=0.5)
        assert hits
        assert hits[0].id in {"a", "c"}
        mmr_hits = store.search(hash_embed("巡检", dim=32), top_k=2, mmr=True)
        assert len(mmr_hits) <= 2


def test_export_import_and_cli(tmp_path: Path) -> None:
    db = tmp_path / "demo.db"
    jsonl = tmp_path / "rows.jsonl"
    rows = [
        {"id": "v1", "vector": [1.0, 0.0], "metadata": {"source": "demo"}, "text": "hello"},
        {"id": "v2", "vector": [0.0, 1.0], "metadata": {"source": "demo"}, "text": "world"},
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert main(["ingest", str(db), str(jsonl)]) == 0
    assert main(["search", str(db), "--vector", "1,0", "--topk", "1"]) == 0
    out = tmp_path / "out.jsonl"
    assert main(["export", str(db), str(out)]) == 0
    assert out.exists()
    assert main(["stats", str(db)]) == 0
    assert main(["collections", str(db)]) == 0
    assert main(["hybrid-search", str(db), "--text", "hello", "--dim", "2"]) in {0, 2}


def test_projection_and_filters() -> None:
    assert match_metadata({"a": 1}, {"a": {"$ne": 2}})
    v = hash_embed("abc", dim=16)
    assert v.shape == (16,)
    from vecstore_lite.embedding import random_projection

    p = random_projection(np.arange(32, dtype=np.float32), 8, seed=1)
    assert p.shape == (8,)


def test_l2_metric(tmp_path: Path) -> None:
    with VecStore(tmp_path / "x.db") as store:
        store.upsert("near", np.array([1.0, 1.0], dtype=np.float32))
        store.upsert("far", np.array([10.0, 10.0], dtype=np.float32))
        hits = store.search([1.1, 0.9], top_k=1, metric="l2")
        assert hits[0].id == "near"
