# 使用指南

## 安装

```bash
cd vecstore-lite
pip install -e ".[dev]"
```

## 集合（Collection）

一个 `.db` 文件可包含多个集合，默认集合名为 `default`：

```python
store.ensure_collection("papers")
store.upsert("p1", [0.1, 0.2], {"year": 2024}, collection="papers", text="论文摘要")
print(store.list_collections())
```

## 元数据过滤

```python
store.search(
    [1, 0, 0],
    metadata_filter={
        "tag": {"$in": ["demo", "prod"]},
        "year": {"$gte": 2023, "$lte": 2026},
    },
)
```

## 混合检索

`hybrid_search` 会：

1. 对查询文本做 FTS5 关键词召回（BM25）
2. 用向量相似度召回
3. 按 `alpha` 做线性融合
4. 可选 MMR 重排

```python
hits = store.hybrid_search(
    query_text="本地 RAG 向量检索",
    top_k=5,
    alpha=0.6,   # 向量权重
    mmr=True,
)
```

若未提供 `query_vector`，默认使用 `hash_embed`（离线、无 API）。

## 随机投影降维

高维向量可在写入时压缩：

```python
store.upsert("x", high_dim_vector, project_to=128, projection_seed=42)
```

注意：查询向量需使用**相同** `project_to` / `seed`，否则空间不一致。

## CLI 速查

```bash
vecstore init demo.db
vecstore ingest demo.db examples/sample.jsonl
vecstore ingest-text demo.db examples/sample_texts.jsonl --dim 64
vecstore search demo.db --vector 1,0,0 --topk 3 --mmr
vecstore hybrid-search demo.db --text "巡检" --alpha 0.55
vecstore export demo.db out.jsonl --collection default
vecstore collections demo.db
vecstore stats demo.db
```

## 与外部嵌入模型集成

```python
def my_embed(text: str):
    # 替换成 OpenAI / DeepSeek / 本地 sentence-transformers 等
    ...

store.ingest_texts(items, embed_fn=my_embed)
store.hybrid_search(query_text="...", embed_fn=my_embed)
```
