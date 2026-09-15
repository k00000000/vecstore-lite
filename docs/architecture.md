# 架构说明

## 设计目标

1. **单文件可移植**：整个库就是一个 SQLite 文件。
2. **结果可解释**：默认精确相似度，而不是黑盒 ANN。
3. **RAG 友好**：同时保留 `text` 与 `metadata`，支持混合检索。
4. **可测试**：不依赖外部服务，CI 可秒级跑通。

## 模块划分

```text
vecstore_lite/
  store.py       # 核心：schema、CRUD、search、hybrid_search
  filters.py     # 元数据过滤 DSL
  embedding.py   # hash_embed / random_projection
  mmr.py         # Maximal Marginal Relevance
  cli.py         # 命令行入口
```

## 存储模型

- `collections`：集合注册表
- `vectors`：`(collection, id)` 主键，向量以 `float32` BLOB 存储
- `vectors_fts`：FTS5 虚表，对 `text` + 扁平化 metadata 建索引

写入路径会同步维护 FTS；删除会双删。

## 检索流水线

### 向量检索

1. 按 collection 拉取候选（可先元数据过滤）
2. NumPy 计算 cosine / l2 / dot
3. `argpartition` Top-K
4. 可选 MMR：在相关度与多样性之间权衡

### 混合检索

1. FTS5 `MATCH` 得到关键词分（BM25 → 归一化）
2. 向量检索得到向量分（min-max 归一化）
3. `score = alpha * vec + (1-alpha) * fts`
4. 可选 MMR

## 权衡

| 优点 | 代价 |
| --- | --- |
| 实现简单、依赖少 | 大数据量需全表向量加载 |
| 精确、可复现 | 不适合千万级在线服务 |
| 单文件备份容易 | 无内置复制集/分片 |

若规模增长，可把同一套 JSONL 迁到 Milvus，同时保留本库做本地开发与回归测试。
