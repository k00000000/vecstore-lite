# VecStore Lite

面向本地 RAG / 语义检索原型的 **嵌入式向量库**。

- 存储：SQLite（单文件可移植）
- 检索：NumPy 精确相似度 + 可选 FTS5 关键词混合
- 接口：Python 库 + `vecstore` CLI
- **无需 Docker / 无需常驻服务 / 无需云账号**

当你还用不上 Milvus/Qdrant，但又希望比「手写 numpy 数组」更完整时，这就是合适的中间层。

## 功能一览

| 能力 | 说明 |
| --- | --- |
| 集合（collection） | 单库多命名空间 |
| upsert / upsert_many | 事务批量写入 |
| search | cosine / l2 / dot，支持元数据过滤 |
| hybrid-search | 向量分 + FTS5 关键词分线性融合 |
| MMR | 多样性重排，减少结果扎堆 |
| 元数据过滤 | 等值、`$in`、`$gte` / `$lte` / `$ne` |
| 离线哈希嵌入 | `hash_embed` / `ingest-text`，无需 API Key |
| 随机投影降维 | `project_to`，压缩高维向量 |
| export / ingest | JSONL 导入导出 |
| CLI | `init` `add` `ingest` `ingest-text` `search` `hybrid-search` `delete` `export` `stats` `collections` |

## 环境要求

- Python 3.10+
- NumPy

## 安装

```bash
pip install -e ".[dev]"
```

## 快速开始（CLI）

```bash
vecstore init demo.db

vecstore add demo.db --id doc1 --vector 1,0,0 --meta "{\"tag\":\"alpha\"}" --text "向量检索入门"
vecstore add demo.db --id doc2 --vector 0.9,0.1,0 --meta "{\"tag\":\"alpha\"}" --text "本地 RAG 实践"
vecstore add demo.db --id doc3 --vector 0,1,0 --meta "{\"tag\":\"beta\"}" --text "其他主题"

vecstore search demo.db --vector 1,0,0 --topk 2 --filter "{\"tag\":\"alpha\"}"
vecstore hybrid-search demo.db --text "本地 RAG" --topk 3
vecstore stats demo.db
```

## 快速开始（库）

```python
from vecstore_lite import VecStore, hash_embed

with VecStore("demo.db") as store:
    store.upsert("doc1", [1.0, 0.0, 0.0], {"tag": "alpha"}, text="向量检索入门")
    store.ingest_texts(
        [
            {"id": "t1", "text": "无人机巡检影像检索", "metadata": {"domain": "uav"}},
            {"id": "t2", "text": "电力线路缺陷检测", "metadata": {"domain": "power"}},
        ],
        dim=64,
    )
    hits = store.hybrid_search(query_text="巡检影像", top_k=5, alpha=0.55, mmr=True)
    for hit in hits:
        print(hit.id, hit.score, hit.text)
```

## JSONL 格式

向量导入：

```json
{"id": "doc1", "vector": [0.1, 0.2, 0.3], "metadata": {"source": "wiki"}, "text": "可选原文"}
```

文本导入（`ingest-text`）：

```json
{"id": "doc1", "text": "本地混合检索示例", "metadata": {"lang": "zh"}}
```

## 何时用它，何时上 Milvus

| 场景 | 更推荐 |
| --- | --- |
| 本地原型 / CI / 单测 / 课程演示 | **vecstore-lite** |
| 百万级向量、分布式、生产 ANN | Milvus / Qdrant 等 |

本项目刻意使用**精确检索**（可加 MMR），结果可复现、易推理；不是近似最近邻引擎。

## 运行测试

```bash
pytest -q
```

## 文档

- [使用指南](docs/usage.md)
- [架构说明](docs/architecture.md)

## License

MIT
