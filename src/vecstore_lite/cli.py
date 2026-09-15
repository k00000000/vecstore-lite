from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .embedding import hash_embed
from .store import VecStore


def _parse_vector(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vecstore",
        description="嵌入式 SQLite 向量库：本地语义检索 / RAG 原型工具。",
    )
    parser.add_argument("--version", action="store_true")
    sub = parser.add_subparsers(dest="command")

    init = sub.add_parser("init", help="创建空库")
    init.add_argument("db")

    add = sub.add_parser("add", help="写入/更新一条向量")
    add.add_argument("db")
    add.add_argument("--id", required=True)
    add.add_argument("--vector", required=True, help="逗号分隔浮点数")
    add.add_argument("--meta", default="{}", help="JSON 元数据")
    add.add_argument("--text", default=None, help="可选原文，用于混合检索")
    add.add_argument("--collection", default="default")
    add.add_argument("--project-to", type=int, default=None, help="随机投影降维目标维度")

    ingest = sub.add_parser("ingest", help="批量导入 JSONL")
    ingest.add_argument("db")
    ingest.add_argument("jsonl")
    ingest.add_argument("--collection", default="default")
    ingest.add_argument("--project-to", type=int, default=None)

    ingest_text = sub.add_parser("ingest-text", help="对文本 JSONL 做离线哈希嵌入并导入")
    ingest_text.add_argument("db")
    ingest_text.add_argument("jsonl", help='每行 {"id","text","metadata"?}')
    ingest_text.add_argument("--collection", default="default")
    ingest_text.add_argument("--dim", type=int, default=64)

    search = sub.add_parser("search", help="向量近邻检索")
    search.add_argument("db")
    search.add_argument("--vector", required=True)
    search.add_argument("--topk", type=int, default=5)
    search.add_argument("--metric", choices=["cosine", "l2", "dot"], default="cosine")
    search.add_argument("--filter", dest="metadata_filter", default=None, help="JSON 过滤条件")
    search.add_argument("--collection", default="default")
    search.add_argument("--mmr", action="store_true", help="启用 MMR 多样性重排")

    hybrid = sub.add_parser("hybrid-search", help="向量 + 关键词混合检索")
    hybrid.add_argument("db")
    hybrid.add_argument("--text", required=True)
    hybrid.add_argument("--vector", default=None, help="可选；缺省则对 --text 做哈希嵌入")
    hybrid.add_argument("--topk", type=int, default=5)
    hybrid.add_argument("--alpha", type=float, default=0.6, help="向量分权重")
    hybrid.add_argument("--filter", dest="metadata_filter", default=None)
    hybrid.add_argument("--collection", default="default")
    hybrid.add_argument("--mmr", action="store_true")
    hybrid.add_argument("--dim", type=int, default=64)

    get = sub.add_parser("get", help="按 id 读取")
    get.add_argument("db")
    get.add_argument("--id", required=True)
    get.add_argument("--collection", default="default")

    delete = sub.add_parser("delete", help="删除一条记录")
    delete.add_argument("db")
    delete.add_argument("--id", required=True)
    delete.add_argument("--collection", default="default")

    export = sub.add_parser("export", help="导出 JSONL")
    export.add_argument("db")
    export.add_argument("out")
    export.add_argument("--collection", default="default")

    stats = sub.add_parser("stats", help="统计信息")
    stats.add_argument("db")

    collections = sub.add_parser("collections", help="列出集合")
    collections.add_argument("db")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0
    if not args.command:
        parser.print_help()
        return 1

    if args.command == "init":
        with VecStore(args.db) as store:
            print(json.dumps(store.stats(), ensure_ascii=False, indent=2))
        return 0

    with VecStore(args.db) as store:
        if args.command == "add":
            store.upsert(
                args.id,
                _parse_vector(args.vector),
                json.loads(args.meta),
                collection=args.collection,
                text=args.text,
                project_to=args.project_to,
            )
            print(json.dumps({"upserted": args.id, "collection": args.collection}, ensure_ascii=False))
            return 0

        if args.command == "ingest":
            count = store.ingest_jsonl(args.jsonl, collection=args.collection, project_to=args.project_to)
            print(json.dumps({"ingested": count, "stats": store.stats()}, ensure_ascii=False, indent=2))
            return 0

        if args.command == "ingest-text":
            items = []
            with Path(args.jsonl).open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
            count = store.ingest_texts(items, collection=args.collection, dim=args.dim)
            print(json.dumps({"ingested": count, "stats": store.stats()}, ensure_ascii=False, indent=2))
            return 0

        if args.command == "search":
            metadata_filter = json.loads(args.metadata_filter) if args.metadata_filter else None
            hits = store.search(
                _parse_vector(args.vector),
                top_k=args.topk,
                metric=args.metric,
                metadata_filter=metadata_filter,
                collection=args.collection,
                mmr=args.mmr,
            )
            print(_hits_json(hits))
            return 0

        if args.command == "hybrid-search":
            metadata_filter = json.loads(args.metadata_filter) if args.metadata_filter else None
            query_vector = _parse_vector(args.vector) if args.vector else None
            hits = store.hybrid_search(
                query_text=args.text,
                query_vector=query_vector,
                top_k=args.topk,
                alpha=args.alpha,
                metadata_filter=metadata_filter,
                collection=args.collection,
                dim=args.dim,
                mmr=args.mmr,
            )
            print(_hits_json(hits))
            return 0

        if args.command == "get":
            row = store.get(args.id, collection=args.collection)
            print(json.dumps(row, ensure_ascii=False, indent=2))
            return 0 if row else 2

        if args.command == "delete":
            ok = store.delete(args.id, collection=args.collection)
            print(json.dumps({"deleted": ok}, ensure_ascii=False))
            return 0 if ok else 2

        if args.command == "export":
            n = store.export_jsonl(args.out, collection=args.collection)
            print(json.dumps({"exported": n, "out": args.out}, ensure_ascii=False))
            return 0

        if args.command == "stats":
            print(json.dumps(store.stats(), ensure_ascii=False, indent=2))
            return 0

        if args.command == "collections":
            print(json.dumps(store.list_collections(), ensure_ascii=False, indent=2))
            return 0

    parser.error(f"未知命令: {args.command}")
    return 1


def _hits_json(hits) -> str:
    return json.dumps(
        [
            {
                "id": h.id,
                "score": h.score,
                "collection": h.collection,
                "metadata": h.metadata,
                "text": h.text,
            }
            for h in hits
        ],
        ensure_ascii=False,
        indent=2,
    )


if __name__ == "__main__":
    sys.exit(main())
