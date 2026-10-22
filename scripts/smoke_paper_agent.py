"""论文调研 Agent —— 工具层端到端冒烟测试（不启动 Web 服务、不消耗对话模型额度）。

为什么需要这个脚本：
    EdgeOne Makers 用自定义加载器把 agents/chat/index.py 当作包内模块加载，
    所以源码里用 `from .._papers import ...` 这种相对导入。直接用 python 跑
    会报 "No module named 'agents.chat'"。这里用 types.ModuleType 造一个
    合成包指向 agents/ 目录，让相对导入在本地也能成立。

用法（仓库根目录）：
    .venv/Scripts/python.exe scripts/smoke_paper_agent.py
    .venv/Scripts/python.exe scripts/smoke_paper_agent.py --query "urban flood" --arxiv-id 2405.01234
    .venv/Scripts/python.exe scripts/smoke_paper_agent.py --skip-network   # 只测切片/检索算法

退出码：0 全通过 / 1 有失败项。
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---- 合成包：让 agents/*.py 里的相对导入可用 -------------------------------
_pkg = types.ModuleType("_paperagent")
_pkg.__path__ = [str(ROOT / "agents")]
sys.modules["_paperagent"] = _pkg

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import importlib  # noqa: E402

papers = importlib.import_module("_paperagent._papers")
rag = importlib.import_module("_paperagent.rag")
tools = importlib.import_module("_paperagent.tools")

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    results.append((name, bool(cond), detail))
    print(f"  [{PASS if cond else FAIL}] {name}" + (f"  — {detail}" if detail else ""))
    return cond


def show(label: str, value: object, width: int = 300) -> None:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    print(f"      {label}: {text[:width]}")


# ---------------------------------------------------------------- 0) 纯函数
def test_pure() -> None:
    print("\n0) 切片与分词（离线）")
    text = ("Retrieval-augmented generation combines a dense retriever with a seq2seq model. "
            "We use batch size 64 and learning rate 3e-5 for fine-tuning. " * 30)
    chunks = rag.chunk_sections([("Introduction", text), ("Experiments", text[:900])])
    check("切片数量 > 1", len(chunks) > 1, f"{len(chunks)} 片")
    check("单片长度未超限", all(len(c[1]) <= rag.CHUNK_SIZE + 60 for c in chunks),
          f"max={max(len(c[1]) for c in chunks)}")
    toks = rag.tokenize("城市道路积水 short-term prediction, RAG")
    check("中英混合分词", "城市" in toks and "rag" in toks and "prediction" in toks,
          f"{toks[:8]}")

    # 归一化后点积 ≈ 1
    v = rag._l2_normalize([3.0, 4.0])
    check("L2 归一化", abs(sum(a * b for a, b in zip(v, v)) - 1.0) < 1e-9, str(v))

    # BM25 能区分相关/不相关
    docs = [("a", rag.tokenize("dense retriever for open domain question answering")),
            ("b", rag.tokenize("weather forecast with graph neural network"))]
    ranked = rag._bm25_rank(rag.tokenize("dense retriever"), docs)
    check("BM25 排序正确", bool(ranked) and ranked[0][0] == "a", str(ranked))


# ---------------------------------------------------------------- 1)+2)
def test_metadata(query: str) -> dict | None:
    print("\n1) 主题检索 search_papers")
    raw = tools._search_papers(query, 5, 0, 0)
    data = json.loads(raw)
    check("检索有结果", data.get("ok"), f"source={data.get('source')} count={data.get('count')}")
    if not data.get("ok"):
        show("error", data.get("detail") or data.get("error"))
        return None
    first = data["papers"][0]
    print(f"      首条：[{first.get('year')}] {first.get('title', '')[:70]}")
    print(f"      paper_id={first['paper_id']}  被引={first.get('citation_count')}  "
          f"arXiv={first.get('arxiv_id') or '-'}  DOI={(first.get('doi') or '-')[:30]}")
    check("字段完整（标题/年份/paper_id）",
          all(first.get(k) not in (None, "") for k in ("title", "year", "paper_id")))

    print("\n2) 精读单篇 get_paper_detail")
    detail = json.loads(tools._get_paper_detail(first["paper_id"]))
    check("详情可取", detail.get("ok"), f"source={detail.get('source')}")
    if detail.get("ok"):
        p = detail["paper"]
        print(f"      {p['title'][:70]}")
        print(f"      摘要 {len(p.get('abstract') or '')} 字符 | venue={p.get('venue') or '-'}")
    return first


# ---------------------------------------------------------------- 3)
def test_citations(paper_id: str) -> None:
    print("\n3) 溯源引用 get_citations")
    data = json.loads(tools._get_citations(paper_id, 5))
    if not data.get("ok"):
        check("引用溯源", False, data.get("error", ""))
        show("detail", data.get("detail", ""))
        return
    check("引用溯源有结果", True, f"source={data.get('source')} count={data.get('count')}")
    for c in data["citations"][:3]:
        print(f"      [{c.get('year')}] {(c.get('title') or '')[:64]} (被引 {c.get('citation_count')})")


# ---------------------------------------------------------------- 4)
def test_arxiv(query: str, days: int) -> str | None:
    print("\n4) 最新预印本 search_arxiv_latest")
    data = json.loads(tools._search_arxiv_latest(query, days, 5))
    check("arXiv 有结果", data.get("ok"), f"count={data.get('count')}")
    if not data.get("ok"):
        show("error", data.get("error"))
        return None
    for p in data["papers"][:3]:
        print(f"      {p.get('submitted_at', '')[:10]}  {p['paper_id']:<22} {p['title'][:52]}")
    return data["papers"][0]["paper_id"]


# ---------------------------------------------------------------- 5)+6)
def test_rag(arxiv_id: str, query: str) -> None:
    print(f"\n5) 全文入库 ingest_paper({arxiv_id})")
    data = json.loads(tools._ingest_paper(arxiv_id, False))
    check("入库成功", data.get("ok"), data.get("error", "") or
          f"source={data.get('fulltext_source')} chunks={data.get('n_chunks')}")
    if not data.get("ok"):
        return
    print(f"      {data.get('title', '')[:70]}")
    print(f"      {data.get('n_chunks')} 片 / {data.get('n_chars')} 字符 → {data.get('store')}")
    pid = data["paper_id"]

    print("\n6) 混合检索 search_passages")
    data = json.loads(tools._search_passages(query, pid, 5))
    check("检索命中", data.get("ok") and data.get("hits"), data.get("error", "") or
          f"candidates={data.get('candidates')} bm25={data.get('bm25_hits')} vec={data.get('vector_hits')}")
    if not data.get("ok"):
        return
    for h in data["hits"][:3]:
        print(f"      [{h['chunk_index']:>3}] ({h['section'][:24]:<24}) rrf={h['rrf_score']:.5f} "
              f"| {h['text'][:70]}…")
    check("片段带 paper_id 与序号",
          all(h.get("paper_id") and h.get("chunk_index") is not None for h in data["hits"]))

    print("\n7) 库内清单 list_ingested")
    data = rag.list_ingested()
    check("清单可读", data.get("ok"), f"count={data.get('count')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="retrieval augmented generation")
    ap.add_argument("--rag-query", default="dense retriever training batch size")
    ap.add_argument("--arxiv-id", default="ARXIV:2005.11401",
                    help="要入库的 arXiv 论文，默认 RAG 原论文")
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--skip-network", action="store_true")
    args = ap.parse_args()

    print("=" * 66)
    print("论文调研 Agent —— 工具层冒烟测试")
    print(f"仓库：{ROOT}")
    print("=" * 66)

    test_pure()
    if args.skip_network:
        return report()

    first = test_metadata(args.query)
    if first:
        test_citations(first["paper_id"])
    test_arxiv(args.query, args.days)
    test_rag(args.arxiv_id, args.rag_query)
    return report()


def report() -> int:
    print("\n" + "=" * 66)
    failed = [name for name, ok, _ in results if not ok]
    for name, ok, detail in results:
        print(f"  {PASS if ok else FAIL}  {name}" + (f"  — {detail}" if detail and not ok else ""))
    print("=" * 66)
    print("全部通过" if not failed else f"失败 {len(failed)} 项：{', '.join(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
