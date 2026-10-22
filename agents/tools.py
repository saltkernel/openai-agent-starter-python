"""检索类工具 —— 用 @function_tool 注册给 Agent。

六个能力与四个后端（Semantic Scholar / OpenAlex / arXiv / 本地向量库）的映射：

    search_papers        主题检索        S2 → OpenAlex 兜底
    get_paper_detail     精读单篇元数据  S2 → OpenAlex 兜底
    get_citations        溯源引用        S2 citations → OpenAlex cites:
    search_arxiv_latest  最新预印本      arXiv Atom（按提交时间倒序）
    ingest_paper         全文入库        arXiv HTML → OA PDF → 摘要
    search_passages      混合检索        BM25 + 向量 + RRF（见 rag.py）

约定：所有工具一律返回 JSON 字符串，且**只搬运事实**——标题、年份、
被引数全部来自接口返回值，不做任何推断，避免模型拿自己的记忆当数据。
"""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any

from agents import function_tool

from . import _papers, rag

ABSTRACT_LIMIT = 900  # 回传给模型的摘要截断长度（省 token）


def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=None)


def _trim(paper: dict, *, with_abstract: bool = True) -> dict:
    """裁掉模型用不到的重字段，保留可溯源所需的全部标识。"""
    out = dict(paper)
    abstract = out.get("abstract") or ""
    if not with_abstract:
        out.pop("abstract", None)
    elif len(abstract) > ABSTRACT_LIMIT:
        out["abstract"] = abstract[:ABSTRACT_LIMIT]
        out["abstract_truncated"] = True
    return out


# ============================================================ 1) 主题检索
def _search_papers(query: str, limit: int, year_from: int, year_to: int) -> str:
    yf = year_from or None
    yt = year_to or None

    papers, err = _papers.search_s2(query, limit, yf, yt)
    source = "semantic_scholar"
    fallback_error = ""
    if not papers:
        fallback_error = err
        papers, err = _papers.search_openalex(query, limit, yf, yt)
        source = "openalex"

    if not papers:
        return _dump({
            "ok": False, "query": query, "source": None,
            "error": "两个数据源都没返回结果",
            "detail": {"semantic_scholar": fallback_error, "openalex": err},
            "hint": "换更通用的英文关键词，或去掉年份限制；不要凭空补论文",
        })

    return _dump({
        "ok": True, "query": query, "source": source,
        "count": len(papers),
        "papers": [_trim(p) for p in papers],
        "note": "paper_id 原样传回给 get_paper_detail / ingest_paper / search_passages",
    })


@function_tool
async def search_papers(
    query: Annotated[str, "检索关键词，英文效果最好，例如 'multimodal large model urban flood inundation'"],
    limit: Annotated[int, "返回条数，1-20，默认 8"] = 8,
    year_from: Annotated[int, "起始年份（含），0 表示不限"] = 0,
    year_to: Annotated[int, "结束年份（含），0 表示不限"] = 0,
) -> str:
    """按主题关键词检索论文，返回标题/作者/年份/被引数/venue/arXivID/DOI/开放获取PDF/摘要。

    这是文献综述的第一步：先拿到候选论文列表（含 paper_id），
    再对其中相关的几篇调用 ingest_paper 入库，最后用 search_passages 取证据。
    """
    return await asyncio.to_thread(_search_papers, query, limit, year_from, year_to)


# ============================================================ 2) 精读单篇
def _get_paper_detail(paper_id: str) -> str:
    paper, source, err = _papers.resolve_paper(paper_id)
    if not paper:
        return _dump({"ok": False, "paper_id": paper_id,
                      "error": f"取不到该论文：{err}",
                      "hint": "paper_id 需带前缀，如 S2:xxxx / DOI:10.xxxx / ARXIV:2005.11401"})
    return _dump({
        "ok": True, "source": source, "paper_id": paper["paper_id"],
        "paper": _trim(paper),
        "note": "若要看方法/超参/实验细节，先 ingest_paper 全文入库，再 search_passages",
    })


@function_tool
async def get_paper_detail(
    paper_id: Annotated[str, "论文标识，带前缀：S2:xxxx / DOI:10.xxxx / ARXIV:2005.11401"],
) -> str:
    """取单篇论文的完整元数据与摘要（标题、作者、年份、venue、被引数、DOI、arXivID、开放获取PDF）。

    只在需要确认某篇论文的基本信息时调用；要读正文细节请用 ingest_paper + search_passages。
    """
    return await asyncio.to_thread(_get_paper_detail, paper_id)


# ============================================================ 3) 溯源引用
def _get_citations(paper_id: str, limit: int) -> str:
    papers, source, err = _papers.get_citations(paper_id, limit)
    if not papers:
        return _dump({"ok": False, "paper_id": paper_id,
                      "error": "未检索到引用该论文的后续工作",
                      "detail": err,
                      "hint": "可能是新论文尚无引用，或该论文不在 S2/OpenAlex 收录范围"})
    return _dump({
        "ok": True, "source": source, "paper_id": paper_id,
        "count": len(papers),
        "citations": [_trim(p, with_abstract=False) for p in papers],
        "note": "这些是引用了目标论文的后续工作（谁改进了它），引用关系由数据源给出",
    })


@function_tool
async def get_citations(
    paper_id: Annotated[str, "被引论文的 paper_id，带前缀：S2:xxxx / DOI:10.xxxx / ARXIV:2005.11401"],
    limit: Annotated[int, "返回条数，1-20，默认 8"] = 8,
) -> str:
    """查某篇论文的后续引用工作（谁引用了它 / 谁改进了它），用于梳理技术演进脉络。

    返回的 cited_by_count 等数字均来自数据源，不得自行改写。
    """
    return await asyncio.to_thread(_get_citations, paper_id, limit)


# ============================================================ 4) 最新预印本
def _search_arxiv_latest(query: str, days: int, limit: int) -> str:
    papers, err = _papers.search_arxiv(query, days, limit)
    if not papers:
        return _dump({"ok": False, "query": query,
                      "error": err or f"最近 {days} 天内没有匹配的预印本",
                      "hint": "放宽关键词或加大 days；arXiv 只覆盖预印本，不含期刊论文"})
    return _dump({
        "ok": True, "source": "arxiv", "query": query, "days": days,
        "count": len(papers),
        "papers": [_trim(p) for p in papers],
        "note": "按提交时间倒序；citation_count 为 null 是因为 arXiv 不提供被引数",
    })


@function_tool
async def search_arxiv_latest(
    query: Annotated[str, "检索关键词，空格分词后按 AND 组合，例如 'multimodal flood monitoring'"],
    days: Annotated[int, "只看最近多少天内提交的预印本，默认 90"] = 90,
    limit: Annotated[int, "返回条数，1-20，默认 8"] = 8,
) -> str:
    """在 arXiv 上按提交时间检索最新预印本（谁最近在做这个方向）。

    用于补充期刊数据库滞后的问题。arXiv 的 paper_id 形如 ARXIV:2405.01234，可直接 ingest。
    """
    return await asyncio.to_thread(_search_arxiv_latest, query, days, limit)


# ============================================================ 5) 全文入库
def _ingest_paper(paper_id: str, force: bool) -> str:
    try:
        result = rag.ingest_paper(paper_id, force=force)
    except Exception as e:  # noqa: BLE001 —— 工具层必须兜住，不能把异常抛给模型
        return _dump({"ok": False, "paper_id": paper_id,
                      "error": f"{type(e).__name__}: {e}",
                      "hint": "检查 EMBEDDING_* 配置与 VECTOR_STORE_PATH 目录权限"})
    result["next"] = "调用 search_passages(query=..., paper_id=...) 取原文片段"
    return _dump(result)


@function_tool
async def ingest_paper(
    paper_id: Annotated[str, "要入库的论文 paper_id，带前缀：ARXIV:2005.11401 / S2:xxxx / DOI:10.xxxx"],
    force: Annotated[bool, "true 表示已入库也重新抓取，默认 false"] = False,
) -> str:
    """抓取论文全文并切片向量化入库（优先 arXiv HTML，回退开放获取 PDF，最后退回摘要）。

    入库后该论文的方法、超参、实验设置等内部细节才能被检索到，是「精读追问」的前提。
    一次只入一篇，避免超时；综述只需入库最相关的 3-6 篇。
    """
    return await asyncio.to_thread(_ingest_paper, paper_id, force)


# ============================================================ 6) 混合检索
def _search_passages(query: str, paper_id: str, top_k: int) -> str:
    try:
        result = rag.hybrid_search(query, paper_id.strip() or None, top_k)
    except Exception as e:  # noqa: BLE001
        return _dump({"ok": False, "query": query,
                      "error": f"{type(e).__name__}: {e}",
                      "hint": "检查向量库是否存在、EMBEDDING_* 是否可用"})

    if not result.get("ok"):
        return _dump(result)

    return _dump({
        **result,
        "note": "text 是原文片段；papers 是命中论文的元数据（标题/作者/年份/venue/DOI/arXiv），"
                "写参考文献表时直接用这里的字段，不要自己编。片段里没写的内容一律不得补写",
    })


@function_tool
async def search_passages(
    query: Annotated[str, "要查的具体问题或关键词，例如 'learning rate and batch size' 或 '消融实验设置'"],
    paper_id: Annotated[str, "限定在某篇论文内检索；留空则在所有已入库论文中检索"] = "",
    top_k: Annotated[int, "返回片段数，1-20，默认 6"] = 6,
) -> str:
    """混合检索：BM25 关键词 + 向量语义，用 RRF 融合，返回带 paper_id 与片段序号的原文片段。

    这是所有事实性结论的唯一证据来源。回答方法/超参/实验设置等问题前必须先调用它。
    若论文已在库中，直接用本工具，不要再用 search_papers 重搜或 ingest_paper 重抓。
    """
    return await asyncio.to_thread(_search_passages, query, paper_id, top_k)


# ============================================================ 7) 库内清单
@function_tool
async def list_ingested_papers() -> str:
    """列出当前已入库（可追问细节）的论文清单及其片段数，用于确认追问范围。"""
    try:
        return _dump(await asyncio.to_thread(rag.list_ingested))
    except Exception as e:  # noqa: BLE001
        return _dump({"ok": False, "error": f"{type(e).__name__}: {e}"})


# 供 chat/index.py 一次性注册
PAPER_TOOLS = [
    search_papers,
    get_paper_detail,
    get_citations,
    search_arxiv_latest,
    ingest_paper,
    search_passages,
    list_ingested_papers,
]
