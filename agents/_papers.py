"""论文数据源层 —— 私有模块（以下划线开头，不暴露为路由）。

职责：把三个异构数据源（Semantic Scholar / OpenAlex / arXiv）收敛成同一套
paper 字典结构，供 tools.py 与 rag.py 消费。本模块**只负责元数据**，
全文抓取在 rag.py。

设计要点：
  1. Semantic Scholar 全局限流 ≥1.2s，429/5xx 指数退避（铁律要求）；
  2. S2 在共享池下极易整站 429（按出口 IP 计数），因此检索/详情/引用
     三处都做了 OpenAlex 兜底 —— 否则演示时 Agent 会直接瘫掉；
  3. paper_id 统一带类型前缀，保证跨数据源可反解：
        S2:5109c9f6...   /  DOI:10.1234/abc  /  ARXIV:2005.11401  /  OA:W712345
"""

from __future__ import annotations

import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

# ---------------------------------------------------------------- 常量
S2_BASE = "https://api.semanticscholar.org/graph/v1"
OPENALEX_BASE = "https://api.openalex.org"
ARXIV_API = "https://export.arxiv.org/api/query"

S2_MIN_INTERVAL = 1.2          # 铁律：S2 相邻请求间隔 ≥1.2s
S2_MAX_RETRIES = 4
S2_BACKOFF_BASE = 2.0          # 2s → 4s → 8s → 16s，带抖动
ARXIV_MIN_INTERVAL = 3.0       # arXiv 官方礼仪：≥3s

UA = "paper-survey-agent/1.0 (research use; contact via env mailto)"

# S2 检索/详情统一字段串（与 check_connectivity.py 保持一致）
S2_FIELDS = (
    "paperId,title,authors,year,venue,publicationVenue,abstract,tldr,"
    "citationCount,referenceCount,influentialCitationCount,externalIds,"
    "isOpenAccess,openAccessPdf,url,publicationDate"
)

# ---------------------------------------------------------------- 限流器
_lock = threading.Lock()
_last_call: dict[str, float] = {}


def _throttle(bucket: str, min_interval: float) -> None:
    """跨线程保证同一数据源相邻请求间隔 ≥ min_interval 秒。"""
    global _last_call
    with _lock:
        now = time.monotonic()
        wait = min_interval - (now - _last_call.get(bucket, 0.0))
        if wait > 0:
            time.sleep(wait)
        _last_call[bucket] = time.monotonic()


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _mailto() -> str:
    """OpenAlex 礼貌池邮箱（.env 里的 OPENALEX_MAILTO）。"""
    return _env("OPENALEX_MAILTO") or _env("UNPAYWALL_EMAIL") or "anonymous@example.com"


# ---------------------------------------------------------------- S2 请求
def s2_get(path: str, params: dict[str, Any], *, timeout: int = 30) -> tuple[int, Any, str]:
    """带限流与指数退避的 S2 GET。

    返回 (status, data, error)。200 时 data 为 dict，其余为 None。
    429 / 5xx 会重试 S2_MAX_RETRIES 次，仍失败则把状态码交给上层兜底。
    """
    key = _env("S2_API_KEY")
    headers = {"x-api-key": key} if key else {}
    url = f"{S2_BASE}{path}"
    last_status, last_err = 0, ""

    for attempt in range(S2_MAX_RETRIES + 1):
        _throttle("s2", S2_MIN_INTERVAL)
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            last_status, last_err = 0, f"{type(e).__name__}: {e}"
        else:
            if r.status_code == 200:
                try:
                    return 200, r.json(), ""
                except ValueError:
                    return 200, None, "响应不是合法 JSON"
            last_status = r.status_code
            last_err = f"HTTP {r.status_code}: {r.text[:180]}"
            # 4xx（除 429）属于请求本身有问题，重试无意义
            if r.status_code not in (429, 500, 502, 503, 504):
                return r.status_code, None, last_err

        if attempt < S2_MAX_RETRIES:
            delay = S2_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 0.4)
            time.sleep(min(delay, 30.0))

    return last_status, None, last_err


# ---------------------------------------------------------------- 归一化
def _s2_authors(raw: list[dict] | None) -> list[str]:
    return [a.get("name", "") for a in (raw or []) if a.get("name")]


def _norm_s2(d: dict) -> dict:
    """S2 论文对象 → 统一结构。"""
    ext = d.get("externalIds") or {}
    arxiv_id = ext.get("ArXiv") or ""
    doi = ext.get("DOI") or ""
    pid = d.get("paperId") or ""
    venue = d.get("venue") or ((d.get("publicationVenue") or {}).get("name") or "")
    pdf = ((d.get("openAccessPdf") or {}).get("url") or "") or ""
    tldr = ((d.get("tldr") or {}).get("text") or "") if isinstance(d.get("tldr"), dict) else ""

    if pid:
        paper_id = f"S2:{pid}"
    elif arxiv_id:
        paper_id = f"ARXIV:{arxiv_id}"
    elif doi:
        paper_id = f"DOI:{doi}"
    else:
        paper_id = f"S2:{d.get('paperId', 'unknown')}"

    return {
        "paper_id": paper_id,
        "title": (d.get("title") or "").strip(),
        "authors": _s2_authors(d.get("authors")),
        "year": d.get("year"),
        "venue": venue,
        "abstract": (d.get("abstract") or "").strip(),
        "tldr": tldr,
        "citation_count": d.get("citationCount"),
        "reference_count": d.get("referenceCount"),
        "influential_citation_count": d.get("influentialCitationCount"),
        "arxiv_id": arxiv_id,
        "doi": doi,
        "open_access_pdf": pdf,
        "url": d.get("url") or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""),
        "source": "semantic_scholar",
    }


def _openalex_abstract(inv: dict | None) -> str:
    """OpenAlex 的 abstract_inverted_index → 正常摘要文本。"""
    if not inv:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions).strip()


def _openalex_arxiv_id(work: dict) -> str:
    """从 OpenAlex 的 locations 里嗅探 arXiv id。"""
    blobs: list[str] = []
    for loc in work.get("locations") or []:
        for f in ("landing_page_url", "pdf_url"):
            if loc.get(f):
                blobs.append(loc[f])
    for u in blobs:
        m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", u)
        if m:
            return m.group(1)
    doi = (work.get("doi") or "").replace("https://doi.org/", "")
    m = re.search(r"arxiv\.(\d{4}\.\d{4,5})", doi, re.I)
    return m.group(1) if m else ""


def _norm_openalex(w: dict) -> dict:
    """OpenAlex work → 统一结构。"""
    doi = (w.get("doi") or "").replace("https://doi.org/", "")
    arxiv_id = _openalex_arxiv_id(w)
    oa = w.get("open_access") or {}
    src = ((w.get("primary_location") or {}).get("source") or {})
    venue = src.get("display_name") or ""
    authors = [
        ((a.get("author") or {}).get("display_name") or "")
        for a in (w.get("authorships") or [])
    ]
    oa_id = (w.get("id") or "").rsplit("/", 1)[-1]

    if arxiv_id:
        paper_id = f"ARXIV:{arxiv_id}"
    elif doi:
        paper_id = f"DOI:{doi}"
    else:
        paper_id = f"OA:{oa_id}"

    return {
        "paper_id": paper_id,
        "title": (w.get("title") or w.get("display_name") or "").strip(),
        "authors": [a for a in authors if a],
        "year": w.get("publication_year"),
        "venue": venue,
        "abstract": _openalex_abstract(w.get("abstract_inverted_index")),
        "tldr": "",
        "citation_count": w.get("cited_by_count"),
        "reference_count": w.get("referenced_works_count"),
        "influential_citation_count": None,
        "arxiv_id": arxiv_id,
        "doi": doi,
        "open_access_pdf": oa.get("oa_url") or "",
        "url": w.get("id") or "",
        "source": "openalex",
    }


# ---------------------------------------------------------------- paper_id 解析
def parse_paper_id(paper_id: str) -> tuple[str, str]:
    """拆分 paper_id 前缀，返回 (kind, value)。无前缀时按 S2 处理。"""
    pid = (paper_id or "").strip()
    m = re.match(r"^(S2|DOI|ARXIV|OA):(.+)$", pid, re.I)
    if m:
        return m.group(1).upper(), m.group(2).strip()
    if pid.lower().startswith(("10.", "https://doi.org/")):
        return "DOI", pid.replace("https://doi.org/", "")
    return "S2", pid


def to_s2_ref(paper_id: str) -> str:
    """转成 S2 可接受的引用形式：paperId / DOI:xxx / arXiv:xxxx。"""
    kind, val = parse_paper_id(paper_id)
    if kind == "S2":
        return val
    if kind == "DOI":
        return f"DOI:{val}"
    if kind == "ARXIV":
        return f"arXiv:{val}"
    return ""  # OA: 前缀无法直接喂给 S2


# ---------------------------------------------------------------- 检索
def search_s2(
    query: str,
    limit: int = 10,
    year_from: int | None = None,
    year_to: int | None = None,
) -> tuple[list[dict], str]:
    """S2 主题检索。返回 (papers, error)。"""
    params: dict[str, Any] = {
        "query": query,
        "limit": max(1, min(limit, 100)),
        "fields": S2_FIELDS,
    }
    if year_from or year_to:
        params["year"] = f"{year_from or ''}-{year_to or ''}"

    status, data, err = s2_get("/paper/search", params)
    if status != 200 or not data:
        return [], err or f"HTTP {status}"
    papers = [_norm_s2(p) for p in (data.get("data") or [])]
    return papers, ""


def search_openalex(
    query: str,
    limit: int = 10,
    year_from: int | None = None,
    year_to: int | None = None,
) -> tuple[list[dict], str]:
    """OpenAlex 主题检索（S2 限流时的兜底，免 Key 配额宽松）。"""
    filters = []
    if year_from:
        filters.append(f"from_publication_date:{year_from}-01-01")
    if year_to:
        filters.append(f"to_publication_date:{year_to}-12-31")

    params: dict[str, Any] = {
        "search": query,
        "per-page": max(1, min(limit, 50)),
        "mailto": _mailto(),
    }
    if filters:
        params["filter"] = ",".join(filters)

    try:
        _throttle("openalex", 0.3)
        r = requests.get(f"{OPENALEX_BASE}/works", params=params,
                         headers={"User-Agent": UA}, timeout=30)
    except requests.RequestException as e:
        return [], f"{type(e).__name__}: {e}"

    if r.status_code != 200:
        return [], f"HTTP {r.status_code}: {r.text[:160]}"
    try:
        data = r.json()
    except ValueError:
        return [], "响应不是合法 JSON"
    return [_norm_openalex(w) for w in (data.get("results") or [])], ""


# ---------------------------------------------------------------- 详情
def get_s2_paper(paper_id: str) -> tuple[dict | None, str]:
    """S2 单篇详情。"""
    ref = to_s2_ref(paper_id)
    if not ref:
        return None, "该 paper_id 不是 S2/DOI/arXiv 形式，无法直查 S2"
    status, data, err = s2_get(f"/paper/{ref}", {"fields": S2_FIELDS})
    if status != 200 or not data:
        return None, err or f"HTTP {status}"
    return _norm_s2(data), ""


def get_openalex_paper(paper_id: str) -> tuple[dict | None, str]:
    """OpenAlex 单篇详情（按 DOI / arXiv id / OpenAlex id 反查）。"""
    kind, val = parse_paper_id(paper_id)
    if kind == "OA":
        url = f"{OPENALEX_BASE}/works/{val}"
    elif kind == "DOI":
        url = f"{OPENALEX_BASE}/works/doi:{val}"
    elif kind == "ARXIV":
        url = f"{OPENALEX_BASE}/works/doi:10.48550/arxiv.{val}"
    else:
        return None, "S2 paperId 无法直接映射到 OpenAlex，请改用 DOI:/ARXIV: 前缀"

    try:
        _throttle("openalex", 0.3)
        r = requests.get(url, params={"mailto": _mailto()},
                         headers={"User-Agent": UA}, timeout=30)
    except requests.RequestException as e:
        return None, f"{type(e).__name__}: {e}"
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}: {r.text[:160]}"
    try:
        return _norm_openalex(r.json()), ""
    except ValueError:
        return None, "响应不是合法 JSON"


def resolve_paper(paper_id: str, *, prefer: str = "s2") -> tuple[dict | None, str, str]:
    """按 paper_id 取元数据，主源失败自动兜底。

    返回 (paper, source_label, error)。两者皆失败时 source_label 为空。
    """
    errors: list[str] = []
    order = ["s2", "openalex"] if prefer == "s2" else ["openalex", "s2"]

    for src in order:
        if src == "s2":
            paper, err = get_s2_paper(paper_id)
        else:
            paper, err = get_openalex_paper(paper_id)
        if paper:
            return paper, src, ""
        errors.append(f"{src}: {err}")

    return None, "", " | ".join(errors)


# ---------------------------------------------------------------- 引用溯源
def get_s2_citations(paper_id: str, limit: int = 10) -> tuple[list[dict], str]:
    """S2 后续引用（谁引用了它）。"""
    ref = to_s2_ref(paper_id)
    if not ref:
        return [], "该 paper_id 无法直查 S2 citations"
    status, data, err = s2_get(
        f"/paper/{ref}/citations",
        {"fields": "title,year,venue,citationCount,externalIds,abstract,"
                   "authors,influentialCitationCount,isOpenAccess,openAccessPdf,url,publicationDate",
         "limit": max(1, min(limit, 100))},
    )
    if status != 200 or not data:
        return [], err or f"HTTP {status}"

    out: list[dict] = []
    for item in data.get("data") or []:
        citing = item.get("citingPaper") or {}
        if not citing:
            continue
        p = _norm_s2(citing)
        # 记录它引用的是哪篇（S2 会返回 isInfluential 等上下文）
        p["is_influential"] = bool(item.get("isInfluential"))
        out.append(p)
    return out, ""


def get_openalex_citations(paper_id: str, limit: int = 10) -> tuple[list[dict], str]:
    """OpenAlex 版本的引用溯源：先定位 work id，再 filter=cites:Wxxx。"""
    kind, val = parse_paper_id(paper_id)
    work_id = val if kind == "OA" else ""

    if not work_id:
        work, err = get_openalex_paper(paper_id)
        if not work:
            return [], f"OpenAlex 定位失败：{err}"
        doi = work.get("doi") or ""
        if not doi:
            return [], "该论文在 OpenAlex 无 DOI，无法做引用溯源"
        try:
            _throttle("openalex", 0.3)
            r = requests.get(f"{OPENALEX_BASE}/works/doi:{doi}",
                             params={"mailto": _mailto(), "select": "id"},
                             headers={"User-Agent": UA}, timeout=30)
            work_id = ((r.json() or {}).get("id") or "").rsplit("/", 1)[-1]
        except Exception as e:  # noqa: BLE001
            return [], f"{type(e).__name__}: {e}"
        if not work_id:
            return [], "拿不到 OpenAlex work id"

    try:
        _throttle("openalex", 0.3)
        r = requests.get(
            f"{OPENALEX_BASE}/works",
            params={"filter": f"cites:{work_id}", "per-page": max(1, min(limit, 50)),
                    "sort": "cited_by_count:desc", "mailto": _mailto()},
            headers={"User-Agent": UA}, timeout=30,
        )
    except requests.RequestException as e:
        return [], f"{type(e).__name__}: {e}"
    if r.status_code != 200:
        return [], f"HTTP {r.status_code}: {r.text[:160]}"
    try:
        return [_norm_openalex(w) for w in (r.json().get("results") or [])], ""
    except ValueError:
        return [], "响应不是合法 JSON"


def get_citations(paper_id: str, limit: int = 10) -> tuple[list[dict], str, str]:
    """引用溯源（S2 主 + OpenAlex 兜底）。返回 (papers, source, error)。"""
    papers, err = get_s2_citations(paper_id, limit)
    if papers:
        return papers, "semantic_scholar", ""
    papers2, err2 = get_openalex_citations(paper_id, limit)
    if papers2:
        return papers2, "openalex", ""
    return [], "", f"S2: {err} | OpenAlex: {err2}"


# ---------------------------------------------------------------- arXiv 最新
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"


def search_arxiv(query: str, days: int = 30, limit: int = 10) -> tuple[list[dict], str]:
    """按提交时间倒序检索 arXiv 预印本。"""
    terms = [t for t in re.split(r"\s+", query.strip()) if t]
    # 多词用 AND 连接，避免 arXiv 把长句当短语而零命中
    search_query = " AND ".join(f"all:{t}" for t in terms) if terms else "all:*"
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max(1, min(limit * 3, 100)),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    try:
        _throttle("arxiv", ARXIV_MIN_INTERVAL)
        r = requests.get(ARXIV_API, params=params, headers={"User-Agent": UA}, timeout=40)
    except requests.RequestException as e:
        return [], f"{type(e).__name__}: {e}"
    if r.status_code != 200:
        return [], f"HTTP {r.status_code}: {r.text[:160]}"

    try:
        root = ET.fromstring(r.content)
    except ET.ParseError as e:
        return [], f"Atom 解析失败：{e}"

    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, days))
    out: list[dict] = []
    for entry in root.findall(f"{_ATOM}entry"):
        published = (entry.findtext(f"{_ATOM}published") or "").strip()
        try:
            dt = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            dt = None
        if dt and dt < cutoff:
            continue

        raw_id = (entry.findtext(f"{_ATOM}id") or "").strip()
        arxiv_id = re.sub(r"^https?://arxiv\.org/abs/", "", raw_id)
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
        pdf_url = ""
        for link in entry.findall(f"{_ATOM}link"):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href", "")
                break
        authors = [
            (a.findtext(f"{_ATOM}name") or "").strip()
            for a in entry.findall(f"{_ATOM}author")
        ]
        comment = (entry.findtext(f"{_ARXIV_NS}comment") or "").strip()

        # arXiv 只有 arXiv id，没有 DOI；落库时用 ARXIV: 前缀
        out.append({
            "paper_id": f"ARXIV:{arxiv_id}",
            "title": " ".join((entry.findtext(f"{_ATOM}title") or "").split()),
            "authors": [a for a in authors if a],
            "year": dt.year if dt else None,
            "venue": "arXiv (preprint)",
            "abstract": " ".join((entry.findtext(f"{_ATOM}summary") or "").split()),
            "tldr": comment,
            "citation_count": None,
            "reference_count": None,
            "influential_citation_count": None,
            "arxiv_id": arxiv_id,
            "doi": "",
            "open_access_pdf": pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "submitted_at": published,
            "source": "arxiv",
        })
        if len(out) >= limit:
            break

    return out, ""
