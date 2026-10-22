"""RAG 层 —— 全文抓取、切片、向量化、SQLite 落地、混合检索。

只做检索与证据供给，不生成任何文字结论 —— 结论一律由模型基于
本模块返回的原文片段书写（见 agents/chat/index.py 的 system instructions）。

对外四个入口（被 tools.py 包成 @function_tool）：
    fetch_fulltext(paper)       全文抓取：arXiv HTML → 开放获取 PDF → 摘要兜底
    ingest_paper(paper_id)      切片 + 向量化 + 落 SQLite
    hybrid_search(query, ...)   BM25 + 向量 + RRF 融合
    list_ingested()             查看库内已入库论文

存储：VECTOR_STORE_PATH（.env，默认 ./data/store.sqlite），
沙箱重启后数据仍在。向量以 float32 BLOB 存储，入库前做 L2 归一化，
检索时用点积即余弦相似度。
"""

from __future__ import annotations

import array
import json
import math
import os
import re
import sqlite3
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import requests

from ._papers import _env, _throttle, parse_paper_id, resolve_paper

# ---------------------------------------------------------------- 常量
EMBED_DIM = 1024                 # BAAI/bge-m3
EMBED_BATCH = 16
CHUNK_SIZE = 1100                # 目标字符数
CHUNK_OVERLAP = 180              # 相邻片段重叠字符数
RRF_K = 60                       # RRF 平滑常数
MAX_CANDIDATES = 6000            # 单次检索最多扫描的片段数（防爆内存）
UA = "paper-survey-agent/1.0 (research use)"

_db_lock = threading.Lock()


# ---------------------------------------------------------------- 路径与连接
def store_path() -> Path:
    """VECTOR_STORE_PATH 相对路径一律相对「仓库根目录」解析。"""
    raw = _env("VECTOR_STORE_PATH", "./data/store.sqlite")
    p = Path(raw)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / raw
    return p


SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id       TEXT PRIMARY KEY,
    title          TEXT,
    authors        TEXT,
    year           INTEGER,
    venue          TEXT,
    doi            TEXT,
    arxiv_id       TEXT,
    url            TEXT,
    abstract       TEXT,
    fulltext_source TEXT,
    n_chunks       INTEGER DEFAULT 0,
    n_chars        INTEGER DEFAULT 0,
    ingested_at    TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id  TEXT PRIMARY KEY,
    paper_id  TEXT NOT NULL,
    idx       INTEGER NOT NULL,
    section   TEXT,
    text      TEXT NOT NULL,
    tokens    TEXT NOT NULL,
    embedding BLOB NOT NULL,
    dim       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);
"""


def db() -> sqlite3.Connection:
    """取连接并确保表结构存在（幂等）。"""
    p = store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=30)
    conn.row_factory = sqlite3.Row
    with _db_lock:
        conn.executescript(SCHEMA)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            pass
    return conn


# ---------------------------------------------------------------- 向量化
def embed_texts(texts: list[str], *, batch_size: int = EMBED_BATCH) -> list[list[float]]:
    """调用 EMBEDDING_* 配置的 OpenAI 兼容 /embeddings 接口。"""
    base = _env("EMBEDDING_BASE_URL")
    key = _env("EMBEDDING_API_KEY")
    model = _env("EMBEDDING_MODEL")
    if not (base and key and model):
        raise RuntimeError("EMBEDDING_BASE_URL / EMBEDDING_API_KEY / EMBEDDING_MODEL 未配置")

    url = f"{base.rstrip('/')}/embeddings"
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        payload = {"model": model, "input": batch}
        last = ""
        for attempt in range(3):
            try:
                r = requests.post(url, json=payload, timeout=90,
                                  headers={"Authorization": f"Bearer {key}"})
                if r.status_code == 200:
                    data = r.json()
                    vecs = [d["embedding"] for d in data["data"]]
                    if len(vecs) != len(batch):
                        raise RuntimeError(f"返回条数不符：{len(vecs)} != {len(batch)}")
                    dim = len(vecs[0])
                    if dim != EMBED_DIM:
                        raise RuntimeError(
                            f"向量维度 {dim} ≠ 期望 {EMBED_DIM}；换过 embedding 模型需重建向量库"
                        )
                    out.extend(vecs)
                    break
                last = f"HTTP {r.status_code}: {r.text[:160]}"
                if r.status_code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(last)
            except requests.RequestException as e:
                last = f"{type(e).__name__}: {e}"
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"向量化失败：{last}")
    return out


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _pack(vec: list[float]) -> bytes:
    return array.array("f", vec).tobytes()


def _unpack(blob: bytes) -> array.array:
    a = array.array("f")
    a.frombytes(blob)
    return a


# ---------------------------------------------------------------- 分词（BM25 用）
_WORD_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "for", "on", "with", "is",
    "are", "was", "were", "be", "by", "as", "at", "from", "that", "this", "it",
    "we", "our", "can", "which", "these", "their", "has", "have", "not", "but",
}


def tokenize(text: str) -> list[str]:
    """英文按词、中文按二元组切分（不引入 jieba，零依赖）。"""
    low = (text or "").lower()
    toks = [t for t in _WORD_RE.findall(low) if t not in _STOP and len(t) > 1]
    for run in _CJK_RE.findall(low):
        if len(run) == 1:
            toks.append(run)
        else:
            toks.extend(run[i:i + 2] for i in range(len(run) - 1))
    return toks


# ---------------------------------------------------------------- 全文抓取
class _HTMLText(HTMLParser):
    """把 arXiv HTML 拆成 (类型, 文本) 块，保留标题层级。"""

    SKIP = {"script", "style", "nav", "footer", "noscript", "svg"}
    HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
    BREAKS = {"p", "li", "div", "br", "tr", "figcaption", "blockquote", "td"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str, int]] = []
        self._buf: list[str] = []
        self._heading_level = 0
        self._skip = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.HEADINGS:
            self._flush()
            self._heading_level = self.HEADINGS[tag]
        elif tag in self.BREAKS:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in self.HEADINGS:
            self._flush()
            self._heading_level = 0
        elif tag in self.BREAKS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._buf.append(data)

    def _flush(self) -> None:
        text = " ".join("".join(self._buf).split())
        self._buf = []
        if not text or len(text) < 2:
            return
        if self._heading_level:
            self.blocks.append(("h", text, self._heading_level))
        else:
            self.blocks.append(("p", text, 0))


def _blocks_to_sections(blocks: list[tuple[str, str, int]]) -> list[tuple[str, str]]:
    """把块序列按 h1–h3 聚成 (小节标题, 正文)。"""
    sections: list[tuple[str, str]] = []
    title = "Body"
    buf: list[str] = []

    def flush() -> None:
        body = " ".join(buf).strip()
        if len(body) > 80:
            sections.append((title, body))

    for kind, text, level in blocks:
        if kind == "h" and level <= 3:
            flush()
            buf = []
            title = text[:120]
        else:
            buf.append(text)
    flush()
    return sections


_PDF_SECTION_RE = re.compile(
    r"^\s*(?:(\d+(?:\.\d+)*)\s+)?"
    r"(abstract|introduction|related work|method(?:s|ology)?|approach|"
    r"experiment(?:s|al setup|al results)?|results?|discussion|"
    r"conclusion(?:s)?|limitations?|references|acknowledg(?:e)?ments?)\s*$",
    re.I,
)


def _pdf_sections(text: str) -> list[tuple[str, str]]:
    """PDF 纯文本按常见章节名切分，切不出来就整篇当一节。"""
    lines = text.split("\n")
    sections: list[tuple[str, str]] = []
    title, buf = "Body", []

    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped) < 60 and _PDF_SECTION_RE.match(stripped):
            body = "\n".join(buf).strip()
            if len(body) > 80:
                sections.append((title, body))
            title, buf = stripped, []
            continue
        buf.append(line)

    body = "\n".join(buf).strip()
    if len(body) > 80:
        sections.append((title, body))
    return sections or [("Body", text)]


def _http_get(url: str, *, timeout: int = 60, min_interval: float = 0.0,
              bucket: str = "web") -> requests.Response | None:
    try:
        if min_interval:
            _throttle(bucket, min_interval)
        r = requests.get(url, timeout=timeout, headers={"User-Agent": UA},
                         allow_redirects=True)
    except requests.RequestException:
        return None
    return r if r.status_code == 200 else None


def _arxiv_html(arxiv_id: str) -> tuple[list[tuple[str, str]], str]:
    """优先 arXiv 官方 HTML；旧论文没有 HTML 时退回 ar5iv。"""
    for url in (
        f"https://arxiv.org/html/{arxiv_id}",
        f"https://arxiv.org/html/{arxiv_id}v1",
        f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}",
    ):
        r = _http_get(url, timeout=45, min_interval=3.0, bucket="arxiv")
        if r is None or len(r.text) < 5000:
            continue
        parser = _HTMLText()
        try:
            parser.feed(r.text)
        except Exception:  # noqa: BLE001 —— 残缺 HTML 不应连带整个入库失败
            continue
        sections = _blocks_to_sections(parser.blocks)
        if sections:
            return sections, url
    return [], ""


def _pdf_text(url: str) -> str:
    """下载 PDF 并用 pypdf 抽文本。"""
    r = _http_get(url, timeout=120)
    if r is None or not r.content[:4] == b"%PDF":
        return ""
    try:
        import io

        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(r.content))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:  # noqa: BLE001
        return ""


def _unpaywall_pdf(doi: str) -> str:
    """DOI → 开放获取 PDF 直链（Unpaywall 免费，必须带 email）。"""
    if not doi:
        return ""
    email = _env("UNPAYWALL_EMAIL") or "anonymous@example.com"
    r = _http_get(f"https://api.unpaywall.org/v2/{doi}", timeout=25)
    if r is None:
        return ""
    try:
        data = r.json()
    except ValueError:
        return ""
    best = data.get("best_oa_location") or {}
    for key in ("url_for_pdf", "url"):
        if best.get(key):
            return best[key]
    return ""


def fetch_fulltext(paper: dict) -> tuple[list[tuple[str, str]], str]:
    """按优先级取全文：arXiv HTML → 开放获取 PDF → 摘要兜底。

    返回 (sections, source)，source ∈ {arxiv_html, pdf, abstract, none}。
    """
    arxiv_id = (paper.get("arxiv_id") or "").strip()
    if arxiv_id:
        sections, _url = _arxiv_html(arxiv_id)
        if sections:
            return sections, "arxiv_html"

    pdf_url = (paper.get("open_access_pdf") or "").strip()
    if not pdf_url:
        pdf_url = _unpaywall_pdf((paper.get("doi") or "").strip())
    if pdf_url:
        text = _pdf_text(pdf_url)
        if text.strip():
            return _pdf_sections(text), "pdf"

    abstract = (paper.get("abstract") or "").strip()
    if abstract:
        return [("Abstract", abstract)], "abstract"
    return [], "none"


# ---------------------------------------------------------------- 切片
_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")


def chunk_sections(
    sections: list[tuple[str, str]],
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[tuple[str, str]]:
    """把小节切成带重叠的片段，尽量在句末断开。返回 [(section, text)]。"""
    chunks: list[tuple[str, str]] = []

    for section, body in sections:
        body = " ".join(body.split())
        if len(body) <= size:
            if len(body) > 60:
                chunks.append((section, body))
            continue

        sentences = _SENT_SPLIT.split(body)
        buf = ""
        for sent in sentences:
            if len(buf) + len(sent) + 1 <= size:
                buf = f"{buf} {sent}".strip()
                continue
            if buf:
                chunks.append((section, buf))
                tail = buf[-overlap:] if overlap else ""
                buf = f"{tail} {sent}".strip()
            else:
                # 单句超长，硬切
                for i in range(0, len(sent), size - overlap):
                    chunks.append((section, sent[i:i + size]))
                buf = ""
        if len(buf.strip()) > 60:
            chunks.append((section, buf.strip()))

    return chunks


# ---------------------------------------------------------------- 入库
def ingest_paper(paper_id: str, force: bool = False) -> dict:
    """抓全文 → 切片 → 向量化 → 写 SQLite。已入库则默认跳过。"""
    paper, src, err = resolve_paper(paper_id)
    if not paper:
        return {"ok": False, "paper_id": paper_id,
                "error": f"取元数据失败：{err}", "hint": "确认 paper_id 前缀（S2:/DOI:/ARXIV:）正确"}

    pid = paper["paper_id"]
    conn = db()
    try:
        row = conn.execute("SELECT n_chunks, fulltext_source FROM papers WHERE paper_id=?",
                           (pid,)).fetchone()
        if row and row["n_chunks"] and not force:
            return {"ok": True, "paper_id": pid, "skipped": True,
                    "n_chunks": row["n_chunks"], "fulltext_source": row["fulltext_source"],
                    "title": paper.get("title"),
                    "message": "已在库中，跳过（如需重抓请传 force=true）"}

        sections, source = fetch_fulltext(paper)
        if not sections:
            return {"ok": False, "paper_id": pid, "error": "抓不到全文也抓不到摘要",
                    "hint": "换一篇有 arXiv HTML 或开放获取 PDF 的论文"}

        chunks = chunk_sections(sections)
        if not chunks:
            return {"ok": False, "paper_id": pid, "error": "切片结果为空"}

        vectors = embed_texts([c[1] for c in chunks])

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        n_chars = sum(len(s[1]) for s in sections)

        with conn:  # 事务：要么整篇替换成功，要么保持原样
            conn.execute("DELETE FROM chunks WHERE paper_id=?", (pid,))
            conn.executemany(
                "INSERT INTO chunks (chunk_id, paper_id, idx, section, text, tokens, embedding, dim)"
                " VALUES (?,?,?,?,?,?,?,?)",
                [
                    (
                        f"{pid}#{i:04d}", pid, i, section, text,
                        " ".join(tokenize(text)),
                        _pack(_l2_normalize(vec)), EMBED_DIM,
                    )
                    for i, ((section, text), vec) in enumerate(zip(chunks, vectors))
                ],
            )
            conn.execute(
                "INSERT INTO papers (paper_id, title, authors, year, venue, doi, arxiv_id,"
                " url, abstract, fulltext_source, n_chunks, n_chars, ingested_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(paper_id) DO UPDATE SET"
                " title=excluded.title, authors=excluded.authors, year=excluded.year,"
                " venue=excluded.venue, doi=excluded.doi, arxiv_id=excluded.arxiv_id,"
                " url=excluded.url, abstract=excluded.abstract,"
                " fulltext_source=excluded.fulltext_source, n_chunks=excluded.n_chunks,"
                " n_chars=excluded.n_chars, ingested_at=excluded.ingested_at",
                (
                    pid, paper.get("title"), json.dumps(paper.get("authors") or [], ensure_ascii=False),
                    paper.get("year"), paper.get("venue"), paper.get("doi"),
                    paper.get("arxiv_id"), paper.get("url"), paper.get("abstract"),
                    source, len(chunks), n_chars, now,
                ),
            )

        return {
            "ok": True, "paper_id": pid, "title": paper.get("title"),
            "fulltext_source": source, "sections": len(sections),
            "n_chunks": len(chunks), "n_chars": n_chars,
            "store": str(store_path()),
            "abstract_only": source == "abstract",
            "message": ("仅拿到摘要（未找到 HTML/PDF），细节追问会缺证据"
                        if source == "abstract" else "全文已入库"),
        }
    finally:
        conn.close()


# ---------------------------------------------------------------- 检索
def _bm25_rank(query_tokens: list[str], docs: list[tuple[str, list[str]]],
               k1: float = 1.5, b: float = 0.75) -> list[tuple[str, float]]:
    """纯 Python BM25（Okapi）。docs = [(chunk_id, tokens)]。"""
    n = len(docs)
    if not n:
        return []
    avgdl = sum(len(t) for _, t in docs) / n or 1.0
    df: Counter = Counter()
    for _cid, toks in docs:
        for t in set(toks):
            df[t] += 1

    q = [t for t in dict.fromkeys(query_tokens) if t in df]
    scores: dict[str, float] = {}
    for cid, toks in docs:
        tf = Counter(toks)
        dl = len(toks) or 1
        s = 0.0
        for t in q:
            f = tf.get(t, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
        if s > 0:
            scores[cid] = s
    return sorted(scores.items(), key=lambda x: -x[1])


def _vector_rank(qvec: list[float], rows: list[sqlite3.Row]) -> list[tuple[str, float]]:
    """向量已归一化，点积即余弦。"""
    scored = []
    for row in rows:
        vec = _unpack(row["embedding"])
        if len(vec) != len(qvec):
            continue
        scored.append((row["chunk_id"], sum(a * b for a, b in zip(qvec, vec))))
    return sorted(scored, key=lambda x: -x[1])


def _rrf(rankings: list[list[tuple[str, float]]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion：只看名次，天然免疫两路分数量纲不一致。"""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (cid, _score) in enumerate(ranking, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


def _canonical_pid(conn: sqlite3.Connection, paper_id: str) -> str:
    """把任意形式的 paper_id 映射到库内实际主键（先查库，避免多打一次 API）。"""
    row = conn.execute("SELECT paper_id FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
    if row:
        return row["paper_id"]
    kind, val = parse_paper_id(paper_id)
    column = {"DOI": "doi", "ARXIV": "arxiv_id"}.get(kind)
    if column:
        row = conn.execute(
            f"SELECT paper_id FROM papers WHERE {column}=? LIMIT 1", (val,)  # noqa: S608
        ).fetchone()
        if row:
            return row["paper_id"]
    return ""


def hybrid_search(query: str, paper_id: str | None = None,
                  top_k: int = 6, pool: int = 40) -> dict:
    """BM25 + 向量检索，RRF 融合，返回带 paper_id / 片段序号的原文。"""
    q = (query or "").strip()
    if not q:
        return {"ok": False, "error": "query 不能为空"}

    conn = db()
    pid: str | None = None
    try:
        if paper_id:
            pid = _canonical_pid(conn, paper_id)
            if not pid:
                # 库内没有 ⇒ 可能是另一种 id 写法，问一次元数据拿规范 id
                paper, _src, _err = resolve_paper(paper_id, prefer="openalex")
                pid = paper["paper_id"] if paper else paper_id
            rows = conn.execute(
                "SELECT chunk_id, paper_id, idx, section, text, tokens, embedding"
                " FROM chunks WHERE paper_id=? ORDER BY idx LIMIT ?", (pid, MAX_CANDIDATES),
            ).fetchall()
            if not rows:
                return {"ok": False, "paper_id": pid,
                        "error": "该论文尚未入库",
                        "hint": f"先调用 ingest_paper(paper_id=\"{paper_id}\")"}
        else:
            rows = conn.execute(
                "SELECT chunk_id, paper_id, idx, section, text, tokens, embedding"
                " FROM chunks ORDER BY paper_id, idx LIMIT ?", (MAX_CANDIDATES,),
            ).fetchall()
            if not rows:
                return {"ok": False, "error": "向量库为空",
                        "hint": "先对目标论文调用 ingest_paper"}

        docs = [(r["chunk_id"], r["tokens"].split()) for r in rows]
        bm25 = _bm25_rank(tokenize(q), docs)

        qvec = _l2_normalize(embed_texts([q])[0])
        vec = _vector_rank(qvec, rows)

        fused = _rrf([bm25[:pool], vec[:pool]])
        top = fused[:max(1, min(top_k, 20))]

        by_id = {r["chunk_id"]: r for r in rows}
        hits = []
        for rank, (cid, score) in enumerate(top, start=1):
            r = by_id[cid]
            hits.append({
                "chunk_id": cid,
                "paper_id": r["paper_id"],
                "chunk_index": r["idx"],
                "section": r["section"],
                "rrf_score": round(score, 6),
                "text": r["text"],
            })

        # 顺带把命中论文的元数据带回：模型据此直接生成参考文献表，无需再调 detail
        pids = sorted({h["paper_id"] for h in hits})
        papers_map: dict[str, dict] = {}
        if pids:
            placeholders = ",".join("?" * len(pids))
            for row in conn.execute(
                f"SELECT paper_id, title, authors, year, venue, doi, arxiv_id, url"  # noqa: S608
                f" FROM papers WHERE paper_id IN ({placeholders})", pids,
            ):
                meta = dict(row)
                try:
                    meta["authors"] = json.loads(meta.get("authors") or "[]")
                except (ValueError, TypeError):
                    meta["authors"] = []
                papers_map[row["paper_id"]] = meta

        return {
            "ok": True,
            "query": q,
            "scope": pid if paper_id else "all_ingested",
            "candidates": len(rows),
            "bm25_hits": len(bm25),
            "vector_hits": len(vec),
            "hits": hits,
            "papers": papers_map,
        }
    finally:
        conn.close()


def list_ingested() -> dict:
    """列出库内已入库论文（供模型确认可追问范围）。"""
    conn = db()
    try:
        rows = conn.execute(
            "SELECT paper_id, title, year, venue, fulltext_source, n_chunks, n_chars, ingested_at"
            " FROM papers ORDER BY ingested_at DESC"
        ).fetchall()
        return {"ok": True, "store": str(store_path()),
                "count": len(rows), "papers": [dict(r) for r in rows]}
    finally:
        conn.close()
