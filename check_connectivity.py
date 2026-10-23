"""最小连通性测试 —— 只读验证，不改动任何业务代码。

依次验证三件事：
  1) 对话模型能否正常返回（OpenAI 兼容 /chat/completions）
  2) 向量模型能否返回 1024 维向量（/embeddings）
  3) Semantic Scholar 能否搜到结果（/graph/v1/paper/search）

用法（在仓库根目录）：
    .venv/Scripts/python.exe check_connectivity.py
    .venv/Scripts/python.exe check_connectivity.py --skip-s2
    .venv/Scripts/python.exe check_connectivity.py --query "city flood inundation"

退出码：0 = 三项全通过；1 = 有失败；2 = 无失败但有降级（限流/维度等）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ENV_PATH = Path(__file__).resolve().parent / ".env"
EXPECTED_DIM = 1024

# 与 agents/chat/index.py 同一套优先级：LLM_* 优先，AI_GATEWAY_* 兜底
_ALIASES = {
    "AI_GATEWAY_API_KEY": ("LLM_API_KEY", "AI_GATEWAY_API_KEY"),
    "AI_GATEWAY_BASE_URL": ("LLM_BASE_URL", "AI_GATEWAY_BASE_URL"),
    "AI_GATEWAY_MODEL": ("LLM_MODEL", "AI_GATEWAY_MODEL"),
}

# 与教程 tools.py 完全一致的字段串与端点
S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = (
    "title,authors,year,venue,abstract,tldr,citationCount,"
    "referenceCount,influentialCitationCount,externalIds,"
    "isOpenAccess,openAccessPdf"
)

REQUIRED = [
    "AI_GATEWAY_API_KEY",
    "AI_GATEWAY_BASE_URL",
    "AI_GATEWAY_MODEL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_MODEL",
]
OPTIONAL_EMPTY_OK = ["S2_API_KEY", "CROSSREF_MAILTO", "OPENALEX_MAILTO", "UNPAYWALL_EMAIL"]
NOT_READ_BY_CODE = ["CROSSREF_MAILTO", "OPENALEX_MAILTO", "UNPAYWALL_EMAIL"]

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
MARK = {PASS: "[PASS]", WARN: "[WARN]", FAIL: "[FAIL]"}

results: list[tuple[str, str, str]] = []


def mask(value: str) -> str:
    v = (value or "").strip()
    if not v:
        return "(空)"
    if len(v) <= 8:
        return "*" * len(v)
    return f"{v[:4]}...{v[-4:]} (len={len(v)})"


def env(name: str, default: str = "") -> str:
    """LLM_* 优先于 AI_GATEWAY_*（线上平台会预置后者并顶掉同名配置）。"""
    for candidate in _ALIASES.get(name, (name,)):
        value = (os.getenv(candidate) or "").strip()
        if value:
            return value
    return default.strip() if default else ""


def line(status: str, text: str) -> None:
    print(f"  {MARK[status]} {text}")


# ---------------------------------------------------------------- 配置体检
def check_env() -> None:
    print("=" * 70)
    print("0. 配置体检   .env ->", ENV_PATH)
    print("=" * 70)

    if not ENV_PATH.exists():
        line(FAIL, "找不到 .env，请先复制 .env.example 并填写")
        results.append(("配置", FAIL, ".env 缺失"))
        print()
        return
    line(PASS, f".env 存在，{ENV_PATH.stat().st_size} 字节")

    missing = [k for k in REQUIRED if not env(k)]
    for k in REQUIRED:
        v = env(k)
        if not v:
            line(FAIL, f"{k:<22} 缺失或为空")
        else:
            line(PASS, f"{k:<22} {mask(v) if 'KEY' in k else v}")
    results.append(
        ("必填变量齐全", FAIL, "缺失 " + ", ".join(missing)) if missing
        else ("必填变量齐全", PASS, f"{len(REQUIRED)} 个必填项全部非空")
    )

    # 占位符检测：非空但明显是模板文字，会直接导致 401/403
    bad = []
    for k in REQUIRED + OPTIONAL_EMPTY_OK:
        v = env(k)
        if not v:
            continue
        low = v.lower()
        if any(t in v for t in ("你的", "没有就留空", "填我")) or any(
            t in low for t in ("your-", "your_", "xxx", "placeholder", "changeme")
        ):
            bad.append(k)
    if bad:
        line(FAIL, f"疑似占位符（非空真值，会报 401/403）：{', '.join(bad)}")
        results.append(("无占位符残留", FAIL, ", ".join(bad)))
    else:
        line(PASS, "无占位符残留")
        results.append(("无占位符残留", PASS, "全部为真实值或真空"))

    s2 = env("S2_API_KEY")
    line(PASS if not s2 else WARN,
         "S2_API_KEY             " + ("真空 → 走共享池（1.2s 节流）" if not s2 else mask(s2) + "（有 Key，走个人配额）"))

    for k in NOT_READ_BY_CODE:
        line(WARN, f"{k:<22} 当前代码零引用（悬空配置，不影响测试）")

    vs = env("VECTOR_STORE_PATH", "./data/store.sqlite")
    p = Path(vs) if Path(vs).is_absolute() else (ENV_PATH.parent / vs)
    line(PASS, f"VECTOR_STORE_PATH      {vs}  ->  {'已存在' if p.exists() else '待创建（首次入库时自动建）'}")

    # 拼接后的实际 URL，提前暴露 /v1 漏写或重复的问题
    for label, base, tail in (
        ("对话", env("AI_GATEWAY_BASE_URL"), "chat/completions"),
        ("向量", env("EMBEDDING_BASE_URL"), "embeddings"),
    ):
        if base:
            line(PASS, f"{label}URL 实际请求      {base.rstrip('/')}/{tail}")
            if base.rstrip("/").endswith("/v1/v1"):
                line(FAIL, f"{label} BASE_URL 出现重复 /v1")
    print()


# ---------------------------------------------------------------- 1) 对话模型
def test_chat(max_try: int = 4, model_override: str = "") -> None:
    print("=" * 70)
    print("1. 对话模型   POST {AI_GATEWAY_BASE_URL}/chat/completions")
    print("=" * 70)
    base, key = env("AI_GATEWAY_BASE_URL"), env("AI_GATEWAY_API_KEY")
    model = model_override or env("AI_GATEWAY_MODEL")
    if model_override:
        print(f"  模型覆盖：{model_override}（仅本次测试，不改 .env）")
    if not (base and key and model):
        line(FAIL, "变量缺失，跳过")
        results.append(("1 对话模型", FAIL, "变量缺失"))
        print()
        return

    url = f"{base.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "直接回复四个字，不要解释：连通正常"}],
        # 注意：glm-4.7-flash 是推理模型，先产 reasoning_content；
        # max_tokens 太小会让 content 为空，看起来像「请求成功但没回答」
        "max_tokens": 512,
        "temperature": 0,
        "stream": False,
    }

    r = None
    for attempt in range(1, max_try + 1):
        t0 = time.perf_counter()
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
        except Exception as e:
            line(FAIL, f"请求异常 {type(e).__name__}: {e}")
            results.append(("1 对话模型", FAIL, f"{type(e).__name__}"))
            print()
            return
        cost = time.perf_counter() - t0
        print(f"  HTTP {r.status_code}  耗时 {cost:.2f}s  第 {attempt}/{max_try} 次")

        if r.status_code == 200:
            break
        if r.status_code == 429 and attempt < max_try:
            # 智谱免费池 1305「当前访问量过大」属瞬时限流，退避即可
            wait = 8 * attempt
            line(WARN, f"429 限流，{wait}s 后重试")
            time.sleep(wait)
            continue
        break

    if r is None or r.status_code != 200:
        code = r.status_code if r is not None else "-"
        body = (r.text[:300].replace("\n", " ") if r is not None else "")
        line(FAIL, f"失败 HTTP {code}：{body}")
        hint = {
            401: "Key 无效或与 BASE_URL 不匹配（智谱 Key 不能用于硅基流动，反之亦然）",
            403: "Key 无该模型权限，或模型未开通",
            404: "BASE_URL 少了/多了 /v1，或模型名不存在",
            429: "限流或免费额度用尽；免费模型高峰期限流很常见，稍后重试或换模型",
        }.get(code, "")
        if hint:
            print(f"        提示：{hint}")
        results.append(("1 对话模型", FAIL, f"HTTP {code}"))
        print()
        return

    try:
        data = r.json()
        msg = data["choices"][0]["message"]
        text = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or "").strip()
    except Exception:
        line(FAIL, f"返回体结构异常：{r.text[:300]}")
        results.append(("1 对话模型", FAIL, "响应解析失败"))
        print()
        return

    print(f"  模型回显：{data.get('model', model)}")
    print(f"  回复内容：{text[:80]!r}")
    if reasoning:
        print(f"  推理摘要：{reasoning[:60]!r}（推理模型会先想再答）")
    usage = data.get("usage") or {}
    print(f"  token 用量：{usage.get('total_tokens', '?')}")

    if text:
        line(PASS, "对话模型连通")
        results.append(("1 对话模型", PASS, f"HTTP 200 / {cost:.2f}s"))
    else:
        line(WARN, "接口 200 但 content 为空，token 被 reasoning 吃光；业务代码不要设过小的 max_tokens")
        results.append(("1 对话模型", WARN, "content 为空（max_tokens 过小）"))
    print()


# ---------------------------------------------------------------- 2) 向量模型
def test_embedding() -> None:
    print("=" * 70)
    print("2. 向量模型   POST {EMBEDDING_BASE_URL}/embeddings")
    print("=" * 70)
    base, key, model = env("EMBEDDING_BASE_URL"), env("EMBEDDING_API_KEY"), env("EMBEDDING_MODEL")
    if not (base and key and model):
        line(FAIL, "变量缺失，跳过")
        results.append(("2 向量模型", FAIL, "变量缺失"))
        print()
        return

    url = f"{base.rstrip('/')}/embeddings"
    # 注意：bge-m3 拒绝 dimensions 参数，这里只传 model + input
    payload = {"model": model, "input": ["城市道路积水短时趋势预测"]}
    t0 = time.perf_counter()
    try:
        r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                          json=payload, timeout=90)
        cost = time.perf_counter() - t0
    except Exception as e:
        line(FAIL, f"请求异常 {type(e).__name__}: {e}")
        results.append(("2 向量模型", FAIL, f"{type(e).__name__}"))
        print()
        return

    print(f"  HTTP {r.status_code}  耗时 {cost:.2f}s")
    if r.status_code != 200:
        line(FAIL, f"失败：{r.text[:300]}")
        if r.status_code == 404:
            print("        提示：EMBEDDING_BASE_URL 必须是 https://api.siliconflow.cn/v1（含 /v1，且只一次）")
        results.append(("2 向量模型", FAIL, f"HTTP {r.status_code}"))
        print()
        return

    try:
        body = r.json()
        vec = body["data"][0]["embedding"]
    except Exception:
        line(FAIL, f"返回体结构异常：{r.text[:300]}")
        results.append(("2 向量模型", FAIL, "响应解析失败"))
        print()
        return

    dim = len(vec)
    print(f"  模型回显：{body.get('model', model)}")
    print(f"  维度：{dim}（期望 {EXPECTED_DIM}）")
    print(f"  前 5 维：{[round(x, 5) for x in vec[:5]]}")
    if dim == EXPECTED_DIM:
        line(PASS, "向量模型连通，维度正确")
        results.append(("2 向量模型", PASS, f"{dim} 维 / {cost:.2f}s"))
    else:
        line(FAIL, f"维度不符：实际 {dim}。换过模型（nomic-embed-text=768）需同步重建向量库")
        results.append(("2 向量模型", FAIL, f"维度 {dim} ≠ {EXPECTED_DIM}"))
    print()


# ---------------------------------------------------------------- 3) Semantic Scholar
def _s2_print(papers: list) -> None:
    print(f"  本次返回 {len(papers)} 条")
    for i, p in enumerate(papers[:3], 1):
        title = (p.get("title") or "").strip()
        print(f"   {i}. [{p.get('year')}] {title[:58]}  (引用 {p.get('citationCount')})")


def test_s2(query: str = "retrieval augmented generation", cool: int = 20) -> None:
    """注意：无 Key 时共享池配额极小，连续调用会触发整站级 429 且无 Retry-After。
    所以这里只打 1 次，失败后冷却再单发 1 次，避免把 IP 越测越黑。
    """
    print("=" * 70)
    print("3. Semantic Scholar   GET /graph/v1/paper/search")
    print("=" * 70)
    s2_key = env("S2_API_KEY")
    headers = {"x-api-key": s2_key} if s2_key else {}
    print(f"  鉴权方式：{'个人 Key ' + mask(s2_key) if s2_key else '无 Key（共享池，配额极小）'}")

    params = {"query": query, "limit": 3, "year": "2019-", "fields": S2_FIELDS}

    def hit() -> tuple[requests.Response | None, float]:
        t0 = time.perf_counter()
        try:
            return requests.get(f"{S2_BASE}/paper/search", params=params, headers=headers, timeout=25), time.perf_counter() - t0
        except Exception as e:
            print(f"  网络异常 {type(e).__name__}: {e}")
            return None, time.perf_counter() - t0

    r, cost = hit()
    if r is not None:
        print(f"  HTTP {r.status_code}  耗时 {cost:.2f}s")
        if r.status_code == 403:
            line(FAIL, "403：x-api-key 被拒绝 → S2_API_KEY 是无效值或占位符，请改回真空")
            results.append(("3 Semantic Scholar", FAIL, "HTTP 403 / Key 无效"))
            print()
            return

    if r is None:
        line(FAIL, "网络层不可达（DNS/超时/代理），与 .env 无关")
        results.append(("3 Semantic Scholar", FAIL, "网络不可达"))
        print()
        return

    if r.status_code == 200:
        data = r.json()
        papers = data.get("data") or []
        print(f"  命中总数 total={data.get('total')}")
        _s2_print(papers)
        if papers:
            line(PASS, "Semantic Scholar 可搜到结果")
            results.append(("3 Semantic Scholar", PASS, f"{len(papers)} 条 / total={data.get('total')}"))
        else:
            line(WARN, "接口通但零结果（换关键词或放宽 year）")
            results.append(("3 Semantic Scholar", WARN, "0 条结果"))
        print()
        return

    if r.status_code != 429:
        line(FAIL, f"失败 HTTP {r.status_code}：{r.text[:250]}")
        results.append(("3 Semantic Scholar", FAIL, f"HTTP {r.status_code}"))
        print()
        return

    # 429：冷却后只发一次轻量探测，判断是「仅 search 端点拥塞」还是「整站限流」
    line(WARN, f"429 共享池限流（响应无 Retry-After），冷却 {cool}s 后单次探测")
    time.sleep(cool)
    try:
        pr = requests.get(f"{S2_BASE}/paper/search/match",
                          params={"query": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                                  "fields": "title,year"}, headers=headers, timeout=20)
    except Exception as e:
        pr = None
        print(f"  /paper/search/match -> 异常 {type(e).__name__}: {e}")

    if pr is not None:
        print(f"  /paper/search/match -> HTTP {pr.status_code}")
    if pr is not None and pr.status_code == 200 and (pr.json().get("data") or []):
        item = pr.json()["data"][0]
        print(f"  命中：[{item.get('year')}] {(item.get('title') or '')[:58]}")
        line(WARN, "S2 服务可达且能出结果，但 /paper/search 被共享池限流")
        print("        影响：search_papers 工具会间歇 429；申请免费 Key 才能稳定")
        results.append(("3 Semantic Scholar", WARN, "search 端点 429，服务本身可用"))
    elif pr is not None and pr.status_code in (403, 404):
        line(FAIL, f"S2 返回 {pr.status_code} —— 请求被拒，与网络无关")
        results.append(("3 Semantic Scholar", FAIL, f"HTTP {pr.status_code}"))
    else:
        code = pr.status_code if pr is not None else "ERR"
        line(WARN, f"整站级 429（连 /match 也返回 {code}）：无 Key 共享池配额已耗尽")
        print("        影响：所有 S2 端点暂不可用，需等待冷却或申请 Key（用 edu.cn 邮箱）")
        results.append(("3 Semantic Scholar", WARN, f"整站 429（{code}），需 Key"))
    print()


# ---------------------------------------------------------------- 附加：备用数据源
def test_extra_fallback(query: str = "retrieval augmented generation") -> None:
    """S2 无 Key 被限流时的备用源体检（OpenAlex 免 Key、配额宽松；arXiv 要求 ≥3s 间隔）。"""
    print("=" * 70)
    print("4. 备用数据源（可选）   OpenAlex / arXiv")
    print("=" * 70)
    mailto = env("OPENALEX_MAILTO")

    try:
        t0 = time.perf_counter()
        r = requests.get("https://api.openalex.org/works",
                         params={"search": query, "per-page": 2, **({"mailto": mailto} if mailto else {})},
                         timeout=25)
        cost = time.perf_counter() - t0
        if r.status_code == 200:
            d = r.json()
            got = d.get("results") or []
            print(f"  OpenAlex HTTP 200  {cost:.2f}s  count={d.get('meta', {}).get('count')}")
            for w in got[:2]:
                print(f"   - [{w.get('publication_year')}] {(w.get('title') or '')[:58]} (引用 {w.get('cited_by_count')})")
            line(PASS, "OpenAlex 可搜到结果（S2 的可行替代）")
            results.append(("4 OpenAlex(备用)", PASS, f"count={d.get('meta', {}).get('count')}"))
        else:
            print(f"  OpenAlex HTTP {r.status_code}: {r.text[:150]}")
            line(WARN, f"OpenAlex HTTP {r.status_code}")
            results.append(("4 OpenAlex(备用)", WARN, f"HTTP {r.status_code}"))
    except Exception as e:
        line(FAIL, f"OpenAlex 异常 {type(e).__name__}: {e}")
        results.append(("4 OpenAlex(备用)", FAIL, type(e).__name__))

    time.sleep(3)  # arXiv 官方要求相邻请求间隔 ≥3 秒
    try:
        r = requests.get("https://export.arxiv.org/api/query",
                         params={"search_query": "all:retrieval augmented generation", "max_results": 2},
                         timeout=40)
        n = r.text.count("<entry>")
        print(f"  arXiv HTTP {r.status_code}  entries={n}")
        if r.status_code == 200 and n:
            line(PASS, "arXiv 可搜到结果")
            results.append(("4 arXiv(备用)", PASS, f"{n} 条"))
        else:
            line(WARN, f"arXiv HTTP {r.status_code}（同样受出口 IP 限流）")
            results.append(("4 arXiv(备用)", WARN, f"HTTP {r.status_code}"))
    except Exception as e:
        line(WARN, f"arXiv 异常 {type(e).__name__}（export.arxiv.org 在国内常被墙/超时）")
        results.append(("4 arXiv(备用)", WARN, type(e).__name__))
    print()


# ---------------------------------------------------------------- 汇总
def summary() -> int:
    print("=" * 70)
    print("汇总")
    print("=" * 70)
    for name, status, detail in results:
        print(f"  {MARK[status]} {name:<20} {detail}")
    fails = [n for n, s, _ in results if s == FAIL]
    warns = [n for n, s, _ in results if s == WARN]
    print()
    if fails:
        print(f"失败项：{', '.join(fails)}")
        return 1
    if warns:
        print(f"无失败，但有降级项：{', '.join(warns)}")
        return 2
    print("全部通过：对话 / 向量(1024) / Semantic Scholar 三项可用。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-chat", action="store_true")
    ap.add_argument("--skip-embed", action="store_true")
    ap.add_argument("--skip-s2", action="store_true")
    ap.add_argument("--query", default="retrieval augmented generation")
    ap.add_argument("--s2-cool", type=int, default=20, help="S2 遇 429 后的冷却秒数")
    ap.add_argument("--chat-model", default="", help="临时替换对话模型名做对照测试，不改 .env")
    ap.add_argument("--extra", action="store_true", help="额外体检备用源 OpenAlex / arXiv")
    args = ap.parse_args()

    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH, override=False)  # 不覆盖已存在的系统环境变量

    check_env()
    if not args.skip_chat:
        test_chat(model_override=args.chat_model)
    if not args.skip_embed:
        test_embedding()
    if not args.skip_s2:
        test_s2(args.query, args.s2_cool)
    if args.extra:
        test_extra_fallback(args.query)
    return summary()


if __name__ == "__main__":
    raise SystemExit(main())
