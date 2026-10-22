"""最小连通性测试（只读，不改任何业务代码）。

三项逐一验证：
  1) 对话模型     POST {AI_GATEWAY_BASE_URL}/chat/completions
  2) 向量模型     POST {EMBEDDING_BASE_URL}/embeddings     必须 1024 维
  3) Semantic Scholar  GET /graph/v1/paper/search         必须能搜到结果

用法（仓库根目录）：
    .venv/Scripts/python.exe check_connectivity_min.py
    .venv/Scripts/python.exe check_connectivity_min.py --query "urban flood forecasting"

退出码：0 全通过 / 1 有失败。
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

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
EXPECTED_DIM = 1024
TIMEOUT = 60
S2_COOL = 15  # S2 遇 429 后的冷却秒数（无 Retry-After，只能经验值）


def env(name: str, default: str = "") -> str:
    """读取并去掉首尾空白 —— .env 里 `KEY= value` 这种写法会带前导空格。"""
    return (os.getenv(name) or default).strip()


def mask(v: str) -> str:
    v = (v or "").strip()
    return "(空)" if not v else (v[:4] + "..." + v[-4:] if len(v) > 8 else "*" * len(v))


def ok(msg: str) -> bool:
    print(f"  [PASS] {msg}")
    return True


def bad(msg: str) -> bool:
    print(f"  [FAIL] {msg}")
    return False


# ---------------------------------------------------------------- 0) 变量体检
def check_vars() -> bool:
    required = ["AI_GATEWAY_API_KEY", "AI_GATEWAY_BASE_URL", "AI_GATEWAY_MODEL",
                "EMBEDDING_API_KEY", "EMBEDDING_BASE_URL", "EMBEDDING_MODEL"]
    print("0) 变量体检")
    if not ENV_PATH.exists():
        return bad(f"找不到 {ENV_PATH}")
    print(f"  .env = {ENV_PATH}")
    missing = [k for k in required if not env(k)]
    for k in required:
        v = env(k)
        print(f"  {'OK ' if v else 'MISS'} {k:<20} {mask(v) if 'KEY' in k else (v or '(空)')}")
    # 占位符残留：非空但是模板文字，会直接 401/403
    ph = [k for k in required if env(k) and env(k).lower().startswith(("your-", "your_", "sk-xxx"))]
    if ph:
        return bad(f"疑似占位符：{ph}")
    if missing:
        return bad(f"缺失必填项：{missing}")
    return ok(f"{len(required)} 个必填项齐全")


# ---------------------------------------------------------------- 1) 对话模型
def test_chat(model_override: str = "") -> bool:
    base, key, model = env("AI_GATEWAY_BASE_URL"), env("AI_GATEWAY_API_KEY"), env("AI_GATEWAY_MODEL")
    if model_override:
        model = model_override
        print(f"\n  模型临时替换为 {model}（仅本次运行，不改 .env）")
    url = f"{base.rstrip('/')}/chat/completions"
    print(f"\n1) 对话模型  POST {url}")
    if not (base and key and model):
        return bad("变量缺失，跳过")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "只回复两个字：正常"}],
        # glm 系列推理模型会先写 reasoning_content，max_tokens 太小会导致 content 为空
        "max_tokens": 512,
        "temperature": 0,
    }
    r, cost = None, 0.0
    for attempt in range(1, 5):
        try:
            t0 = time.perf_counter()
            r = requests.post(url, headers={"Authorization": f"Bearer {key}",
                                            "Content-Type": "application/json"},
                              json=payload, timeout=TIMEOUT)
            cost = time.perf_counter() - t0
        except Exception as e:
            return bad(f"请求异常 {type(e).__name__}: {e}")

        print(f"  HTTP {r.status_code}  {cost:.2f}s  (第 {attempt}/4 次)")
        # 智谱免费池 1305「访问量过大」是容量型限流，与配置无关，退避重试即可
        if r.status_code == 429 and attempt < 4:
            wait = 10 * attempt
            print(f"  限流，{wait}s 后重试…")
            time.sleep(wait)
            continue
        break

    if r is None or r.status_code != 200:
        if r is None:
            return bad("未拿到响应")
        hint = {401: "Key 与 BASE_URL 不匹配", 403: "无该模型权限",
                404: "BASE_URL 少/多 /v1 或模型名不存在",
                429: "重试后仍限流：免费模型高峰期限流，稍后再测或换模型"}.get(r.status_code, "")
        return bad(f"HTTP {r.status_code}: {r.text[:200]}" + (f"  → {hint}" if hint else ""))
    try:
        msg = r.json()["choices"][0]["message"]
    except Exception:
        return bad(f"响应结构异常：{r.text[:200]}")

    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or "").strip()
    print(f"  模型回显：{r.json().get('model', model)}")
    print(f"  回答：{content[:60]!r}")
    if reasoning:
        print(f"  思考片段：{reasoning[:50]!r}")
    if not content:
        return bad("HTTP 200 但 content 为空（max_tokens 被 reasoning 吃光）")
    return ok(f"对话模型可用，{cost:.2f}s")


# ---------------------------------------------------------------- 2) 向量模型
def test_embedding() -> bool:
    base, key, model = env("EMBEDDING_BASE_URL"), env("EMBEDDING_API_KEY"), env("EMBEDDING_MODEL")
    url = f"{base.rstrip('/')}/embeddings"
    print(f"\n2) 向量模型  POST {url}")
    if not (base and key and model):
        return bad("变量缺失，跳过")
    try:
        t0 = time.perf_counter()
        # bge-m3 不接受 dimensions 参数，只传 model + input
        r = requests.post(url, headers={"Authorization": f"Bearer {key}"},
                          json={"model": model, "input": ["城市道路积水短时趋势预测"]}, timeout=TIMEOUT)
        cost = time.perf_counter() - t0
    except Exception as e:
        return bad(f"请求异常 {type(e).__name__}: {e}")

    print(f"  HTTP {r.status_code}  {cost:.2f}s")
    if r.status_code != 200:
        return bad(f"HTTP {r.status_code}: {r.text[:200]}")
    try:
        vec = r.json()["data"][0]["embedding"]
    except Exception:
        return bad(f"响应结构异常：{r.text[:200]}")

    dim = len(vec)
    print(f"  模型回显：{r.json().get('model', model)}")
    print(f"  维度：{dim}（期望 {EXPECTED_DIM}）")
    print(f"  前 3 维：{[round(x, 5) for x in vec[:3]]}")
    if dim != EXPECTED_DIM:
        return bad(f"维度不符：{dim} ≠ {EXPECTED_DIM}，换模型后需重建向量库")
    return ok(f"向量模型可用，{dim} 维，{cost:.2f}s")


# ---------------------------------------------------------------- 3) Semantic Scholar
def test_s2(query: str) -> bool:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    key = env("S2_API_KEY")
    headers = {"x-api-key": key} if key else {}
    print(f"\n3) Semantic Scholar  GET {url}")
    print(f"  鉴权：{'个人 Key ' + mask(key) if key else '无 Key（共享池，易 429）'}")
    try:
        r, cost = None, 0.0
        for attempt in range(1, 3):
            t0 = time.perf_counter()
            r = requests.get(url, headers=headers, timeout=40, params={
                "query": query, "limit": 3,
                "fields": "title,year,citationCount,externalIds,abstract",
            })
            cost = time.perf_counter() - t0
            print(f"  HTTP {r.status_code}  {cost:.2f}s  (第 {attempt}/2 次)  查询：{query!r}")
            if r.status_code == 429 and attempt < 2:
                print(f"  限流，冷却 {S2_COOL}s 后重试…")
                time.sleep(S2_COOL)
                continue
            break
    except Exception as e:
        return bad(f"网络异常 {type(e).__name__}: {e}")

    if r.status_code == 403:
        return bad("403：x-api-key 无效 → S2_API_KEY 请改回空值")
    if r.status_code == 429:
        return bad(f"429：限流（冷却 {S2_COOL}s 重试后仍失败），稍后再测或换 Key")
    if r.status_code != 200:
        return bad(f"HTTP {r.status_code}: {r.text[:200]}")

    data = r.json()
    papers = data.get("data") or []
    print(f"  total={data.get('total')}  本次返回 {len(papers)} 条")
    for i, p in enumerate(papers, 1):
        print(f"   {i}. [{p.get('year')}] {(p.get('title') or '')[:56]} (引用 {p.get('citationCount')})")
    if not papers:
        return bad("接口通但 0 结果，换个 query 试试")
    return ok(f"Semantic Scholar 可用，{len(papers)} 条结果，{cost:.2f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="retrieval augmented generation")
    ap.add_argument("--skip-chat", action="store_true")
    ap.add_argument("--skip-embed", action="store_true")
    ap.add_argument("--skip-s2", action="store_true")
    ap.add_argument("--model", default="", help="临时替换对话模型名（不改 .env），用于对照测试")
    args = ap.parse_args()

    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH, override=False)

    checks = [("变量体检", check_vars)]
    if not args.skip_chat:
        checks.append(("对话模型", lambda: test_chat(args.model)))
    if not args.skip_embed:
        checks.append(("向量模型", lambda: test_embedding()))
    if not args.skip_s2:
        checks.append(("Semantic Scholar", lambda: test_s2(args.query)))

    results = [(name, fn()) for name, fn in checks]

    print("\n" + "=" * 60)
    for name, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    failed = [n for n, p in results if not p]
    print("=" * 60)
    print("全部通过" if not failed else f"失败：{', '.join(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
