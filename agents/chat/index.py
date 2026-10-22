"""
Agent handler — EdgeOne Makers
========================================

The file path agents/chat/index.py is auto-mapped to **POST /chat**.
(EdgeOne Makers convention: directory name = route name, `index` is the default entry.)

`context` contract:
    context.request.body    — dict, request body
    context.request.signal  — asyncio.Event, set when /chat/stop is called
    context.conversation_id — conversation ID
    context.run_id          — run ID for this invocation
    context.store           — EdgeOne built-in store; provides openai_session() and friends
"""

from typing import Any, AsyncGenerator
import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, OpenAIChatCompletionsModel, Runner

from .._logger import create_logger
from ..rag import list_ingested
from ..tools import PAPER_TOOLS


# override=True：EdgeOne CLI (makers dev) 会向进程注入自家的 AI_GATEWAY_API_KEY /
# AI_GATEWAY_BASE_URL（内置 Makers 网关），默认 load_dotenv 不覆盖已有变量，会导致
# .env 里的智谱配置被顶掉 → 401 "API key not found"。这里强制让 .env 赢回来。
load_dotenv(override=True)

logger = create_logger("chat")


# ========== LLM Model ==========
llm_client = AsyncOpenAI(
    api_key=os.getenv("AI_GATEWAY_API_KEY"),
    base_url=os.getenv("AI_GATEWAY_BASE_URL"),
)

llm_model = OpenAIChatCompletionsModel(
    model=os.getenv("AI_GATEWAY_MODEL", "@makers/deepseek-v4-flash"),
    openai_client=llm_client,
)


# ========== Agent 定位 ==========
# 这个 Agent 的产出是「可溯源的文献综述」，不是聊天。指令里三条硬约束互相咬合：
#   工具分工（检索/入库/取证据）→ 引用规则（[n] + 文献表）→ 反编造（无证据就说没有）
INSTRUCTIONS = """你是「论文调研 Agent」，服务对象是需要写文献综述的研究者。你的唯一任务是：把检索到的论文原文，整理成可逐条溯源的综述正文。

# 可用工具与分工（不要跳步）
- search_papers：按主题检索论文，拿候选列表和 paper_id。这是任何调研的第一步。
- search_arxiv_latest：查最新预印本，用来补上期刊数据库的时间滞后。
- get_paper_detail：只看某篇的元数据与摘要，不产生正文证据。
- ingest_paper：把论文全文抓取、切片、向量化入库。只有入库后的论文才能被读细节。
- search_passages：混合检索（BM25+向量+RRF），返回带 paper_id 与片段序号的原文。**这是唯一允许的事实来源。**
- get_citations：查谁引用了某篇论文，用来讲技术演进脉络。
- list_ingested_papers：确认当前可追问的论文范围。

标准流程：search_papers → 挑 3-6 篇最相关的 ingest_paper → 针对每个论点逐条 search_passages → 成文。论文多时先入库再检索，不要凭标题猜内容。

**先查库，再联网**：任何涉及论文内容或细节的问题，第一步调用 list_ingested_papers。库里已有的论文不要再 search_papers 重搜、也不要重复 ingest_paper，直接用 search_passages 取证据。只有库里没有覆盖用户问的主题时，才联网检索并入库。

# 铁律（违反任何一条都视为任务失败）
1. 任何事实性结论都必须来自 search_passages 或 get_paper_detail 返回的原文片段，不得凭记忆、常识或标题作答。写作前先问自己：这句话对应哪条片段？
2. 每条结论后必须紧跟引用标注。标注一律是「方括号 + 阿拉伯数字」：`[1]`、`[2]`。严禁写成 `[n]`、`[n1]`、`[^1]`、`[文献1]`、`[1,2]`（需要两篇就写 `[1][2]`）。编号从 1 开始，按正文首次出现顺序递增，同一个数字始终指向同一篇文献。
3. 每次回答的末尾必须输出参考文献表，标题固定为 `## 参考文献`，与正文编号一一对应、不重不漏。每条格式：
   `[1] 标题. 第一作者 et al. (年份). venue. DOI: 10.xxxx 或 https://arxiv.org/abs/xxxx`
   其中标题、作者、年份、venue、DOI、arXiv 全部取自工具返回的 `papers` 字段；字段缺失就写「数据源未提供」，不得猜。
4. 检索不到、片段不足、或片段里没有相关内容时，直接输出「未检索到支撑证据」，并说明缺哪一类证据。禁止用常识补全，禁止编造论文、作者、数据。
5. 被引数、年份、作者、venue、DOI 等数字与字符串，必须原样来自工具返回值，不得推算、四舍五入或凭印象填写。工具没给就写「数据源未提供」。
6. 一句话只归因一篇文献，不得把多篇论文的结论合并成一句再挂一个引用。需要对比时拆成多句，各带各的标注。
7. 标注只允许出现在有片段支撑的句子后面；不要给过渡句、你自己的评价随意挂引用。

# 输出格式
- 默认用中文写作，论文标题、术语、方法名保留英文原文。
- 综述结构：研究现状 → 方法与技术路线 → 数据集与评测 → 局限与开放问题。每节 2-5 段，段落之间用引用串起证据链。
- 篇幅服从证据量：只有 2 篇论文有片段，就写 2 篇能支撑的内容，不要为凑字数扩写。
- 即使回答只有一句话，也必须带 `[1]` 和 `## 参考文献`，不能省略文献表。
- 若某篇论文只入库了摘要（工具返回 abstract_only=true），明确声明「该篇仅有摘要级证据，细节未验证」，不要假装读过正文。

# 引用格式样例（只照格式，内容不得复用；实际编号与文献以工具返回为准）
用户问：RAG 的检索器是怎么训练的？

答：RAG 的检索器用 DPR 初始化，并在训练时与生成器联合优化，检索文档的边际化作为隐变量处理 `[1]`。消融显示，冻结检索器会全面拉低下游任务表现，说明联合训练检索端是必要的 `[1]`。

## 参考文献
[1] Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. Lewis et al. (2020). NeurIPS. https://arxiv.org/abs/2005.11401


# 追问场景
用户追问某篇论文的方法、超参、实验设置时：
1. 先确认这篇论文是否已入库——用 list_ingested_papers，或在对话上文里找它的 paper_id；
2. 调用 search_passages 时**必须带 paper_id**，把检索范围限定在该篇。不带 paper_id 的全库检索会把别的论文片段混进来，是错误归因的主要来源；
3. 片段里没有的具体数值（如学习率、batch size），直接说「该论文入库片段中未提及此数值」，不得给出常见默认值。

# 效率约束
- 同一个问题最多检索 3 次。第一次没命中就换关键词，仍不中就直接说「未检索到支撑证据」，不要反复换词试探。
- 一次回答最多入库 6 篇论文；用户没点名要综述时，不要主动批量入库。
- 检索结果若明显与问题主题不符（例如问 DPR 却返回城市积水论文），判定为不相关，不要拿来作答。"""


# ========== 库内清单注入 ==========
def _library_hint(limit: int = 20) -> str:
    """把已入库论文清单拼进 instructions。

    为什么必须在指令里给出清单：模型看不到本地 SQLite，只能靠工具自查；
    实测弱模型会直接联网重搜「已经入库的论文」，白烧配额还答错。
    每轮注入一份很短的清单（paper_id + 标题 + 年份）成本极低，效果确定。
    """
    try:
        lib = list_ingested()
    except Exception as e:  # noqa: BLE001 —— 向量库异常不应挡住聊天
        logger.error(f"[library-hint] 读取失败：{type(e).__name__}: {e}")
        return ""
    if not lib.get("count"):
        return "\n\n# 当前向量库\n空。用户若要问论文细节，先 search_papers → ingest_paper 入库。"

    lines = []
    for p in lib["papers"][:limit]:
        year = p.get("year") or "-"
        lines.append(f"- {p['paper_id']} | {p['title'][:90]} ({year}) | {p['n_chunks']} 片段")
    more = "" if lib["count"] <= limit else f"\n（另有 {lib['count'] - limit} 篇未列出）"
    return (
        "\n\n# 当前向量库（已入库，可直接 search_passages，勿重复联网检索或重复入库）\n"
        + "\n".join(lines) + more
        + "\n用户问到的论文若在此清单内，直接用其 paper_id 检索；不在清单内才联网。"
    )


# ========== Agent ==========
agent = Agent(
    name="Paper Survey Assistant",
    instructions=INSTRUCTIONS,
    tools=PAPER_TOOLS,
    model=llm_model,
)


# ========== SSE Helper ==========
def sse_event(event: str, data: dict) -> str:
    """Format a single SSE event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ========== Event Stream Generator ==========
async def _event_stream(
    message: str,
    session=None,
    cancel_signal: asyncio.Event | None = None,
    run_agent: Agent | None = None,
) -> AsyncGenerator[str, None]:
    """Async generator that yields SSE frames from the agent stream.

    Accepts cancel_signal to enable cancellation — when set,
    the loop breaks and the SDK releases its connection.
    run_agent 允许传入「注入了库内清单」的克隆体，None 时用模块级 agent。
    """
    result = Runner.run_streamed(run_agent or agent, input=message, session=session)

    async for event in result.stream_events():
        # Check cancel between each event
        if cancel_signal and cancel_signal.is_set():
            break

        # Raw token delta → push per-character
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            logger.log(f"[stream] text_delta: {repr(event.data.delta)}")
            yield sse_event("text_delta", {"delta": event.data.delta})

        # Semantic event → tool called
        elif event.type == "run_item_stream_event":
            if event.name == "tool_called":
                tool_name = (
                    getattr(event.item, "name", None)
                    or getattr(getattr(event.item, "raw_item", None), "name", None)
                )
                if tool_name:
                    logger.log(f"[stream] tool_called: {tool_name}")
                    yield sse_event("tool_called", {"tool": tool_name})


# ========== Core Handler ==========
async def handler(context: Any) -> AsyncGenerator[str, None]:
    """EdgeOne Makers entry point.

    Uses OpenAI Agents SDK session for automatic memory:
      - SDK calls session.get_items() to inject history
      - SDK calls session.add_items() to persist assistant replies and tool results
    """
    request = context.request
    body = request.body
    message = body.get("message") if isinstance(body, dict) else None

    if not message:
        yield sse_event("error", {"message": "'message' is required"})
        yield sse_event("done", {})
        return

    # Accept both camelCase (chat handler historical convention) and snake_case
    # (cloud-functions convention) as a body field name for the user id.
    raw_user_id = ""
    if isinstance(body, dict):
        raw_user_id = body.get("userId") or body.get("user_id") or ""
    user_id = str(raw_user_id).strip() or None

    # Session for memory persistence
    cid = context.conversation_id

    logger.log(f"[request] cid={cid}, uid={user_id or '-'}, message={message[:50]!r}")

    # Write a user-indexed copy of the user message ONLY on the first turn of
    # this conversation, so /conversations (which scans the
    # user_conversation_index prefix) can list this thread.
    #
    # Why first-turn-only:
    #   - The OpenAI Agents SDK Session adapter does NOT pass user_id when it
    #     persists turns, so without a user-indexed write the user index stays
    #     empty and list_conversations(user_id=...) returns [].
    #   - But the index lives at the (user_id, conversation_id) level — once
    #     a single message has been written under this user_id, the
    #     conversation is reachable. Writing a user-indexed copy on every turn
    #     just produces duplicates that downstream consumers have to dedupe
    #     (see cloud-functions/conversations/index.py and history/index.py)
    #     and surfaces a duplicated user message in OpenAI Agents SDK traces
    #     because the SDK's session.get_items() will re-feed it as input
    #     alongside the new `message` argument.
    #   - We therefore probe with get_messages(limit=1); if the conversation
    #     already has any record, the index is established and we skip.
    #
    # The /history route already filters by metadata.agent_sdk_session and
    # dedupes adjacent items, so the chat window stays clean either way —
    # but the trace dashboard shows raw SDK input, hence the visible doubling.
    body_keys_preview = list(body.keys()) if isinstance(body, dict) else None
    logger.log(
        f"[user-index] body keys={body_keys_preview!r} "
        f"raw_user_id={raw_user_id!r} normalized_user_id={user_id!r} cid={cid!r}"
    )
    if user_id and cid:
        try:
            existing = await context.store.get_messages(
                conversation_id=cid, limit=1
            )
            already_indexed = bool(existing)
        except Exception as e:
            # If the probe fails, fall through to writing — better to leave
            # a duplicate than to lose the user index entirely.
            logger.error(
                f"[user-index] probe failed: {type(e).__name__}: {e}; "
                f"will write user-index entry anyway"
            )
            already_indexed = False

        if already_indexed:
            logger.log(
                f"[user-index] SKIPPED user-index write: cid={cid} already has "
                f"messages, user index for user_id={user_id!r} is established"
            )
        else:
            try:
                # NOTE: the Python store.append_message() does NOT accept a
                # `message_id` kwarg (unlike the TS counterpart) — the SDK
                # auto-generates one. Passing it raises TypeError, which
                # silently kills the user-index write.
                msg_id = await context.store.append_message(
                    conversation_id=cid,
                    role="user",
                    content=message,
                    user_id=user_id,
                )
                logger.log(
                    f"[user-index] WROTE first-turn user-indexed message: cid={cid} "
                    f"user_id={user_id} msg_id={msg_id!r}"
                )
            except Exception as e:
                # Non-fatal — chat itself should keep working even if the
                # user-index write fails.
                logger.error(f"[user-index] FAILED to write user index: {type(e).__name__}: {e}")
    else:
        logger.log(
            f"[user-index] SKIPPED user-index write: user_id={user_id!r} cid={cid!r} "
            f"(both must be truthy to trigger the write — this is why "
            f"/conversations would return empty for this user)"
        )

    session = context.store.openai_session(cid) if cid else None

    # Cancel signal (asyncio.Event), set when /chat/stop is called
    cancel_signal = request.signal
    stopped = False

    # 每轮把库内清单拼进 instructions：模型看不到 SQLite，不注入就会盲目联网重搜
    run_agent = agent.clone(instructions=INSTRUCTIONS + _library_hint())

    try:
        async for frame in _event_stream(message, session, cancel_signal, run_agent):
            if cancel_signal.is_set():
                stopped = True
                break
            yield frame
    except asyncio.CancelledError:
        stopped = True
        logger.log("[stream] cancelled")
    except Exception as e:
        logger.error(f"[stream] error: {type(e).__name__}: {e}")
        # Try to surface upstream HTTP error body (e.g. OpenAI SDK APIStatusError) to the TRACE panel
        detail: Any = str(e)
        status: Any = None
        response = getattr(e, "response", None)
        if response is not None:
            status = getattr(response, "status_code", None)
            try:
                body_text = response.text if hasattr(response, "text") else None
                if callable(body_text):
                    body_text = body_text()
                if body_text:
                    try:
                        detail = json.loads(body_text)
                    except (json.JSONDecodeError, ValueError, TypeError):
                        detail = body_text
            except Exception:
                pass
        body_attr = getattr(e, "body", None)
        if body_attr and detail == str(e):
            detail = body_attr
        yield sse_event("error", {
            "message": str(e),
            "errorType": type(e).__name__,
            "status": status,
            "detail": detail,
        })
    finally:
        yield sse_event("done", {"stopped": stopped})
