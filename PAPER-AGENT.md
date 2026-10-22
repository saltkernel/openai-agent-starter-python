# 论文调研 Agent —— 搭建与验收指南

把原来的 Demo（天气/翻译示例工具）改造成「输入研究方向 → 检索论文 → 全文入库 → 出可溯源综述 → 追问细节」的研究助手。

---

## 一、已经做完的部分（你不用动手）

| 文件 | 作用 |
|---|---|
| `agents/_papers.py` | 数据源层：S2 限流（≥1.2s）+ 429 指数退避、OpenAlex 兜底、arXiv Atom 解析、paper_id 统一归一化 |
| `agents/tools.py` | 7 个 `@function_tool`：主题检索 / 精读单篇 / 引用溯源 / 最新预印本 / 全文入库 / 混合检索 / 库内清单 |
| `agents/rag.py` | 全文抓取（arXiv HTML → ar5iv → OA PDF → 摘要）、切片、bge-m3 向量化、SQLite 落地、BM25+向量+RRF 融合 |
| `agents/chat/index.py` | 换成论文调研的 system instructions（五条铁律 + 引用格式），注册新工具，每轮注入库内清单 |
| `api-schema.json` | 每个工具的 name / description / method / endpoint / 参数 / 返回字段 |
| `scripts/smoke_paper_agent.py` | 不依赖 Web 服务的端到端冒烟测试 |
| `check_connectivity_min.py` | 三项连通性体检（对话 / 向量 1024 维 / S2） |

实测已通过：88 片、77 片、103 片三篇论文入库；`ARXIV:` 前缀能命中以 `S2:` 入库的论文；单篇追问能产出 `[1]` + 参考文献表。

---

## 二、你要手动做的（按编号顺序）

### 步骤 1｜补装两个依赖

```bash
cd openai-agent-starter-python
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

新增的是 `pypdf`（PDF 全文解析）和 `requests`（HTTP）。requirements.txt 已改好，云端部署会自动装。

### 步骤 2｜核对 `.env`

必须非空：`AI_GATEWAY_API_KEY`、`AI_GATEWAY_BASE_URL`、`EMBEDDING_API_KEY`、`EMBEDDING_BASE_URL`、`EMBEDDING_MODEL`。
建议填写：`OPENALEX_MAILTO`、`UNPAYWALL_EMAIL`（真实邮箱，Unpaywall 不给假邮箱服务）、`S2_API_KEY`（有 Key 才不容易 429）。
`VECTOR_STORE_PATH` 默认 `./data/store.sqlite`，已加进 `.gitignore`。

验证：

```bash
.venv/Scripts/python.exe check_connectivity_min.py
```

三项全 PASS 再往下走。`S2` 报 429 属正常（按 IP 限流），冷却几分钟即可，代码里也有 OpenAlex 兜底。

### 步骤 3｜跑冒烟测试（不花对话模型额度）

```bash
.venv/Scripts/python.exe scripts/smoke_paper_agent.py
```

预期：14 项全 PASS，并打印切片数、检索命中、库内清单。

想换目标论文：`--arxiv-id ARXIV:2405.01234`。只想验算法：`--skip-network`。

### 步骤 4｜装 EdgeOne CLI 并起服务

```bash
npm install
npm i -g edgeone        # 已装可跳过
npm run dev:agents      # 或 edgeone makers dev
```

打开 `http://localhost:8080`，前端聊天页；`http://localhost:8080/agent-metrics` 是本地观测面板。

### 步骤 5｜按验收用例对话

| # | 输入 | 期望表现 |
|---|---|---|
| 1 | `帮我调研多模态大模型在城市积水监测中的应用` | 依次调 `search_papers` → `ingest_paper`（3-6 篇）→ `search_passages`；输出分节综述，每条结论带 `[1]`，末尾有 `## 参考文献` |
| 2 | `第二篇论文的 batch size 和学习率是多少？` | 先 `list_ingested_papers`（或直接用注入的清单），再 `search_passages(paper_id=...)`；片段里没有就答「该论文入库片段中未提及此数值」 |
| 3 | `谁改进了这篇论文？` | 调 `get_citations`，返回后续工作列表 |
| 4 | `最近三个月 arXiv 上有哪些新工作？` | 调 `search_arxiv_latest`，按提交时间倒序 |
| 5 | `这个方向有哪些综述？`（库里没有） | 调 `search_papers`，不得凭空列文献 |

判断标准：**随便抽一句带 `[n]` 的话，去 `search_passages` 返回的片段里能对上原文**，就算通过。

### 步骤 6｜部署（可选）

```bash
edgeone makers deploy      # 或走 Makers 控制台，以 CLI 提示为准
```

部署前确认控制台里也配了同一套环境变量（`.env` 不会上传）。

> ⚠️ 云上沙箱的 `data/store.sqlite` 会在重新部署/实例回收后重置。演示够用；要长期保存就换外部存储（对象存储或云数据库）。

### 步骤 7｜清库重建（改过 embedding 模型时）

直接删文件即可，下次 `ingest_paper` 自动建表：

```bash
rm -rf data/
```

---

## 三、几个需要你知道的设计取舍

1. **S2 一定会 429**。共享池按出口 IP 计数，实测连续调用几分钟就被封。所以检索/详情/引用三处都接了 OpenAlex 兜底，返回体的 `source` 字段会告诉你数据实际来自哪。
2. **弱模型会违反铁律**。`glm-4-flash` 实测能答对内容但会丢 `[n]` 和文献表；`glm-4.7-flash` 遵守得好但高峰期常 1305 限流。要稳定产出合规综述，建议换成更强的模型（4.7 或付费模型）。
3. **`tools.py` / `rag.py` 不是下划线开头**。EdgeOne 约定 `_` 前缀才是私有模块，非下划线文件有被当成路由暴露的风险。若部署时报路由冲突，改名成 `_tools.py` / `_rag.py` 并同步 `index.py` 的 import 即可，不影响逻辑。
4. **每次请求会把库内清单拼进 instructions**。这是为了让模型知道「哪些论文已经能追问」，否则实测它会联网重搜已入库论文。清单上限 20 篇、每行 90 字，成本可忽略。
5. **`agents/_tools.py` 里的示例工具已不再调用**，保留未删，可自行移除。

---

## 四、常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `EMBEDDING_* 未配置` | .env 缺项 | 补 `.env`，跑步骤 2 |
| `向量维度 xxx ≠ 1024` | 换了 embedding 模型 | 删 `data/` 重建 |
| 追问答「该论文尚未入库」 | 只 search 了没 ingest | 让它先 `ingest_paper`，或看 `abstract_only` |
| 入库很慢 | PDF 有几十页 | 正常，18MB PDF 约 18s；arXiv HTML 通常 1s 内 |
| 综述里出现库里没有的论文 | 模型编造 | 提高模型能力，或在 instructions 里再加一条硬约束 |
| `MaxTurnsExceeded` | 模型反复检索 | 已加「同一问题最多检索 3 次」约束；仍超出说明模型太弱 |
