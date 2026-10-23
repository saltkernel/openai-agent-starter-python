const zh = {
  // Header
  "app.title": "论文调研 Agent",
  "app.subtitle": "可溯源的文献综述助手",

  // Empty state
  "empty.title": "论文调研 Agent",
  "empty.hint": "我检索论文原文、逐条切片入库，回答中每个结论都标注出处。检索不到证据时我会直说，不凭常识编造。",
  "empty.features": "混合检索 · 引用溯源 · 全文入库",

  // Chat input
  "chat.placeholder": "输入论文主题或问题…  ⏎ 发送 · Shift+⏎ 换行",
  "chat.hint": "由 OpenAI Agents SDK + EdgeOne Makers 驱动 · 结论可逐条溯源",

  // Preset questions
  "preset.1": "帮我综述一下 RAG 在检索器训练上的进展",
  "preset.2": "这篇论文的实验设置是什么？用的什么数据集？",

  // Tool indicators —— 键名与 agents/tools.py 的 PAPER_TOOLS 一一对应
  "tool.searchPapers": "论文检索",
  "tool.paperDetail": "论文详情",
  "tool.citations": "引用溯源",
  "tool.arxivLatest": "最新预印本",
  "tool.ingestPaper": "全文入库",
  "tool.searchPassages": "片段检索",
  "tool.listIngested": "库内清单",

  // Status & errors
  "status.error": "⚠️ 请求失败，请检查后端服务是否启动。",
  "status.stopped": "⏹ *已停止生成*",
  "status.backendError": "⚠️ 后端中断请求失败，服务端可能仍在运行。",

  // Debug panel
  "debug.title": "传输流",
  "debug.events": "事件",
  "debug.clear": "清除",
  "debug.empty": "等待 SSE 事件...",
  "debug.emptyHint": "发送消息后，所有原始后端数据将在此处显示。",

  // Conversation sidebar
  "sidebar.label": "会话列表",
  "sidebar.title": "会话",
  "sidebar.newChat": "新建聊天",
  "sidebar.loading": "正在加载会话...",
  "sidebar.loadMore": "加载更多",
  "sidebar.loadingMore": "加载中...",
  "sidebar.emptyTitle": "暂无会话",
  "sidebar.emptyHint": "点击「新建聊天」开始第一段对话。",
  "sidebar.delete": "删除会话",
  "sidebar.deleteConfirm": "确定要永久删除这个会话吗？此操作不可恢复。",

  // Aria labels (button hover/screen-reader)
  "aria.send": "发送",
  "aria.clearHistory": "清除历史",
  "aria.stopGeneration": "停止生成",

  // Language toggle
  "lang.switch": "English",

  // ─── Floating bottom-right action badges ─────────────────────────────
  "floatingLink.deploy": "一键部署",
  "floatingLink.github": "GitHub",
} as const;

export default zh;
