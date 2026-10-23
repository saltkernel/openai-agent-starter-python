const en = {
  // Header
  "app.title": "Paper Survey Agent",
  "app.subtitle": "A literature-review assistant with traceable citations",

  // Empty state
  "empty.title": "Paper Survey Agent",
  "empty.hint": "I retrieve paper full texts, chunk and index them, then answer with a citation on every claim. When the evidence is missing, I say so instead of guessing.",
  "empty.features": "Hybrid Retrieval · Traceable Citations · Full-text Index",

  // Chat input
  "chat.placeholder": "Ask about a paper or topic...  ⏎ Send · Shift+⏎ Newline",
  "chat.hint": "Powered by OpenAI Agents SDK + EdgeOne Makers · Every claim is traceable",

  // Preset questions
  "preset.1": "Summarise the progress of retriever training in RAG",
  "preset.2": "What is the experimental setup of this paper? Which datasets were used?",

  // Tool indicators —— keys mirror PAPER_TOOLS in agents/tools.py
  "tool.searchPapers": "Search",
  "tool.paperDetail": "Details",
  "tool.citations": "Citations",
  "tool.arxivLatest": "arXiv",
  "tool.ingestPaper": "Ingest",
  "tool.searchPassages": "Passages",
  "tool.listIngested": "Library",

  // Status & errors
  "status.error": "Request failed. Please check if the backend service is running.",
  "status.stopped": "⏹ *Generation stopped*",
  "status.backendError": "Backend abort request failed. The server may still be running.",

  // Debug panel
  "debug.title": "Trace",
  "debug.events": "events",
  "debug.clear": "Clear",
  "debug.empty": "Waiting for SSE events...",
  "debug.emptyHint": "After sending a message, all raw backend data will be displayed here.",

  // Conversation sidebar
  "sidebar.label": "Conversation list",
  "sidebar.title": "Chats",
  "sidebar.newChat": "New chat",
  "sidebar.loading": "Loading conversations...",
  "sidebar.loadMore": "Load more",
  "sidebar.loadingMore": "Loading...",
  "sidebar.emptyTitle": "No conversations yet",
  "sidebar.emptyHint": "Click \"New chat\" to start your first conversation.",
  "sidebar.delete": "Delete conversation",
  "sidebar.deleteConfirm": "Permanently delete this conversation? This cannot be undone.",

  // Aria labels (button hover/screen-reader)
  "aria.send": "Send",
  "aria.clearHistory": "Clear history",
  "aria.stopGeneration": "Stop generation",

  // Language toggle
  "lang.switch": "中文",

  // ─── Floating bottom-right action badges ─────────────────────────────
  "floatingLink.deploy": "Deploy",
  "floatingLink.github": "GitHub",
} as const;

export default en;
