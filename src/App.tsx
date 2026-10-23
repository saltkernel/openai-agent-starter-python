import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import type {
  Message,
  ToolLampState,
  ConversationSummary,
} from './types';
import {
  fetchConversationHistory,
  sendMessageStream,
  stopAgent,
  listConversations,
  deleteConversation,
} from './api';
import type { RawSseEvent } from './api';
import ToolIndicators from './components/ToolIndicators';
import ChatWindow from './components/ChatWindow';
import ChatInput from './components/ChatInput';
import DebugPanel from './components/DebugPanel';
import CitationPanel from './components/CitationPanel';
import ConversationSidebar from './components/ConversationSidebar';
import GitHubLink from './components/GitHubLink';
import DeployLink from './components/DeployLink';
import { I18nProvider, LangToggle, useT, MessageKeys } from './i18n';
import { deleteSnapshot, loadSnapshot, saveSnapshot } from './lib/chatUiStore';
import styles from './App.module.css';

/**
 * 工具指示灯映射。
 *
 * 键必须与 agents/tools.py 里 PAPER_TOOLS 的函数名（= 后端 tool_called 事件里的
 * tool 字段）逐一对应，否则工具真被调用时指示灯不会亮（LAMP_I18N_KEYS 查不到 key）。
 *
 * 七个工具都要在 LAMP_I18N_KEYS / LAMP_ICONS 里登记；LAMP_IDS 只决定顶部「显示」
 * 哪几个——头顶空间有限，只放最核心的四个，其余三个仍然能正常点亮/无灯但文案齐全。
 */
const ALL_TOOL_IDS = [
  'search_papers',
  'get_paper_detail',
  'get_citations',
  'search_arxiv_latest',
  'ingest_paper',
  'search_passages',
  'list_ingested_papers',
] as const;

/** 顶部实际渲染的四个（其余三个不占位，但仍登记文案以便将来放开） */
const LAMP_IDS = ['search_papers', 'get_paper_detail', 'ingest_paper', 'search_passages'] as const;

const LAMP_ICONS: Record<string, string> = {
  search_papers: '🔍',
  get_paper_detail: '📄',
  get_citations: '🔗',
  search_arxiv_latest: '📰',
  ingest_paper: '📥',
  search_passages: '🧩',
  list_ingested_papers: '📚',
};

const LAMP_I18N_KEYS: Record<string, string> = {
  search_papers: 'tool.searchPapers',
  get_paper_detail: 'tool.paperDetail',
  get_citations: 'tool.citations',
  search_arxiv_latest: 'tool.arxivLatest',
  ingest_paper: 'tool.ingestPaper',
  search_passages: 'tool.searchPassages',
  list_ingested_papers: 'tool.listIngested',
};

// 编译期一致性护栏：登记表必须覆盖全部工具，少一个就类型报错，避免静默漏灯
type _ToolIdCoverage =
  | Exclude<(typeof ALL_TOOL_IDS)[number], keyof typeof LAMP_ICONS>
  | Exclude<(typeof ALL_TOOL_IDS)[number], keyof typeof LAMP_I18N_KEYS>;
const _assertToolCoverage: [_ToolIdCoverage] extends [never] ? true : never = true;
void _assertToolCoverage;

const CONVERSATION_ID_STORAGE_KEY = 'eo_conversation_id';

/**
 * eo-uuid: stable per-browser identifier. Generated client-side on first
 * visit and persisted in localStorage. Used as the `userId` argument to all
 * memory-store calls so that listConversations / clear / delete can scope
 * results to "this browser's conversations only". Cross-template aware:
 * intentionally shares the same localStorage key with claude-agent-starter
 * so the same browser sees the same identity in either template.
 */
const EO_USER_ID_STORAGE_KEY = 'eo-uuid';
const CONVERSATIONS_PAGE_SIZE = 20;

/** Returns existing conversation ID from localStorage, or null if first visit */
function getExistingConversationId(): string | null {
  return localStorage.getItem(CONVERSATION_ID_STORAGE_KEY);
}

/** Returns existing or creates a new conversation ID */
function getOrCreateConversationId(): string {
  const cached = getExistingConversationId();
  if (cached) return cached;

  const conversationId = crypto.randomUUID();
  localStorage.setItem(CONVERSATION_ID_STORAGE_KEY, conversationId);
  return conversationId;
}

function getOrCreateEoUuid(): string {
  const cached = localStorage.getItem(EO_USER_ID_STORAGE_KEY);
  if (cached) return cached;
  const eoUuid = crypto.randomUUID();
  localStorage.setItem(EO_USER_ID_STORAGE_KEY, eoUuid);
  return eoUuid;
}

// Module-level dedup flag — outside React lifecycle, unaffected by StrictMode
let _historyFetchInFlight = false;

export default function App() {
  return (
    <I18nProvider>
      <LangToggle />
      <AppInner />
    </I18nProvider>
  );
}

function AppInner() {
  const { t } = useT();
  const buildLamps = useCallback((): ToolLampState[] =>
    LAMP_IDS.map(id => ({
      id,
      label: t(LAMP_I18N_KEYS[id] as MessageKeys),
      icon: LAMP_ICONS[id],
      active: false,
      animKey: 0,
    })),
  [t]);

  const [messages, setMessages] = useState<Message[]>([]);
  const [lamps, setLamps]       = useState<ToolLampState[]>(buildLamps);
  const [loading, setLoading]   = useState(false);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [debugEvents, setDebugEvents] = useState<RawSseEvent[]>([]);
  const [rightPanelMode, setRightPanelMode] = useState<'citations' | 'debug'>('citations');

  // Conversation list (sidebar) state
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [conversationsLoading, setConversationsLoading] = useState(true);
  const [conversationsLoadingMore, setConversationsLoadingMore] = useState(false);
  const [nextCursor, setNextCursor] = useState<string | undefined>(undefined);
  const [activeConversationId, setActiveConversationId] = useState<string>(() => getOrCreateConversationId());

  const botMsgIdRef = useRef<string>('');
  const abortCtrlRef = useRef<AbortController | null>(null);
  const hadExistingConversationIdRef = useRef(getExistingConversationId() !== null);
  const conversationIdRef = useRef<string>(activeConversationId);
  const eoUuidRef = useRef<string>(getOrCreateEoUuid());
  const initDoneRef = useRef(false);
  const snapshotTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Keep conversationIdRef in sync with the activeConversationId state.
  useEffect(() => {
    conversationIdRef.current = activeConversationId;
  }, [activeConversationId]);

  // Update lamp labels when language changes
  useEffect(() => {
    setLamps(prev =>
      prev.map(l => ({
        ...l,
        label: t(LAMP_I18N_KEYS[l.id] as MessageKeys),
      }))
    );
  }, [t]);

  // Persist a UI snapshot of the current conversation's messages to IndexedDB
  // (debounced) so a refresh restores instantly without hitting /history.
  useEffect(() => {
    if (messages.length === 0) return;
    if (!initDoneRef.current) return;

    if (snapshotTimerRef.current) clearTimeout(snapshotTimerRef.current);
    snapshotTimerRef.current = setTimeout(() => {
      saveSnapshot(conversationIdRef.current, messages).catch(err => {
        console.warn('[chatUiStore] snapshot save failed:', err);
      });
    }, 500);

    return () => {
      if (snapshotTimerRef.current) clearTimeout(snapshotTimerRef.current);
    };
  }, [messages]);

  /** Load full message list for a conversation: snapshot first, then /history reconciliation. */
  const loadConversation = useCallback(async (convId: string) => {
    setHistoryLoading(true);
    let restoredFromSnapshot = false;
    let snapshotMessageCount = 0;

    try {
      const snapshot = await loadSnapshot(convId).catch(() => [] as Message[]);
      snapshotMessageCount = snapshot.length;
      if (snapshot.length > 0) {
        restoredFromSnapshot = true;
        setMessages(snapshot);
        setHistoryLoading(false);
      }

      const history = await fetchConversationHistory(convId, eoUuidRef.current);
      if (history.length > 0) {
        if (!restoredFromSnapshot || history.length > snapshotMessageCount) {
          setMessages(history);
        }
        saveSnapshot(convId, history).catch(() => {});
      } else if (!restoredFromSnapshot) {
        setMessages([]);
      }
    } finally {
      setHistoryLoading(false);
      initDoneRef.current = true;
    }
  }, []);

  /** Refresh sidebar conversations list. mode='replace' → reload from start; 'append' → add next page. */
  const refreshConversations = useCallback(async (mode: 'replace' | 'append', cursor?: string) => {
    if (mode === 'append') {
      setConversationsLoadingMore(true);
    } else {
      setConversationsLoading(true);
    }
    try {
      const res = await listConversations({
        userId: eoUuidRef.current,
        limit: CONVERSATIONS_PAGE_SIZE,
        order: 'desc',
        after: cursor,
      });

      setNextCursor(res.nextCursor);

      if (mode === 'append') {
        setConversations(prev => {
          const seen = new Set(prev.map(c => c.id));
          const merged = [...prev];
          for (const c of res.conversations) {
            if (!seen.has(c.id)) merged.push(c);
          }
          return merged;
        });
      } else {
        setConversations(res.conversations);
      }
    } finally {
      if (mode === 'append') {
        setConversationsLoadingMore(false);
      } else {
        setConversationsLoading(false);
      }
    }
  }, []);

  // Initial load: history (only if previously visited) + conversations list
  useEffect(() => {
    if (_historyFetchInFlight) {
      void refreshConversations('replace');
      return;
    }
    _historyFetchInFlight = true;

    if (!hadExistingConversationIdRef.current) {
      // First visit: skip /history fetch (would be empty). Still kick off the
      // conversations list so the sidebar shows the user's other browsers'
      // conversations if any (rare but possible).
      setHistoryLoading(false);
      initDoneRef.current = true;
      void refreshConversations('replace').finally(() => {
        _historyFetchInFlight = false;
      });
      return;
    }

    void loadConversation(conversationIdRef.current).finally(() => {
      void refreshConversations('replace').finally(() => {
        _historyFetchInFlight = false;
      });
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** Update the current bot message's content via an updater function. */
  const updateBotMessage = useCallback((updater: (content: string) => string) => {
    setMessages(prev =>
      prev.map(m =>
        m.id === botMsgIdRef.current
          ? { ...m, content: updater(m.content) }
          : m
      )
    );
  }, []);

  /** Clear the assistant message's `streaming` flag (hides the blinking caret). */
  const clearBotStreaming = useCallback(() => {
    setMessages(prev => {
      let changed = false;
      const next = prev.map(m => {
        if (m.id === botMsgIdRef.current && m.streaming) {
          changed = true;
          const { streaming, ...rest } = m;
          return rest;
        }
        return m;
      });
      return changed ? next : prev;
    });
  }, []);

  const finishStream = useCallback(() => {
    setLoading(false);
    abortCtrlRef.current = null;
  }, []);

  const handleSend = useCallback(async (text: string) => {
    initDoneRef.current = true;
    setRightPanelMode('debug');

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      timestamp: Date.now(),
    };

    const botMsgId = crypto.randomUUID();
    botMsgIdRef.current = botMsgId;
    const botMsg: Message = {
      id: botMsgId,
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
      streaming: true,
    };

    setMessages(prev => [...prev, userMsg, botMsg]);
    setLoading(true);

    /**
     * Optimistic sidebar update — fires as soon as the backend emits its first
     * SSE event (matches ChatGPT's "new chat appears the moment streaming
     * starts" UX). For brand-new conversations we prepend a synthesized summary;
     * for existing ones we just bump them to the top.
     *
     * Server reconciliation still happens in onDone() via refreshConversations,
     * which can correct the title if the runtime later overrides it.
     */
    let sidebarPrimed = false;
    const cleanedText = text.replace(/\s+/g, ' ').trim();
    const optimisticTitle =
      cleanedText.length === 0 ? 'New chat'
        : cleanedText.length <= 8 ? cleanedText
          : `${cleanedText.slice(0, 8)}...`;

    const primeSidebar = () => {
      if (sidebarPrimed) return;
      sidebarPrimed = true;

      const convId = conversationIdRef.current;
      const now = Date.now();

      setConversations(prev => {
        const idx = prev.findIndex(c => c.id === convId);
        if (idx === -1) {
          const summary: ConversationSummary = {
            id: convId,
            title: optimisticTitle,
            lastMessageAt: now,
            userId: eoUuidRef.current,
          };
          return [summary, ...prev];
        }
        const next = [...prev];
        const [moved] = next.splice(idx, 1);
        next.unshift({ ...moved, lastMessageAt: now });
        return next;
      });
    };

    const ctrl = sendMessageStream(text, {
      onTextDelta(delta) {
        updateBotMessage(content => content + delta);
      },

      onToolCalled(toolName) {
        setLamps(prev =>
          prev.map(l =>
            l.id === toolName
              ? { ...l, active: true, animKey: l.animKey + 1 }
              : l
          )
        );
        setTimeout(() => {
          setLamps(prev =>
            prev.map(l => (l.id === toolName ? { ...l, active: false } : l))
          );
        }, 1000);
      },

      onRawEvent(event) {
        // Every backend SSE frame flows through here, so this is the cheapest
        // hook for "first byte from backend".
        primeSidebar();

        // Coalesce consecutive text_delta events into a single growing entry,
        // so a multi-paragraph response doesn't flood the debug panel with
        // hundreds of one-token rows.
        if (event.eventType === 'text_delta') {
          const delta = (event.data as { delta?: string } | null)?.delta ?? '';
          setRightPanelMode('debug');
          setDebugEvents(prev => {
            const last = prev[prev.length - 1];
            if (last && last.eventType === 'text_delta') {
              const prevDelta = (last.data as { delta?: string } | null)?.delta ?? '';
              const merged: RawSseEvent = {
                ...last,
                data: { delta: prevDelta + delta },
                raw: last.raw + delta,
                timestamp: event.timestamp,
              };
              return [...prev.slice(0, -1), merged];
            }
            return [...prev, event];
          });
          return;
        }
        setRightPanelMode('debug');
        setDebugEvents(prev => [...prev, event]);
      },

      onDone() {
        clearBotStreaming();
        finishStream();
        // Reconcile with backend so the title (and any other fields the runtime
        // synthesized) reflect the server's authoritative state.
        void refreshConversations('replace');
      },

      onError(err?: Error) {
        clearBotStreaming();
        // 后端会把「上游到底怎么失败的」放在 message 里（鉴权失败时还会带 base_url /
        // model / key 来源）。以前这里直接丢掉，只显示「请检查后端服务是否启动」，
        // 结果 401 Invalid API key 和「函数没起来」长得一模一样，线上排障只能靠猜。
        const detail = (err?.message || '').replace(/\s+/g, ' ').trim();
        const shown = detail
          ? `⚠️ ${detail.length > 400 ? `${detail.slice(0, 400)}…` : detail}`
          : t("status.error");
        updateBotMessage(content => content || shown);
        finishStream();
      },
    }, conversationIdRef.current, {
      userId: eoUuidRef.current,
      userMsgId: userMsg.id,
      botMsgId,
    });

    abortCtrlRef.current = ctrl;
  }, [updateBotMessage, clearBotStreaming, finishStream, refreshConversations, t]);

  const handleClearHistory = useCallback(() => {
    const oldConvId = conversationIdRef.current;

    if (abortCtrlRef.current) {
      abortCtrlRef.current.abort();
      abortCtrlRef.current = null;
    }

    // The trash button in ChatInput is the same affordance as the trash
    // icon on a sidebar item: it should DELETE the conversation entirely,
    // not just clear its messages. Using `clearConversationHistory` here
    // would leave the old conversation in the sidebar with an empty body
    // and a fallback "New chat" title — confusing for users who clicked
    // trash expecting "make this thread go away".
    //
    // Optimistically drop from the sidebar so the user sees the row
    // disappear immediately; the network call is fire-and-forget.
    setConversations(prev => prev.filter(c => c.id !== oldConvId));

    deleteConversation(oldConvId, eoUuidRef.current).then(ok => {
      if (!ok) {
        console.warn('[delete-conversation] backend request failed');
      }
    }).finally(() => {
      // Reconcile with backend in case state diverged.
      void refreshConversations('replace');
    });

    deleteSnapshot(oldConvId).catch(() => {});

    const newId = crypto.randomUUID();
    localStorage.setItem(CONVERSATION_ID_STORAGE_KEY, newId);
    conversationIdRef.current = newId;
    setActiveConversationId(newId);
    setMessages([]);
    setDebugEvents([]);
    setRightPanelMode('citations');
    setLoading(false);
    initDoneRef.current = false;
  }, [refreshConversations]);

  const handleStop = useCallback(() => {
    // 1. Immediately abort frontend SSE read
    if (abortCtrlRef.current) {
      abortCtrlRef.current.abort();
      abortCtrlRef.current = null;
    }

    // 2. Optimistic UI: show stopped immediately without waiting for backend
    updateBotMessage(content => content ? content + '\n\n' + t("status.stopped") : t("status.stopped"));
    setLoading(false);

    // 3. Backend abort async — notify user on failure
    stopAgent(conversationIdRef.current).then(ok => {
      if (!ok) {
        updateBotMessage(content => content + '\n\n' + t("status.backendError"));
      }
    });
  }, [updateBotMessage, t]);

  /** User clicked a conversation in the sidebar. */
  const handleSelectConversation = useCallback((id: string) => {
    if (loading) return;
    if (id === conversationIdRef.current) return;

    localStorage.setItem(CONVERSATION_ID_STORAGE_KEY, id);
    conversationIdRef.current = id;
    setActiveConversationId(id);
    setRightPanelMode('citations');
    void loadConversation(id);
  }, [loading, loadConversation]);

  /** User clicked "New chat" in the sidebar. */
  const handleCreateConversation = useCallback(() => {
    if (loading) return;

    const newId = crypto.randomUUID();
    localStorage.setItem(CONVERSATION_ID_STORAGE_KEY, newId);
    conversationIdRef.current = newId;
    setActiveConversationId(newId);
    setMessages([]);
    setDebugEvents([]);
    setRightPanelMode('citations');
    initDoneRef.current = false;
    setHistoryLoading(false);
  }, [loading]);

  const handleLoadMoreConversations = useCallback(() => {
    if (!nextCursor || conversationsLoadingMore) return;
    void refreshConversations('append', nextCursor);
  }, [nextCursor, conversationsLoadingMore, refreshConversations]);

  /**
   * User clicked the trash icon on a sidebar item.
   *
   * Optimistic delete: immediately remove the item from local UI state and
   * fire-and-forget the backend request. We don't await or block the user —
   * if the network call fails, we log it but don't roll back, since reloading
   * the page will reconcile via /conversations anyway.
   */
  const handleDeleteConversation = useCallback((id: string) => {
    if (loading) return;
    if (!id) return;

    const confirmed = window.confirm(t('sidebar.deleteConfirm'));
    if (!confirmed) return;

    const isActive = id === conversationIdRef.current;

    setConversations(prev => prev.filter(c => c.id !== id));

    if (isActive) {
      const newId = crypto.randomUUID();
      localStorage.setItem(CONVERSATION_ID_STORAGE_KEY, newId);
      conversationIdRef.current = newId;
      setActiveConversationId(newId);
      setMessages([]);
      setDebugEvents([]);
      setRightPanelMode('citations');
      initDoneRef.current = false;
      setHistoryLoading(false);
    }

    void deleteSnapshot(id).catch(() => {});

    void deleteConversation(id, eoUuidRef.current).catch(e => {
      console.warn('[delete-conversation] backend request failed:', e);
    });
  }, [loading, t]);

  const sidebarHasMore = useMemo(() => Boolean(nextCursor), [nextCursor]);

  return (
    <div className={styles.shell}>
      <div className={styles.stage}>
        <ConversationSidebar
          conversations={conversations}
          activeConversationId={activeConversationId}
          loading={conversationsLoading}
          loadingMore={conversationsLoadingMore}
          hasMore={sidebarHasMore}
          disabled={loading}
          onSelect={handleSelectConversation}
          onCreate={handleCreateConversation}
          onLoadMore={handleLoadMoreConversations}
          onDelete={handleDeleteConversation}
        />

        <div className={styles.chatPanel}>
          <header className={styles.header}>
            <div className={styles.headerLeft}>
              <span className={styles.logo}>⬡</span>
              <div>
                <p className={styles.title}>{t("app.title")}</p>
                <p className={styles.subtitle}>{t("app.subtitle")}</p>
              </div>
            </div>
            <ToolIndicators lamps={lamps} />
          </header>

          <div className={styles.chatWindowShell}>
            <ChatWindow messages={messages} loading={loading} />
            {historyLoading && messages.length === 0 && (
              <div className={styles.historyOverlay}>
                <div className={styles.historySpinner} />
              </div>
            )}
          </div>
          <ChatInput onSend={handleSend} onStop={handleStop} onClear={handleClearHistory} disabled={loading} />
        </div>

        <div className={styles.codePanel}>
          {rightPanelMode === 'citations' ? (
            <CitationPanel
              messages={messages}
              onShowDebug={() => setRightPanelMode('debug')}
              debugActive={false}
            />
          ) : (
            <DebugPanel events={debugEvents} onClear={() => setDebugEvents([])} />
          )}
        </div>
      </div>
      <GitHubLink />
      <DeployLink />
    </div>
  );
}
