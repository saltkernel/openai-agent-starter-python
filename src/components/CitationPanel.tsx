import { useMemo } from 'react';
import type { Message } from '../types';
import styles from './CitationPanel.module.css';

/**
 * 引用与出处面板 —— 从最近一条 AI 回答的「## 参考文献」表格里实时解析。
 *
 * 数据来源是后端铁律强制的文献表（agents/chat/index.py INSTRUCTIONS 第 3 条），
 * 格式固定：`[n] 标题. 作者 et al. (年份). venue. DOI: xx 或 https://arxiv.org/abs/xx`
 * 所以前端解析即可拿到结构化数据，不需要新增后端事件。
 */

export interface ParsedRef {
  no: number;
  title: string;
  meta: string;   // 作者 / 年份 / venue 等剩余信息
  url?: string;   // DOI 链接或 arXiv 链接（能识别出来的话）
}

const REF_LINE = /^\s*\[(\d+)\]\s*(.+?)\s*$/;

export function parseReferences(content: string): ParsedRef[] {
  if (!content) return [];
  const idx = content.lastIndexOf('## 参考文献');
  if (idx === -1) return [];

  const refs: ParsedRef[] = [];
  for (const line of content.slice(idx).split('\n').slice(1)) {
    const m = line.match(REF_LINE);
    if (!m) continue;
    const no = Number(m[1]);
    const body = m[2].trim();

    // 找链接：arXiv 链接或 DOI 链接（markdown 链接或裸文本都认）
    let url: string | undefined;
    const linkMatch = body.match(/https?:\/\/[^\s)]+/);
    if (linkMatch) {
      url = linkMatch[0];
    } else {
      const doiMatch = body.match(/DOI:\s*(10\.\S+?)[.,;]?$/i);
      if (doiMatch) url = `https://doi.org/${doiMatch[1]}`;
    }

    // 标题 = 第一个句号前的部分（与文献表约定格式一致）；剩余作为元信息
    const dot = body.indexOf('. ');
    const title = dot > 0 ? body.slice(0, dot) : body;
    const meta = dot > 0 ? body.slice(dot + 2) : '';

    refs.push({ no, title, meta: meta.replace(/https?:\/\/\S+/, '').trim(), url });
  }
  return refs.sort((a, b) => a.no - b.no);
}

interface Props {
  messages: Message[];
  onShowDebug: () => void;
  debugActive: boolean;
}

export default function CitationPanel({ messages, onShowDebug, debugActive }: Props) {
  const refs = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i];
      if (msg.role !== 'assistant') continue;
      const parsed = parseReferences(msg.content);
      if (parsed.length) return parsed;
    }
    return [];
  }, [messages]);

  return (
    <div className={styles.panel}>
      <div className={styles.head}>
        <span className={styles.title}>引用与出处</span>
        <div className={styles.headRight}>
          {refs.length > 0 && <span className={styles.count}>{refs.length}</span>}
          <button
            className={`${styles.debugBtn} ${debugActive ? styles.debugBtnActive : ''}`}
            onClick={onShowDebug}
            title="查看传输流"
          >
            传输流
          </button>
        </div>
      </div>

      {refs.length > 0 && (
        <p className={styles.summary}>本条回答引用了 {refs.length} 篇文献，点击卡片可跳转原文。</p>
      )}

      <div className={styles.list}>
        {refs.map(r => (
          <a
            key={r.no}
            className={styles.card}
            href={r.url}
            target="_blank"
            rel="noreferrer"
            style={r.url ? undefined : { pointerEvents: 'none' }}
          >
            <span className={styles.badge}>{r.no}</span>
            <span className={styles.cardBody}>
              <span className={styles.refTitle}>{r.title}</span>
              {r.meta && <span className={styles.refMeta}>{r.meta}</span>}
              {r.url && (
                <span className={styles.refUrl}>
                  {r.url.includes('arxiv.org') ? 'arXiv' : r.url.includes('doi.org') ? 'DOI' : '链接'}
                </span>
              )}
            </span>
          </a>
        ))}

        {refs.length === 0 && (
          <div className={styles.empty}>
            <p className={styles.emptyTitle}>暂无引用</p>
            <p className={styles.emptyHint}>
              当回答包含「## 参考文献」时，引用的文献会以卡片形式出现在这里，点击可跳转原文。
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
