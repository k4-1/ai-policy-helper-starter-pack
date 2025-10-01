"use client";
import React from "react";
import { apiAsk } from "../lib/api";
import styles from "./Chat.module.css";

type Message = {
  role: "user" | "assistant";
  content: string;
  citations?: { title: string; section?: string }[];
  chunks?: { title: string; section?: string; text: string }[];
  timestamp?: Date;
};

export default function Chat() {
  const [messages, setMessages] = React.useState<Message[]>([]);
  const [q, setQ] = React.useState("");
  const [loading, setLoading] = React.useState(false);
  const [expandedChunks, setExpandedChunks] = React.useState<Set<number>>(new Set());
  const messagesEndRef = React.useRef<HTMLDivElement>(null);
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  React.useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const adjustTextareaHeight = () => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";
    }
  };

  React.useEffect(() => {
    adjustTextareaHeight();
  }, [q]);

  const send = async () => {
    if (!q.trim() || loading) return;
    
    const userMessage: Message = { 
      role: "user" as const, 
      content: q.trim(),
      timestamp: new Date()
    };
    
    setMessages((m) => [...m, userMessage]);
    setLoading(true);
    setQ("");
    
    try {
      const res = await apiAsk(userMessage.content);
      const assistantMessage: Message = {
        role: "assistant",
        content: res.answer,
        citations: res.citations,
        chunks: res.chunks,
        timestamp: new Date()
      };
      setMessages((m) => [...m, assistantMessage]);
    } catch (e: any) {
      setMessages((m) => [
        ...m,
        { 
          role: "assistant", 
          content: "I apologize, but I encountered an error while processing your request. Please try again.",
          timestamp: new Date()
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  const toggleChunks = (messageIndex: number) => {
    setExpandedChunks(prev => {
      const newSet = new Set(prev);
      if (newSet.has(messageIndex)) {
        newSet.delete(messageIndex);
      } else {
        newSet.add(messageIndex);
      }
      return newSet;
    });
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className={styles.chatContainer}>
      <div className={styles.chatHeader}>
        <h2 className={styles.chatTitle}>
          <div className={styles.chatIcon}>🤖</div>
          AI Policy Assistant
        </h2>
      </div>

      <div className={styles.messagesContainer}>
        {messages.length === 0 && (
          <div className={styles.emptyState}>
            <div className={styles.emptyStateIcon}>💬</div>
            <div className={styles.emptyStateTitle}>Start a conversation</div>
            <div className={styles.emptyStateText}>
              Ask me anything about policies, products, or procedures. I'm here to help!
            </div>
          </div>
        )}

        {messages.map((message, index) => (
          <div key={index} className={styles.messageGroup}>
            <div className={`${styles.messageWrapper} ${message.role === "user" ? styles.user : ""}`}>
              <div className={`${styles.avatar} ${styles[message.role]}`}>
                {message.role === "user" ? "You" : "AI"}
              </div>
              
              <div className={styles.messageContent}>
                <div className={`${styles.messageBubble} ${styles[message.role]}`}>
                  {message.content}
                </div>
                
                {message.timestamp && (
                  <div className={`${styles.messageTime} ${message.role === "user" ? styles.user : ""}`}>
                    {formatTime(message.timestamp)}
                  </div>
                )}

                {message.citations && message.citations.length > 0 && (
                  <div className={styles.citations}>
                    {message.citations.map((citation, idx) => (
                      <span 
                        key={idx} 
                        className={styles.citationBadge} 
                        title={citation.section || ""}
                      >
                        📄 {citation.title}
                      </span>
                    ))}
                  </div>
                )}

                {message.chunks && message.chunks.length > 0 && (
                  <div className={styles.chunks}>
                    <button 
                      className={styles.chunksToggle}
                      onClick={() => toggleChunks(index)}
                    >
                      {expandedChunks.has(index) ? "▼" : "▶"} 
                      View supporting information ({message.chunks.length})
                    </button>
                    
                    {expandedChunks.has(index) && (
                      <div className={styles.chunksContent}>
                        {message.chunks.map((chunk, idx) => (
                          <div key={idx} className={styles.chunkItem}>
                            <div className={styles.chunkTitle}>
                              {chunk.title}
                              {chunk.section && ` — ${chunk.section}`}
                            </div>
                            <div className={styles.chunkText}>{chunk.text}</div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}

        {loading && (
          <div className={styles.messageGroup}>
            <div className={styles.messageWrapper}>
              <div className={`${styles.avatar} ${styles.assistant}`}>AI</div>
              <div className={styles.typingIndicator}>
                <span className={styles.typingText}>AI is thinking</span>
                <div className={styles.loadingDots}>
                  <div className={styles.loadingDot}></div>
                  <div className={styles.loadingDot}></div>
                  <div className={styles.loadingDot}></div>
                </div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className={styles.inputContainer}>
        <div className={styles.inputWrapper}>
          <textarea
            ref={textareaRef}
            className={styles.textInput}
            placeholder="Ask about policies, products, or procedures..."
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
            rows={1}
          />
          <button
            className={styles.sendButton}
            onClick={send}
            disabled={loading || !q.trim()}
            title="Send message"
          >
            {loading ? (
              <div className={styles.loadingDots}>
                <div className={styles.loadingDot}></div>
                <div className={styles.loadingDot}></div>
                <div className={styles.loadingDot}></div>
              </div>
            ) : (
              "➤"
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
