"use client";
import React from "react";
import { apiAsk, ApiException } from "../lib/api";
import styles from "./Chat.module.css";

type Message = {
  role: "user" | "assistant";
  content: string;
  citations?: { title: string; section?: string }[];
  chunks?: { text: string; sources: { title: string; section?: string }[]; source_count?: number }[];
  timestamp?: Date;
  isError?: boolean;
  errorType?: string;
  suggestions?: string[];
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
    } catch (error: any) {
      let errorMessage: Message = {
        role: "assistant",
        content: "I apologize, but I encountered an error while processing your request.",
        timestamp: new Date(),
        isError: true,
      };

      if (error instanceof ApiException) {
        // Use the user-friendly message from the API if available, otherwise fall back to a generic message
        let userFriendlyMessage = error.userMessage || "Something went wrong. Please try again.";
        let suggestions = error.suggestions || [];
        let errorType = error.errorType || 'unknown';

        // If no user message was provided by the API, create one based on error type or status
        if (!error.userMessage) {
          if (error.errorType) {
            switch (error.errorType) {
              case 'authentication_error':
                userFriendlyMessage = "There's an issue with the AI service configuration. Please contact your administrator.";
                break;
              case 'rate_limit_error':
                userFriendlyMessage = "Too many requests. Please wait a moment before trying again.";
                break;
              case 'timeout_error':
                userFriendlyMessage = "Your request took too long to process. Please try again.";
                break;
              case 'connection_error':
                userFriendlyMessage = "Unable to connect to the server. Please check your internet connection.";
                break;
              case 'validation_error':
                userFriendlyMessage = "Please check your input and try again.";
                break;
              case 'service_unavailable':
                userFriendlyMessage = "The service is temporarily unavailable. Please try again later.";
                break;
              case 'not_found':
                userFriendlyMessage = "No relevant information found for your question. Try rephrasing.";
                break;
              case 'internal_error':
                userFriendlyMessage = "Our servers are experiencing issues. Please try again later.";
                break;
              default:
                if (error.status) {
                  switch (error.status) {
                    case 400:
                      userFriendlyMessage = "There was an issue with your request. Please try rephrasing your question.";
                      break;
                    case 401:
                      userFriendlyMessage = "Authentication failed. Please refresh the page and try again.";
                      break;
                    case 403:
                      userFriendlyMessage = "You don't have permission to perform this action.";
                      break;
                    case 429:
                      userFriendlyMessage = "Too many requests. Please wait a moment before trying again.";
                      break;
                    case 500:
                      userFriendlyMessage = "Our servers are experiencing issues. Please try again later.";
                      break;
                    case 503:
                      userFriendlyMessage = "The service is temporarily unavailable. Please try again later.";
                      break;
                  }
                }
            }
          }
        }

        errorMessage = {
          role: "assistant",
          content: userFriendlyMessage,
          timestamp: new Date(),
          isError: true,
          errorType: errorType,
          suggestions: suggestions.length > 0 ? suggestions : [
            "Try rephrasing your question",
            "Check your internet connection", 
            "Refresh the page and try again",
            "Contact support if the problem persists"
          ],
        };
      } else {
        errorMessage.content = error.message || 'An unexpected error occurred. Please try again.';
        errorMessage.errorType = 'UNKNOWN';
        errorMessage.suggestions = [
          "Try refreshing the page",
          "Check your internet connection",
          "Contact support if the problem persists"
        ];
      }

      setMessages((m) => [...m, errorMessage]);
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

  const formatMessageContent = (content: string, isError: boolean = false) => {
    return content.split('\n').map((line, index) => {
      if (line.startsWith('**') && line.endsWith('**')) {
        return <strong key={index} className={isError ? styles.errorTitle : ''}>{line.slice(2, -2)}</strong>;
      }
      if (line.startsWith('• ')) {
        return <li key={index} style={{ marginLeft: '20px' }}>{line.slice(2)}</li>;
      }
      return <span key={index}>{line}<br /></span>;
    });
  };

  return (
    <div className={styles.chatContainer}>
      <div className={styles.chatHeader}>
        <div className={styles.chatTitle}>
          <span className={styles.chatIcon}>💬</span>
          AI Policy Assistant
        </div>
      </div>

      <div className={styles.messagesContainer}>
        {messages.length === 0 && (
          <div className={styles.emptyState}>
            <div className={styles.emptyStateIcon}>🤖</div>
            <div className={styles.emptyStateTitle}>Welcome to AI Policy Assistant</div>
            <div className={styles.emptyStateText}>
              Ask me anything about policies, products, or procedures. I'm here to help!
            </div>
          </div>
        )}

        {messages.map((message, index) => (
          <div key={index} className={styles.messageGroup}>
            <div className={`${styles.messageWrapper} ${message.role === "user" ? styles.user : ""}`}>
              <div className={`${styles.avatar} ${styles[message.role]}`}>
                {message.role === "user" ? "U" : "AI"}
              </div>
              <div className={styles.messageContent}>
                {message.isError ? (
                  <div className={styles.errorContainer}>
                    <div className={styles.errorHeader}>
                      <div className={styles.errorIcon}>
                        {message.errorType === 'authentication_error' ? '🔑' :
                         message.errorType === 'not_found' ? '🔍' :
                         message.errorType === 'validation_error' ? '📝' : '⚠️'}
                      </div>
                      <div className={styles.errorHeaderText}>
                        <div className={styles.errorTitle}>
                          {message.errorType === 'authentication_error' ? 'Authentication Issue' :
                           message.errorType === 'not_found' ? 'No Results Found' :
                           message.errorType === 'validation_error' ? 'Input Error' :
                           message.errorType === 'rate_limit_error' ? 'Too Many Requests' :
                           message.errorType === 'timeout_error' ? 'Request Timeout' :
                           message.errorType === 'connection_error' ? 'Connection Problem' :
                           message.errorType === 'service_unavailable' ? 'Service Unavailable' :
                           message.errorType === 'internal_error' ? 'Server Issue' : 'Something went wrong'}
                        </div>
                        <div className={styles.errorSubtitle}>We encountered an issue processing your request</div>
                      </div>
                    </div>
                    <div className={styles.errorBody}>
                      <div className={styles.errorMessage}>
                        {message.content.split('\n').find(line => !line.includes('**') && line.trim()) || 'An unexpected error occurred'}
                      </div>
                      {message.suggestions && message.suggestions.length > 0 && (
                        <div className={styles.errorSuggestions}>
                          <div className={styles.errorSuggestionsTitle}>Here's what you can try:</div>
                          <div className={styles.errorSuggestionsList}>
                            {message.suggestions.map((suggestion, suggestionIdx) => (
                              <div key={suggestionIdx} className={styles.errorSuggestionItem}>
                                <span className={styles.errorSuggestionBullet}>•</span>
                                <span>{suggestion}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className={`${styles.messageBubble} ${styles[message.role]}`}>
                    {formatMessageContent(message.content, message.isError)}
                  </div>
                )}
                {message.timestamp && (
                  <div className={`${styles.messageTime} ${styles[message.role]}`}>
                    {formatTime(message.timestamp)}
                  </div>
                )}
              </div>
            </div>

            {message.citations && message.citations.length > 0 && (
              <div className={styles.citations}>
                {/* Deduplicate citations by title to avoid repetitive sources */}
                {Array.from(new Map(message.citations.map(citation => [citation.title, citation])).values())
                  .map((citation, citIndex) => (
                  <span key={citIndex} className={styles.citationBadge} title={citation.section || ""}>
                    {citation.title}
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
                  {expandedChunks.has(index) ? "Hide" : "Show"} relevant excerpts ({message.chunks.length})
                </button>
                {expandedChunks.has(index) && (
                  <div className={styles.chunksContent}>
                    {message.chunks.map((item: any, chunkIndex: number) => (
                      <div key={chunkIndex} className={styles.chunk}>
                        <div className={styles.chunkHeader}>
                          <strong>Relevant excerpt</strong>
                          <span className={styles.citationBadge}>Sources ({item.source_count || (item.sources?.length || 0)})</span>
                        </div>
                        <div className={styles.chunkText}>{item.text}</div>
                        <div className={styles.chunksContent}>
                          {item.sources?.map((src: any, i: number) => (
                            <span key={i} className={styles.citationBadge} title={src.section || ""}>
                              {src.title}{src.section ? ` — ${src.section}` : ""}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
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
