"use client";
import React from "react";
import { apiIngest, apiMetrics, ApiException } from "../lib/api";

export default function AdminPanel() {
  const [metrics, setMetrics] = React.useState<any>(null);
  const [busy, setBusy] = React.useState(false);
  const [refreshing, setRefreshing] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const refresh = async () => {
    setRefreshing(true);
    setError(null);
    try {
      console.log("Fetching metrics...");
      const m = await apiMetrics();
      console.log("Metrics received:", m);
      setMetrics(m);
    } catch (err) {
      console.error("Error fetching metrics:", err);
      
      if (err instanceof ApiException) {
        let errorTitle = "Something went wrong";
        let errorMessage = "An unexpected error occurred. Please try again or contact support.";
        
        // Use the user-friendly message from the backend if available
        if (err.userMessage) {
          errorMessage = err.userMessage;
        }
        
        // Set title based on error type or status
        if (err.errorType) {
          switch (err.errorType) {
            case 'authentication_error':
              errorTitle = "Authentication Issue";
              break;
            case 'rate_limit_error':
              errorTitle = "Too Many Requests";
              break;
            case 'timeout_error':
              errorTitle = "Request Timeout";
              break;
            case 'connection_error':
              errorTitle = "Connection Problem";
              break;
            case 'service_unavailable':
              errorTitle = "Service Unavailable";
              break;
            case 'validation_error':
              errorTitle = "Invalid Input";
              break;
            case 'not_found':
              errorTitle = "No Results Found";
              break;
            default:
              errorTitle = "Server Issue";
          }
        } else {
          // Fallback to status-based handling if no error type
          switch (err.status) {
            case 0:
              errorTitle = "Connection Problem";
              break;
            case 408:
              errorTitle = "Request Timeout";
              break;
            case 500:
              errorTitle = "Server Issue";
              break;
          }
        }
        
        // Add suggestions if available
        let fullMessage = `${errorTitle}: ${errorMessage}`;
        if (err.suggestions && err.suggestions.length > 0) {
          fullMessage += "\n\nSuggestions:\n" + err.suggestions.map(s => `• ${s}`).join('\n');
        }
        
        setError(fullMessage);
      } else {
        console.error('Unexpected error in refresh:', err);
        setError('Oops! Something went wrong while refreshing. Please try again.');
      }
    } finally {
      setRefreshing(false);
    }
  };

  const ingest = async () => {
    setBusy(true);
    setError(null);
    try {
      console.log("Starting ingestion...");
      await apiIngest();
      console.log("Ingestion completed, refreshing metrics...");
      await refresh();
    } catch (err) {
      console.error("Error during ingestion:", err);
      
      if (err instanceof ApiException) {
        let errorTitle = "Ingestion Failed";
        let errorMessage = "An unexpected error occurred during ingestion. Please try again or contact support.";
        
        // Use the user-friendly message from the backend if available
        if (err.userMessage) {
          errorMessage = err.userMessage;
        }
        
        // Set title based on error type or status
        if (err.errorType) {
          switch (err.errorType) {
            case 'authentication_error':
              errorTitle = "Authentication Issue";
              break;
            case 'rate_limit_error':
              errorTitle = "Too Many Requests";
              break;
            case 'timeout_error':
              errorTitle = "Ingestion Timeout";
              break;
            case 'connection_error':
              errorTitle = "Connection Problem";
              break;
            case 'service_unavailable':
              errorTitle = "Service Unavailable";
              break;
            case 'validation_error':
              errorTitle = "Invalid Documents";
              break;
            case 'not_found':
              errorTitle = "No Documents Found";
              break;
            default:
              errorTitle = "Server Issue";
          }
        } else {
          // Fallback to status-based handling if no error type
          switch (err.status) {
            case 0:
              errorTitle = "Connection Problem";
              break;
            case 408:
              errorTitle = "Ingestion Timeout";
              break;
            case 500:
              errorTitle = "Server Issue";
              break;
          }
        }
        
        // Add suggestions if available
        let fullMessage = `${errorTitle}: ${errorMessage}`;
        if (err.suggestions && err.suggestions.length > 0) {
          fullMessage += "\n\nSuggestions:\n" + err.suggestions.map(s => `• ${s}`).join('\n');
        }
        
        setError(fullMessage);
      } else {
        console.error('Unexpected error in ingest:', err);
        setError('Oops! Something went wrong during ingestion. Please try again.');
      }
    } finally {
      setBusy(false);
    }
  };

  React.useEffect(() => {
    refresh();
  }, []);

  return (
    <div className="card">
      <h2>Admin</h2>
      
      {error && (
        <div style={{
          background: "#fee2e2",
          border: "1px solid #fecaca",
          color: "#dc2626",
          padding: "12px",
          borderRadius: "8px",
          marginBottom: "16px",
          fontSize: "14px"
        }}>
          <strong>Error:</strong> {error}
        </div>
      )}
      
      <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
        <button
          onClick={ingest}
          disabled={busy || refreshing}
          style={{
            padding: "8px 12px",
            borderRadius: 8,
            border: "1px solid #111",
            background: busy ? "#f3f4f6" : "#fff",
            cursor: busy || refreshing ? "not-allowed" : "pointer",
            opacity: busy || refreshing ? 0.6 : 1,
          }}
        >
          {busy ? "Indexing..." : "Ingest sample docs"}
        </button>
        <button
          onClick={refresh}
          disabled={busy || refreshing}
          style={{
            padding: "8px 12px",
            borderRadius: 8,
            border: "1px solid #111",
            background: refreshing ? "#f3f4f6" : "#fff",
            cursor: busy || refreshing ? "not-allowed" : "pointer",
            opacity: busy || refreshing ? 0.6 : 1,
          }}
        >
          {refreshing ? "Refreshing..." : "Refresh metrics"}
        </button>
      </div>
      {metrics && (
        <div className="code">
          <pre>{JSON.stringify(metrics, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
