"use client";
import React from "react";
import { apiIngest, apiMetrics } from "../lib/api";

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
      setError(err instanceof Error ? err.message : "Failed to fetch metrics");
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
      setError(err instanceof Error ? err.message : "Failed to ingest documents");
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
