import AdminPanel from "../components/AdminPanel";
import Chat from "../components/Chat";
import Tabs from "../components/Tabs";

export default function Page() {
  const tabs = [
    {
      id: "admin",
      label: "📊 Admin Panel",
      content: <AdminPanel />
    },
    {
      id: "chat",
      label: "💬 AI Chat",
      content: <Chat />
    }
  ];

  return (
    <div style={{ padding: "0 24px" }}>
      <div style={{ textAlign: "center", marginBottom: "32px" }}>
        <h1 style={{ 
          fontSize: "2.5rem", 
          fontWeight: "700", 
          background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          marginBottom: "16px"
        }}>
          AI Policy & Product Helper
        </h1>
        <p style={{ 
          color: "#6b7280", 
          fontSize: "1.1rem", 
          maxWidth: "600px", 
          margin: "0 auto 24px",
          lineHeight: "1.6"
        }}>
          Local-first RAG starter. Ingest sample docs, ask questions, and see citations.
        </p>
        
        <div className="card" style={{ 
          maxWidth: "700px", 
          margin: "0 auto",
          textAlign: "left",
          background: "linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)",
          border: "1px solid #cbd5e1"
        }}>
          <h3 style={{ 
            color: "#374151", 
            marginBottom: "16px",
            display: "flex",
            alignItems: "center",
            gap: "8px"
          }}>
            🚀 How to test
          </h3>
          <ol style={{ 
            color: "#4b5563", 
            lineHeight: "1.6",
            paddingLeft: "20px"
          }}>
            <li style={{ marginBottom: "8px" }}>
              Go to <b>Admin Panel</b> tab and click <b>Ingest sample docs</b>.
            </li>
            <li style={{ marginBottom: "8px" }}>
              Switch to <b>AI Chat</b> tab and ask: <i>"Can a customer return a damaged blender after 20 days?"</i>
            </li>
            <li>
              Try another question: <i>"What's the shipping SLA to East Malaysia for bulky items?"</i>
            </li>
          </ol>
        </div>
      </div>

      <Tabs tabs={tabs} defaultTab="chat" />
    </div>
  );
}
