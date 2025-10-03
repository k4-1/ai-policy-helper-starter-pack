def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_ingest_and_ask(client):
    r = client.post("/api/ingest")
    assert r.status_code == 200
    # Ask a deterministic question
    r2 = client.post("/api/ask", json={"query":"What is the refund window for small appliances?"})
    assert r2.status_code == 200
    data = r2.json()
    assert "citations" in data and len(data["citations"]) > 0
    assert "answer" in data and isinstance(data["answer"], str)

def test_streaming_endpoint(client):
    r = client.post("/api/ingest")
    assert r.status_code == 200
    r2 = client.post("/api/ask/stream", json={"query":"What is the refund window for small appliances?", "stream": True})
    assert r2.status_code == 200
    # StreamingResponse returns an iterator of bytes; consume few events
    events = []
    for idx, chunk in enumerate(r2.iter_lines()):
        if idx > 5:
            break
        if chunk:
            s = chunk.decode() if isinstance(chunk, (bytes, bytearray)) else str(chunk)
            if s.startswith("data: "):
                events.append(s[len("data: "):])
    assert any("\"type\":\"start\"" in e or '"type":"start"' in e for e in events)
    assert any('"type":"chunk"' in e for e in events) or any('"type":"end"' in e for e in events)

def test_feedback_endpoint(client):
    # generate an answer first
    client.post("/api/ingest")
    r2 = client.post("/api/ask", json={"query":"What is the refund window for small appliances?"})
    assert r2.status_code == 200
    answer = r2.json()["answer"]
    fb = {
        "query": "What is the refund window for small appliances?",
        "answer": answer,
        "helpful": True,
        "comment": "Helpful",
        "rating": 5
    }
    r3 = client.post("/api/feedback", json=fb)
    assert r3.status_code == 200
    resp = r3.json()
    assert "id" in resp and resp["status"] == "ok"
