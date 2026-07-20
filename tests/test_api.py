import pytest
import httpx

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

# Global session_id for reuse across tests
session_id = None

def create_test_pdf() -> bytes:
    """Returns a raw bytes string of a valid PDF (sample_policy.pdf)."""
    with open("sample_policy.pdf", "rb") as f:
        return f.read()

def test_01_health_check():
    """Test the /health endpoint to ensure the service is running and returns expected fields."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model" in data
    assert "embeddings" in data
    assert data["status"] == "ok"

def test_02_upload_valid_pdf():
    """Test uploading a valid minimal PDF and expect a successful response with chunks."""
    global session_id
    pdf_bytes = create_test_pdf()
    files = {"file": ("test_policy.pdf", pdf_bytes, "application/pdf")}
    
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    
    assert "session_id" in data
    assert "total_chunks" in data
    assert "document_name" in data
    assert "message" in data
    assert data["total_chunks"] > 0
    
    session_id = data["session_id"]

def test_03_upload_non_pdf_rejected():
    """Test uploading a non-PDF file and expect a 400 or 422 error."""
    files = {"file": ("test.txt", b"This is a text file.", "text/plain")}
    response = client.post("/upload", files=files)
    assert response.status_code in [400, 422]

def test_04_ask_question():
    """Test asking a valid question using the session_id from test_02."""
    global session_id
    assert session_id is not None, "session_id must be populated from test_02"
    
    payload = {
        "session_id": session_id,
        "question": "What is the company leave policy?"
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "answer" in data
    assert "sources" in data
    assert "confidence" in data
    assert "session_id" in data
    assert len(data["answer"]) > 10
    assert data["confidence"] in ["High", "Medium", "Low"]

def test_05_ask_invalid_session():
    """Test asking a question with a fake session_id and expect a 404 error."""
    payload = {
        "session_id": "invalid-fake-session-xyz",
        "question": "What are the working hours?"
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 404

def test_06_ask_empty_question():
    """Test asking an empty question and expect a 400 or 422 error."""
    global session_id
    payload = {
        "session_id": session_id or "test",
        "question": ""
    }
    response = client.post("/ask", json=payload)
    assert response.status_code in [400, 422]

def test_07_get_history():
    """Test fetching chat history for a valid session and expect a non-empty list."""
    global session_id
    response = client.get(f"/history/{session_id}")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0

def test_08_get_history_empty_session():
    """Test fetching chat history for a nonexistent session and expect an empty list."""
    response = client.get("/history/nonexistent-session-id")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert data == []

def test_09_cors_headers():
    """Test that CORS middleware is properly configured by sending an OPTIONS request."""
    headers = {"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"}
    response = client.options("/health", headers=headers)
    assert response.status_code in [200, 204]

def test_10_delete_session():
    """Test deleting a session to ensure data is removed."""
    # First, upload a new PDF to get a fresh session
    pdf_bytes = create_test_pdf()
    files = {"file": ("test_policy_delete.pdf", pdf_bytes, "application/pdf")}
    upload_resp = client.post("/upload", files=files)
    assert upload_resp.status_code == 200
    new_session_id = upload_resp.json()["session_id"]
    
    # Now delete the session
    del_resp = client.delete(f"/session/{new_session_id}")
    assert del_resp.status_code == 200
    
    # Try asking a question, expect 404
    payload = {
        "session_id": new_session_id,
        "question": "Are there remote work options?"
    }
    ask_resp = client.post("/ask", json=payload)
    assert ask_resp.status_code == 404
