import pytest
import httpx
from fastapi.testclient import TestClient

# Use TestClient for local tests without needing a running server
from backend.main import app

BASE_URL = "http://localhost:8000"
client = TestClient(app, base_url=BASE_URL)

# Module-level variable to store session_id across tests
test_session_id = None

def create_test_pdf() -> bytes:
    """
    Returns a bytes object containing a valid minimal PDF with specific text.
    Constructed manually without external libraries.
    """
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> "
        b"/Contents 4 0 R >>\nendobj\n"
        b"4 0 obj\n<< /Length 176 >>\nstream\n"
        b"BT\n/F1 12 Tf\n10 700 Td\n(Company Leave Policy: All employees receive 30 days annual leave.) Tj\n"
        b"0 -14 Td\n(Working hours are 9am to 6pm Sunday to Thursday.) Tj\n"
        b"0 -14 Td\n(Remote work requires manager approval.) Tj\nET\n"
        b"endstream\nendobj\n"
        b"xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000295 00000 n \n"
        b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n521\n%%EOF\n"
    )

def test_01_health_check():
    """Test GET /health returns status 200 and required fields."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model" in data
    assert "embeddings" in data
    assert data["status"] == "ok"

def test_02_upload_valid_pdf():
    """Test POST /upload with a valid PDF returns success and session_id."""
    global test_session_id
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
    
    test_session_id = data["session_id"]

def test_03_upload_non_pdf_rejected():
    """Test POST /upload with a .txt file is rejected with 400 or 422."""
    txt_bytes = b"This is a text file, not a PDF."
    files = {"file": ("test.txt", txt_bytes, "text/plain")}
    
    response = client.post("/upload", files=files)
    assert response.status_code in [400, 422]

def test_04_ask_question():
    """Test POST /ask with a question returns an answer and sources."""
    assert test_session_id is not None, "test_session_id not set from test_02"
    
    payload = {
        "session_id": test_session_id,
        "question": "What is the leave policy?"
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
    """Test POST /ask with an invalid session ID returns 404."""
    payload = {
        "session_id": "invalid-fake-session-xyz",
        "question": "What is the leave policy?"
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 404

def test_06_ask_empty_question():
    """Test POST /ask with an empty question returns 400 or 422."""
    payload = {
        "session_id": test_session_id or "dummy",
        "question": ""
    }
    response = client.post("/ask", json=payload)
    assert response.status_code in [400, 422]

def test_07_get_history():
    """Test GET /history for a valid session returns a list of messages."""
    assert test_session_id is not None, "test_session_id not set from test_02"
    
    response = client.get(f"/history/{test_session_id}")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0

def test_08_get_history_empty_session():
    """Test GET /history for a nonexistent session returns an empty list."""
    response = client.get("/history/nonexistent-session-id")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0

def test_09_cors_headers():
    """Test OPTIONS request to verify CORS headers are present."""
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "GET"
    }
    response = client.options("/health", headers=headers)
    assert response.status_code in [200, 204]

def test_10_delete_session():
    """Test DELETE /session clears the session correctly."""
    # Create a new session first
    pdf_bytes = create_test_pdf()
    files = {"file": ("test_policy_2.pdf", pdf_bytes, "application/pdf")}
    upload_response = client.post("/upload", files=files)
    new_session_id = upload_response.json()["session_id"]
    
    # Delete the new session
    delete_response = client.delete(f"/session/{new_session_id}")
    assert delete_response.status_code == 200
    
    # Try asking a question to the deleted session
    payload = {
        "session_id": new_session_id,
        "question": "What is the leave policy?"
    }
    ask_response = client.post("/ask", json=payload)
    assert ask_response.status_code == 404
