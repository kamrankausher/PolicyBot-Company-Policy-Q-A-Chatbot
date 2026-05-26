import pytest
import httpx

BASE_URL = "http://localhost:8000"

# Global session_id for reuse across tests
session_id = None

def create_test_pdf() -> bytes:
    """Returns a raw bytes string of a minimal valid PDF containing the required text."""
    # We provide an imperfect xref, most parsers (including PyMuPDF) can recover
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length 176 >>\nstream\nBT\n/F1 12 Tf\n"
        b"72 700 Td\n(Company Leave Policy: All employees receive 30 days annual leave.) Tj\n"
        b"0 -14 Td\n(Working hours are 9am to 6pm Sunday to Thursday.) Tj\n"
        b"0 -14 Td\n(Remote work requires manager approval.) Tj\nET\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000244 00000 n \n"
        b"0000000471 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n559\n%%EOF\n"
    )

def test_01_health_check():
    """Test the /health endpoint to ensure the service is running and returns expected fields."""
    response = httpx.get(f"{BASE_URL}/health")
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
    
    response = httpx.post(f"{BASE_URL}/upload", files=files, timeout=30.0)
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
    response = httpx.post(f"{BASE_URL}/upload", files=files)
    assert response.status_code in [400, 422]

def test_04_ask_question():
    """Test asking a valid question using the session_id from test_02."""
    global session_id
    assert session_id is not None, "session_id must be populated from test_02"
    
    payload = {
        "session_id": session_id,
        "question": "What is the company leave policy?"
    }
    response = httpx.post(f"{BASE_URL}/ask", json=payload, timeout=60.0)
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
    response = httpx.post(f"{BASE_URL}/ask", json=payload)
    assert response.status_code == 404

def test_06_ask_empty_question():
    """Test asking an empty question and expect a 400 or 422 error."""
    global session_id
    payload = {
        "session_id": session_id or "test",
        "question": ""
    }
    response = httpx.post(f"{BASE_URL}/ask", json=payload)
    assert response.status_code in [400, 422]

def test_07_get_history():
    """Test fetching chat history for a valid session and expect a non-empty list."""
    global session_id
    response = httpx.get(f"{BASE_URL}/history/{session_id}")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0

def test_08_get_history_empty_session():
    """Test fetching chat history for a nonexistent session and expect an empty list."""
    response = httpx.get(f"{BASE_URL}/history/nonexistent-session-id")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert data == []

def test_09_cors_headers():
    """Test that CORS middleware is properly configured by sending an OPTIONS request."""
    headers = {"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"}
    response = httpx.options(f"{BASE_URL}/health", headers=headers)
    assert response.status_code in [200, 204]

def test_10_delete_session():
    """Test deleting a session to ensure data is removed."""
    # First, upload a new PDF to get a fresh session
    pdf_bytes = create_test_pdf()
    files = {"file": ("test_policy_delete.pdf", pdf_bytes, "application/pdf")}
    upload_resp = httpx.post(f"{BASE_URL}/upload", files=files)
    assert upload_resp.status_code == 200
    new_session_id = upload_resp.json()["session_id"]
    
    # Now delete the session
    del_resp = httpx.delete(f"{BASE_URL}/session/{new_session_id}")
    assert del_resp.status_code == 200
    
    # Try asking a question, expect 404
    payload = {
        "session_id": new_session_id,
        "question": "Are there remote work options?"
    }
    ask_resp = httpx.post(f"{BASE_URL}/ask", json=payload)
    assert ask_resp.status_code == 404
