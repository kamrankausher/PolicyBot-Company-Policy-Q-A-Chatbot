# PolicyBot — AI Company Policy Q&A Chatbot

PolicyBot is a modern, AI-powered chatbot that lets employees instantly ask questions and receive answers strictly based on uploaded company policy documents.

## Live Demo
https://policybot-company-policy-q-a-chatbot-9v89.onrender.com

## What Problem It Solves
Employees often struggle to quickly find specific answers in lengthy company policy handbooks (e.g., leave policies, remote work guidelines). PolicyBot eliminates manual searching by instantly extracting exactly what you need, ensuring accurate answers accompanied by source citations.

## How It Works
1. **Document Upload**: You upload a company policy PDF.
2. **Text Extraction & Chunking**: The backend extracts text from the PDF and splits it into smaller overlapping chunks.
3. **Embedding**: These chunks are converted into vector representations using a Sentence Transformer model and stored in a local ChromaDB vector database.
4. **Retrieval**: When you ask a question, the system embeds your question and searches ChromaDB for the most relevant document chunks.
5. **Generation**: The retrieved chunks are sent to the Gemini 1.5 Flash API as context, which generates a concise answer strictly based on those excerpts.

```text
User uploads PDF -> Extracted -> Chunked -> Embedded -> Stored in ChromaDB
User asks Question -> Embedded -> Vector Search in ChromaDB -> Retrieves Context
Question + Context -> Gemini LLM -> Final Answer + Confidence Score
```

## Tech Stack

| Layer | Technology | Why I Used It |
| --- | --- | --- |
| **Backend** | FastAPI | High performance, automatic validation with Pydantic, and built-in interactive API docs (Swagger UI). |
| **Frontend** | Vanilla HTML/CSS/JS | No frameworks required. Features an **Ultra-Premium UI** with bespoke glassmorphism, animated mesh gradient backgrounds, floating input docks, Space Grotesk typography, and fluid cubic-bezier micro-animations. |
| **Vector DB** | ChromaDB | Fast, open-source local vector database ideal for indexing document embeddings. |
| **Embeddings** | sentence-transformers | `all-MiniLM-L6-v2` is compact and generates high-quality semantic embeddings locally. |
| **LLM** | Google Gemini API | `gemini-1.5-flash` provides incredibly fast, cost-effective, and highly accurate text generation. |
| **PDF Parsing** | PyMuPDF | Robust and accurate text extraction from PDF documents. |

## Project Structure
```text
policybot/
├── backend/
│   ├── main.py          # FastAPI application entry point and endpoints
│   ├── ingestor.py      # PDF extraction, chunking, and ChromaDB insertion
│   ├── retriever.py     # Similarity search using ChromaDB
│   ├── answerer.py      # Gemini prompt generation and LLM calling
│   └── models.py        # Pydantic data validation models
├── frontend/
│   └── index.html       # The complete Single-Page Application (UI)
├── tests/
│   └── test_api.py      # Automated tests for all backend endpoints
├── requirements.txt     # Python dependencies
├── .env.example         # Template for environment variables
├── render.yaml          # Infrastructure as Code configuration for Render deployment
└── README.md            # Project documentation
```

## Local Setup

Step by step:
1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd policybot
   ```
2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows: venv\Scripts\activate
   # On Mac/Linux: source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Create a `.env` file:**
   Copy `.env.example` to `.env` and add your Google Gemini API key:
   ```bash
   cp .env.example .env
   ```
5. **Run the backend:**
   ```bash
   uvicorn backend.main:app --reload
   ```
6. **Open the frontend:**
   Simply double-click the `frontend/index.html` file in your browser, or serve it using any local HTTP server (e.g., `python -m http.server`).
7. **Note on First Run:**
   The first time you upload a document, it will automatically download the ~80MB `all-MiniLM-L6-v2` model from Hugging Face. Subsequent runs will use the cached model.

## Running Tests

To run the automated test suite, execute the following command while your backend is running locally on port 8000:
```bash
pytest tests/test_api.py -v
```
**What each test checks:**
- **test_01_health_check**: Validates the `/health` endpoint is active.
- **test_02_upload_valid_pdf**: Tests PDF extraction, embedding, and successful DB storage.
- **test_03_upload_non_pdf_rejected**: Ensures non-PDF uploads are gracefully rejected.
- **test_04_ask_question**: Verifies the end-to-end RAG pipeline, retrieving an answer and confidence score.
- **test_05_ask_invalid_session**: Verifies 404 response for non-existent session queries.
- **test_06_ask_empty_question**: Confirms empty questions are rejected (400/422).
- **test_07_get_history**: Checks that chat history is stored and retrieved correctly.
- **test_08_get_history_empty_session**: Verifies an empty list is returned for a non-existent history query.
- **test_09_cors_headers**: Asserts that CORS is properly configured.
- **test_10_delete_session**: Checks that deleting a session correctly removes the ChromaDB collection and history.

## API Reference

| Method | Path | Description | Request Body | Response Body |
| --- | --- | --- | --- | --- |
| `POST` | `/upload` | Upload and index a PDF policy document. | `multipart/form-data` (file) | `{ message, document_name, total_chunks, session_id }` |
| `POST` | `/ask` | Ask a question based on the uploaded document. | JSON: `{ session_id, question }` | `{ answer, sources, confidence, session_id }` |
| `GET`  | `/history/{session_id}` | Retrieve full chat history for a session. | None | `[ { role, content, sources } ]` |
| `DELETE` | `/session/{session_id}` | Clear a session from the DB and history. | None | `{ message: "Session cleared" }` |
| `GET`  | `/health` | Check backend health status. | None | `{ status, model, embeddings }` |

## Deployment

**Unified Deployment (Render.com)**
This project is built as a unified service. The FastAPI backend automatically serves the `index.html` frontend from the root URL `/`, meaning you only need to deploy a single web service.

1. Push your repository to GitHub.
2. Sign up on [Render.com](https://render.com) and click "New Web Service".
3. Connect your GitHub repository.
4. Render will automatically detect the `render.yaml` configuration and deploy the application.
5. In your Render Dashboard, add the `GEMINI_API_KEY` to your environment variables.
6. Once deployed, simply visit your Render URL to see the fully functioning UI and Backend.

## How I Would Explain This in an Interview
"PolicyBot is an AI-powered QA assistant that allows employees to securely query company policy documents and get instant, cited answers. It uses a Retrieval-Augmented Generation (RAG) architecture, which means it extracts text from PDFs, stores semantic chunks in a local vector database, and retrieves only the most relevant sections when a user asks a question. By combining FastAPI, ChromaDB, and the Google Gemini API, the system seamlessly generates accurate responses backed by direct excerpts from the provided document. Ultimately, this eliminates the friction of manually searching through massive policy handbooks, saving significant time while ensuring compliance and transparency."

## Author
- **Name:** Your Name
- **GitHub:** [your-github](https://github.com/your-username)
- **LinkedIn:** [your-linkedin](https://linkedin.com/in/your-profile)
- **Email:** your.email@example.com
