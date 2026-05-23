# PolicyBot — AI Company Policy Q&A Chatbot

**PolicyBot is an AI-powered assistant that reads company policies and provides instant, source-cited answers to employee questions.**

## Live Demo
[Link to Live Demo] *(Update this link after deploying to Netlify)*

## What Problem It Solves
Employees often struggle to find specific information buried in lengthy company handbooks and policy PDFs. PolicyBot eliminates the need for manual searching by allowing users to ask questions in natural language and receive immediate, accurate answers drawn directly from the uploaded document.

## How It Works
1. **Upload**: A user uploads a policy PDF via the frontend.
2. **Extraction**: The backend extracts text from the PDF and splits it into manageable, overlapping chunks.
3. **Embedding**: Each chunk is converted into a vector representation using sentence-transformers and stored in a local ChromaDB vector database.
4. **Retrieval**: When a question is asked, it is converted into a vector and used to search ChromaDB for the most relevant policy chunks.
5. **Generation**: The retrieved chunks and the question are sent to the Google Gemini API, which generates an accurate answer strictly based on the provided context.

```text
User ──> Upload PDF ──> PyMuPDF extracts text ──> Sentence Transformers ──> ChromaDB
                                                                             │
User ──> Asks Question ──────────────────────────────────────────────────────┘
                                                                             │
Gemini 1.5 Flash <── Retrieves context & prompts ────────────────────────────┘
         │
         └──> Answers User
```

## Tech Stack
| Layer | Technology | Why I Used It |
|---|---|---|
| **Frontend** | Vanilla HTML/CSS/JS | Simple, lightweight, requires no build step. |
| **Backend** | FastAPI | High performance, native async support, and auto-generated API docs. |
| **PDF Parsing** | PyMuPDF | Fast and reliable text extraction from PDF files. |
| **Vector DB** | ChromaDB | Lightweight local vector database perfect for document retrieval. |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | High-quality local embeddings for efficient semantic search. |
| **LLM** | Google Gemini 1.5 Flash | Fast, intelligent generation with generous free tier limits. |

## Project Structure
```text
policybot/
├── .env                  # Environment variables (not committed)
├── .env.example          # Template for environment variables
├── .gitignore            # Ignored files and folders
├── README.md             # Project documentation
├── render.yaml           # Deployment configuration for Render
├── requirements.txt      # Python dependencies
├── backend/              # FastAPI application
│   ├── main.py           # API endpoints and entry point
│   ├── ingestor.py       # PDF parsing, chunking, and embedding logic
│   ├── retriever.py      # ChromaDB similarity search
│   ├── answerer.py       # Gemini prompt building and generation
│   ├── database.py       # ChromaDB persistent client setup
│   └── models.py         # Pydantic models for validation
├── frontend/             # Frontend assets
│   └── index.html        # Main user interface
└── tests/                # Automated tests
    └── test_api.py       # Pytest suite testing API endpoints
```

## Local Setup
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd policybot
   ```
2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```
   *Activate it (Windows):* `venv\Scripts\activate`
   *Activate it (Mac/Linux):* `source venv/bin/activate`
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Create environment variables:**
   Copy `.env.example` to `.env` and insert your Gemini API Key:
   ```bash
   cp .env.example .env
   ```
5. **Run the backend:**
   ```bash
   uvicorn backend.main:app --reload
   ```
6. **Open the frontend:**
   Simply open `frontend/index.html` in your web browser.
7. **First run note:**
   The first time you run the backend or upload a PDF, it will download the `all-MiniLM-L6-v2` embedding model (around 90MB). This may take a moment.

## Running Tests
To run the automated test suite, use pytest:
```bash
pytest tests/test_api.py -v
```
**What the tests check:**
- Health endpoint status and required fields
- Successful upload of a valid PDF and chunk creation
- Rejection of non-PDF files
- Asking a valid question and receiving a confident answer
- Handling invalid session IDs or empty questions appropriately
- Fetching and verifying chat history
- CORS header availability
- Session deletion and subsequent unavailability

## API Reference
| Method | Path | Description | Request Body | Response Body |
|---|---|---|---|---|
| **GET** | `/health` | Check service health | None | `{"status": "ok", "model": "gemini-1.5-flash", "embeddings": "all-MiniLM-L6-v2"}` |
| **POST** | `/upload` | Upload and index PDF | `multipart/form-data` (file) | `UploadResponse` (session_id, total_chunks, message) |
| **POST** | `/ask` | Ask a question | `QuestionRequest` (session_id, question) | `AnswerResponse` (answer, sources, confidence, session_id) |
| **GET** | `/history/{session_id}` | Get session chat history | None | `List[ChatMessage]` |
| **DELETE** | `/session/{session_id}` | Delete session context | None | `{"message": "Session cleared"}` |

## Deployment
### Backend (Render.com)
1. Push your repository to GitHub.
2. Sign in to Render and create a new **Web Service**.
3. Connect your repository.
4. Render will automatically detect the `render.yaml` configuration.
5. Go to the Environment variables section on the Render dashboard and manually add your `GEMINI_API_KEY`.
6. Deploy the service.

### Frontend (Netlify)
1. Update the `API_BASE` variable in `frontend/index.html` to point to your new Render backend URL.
2. Deploy the `frontend/` folder directly to Netlify via drag-and-drop or Git integration.

## How I Would Explain This in an Interview
1. **What the project does:** PolicyBot is an AI-powered application that allows employees to upload company policy documents and ask questions to get instant, accurate answers.
2. **What RAG means in simple terms:** It uses Retrieval-Augmented Generation, meaning it searches the uploaded document for relevant information first, and then uses an AI model to summarize that information into a coherent answer.
3. **How the tech stack fits together:** I built the backend with FastAPI for speed, used ChromaDB and sentence-transformers locally to handle the document search, and integrated Google's Gemini API to generate the final answers, all tied to a lightweight vanilla JavaScript frontend.
4. **What real-world problem it solves:** It saves employees hours of manually searching through dense, multi-page company handbooks by providing immediate, context-aware answers with exact source citations.

## Author
**Senior Software Engineer**
- GitHub: [Your GitHub URL]
- LinkedIn: [Your LinkedIn URL]
- Email: [Your Email]
