# StudyPilot AI

StudyPilot AI is a local-first study assistant built with Streamlit. It indexes PDF, DOCX, and TXT study files into a persistent ChromaDB collection, retrieves source passages for grounded answers, creates document summaries and evidence-linked quizzes with Ollama, and provides a separate external web-search page.

## Requirements

- Windows 11 or another supported desktop OS
- Python 3.11 (recommended; Python 3.10+ is expected to work with the dependencies)
- Ollama for chat, summarization, and quiz generation
- Internet access for first-time Python/model downloads and optional web search

Install the Python packages listed in `requirements.txt` in an isolated virtual environment. Python dependencies provide the interface, document parsers, vector database, local embeddings, and web search. Ollama is a separate application used for text generation; install it and pull at least one model before using chat, summaries, or quizzes.

## Windows PowerShell setup

From the project folder:

```powershell
cd "D:\RAG project\rag_project"
.\setup.bat
```

`setup.bat` prefers Python 3.11 and accepts Python 3.10 or newer, creates an isolated `.venv`, installs the dependencies, and creates `.env` from the safe example configuration without overwriting an existing `.env`. After setup, double-click `run.bat` whenever you want to start the app. It creates the environment on first run and repairs it if Streamlit is missing. Keep the opened terminal window running while you use StudyPilot; closing it stops the app. Streamlit opens the local page in your browser.

You can also use PowerShell commands directly:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Install [Ollama](https://ollama.com/) and make sure its local service is running. If it is not already running, open the Ollama app or start `ollama serve` in a separate terminal. Pull at least one supported model if it is not already installed:

```powershell
ollama pull llama3.2:3b
# or
ollama pull qwen2.5:3b
ollama list
```

StudyPilot never downloads an LLM automatically. In the app, the sidebar displays the models actually reported by the Ollama API and marks an uninstalled configured model.

Start the application:

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).

## Getting the project from GitHub

Clone the repository, then run `setup.bat` once (or let `run.bat` perform setup automatically) and use `run.bat` whenever you want to start the app:

```powershell
git clone https://github.com/bodicherlanikhilsriraj-code/document_analyser.git
cd document_analyser\rag_project
.\setup.bat
.\run.bat
```

Python dependencies and Ollama models are installed separately. The app can open without Ollama, but chat, summaries, and quizzes need Ollama running with a model installed. On the first document indexing, the embedding model is downloaded and cached locally.

Before pushing to GitHub, check that your study materials and local settings are not staged. `.env`, `data/` (uploaded files, databases, and vectors), and Python environments are excluded by `.gitignore`. Commit code, tests, and configuration examples only; do not upload personal documents, `.env` values, or downloaded model files.

The GitHub Actions workflow runs the test suite on pushes and pull requests using Windows and Python 3.11.

## First use

1. Open **My Documents** and upload PDF, DOCX, or TXT material.
2. Indexing extracts text, chunks it with page metadata where available, generates local sentence-transformer embeddings, and persists vectors and document metadata under `data/`.
3. Select indexed files in **AI Study Chat**, ask a question, and expand the source references to inspect filenames, PDF pages, and retrieved excerpts.
4. Use **Summarizer** for structured, hierarchical summaries of a whole document or retrieved chapter/topic passages, or **Quiz Generator** for evidence-checked multiple-choice practice.
5. Use **Web Research** for external search. Only the query entered on that page is sent to the search provider; uploaded notes are never added to search queries.

The first embedding operation downloads `all-MiniLM-L6-v2` if it is not cached. Embeddings run on CPU by default. Summaries use hierarchical reduction and are capped at 64 chunks per request to keep local model usage bounded; split larger material into sections. A scanned/image-only PDF is detected but OCR is not included; OCR it before uploading.

## Configuration

Copy `.env.example` to `.env` and adjust settings as needed:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Local Ollama API address |
| `OLLAMA_MODEL` | `Qwen2.5:3b` | Initial model selection (falls back to an installed model in the app selector) |
| `OLLAMA_TIMEOUT_SECONDS` | `180` | Generation request timeout |
| `OLLAMA_TEMPERATURE` | `0.2` | Default generation temperature |
| `OLLAMA_MAX_OUTPUT_TOKENS` | `1024` | Default output limit |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local vector embedding model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1200` / `180` | Default text chunk settings |
| `RETRIEVAL_COUNT` | `5` | Default number of passages retrieved |
| `RELEVANCE_THRESHOLD` | `0.15` | Minimum cosine similarity |
| `MAX_UPLOAD_MB` | `25` | Per-file upload limit |
| `STUDYPILOT_DATA_DIR` | `./data` | Optional local storage root |

The Settings page changes chunk/retrieval controls for the current Streamlit session. Chunking changes apply to subsequent indexing or explicit re-indexing.

## Tests

Run the automated suite from the project directory:

```powershell
python -m pytest
```

Tests use temporary directories and mocked model/search responses; they do not modify the real `data/` store or require a running Ollama service. ChromaDB integration tests and PDF/DOCX extraction tests are skipped when their runtime dependencies are absent. The standard-library test runner is also supported:

```powershell
python -m unittest discover -s tests -v
```

## Architecture

- `app.py` — Streamlit application setup, local service lifecycle, navigation, and sidebar controls.
- `components/pages.py` — dashboard, document library, chat, summary, quiz, web research, history, and settings pages.
- `core/config.py` — validated environment-backed settings and local storage paths.
- `core/documents.py`, `core/chunking.py` — bounded PDF/DOCX/TXT extraction and deterministic page-aware chunking.
- `core/embeddings.py`, `core/retrieval.py` — lazy local embeddings, SQLite document metadata, and persistent ChromaDB vectors.
- `core/llm.py`, `core/rag.py` — Ollama API integration, grounded prompts, insufficiency handling, and citations.
- `core/study_tools.py`, `core/quiz_service.py` — hierarchical summaries and strict Pydantic quiz validation.
- `core/web_search.py` — DuckDuckGo search adapter and HTTP(S) URL validation.
- `tests/` — isolated unit and optional integration tests.

## Privacy and limitations

Uploaded study documents, extracted source files, document metadata, and vectors are stored locally under the configured data directory. Chat and completed study activity are held in the active Streamlit session; they are not written to a persistent conversation database. The local Ollama service receives prompts containing the selected/retrieved study passages. Web search sends only the query typed on the Web Research page to the search provider.

The app does not claim perfect answer accuracy. Local 3B models can miss details; generated summaries and quiz questions should be checked against course material. Web results show the provider's titles/snippets and links; StudyPilot does not claim it opened or verified the linked articles. Removing documents from Settings deletes their local source copies and vectors.
