# StudyPilot AI

StudyPilot AI is a local-first study assistant. The application source, setup instructions, and tests are in [`rag_project/`](./rag_project/).

## Windows quick start

Install Python 3.11 or newer and [Ollama](https://ollama.com/). From the repository folder:

```powershell
cd rag_project
.\setup.bat
.\run.bat
```

`run.bat` starts the app at `http://localhost:8501`. Ollama must be running with at least one model installed for chat, summaries, and quizzes. See the [full setup and usage guide](./rag_project/README.md) for first-time setup, model installation, configuration, troubleshooting, and privacy details.

GitHub Actions runs the test suite on pushes and pull requests. Local `.env` settings, virtual environments, uploaded documents, and vector/database data are ignored by Git and should remain on your computer.
