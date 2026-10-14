"""StudyPilot AI — local-first study assistant."""

from __future__ import annotations

import logging

import streamlit as st

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

st.set_page_config(
    page_title="StudyPilot AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    from core.config import Settings
    from core.errors import StudyPilotError
    from core.llm import OllamaClient
    from core.retrieval import DocumentStore
    from components.pages import render_page
except ImportError as exc:
    logging.error("StudyPilot dependencies are incomplete: %s", exc)
    st.error(
        "Some project dependencies are missing. From this folder, run "
        "`python -m pip install -r requirements.txt`, then restart StudyPilot."
    )
    st.stop()

STYLE = """
<style>
:root {
  color-scheme: dark;
  --sp-bg: #0b1020;
  --sp-panel: #111a2d;
  --sp-panel-2: #151f35;
  --sp-border: #26334c;
  --sp-text: #e9eef8;
  --sp-muted: #a4b2ca;
  --sp-indigo: #8275f5;
  --sp-teal: #5bd7c2;
}
html, body, [data-testid="stAppViewContainer"] {
  background: radial-gradient(ellipse at 80% -15%, #25204b 0%, transparent 42%),
              radial-gradient(ellipse at 0% 35%, #102b3b 0%, transparent 28%),
              var(--sp-bg);
  color: var(--sp-text);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #11182a 0%, #0d1424 100%);
  border-right: 1px solid var(--sp-border);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stCaptionContainer"] { color: var(--sp-muted); }
h1, h2, h3 { color: var(--sp-text); letter-spacing: -0.025em; }
h1 { font-size: clamp(2rem, 3.2vw, 3rem) !important; }
[data-testid="stMetric"] {
  background: linear-gradient(145deg, rgba(25, 36, 60, .92), rgba(17, 26, 45, .92));
  border: 1px solid var(--sp-border);
  border-radius: 16px;
  padding: 18px 20px;
  box-shadow: 0 12px 32px rgba(0, 0, 0, .16);
}
[data-testid="stMetricLabel"] { color: var(--sp-muted); }
[data-testid="stMetricValue"] { color: var(--sp-teal); }
[data-testid="stExpander"], [data-testid="stForm"] {
  border: 1px solid var(--sp-border);
  border-radius: 14px;
  background: rgba(17, 26, 45, .72);
}
.stButton > button, .stDownloadButton > button, [data-testid="stLinkButton"] a {
  border-radius: 10px;
  border: 1px solid #3b4560;
  transition: border-color .15s ease, background-color .15s ease;
}
.stButton > button[kind="primary"] {
  color: #090e1b;
  background: linear-gradient(110deg, #9a8dfd, #65dbc5);
  border: 0;
  font-weight: 700;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--sp-teal);
}
button:focus-visible, a:focus-visible, input:focus-visible {
  outline: 3px solid var(--sp-teal) !important;
  outline-offset: 2px !important;
}
[data-testid="stChatMessage"] {
  border: 1px solid rgba(62, 78, 108, .54);
  border-radius: 14px;
  background: rgba(17, 26, 45, .66);
}
[data-testid="stFileUploader"] section {
  border: 1px dashed #51617f;
  border-radius: 14px;
  background: rgba(17, 26, 45, .42);
}
@media (max-width: 720px) {
  [data-testid="stMetric"] { padding: 12px; }
  [data-testid="stMainBlockContainer"] { padding: 1rem; }
}
</style>
"""

NAV_ITEMS = [
    "Dashboard",
    "My Documents",
    "AI Study Chat",
    "Summarizer",
    "Quiz Generator",
    "Web Research",
    "Study History",
    "Settings",
]


@st.cache_resource(show_spinner=False)
def _document_store(settings: Settings) -> DocumentStore:
    return DocumentStore(settings)


@st.cache_resource(show_spinner=False)
def _ollama_client(base_url: str, timeout: int) -> OllamaClient:
    return OllamaClient(base_url, timeout)


@st.cache_resource(ttl=15, show_spinner=False)
def _ollama_status(base_url: str, timeout: int):
    return _ollama_client(base_url, timeout).status()


st.markdown(STYLE, unsafe_allow_html=True)
try:
    settings = Settings.from_env()
    store = _document_store(settings)
    ollama = _ollama_client(settings.ollama_url, settings.request_timeout)
    model_status = _ollama_status(settings.ollama_url, settings.request_timeout)
except Exception as exc:
    logging.exception("StudyPilot could not initialize.")
    if isinstance(exc, StudyPilotError):
        st.error(str(exc))
    else:
        st.error(
            "StudyPilot could not initialize its local settings or document index. "
            "Check the installation and data-directory permissions, then restart the app."
        )
    st.stop()

with st.sidebar:
    st.markdown("## StudyPilot AI")
    st.caption("Your personal study space")
    st.markdown("---")
    st.radio("Navigate", NAV_ITEMS, key="navigation", label_visibility="collapsed")
    st.markdown("---")
    models = list(model_status.models)
    choices = list(dict.fromkeys([*models, settings.default_model]))
    st.session_state.setdefault("model_choice", settings.default_model)
    if st.session_state.model_choice not in choices:
        st.session_state.model_choice = settings.default_model
    model = st.selectbox(
        "Ollama model",
        choices,
        key="model_choice",
        format_func=lambda name: (
            f"{name} · installed" if name in models else f"{name} · not installed"
        ),
    )
    if model_status.available:
        if models:
            st.success(f"Ollama ready · {len(models)} model(s) detected")
        else:
            st.warning("Ollama is running, but no models were detected.")
    else:
        st.warning(f"Ollama unavailable: {model_status.detail}")
    with st.expander("Generation controls"):
        st.session_state.setdefault("temperature", settings.temperature)
        st.session_state.setdefault("max_tokens", settings.max_output_tokens)
        st.slider("Temperature", 0.0, 2.0, step=0.1, key="temperature")
        st.slider("Maximum output tokens", 64, 8192, step=64, key="max_tokens")
    st.caption("Local documents and the vector index are stored on this computer.")

selected_page = st.session_state.get("navigation", "Dashboard")
if selected_page not in NAV_ITEMS:
    selected_page = "Dashboard"
try:
    render_page(
        selected_page,
        settings,
        store,
        ollama,
        model,
        float(st.session_state.temperature),
        int(st.session_state.max_tokens),
        model_status,
    )
except Exception as exc:
    logging.exception("Page rendering failed.")
    st.error("This page could not be loaded. Your locally stored study data was not intentionally changed.")
