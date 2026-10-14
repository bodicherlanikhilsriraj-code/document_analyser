"""Functional Streamlit pages for StudyPilot AI."""

from __future__ import annotations

import logging
import html
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st

from core.chunking import split_pages
from core.config import Settings
from core.documents import document_hash, extract_document
from core.errors import StudyPilotError
from core.llm import OllamaClient
from core.quiz_service import QuizQuestion, generate_quiz, score_quiz
from core.rag import answer_question
from core.retrieval import DocumentRecord, DocumentStore, Passage
from core.study_tools import summarize_document
from core.web_search import search_web

logger = logging.getLogger(__name__)


def _activity(kind: str, title: str, details: str = "") -> None:
    events = st.session_state.setdefault("study_history", [])
    events.insert(
        0,
        {"kind": kind, "title": title, "details": details, "time": datetime.now().strftime("%b %d, %H:%M")},
    )
    del events[50:]


def _friendly_error(error: Exception) -> None:
    if isinstance(error, StudyPilotError):
        st.error(str(error))
    elif isinstance(error, (ValueError, OSError)):
        st.error(str(error))
    else:
        logger.exception("StudyPilot operation failed.")
        st.error("That operation failed unexpectedly. Check the local app log and try again.")


def _source_panel(sources: tuple[Passage, ...] | list[Passage]) -> None:
    if not sources:
        return
    with st.expander(f"Sources used ({len(sources)})", expanded=False):
        for source in sources:
            where = f" · page {source.page_number}" if source.page_number else ""
            st.text(f"[{source.source_id}] {source.filename}{where}")
            st.text(source.text[:900] + ("…" if len(source.text) > 900 else ""))


def _document_options(documents: list[DocumentRecord]) -> dict[str, str]:
    return {
        f"{html.escape(record.filename, quote=False)} · {record.chunk_count} chunks": record.document_id
        for record in documents
    }


def _index_uploads(
    uploaded_files: list[Any], settings: Settings, store: DocumentStore
) -> None:
    progress = st.progress(0, text="Preparing uploads…")
    succeeded = 0
    duplicates = 0
    failures: list[str] = []
    for index, uploaded in enumerate(uploaded_files):
        safe_name = uploaded.name.replace("\\", "/").rsplit("/", 1)[-1]
        try:
            with st.status("Processing selected file…", expanded=False) as status:
                st.text(safe_name)
                if Path(safe_name).suffix.lower() not in {".pdf", ".docx", ".txt"}:
                    raise ValueError("Upload a PDF, DOCX, or TXT file.")
                max_bytes = settings.max_upload_mb * 1024 * 1024
                if uploaded.size > max_bytes:
                    raise ValueError(f"The file exceeds the {settings.max_upload_mb} MB limit.")
                content = uploaded.getvalue()
                if len(content) > max_bytes:
                    raise ValueError(f"The file exceeds the {settings.max_upload_mb} MB limit.")
                digest = document_hash(content)
                existing = store.find_by_hash(digest)
                if existing:
                    duplicates += 1
                    st.info("This file was already indexed; no duplicate was added.")
                    st.text(f"{safe_name} → {existing.filename}")
                    status.update(label="Skipped duplicate file", state="complete")
                else:
                    pages = extract_document(safe_name, content, max_bytes)
                    chunk_size = st.session_state.get("chunk_size", settings.chunk_size)
                    chunk_overlap = st.session_state.get("chunk_overlap", settings.chunk_overlap)
                    if chunk_overlap >= chunk_size:
                        raise ValueError("Chunk overlap must be smaller than chunk size.")
                    chunks = split_pages(digest, pages, chunk_size, chunk_overlap)
                    status.update(label=f"Embedding and saving {len(chunks)} passages…")
                    record, created = store.index_document(safe_name, content, digest, chunks)
                    if created:
                        succeeded += 1
                        st.success(
                            f"Indexed {len(pages)} text section(s) into "
                            f"{record.chunk_count} passages."
                        )
                        st.text(record.filename)
                    status.update(label="File indexed", state="complete")
                _activity("Document", safe_name, "Document indexed or duplicate detected.")
        except Exception as exc:
            failures.append(f"{safe_name}: {exc}")
            st.text(safe_name)
            _friendly_error(exc)
        progress.progress((index + 1) / len(uploaded_files), text=f"Processed {index + 1}/{len(uploaded_files)}")
    progress.empty()
    if succeeded:
        st.success(f"Finished indexing {succeeded} document(s).")
    if duplicates:
        st.caption(f"Skipped {duplicates} duplicate upload(s).")
    if failures:
        st.warning(f"{len(failures)} upload(s) need attention.")


def render_dashboard(store: DocumentStore, model_status: Any) -> None:
    st.title("Welcome to StudyPilot")
    st.write("Your personal study space for exploring notes, preparing for exams, and finding sources.")
    records = store.list_documents()
    first, second, third = st.columns(3)
    first.metric("Study documents", len(records))
    second.metric("Indexed passages", store.chunk_count())
    third.metric("Ollama", "Connected" if model_status.available else "Offline")
    st.markdown("---")
    st.subheader("Pick up where you left off")
    targets = [
        ("AI Study Chat", "Ask questions grounded in your notes."),
        ("Summarizer", "Turn a document into a structured revision guide."),
        ("Quiz Generator", "Practice a topic and check explanations."),
        ("Web Research", "Explore external sources separately from your notes."),
    ]
    columns = st.columns(4)
    for column, (page, description) in zip(columns, targets):
        with column:
            st.markdown(f"#### {page}")
            st.caption(description)
            st.button(
                f"Open {page}",
                key=f"dashboard_{page}",
                use_container_width=True,
                on_click=lambda target=page: st.session_state.update({"navigation": target}),
            )
    st.markdown("---")
    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("Your library")
        if records:
            for record in records[:5]:
                st.text(record.filename)
                st.caption(f"{record.chunk_count} indexed passages · added {record.created_at[:10]}")
        else:
            st.info("Your library is empty. Open My Documents to add lecture notes.")
            st.button(
                "Add study material",
                key="dashboard_upload",
                on_click=lambda: st.session_state.update({"navigation": "My Documents"}),
            )
    with right:
        st.subheader("Recent study activity")
        events = st.session_state.get("study_history", [])
        if events:
            for event in events[:6]:
                st.text(event["title"])
                st.caption(f"{event['kind']} · {event['time']}")
        else:
            st.caption("Your recent chats, summaries, and quizzes will appear here.")


def render_documents(settings: Settings, store: DocumentStore) -> None:
    st.title("My Documents")
    st.write("Import and manage local study files. Uploaded content stays on this computer.")
    uploaded = st.file_uploader(
        "Choose study material",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help=f"PDF, DOCX, and TXT files up to {settings.max_upload_mb} MB each.",
    )
    if st.button("Index selected files", type="primary", disabled=not uploaded):
        if uploaded:
            _index_uploads(uploaded, settings, store)
    st.markdown("---")
    records = store.list_documents()
    st.subheader(f"Library · {len(records)} document(s)")
    if not records:
        st.info("No documents yet. Add a PDF, DOCX, or TXT file above.")
        return

    for record in records:
        with st.expander(
            f"Document {record.document_id[:8]} · {record.chunk_count} passages",
            expanded=False,
        ):
            st.text(record.filename)
            st.caption(f"Indexed {record.created_at[:10]} · document ID {record.document_id[:12]}")
            col_reindex, col_delete = st.columns([1, 1])
            with col_reindex:
                if st.button("Re-index from local copy", key=f"reindex_{record.document_id}"):
                    try:
                        with st.spinner(f"Re-indexing {record.filename}…"):
                            updated = store.reindex_document(
                                record.document_id,
                                st.session_state.get("chunk_size", settings.chunk_size),
                                st.session_state.get("chunk_overlap", settings.chunk_overlap),
                            )
                        st.success(f"Re-indexed {updated.chunk_count} passages.")
                        st.text(updated.filename)
                        st.rerun()
                    except Exception as exc:
                        _friendly_error(exc)
            with col_delete:
                confirmed = st.checkbox(
                    "Confirm removal",
                    key=f"confirm_delete_{record.document_id}",
                )
                if st.button(
                    "Delete document",
                    key=f"delete_{record.document_id}",
                    disabled=not confirmed,
                ):
                    try:
                        store.delete_document(record.document_id)
                        st.success("Removed the document and its local index.")
                        st.text(record.filename)
                        st.rerun()
                    except Exception as exc:
                        _friendly_error(exc)


def render_chat(
    settings: Settings,
    store: DocumentStore,
    ollama: OllamaClient,
    model: str,
    temperature: float,
    max_tokens: int,
) -> None:
    st.title("AI Study Chat")
    st.caption("Answers are grounded in selected passages from your documents.")
    records = store.list_documents()
    if not records:
        st.info("Add study documents before asking questions.")
        return
    labels = _document_options(records)
    selected_labels = st.multiselect(
        "Search within",
        list(labels),
        default=list(labels),
        key="chat_document_selection",
    )
    selected_ids = [labels[label] for label in selected_labels]
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("New conversation", key="new_chat"):
            st.session_state.chat_messages = []
            st.rerun()
    with col2:
        st.caption(f"Model: {model}")

    messages = st.session_state.setdefault("chat_messages", [])
    for item in messages[-100:]:
        with st.chat_message(item["role"]):
            if item["role"] == "user":
                st.text(item["content"])
            else:
                st.markdown(item["content"])
            if item["role"] == "assistant":
                _source_panel(item.get("sources", []))
    question = st.chat_input("Ask about your selected study materials", max_chars=2000)
    if not question:
        return
    if not selected_ids:
        st.warning("Select at least one document to search.")
        return

    messages.append({"role": "user", "content": question})
    if len(messages) > 100:
        del messages[:-100]
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your notes and preparing an answer…"):
                history = [
                    {"role": item["role"], "content": item["content"]}
                    for item in messages[:-1][-4:]
                ]
                result = answer_question(
                    question,
                    store,
                    ollama,
                    model,
                    document_ids=selected_ids,
                    history=history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    retrieval_count=st.session_state.get(
                        "retrieval_count", settings.retrieval_count
                    ),
                    relevance_threshold=st.session_state.get(
                        "relevance_threshold", settings.relevance_threshold
                    ),
                )
            st.markdown(result.answer)
            _source_panel(result.sources)
            st.code(result.answer, language=None)
            messages.append(
                {
                    "role": "assistant",
                    "content": result.answer,
                    "sources": list(result.sources),
                }
            )
            _activity("Chat", question[:100], "Grounded" if result.grounded else "Insufficient evidence in selected notes.")
        except Exception as exc:
            messages.pop()
            _friendly_error(exc)


def render_summary(
    settings: Settings,
    store: DocumentStore,
    ollama: OllamaClient,
    model: str,
    temperature: float,
    max_tokens: int,
) -> None:
    st.title("Summarizer")
    st.write("Create a structured revision summary of one indexed document.")
    records = store.list_documents()
    if not records:
        st.info("Add and index a document before requesting a summary.")
        return
    labels = _document_options(records)
    label = st.selectbox("Document", list(labels), key="summary_document")
    topic_scope = st.text_input(
        "Optional chapter or topic",
        max_chars=200,
        placeholder="Leave blank to summarize the whole document",
    )
    st.caption("A chapter/topic scope summarizes retrieved passages from that document.")
    mode = st.selectbox("Summary style", ["Concise", "Detailed", "Exam revision"])
    if st.button("Generate summary", type="primary"):
        st.session_state.summary_result = None
        try:
            with st.spinner("Reading the document and building a grounded summary…"):
                summary = summarize_document(
                    labels[label],
                    store,
                    ollama,
                    model,
                    mode=mode,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    chunk_size=st.session_state.get("chunk_size", settings.chunk_size),
                    chunk_overlap=st.session_state.get(
                        "chunk_overlap", settings.chunk_overlap
                    ),
                    topic=topic_scope.strip() or None,
                    retrieval_count=st.session_state.get(
                        "retrieval_count", settings.retrieval_count
                    ),
                    relevance_threshold=st.session_state.get(
                        "relevance_threshold", settings.relevance_threshold
                    ),
                )
            st.session_state.summary_result = {
                "document": label,
                "mode": mode,
                "topic": topic_scope.strip(),
                "text": summary,
            }
            _activity(
                "Summary",
                labels[label],
                f"{mode} · {topic_scope.strip() or 'whole document'}",
            )
        except Exception as exc:
            _friendly_error(exc)
    result = st.session_state.get("summary_result")
    if result:
        st.markdown("---")
        st.markdown(result["text"])
        stem = Path(result["document"].split(" · ")[0]).stem or "study-summary"
        safe_stem = "".join(ch for ch in stem if ch.isalnum() or ch in "-_ ")[:60].strip()
        st.download_button(
            "Download summary as Markdown",
            data=(
                f"# {result['document']} — {result['mode']} summary\n\n"
                f"Scope: {result['topic'] or 'whole document'}\n\n{result['text']}"
            ),
            file_name=f"{safe_stem or 'study-summary'}-summary.md",
            mime="text/markdown",
        )


def render_quiz(
    settings: Settings,
    store: DocumentStore,
    ollama: OllamaClient,
    model: str,
    temperature: float,
    max_tokens: int,
) -> None:
    st.title("Quiz Generator")
    st.write("Create practice questions from the evidence in your selected notes.")
    records = store.list_documents()
    if not records:
        st.info("Add study documents before creating a quiz.")
        return
    labels = _document_options(records)
    selected_labels = st.multiselect(
        "Use documents",
        list(labels),
        default=list(labels),
        key="quiz_document_selection",
    )
    with st.form("quiz_setup"):
        topic = st.text_input("Topic", max_chars=200, placeholder="e.g. cellular respiration")
        col_count, col_level = st.columns(2)
        question_count = col_count.slider("Number of questions", 1, 10, 5)
        difficulty = col_level.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
        make_quiz = st.form_submit_button("Generate quiz", type="primary")
    if make_quiz:
        st.session_state.quiz_state = None
        selected_ids = [labels[item] for item in selected_labels]
        try:
            with st.spinner("Finding relevant notes and validating quiz questions…"):
                questions = generate_quiz(
                    topic,
                    store,
                    ollama,
                    model,
                    document_ids=selected_ids,
                    question_count=question_count,
                    difficulty=difficulty,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    retrieval_count=st.session_state.get(
                        "retrieval_count", settings.retrieval_count
                    ),
                    relevance_threshold=st.session_state.get(
                        "relevance_threshold", settings.relevance_threshold
                    ),
                )
                passages = store.search(
                    topic,
                    document_ids=selected_ids,
                    count=st.session_state.get("retrieval_count", settings.retrieval_count),
                    threshold=st.session_state.get(
                        "relevance_threshold", settings.relevance_threshold
                    ),
                )
            st.session_state.quiz_state = {
                "quiz_id": uuid4().hex,
                "topic": topic,
                "questions": questions,
                "sources": {source.source_id: source for source in passages},
                "answers": {},
                "current": 0,
                "complete": False,
                "saved": False,
            }
            _activity("Quiz", topic, f"{len(questions)} questions · {difficulty}")
        except Exception as exc:
            _friendly_error(exc)
    state = st.session_state.get("quiz_state")
    if not state:
        st.caption("Each question is validated against quoted evidence from your study materials.")
        return
    questions: tuple[QuizQuestion, ...] = state["questions"]
    current = state["current"]
    st.text(state["topic"])
    if not state["complete"]:
        question = questions[current]
        st.progress(current / len(questions), text=f"Question {current + 1} of {len(questions)}")
        st.text(question.question)
        with st.form(f"quiz_answer_{state['quiz_id']}_{current}"):
            selected = st.radio(
                "Choose an answer",
                [None, 0, 1, 2, 3],
                format_func=lambda value: (
                    "Skip this question"
                    if value is None
                    else f"{'ABCD'[value]}. {question.options[value]}"
                ),
                index=0,
                key=f"quiz_choice_{state['quiz_id']}_{current}",
            )
            submitted = st.form_submit_button("Submit answer")
        if submitted:
            state["answers"][current] = selected
            state["current"] += 1
            if state["current"] >= len(questions):
                state["complete"] = True
            st.rerun()
        if question.source_ids:
            for source_id in question.source_ids:
                source = state["sources"].get(source_id)
                if source:
                    st.text(
                        f"Evidence source: {source.filename}"
                        f"{f', page {source.page_number}' if source.page_number else ''}"
                    )
    else:
        score, total = score_quiz(questions, state["answers"])
        st.success(f"Quiz complete · {score} / {total} correct")
        for index, question in enumerate(questions):
            selected = state["answers"].get(index)
            is_correct = selected == question.answer_index
            label = "Unanswered" if selected is None else ("Correct" if is_correct else "Review")
            with st.expander(f"Question {index + 1} · {label}"):
                st.text(question.question)
                st.text(f"Your answer: {'No answer' if selected is None else question.options[selected]}")
                st.text(f"Correct answer: {question.options[question.answer_index]}")
                st.text(question.explanation)
                st.text(f"Evidence: “{question.evidence}”")
                _source_panel(
                    [
                        state["sources"][source_id]
                        for source_id in question.source_ids
                        if source_id in state["sources"]
                    ]
                )
        if not state["saved"]:
            _activity("Quiz completed", state["topic"], f"Score: {score}/{total}")
            state["saved"] = True
        if st.button("Start another quiz"):
            st.session_state.quiz_state = None
            st.rerun()
        st.caption("Model-generated questions may need academic verification.")


def render_research() -> None:
    st.title("Web Research")
    st.caption("External web results are separate from your uploaded study notes.")
    st.info(
        "Search sends only the query you enter to the search provider. Study documents are "
        "never included in search queries."
    )
    with st.form("web_search"):
        query = st.text_input("Research question", max_chars=300)
        count = st.slider("Maximum results", 1, 10, 5)
        submitted = st.form_submit_button("Search the web", type="primary")
    if submitted:
        st.session_state.pop("web_results", None)
        try:
            with st.spinner("Searching the web…"):
                results = search_web(query, count)
            st.session_state.web_results = results
            st.session_state.web_query = query
            _activity("Web research", query[:100], f"{len(results)} result(s)")
        except Exception as exc:
            _friendly_error(exc)
    if "web_results" not in st.session_state:
        return
    results = st.session_state.web_results
    st.subheader("Search results")
    st.text(st.session_state.get("web_query", ""))
    if not results:
        st.info("No results were returned. Try a more specific query.")
        return
    for result in results:
        st.text(result.title)
        st.caption(result.domain)
        st.text(result.snippet or "No search snippet was provided.")
        st.link_button("Open original source", result.url)
        st.markdown("---")
    st.caption("Results show search-engine titles and snippets; the articles have not been opened or verified.")


def render_history() -> None:
    st.title("Study History")
    st.caption("Activity in this view is held in the current Streamlit session only.")
    events = st.session_state.get("study_history", [])
    if not events:
        st.info("Your recent study activity will appear here.")
        return
    if st.button("Clear study activity"):
        st.session_state.study_history = []
        st.rerun()
    for event in events:
        st.text(event["title"])
        st.caption(f"{event['kind']} · {event['time']}")
        st.text(event["details"])
        st.markdown("---")


def render_settings(settings: Settings, store: DocumentStore, status: Any) -> None:
    st.title("Settings")
    st.write("Manage local retrieval behavior and understand where your study files are stored.")
    st.subheader("Retrieval and chunking")
    st.session_state.setdefault("retrieval_count", settings.retrieval_count)
    st.session_state.setdefault("relevance_threshold", settings.relevance_threshold)
    st.session_state.setdefault("chunk_size", settings.chunk_size)
    st.session_state.setdefault("chunk_overlap", settings.chunk_overlap)
    st.slider("Retrieved passages", 1, 20, key="retrieval_count")
    st.slider("Minimum relevance (cosine similarity)", -1.0, 1.0, step=0.05, key="relevance_threshold")
    st.slider("Chunk size in characters", 200, 6000, step=100, key="chunk_size")
    max_overlap = min(1500, st.session_state.chunk_size - 1)
    if st.session_state.chunk_overlap > max_overlap:
        st.session_state.chunk_overlap = max_overlap
    st.slider("Chunk overlap in characters", 0, max_overlap, step=20, key="chunk_overlap")
    if st.session_state.chunk_overlap >= st.session_state.chunk_size:
        st.warning("Chunk overlap must remain smaller than chunk size.")
    st.caption("Chunking changes apply to new uploads and re-indexing; existing chunks are unchanged until re-indexed.")
    st.markdown("---")
    st.subheader("Local services")
    st.text(f"Ollama: {'Available' if status.available else 'Not reachable'}")
    if status.available:
        st.text("Installed models: " + (", ".join(status.models) or "None detected"))
    else:
        st.caption("Start the Ollama app to use chat, summaries, and quizzes.")
    st.text(f"Local data directory: {settings.data_dir}")
    st.text(f"Embedding model: {settings.embedding_model} (CPU)")
    st.write("The first embedding run may download the model files to the local machine.")
    st.markdown("---")
    st.subheader("Remove imported study material")
    st.warning("This permanently deletes indexed vectors and local uploaded copies.")
    confirmed = st.checkbox("I understand this removes all imported study material.")
    if st.button(
        "Remove all study documents",
        disabled=not confirmed or not store.list_documents(),
        type="secondary",
    ):
        try:
            for record in store.list_documents():
                store.delete_document(record.document_id)
            st.success("All indexed study documents and their local copies were removed.")
            st.rerun()
        except Exception as exc:
            _friendly_error(exc)


def render_page(
    page: str,
    settings: Settings,
    store: DocumentStore,
    ollama: OllamaClient,
    model: str,
    temperature: float,
    max_tokens: int,
    model_status: Any,
) -> None:
    if page == "Dashboard":
        render_dashboard(store, model_status)
    elif page == "My Documents":
        render_documents(settings, store)
    elif page == "AI Study Chat":
        render_chat(settings, store, ollama, model, temperature, max_tokens)
    elif page == "Summarizer":
        render_summary(settings, store, ollama, model, temperature, max_tokens)
    elif page == "Quiz Generator":
        render_quiz(settings, store, ollama, model, temperature, max_tokens)
    elif page == "Web Research":
        render_research()
    elif page == "Study History":
        render_history()
    elif page == "Settings":
        render_settings(settings, store, model_status)
