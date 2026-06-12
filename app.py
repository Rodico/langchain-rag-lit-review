"""
Streamlit web interface for the Literature Review RAG system.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📚 Literature Review RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models…")
def get_components():
    from src.ingestion import DocumentChunker, DocumentLoader
    from src.retrieval import VectorStoreManager
    return DocumentLoader(), DocumentChunker(), VectorStoreManager()


@st.cache_resource(show_spinner="Initialising Q&A chain…")
def get_qa_chain(_vsm):
    from src.generation import LiteratureQAChain
    return LiteratureQAChain(_vsm.get_retriever(), session_id="streamlit")


@st.cache_resource(show_spinner="Initialising review generator…")
def get_review_generator(_vsm):
    from src.generation import ReviewGenerator
    return ReviewGenerator(_vsm.get_retriever())


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 Lit Review RAG")
    st.caption("Powered by LangChain + ChromaDB + OpenAI")

    api_key = st.text_input("OpenAI API Key", type="password", key="api_key")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()
    st.subheader("📂 Ingest Papers")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    arxiv_id = st.text_input("or arXiv ID (e.g. 2303.08774)")

    if st.button("⚡ Ingest", use_container_width=True):
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("Please enter your OpenAI API key first.")
        else:
            loader, chunker, vsm = get_components()
            with st.spinner("Processing…"):
                all_docs = []

                for f in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix) as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name
                    try:
                        docs = loader.load_file(tmp_path)
                        all_docs.extend(docs)
                    except Exception as e:
                        st.warning(f"Could not load {f.name}: {e}")
                    finally:
                        os.unlink(tmp_path)

                if arxiv_id.strip():
                    try:
                        docs = loader.load_arxiv(arxiv_id.strip())
                        all_docs.extend(docs)
                    except Exception as e:
                        st.warning(f"arXiv fetch failed: {e}")

                if all_docs:
                    chunks = chunker.split(all_docs)
                    chunks = chunker.filter_short_chunks(chunks)
                    vsm.add_documents(chunks)
                    st.success(f"✅ Ingested {len(chunks)} chunks from {len(all_docs)} pages!")
                    st.cache_resource.clear()
                else:
                    st.warning("No documents loaded.")

    st.divider()
    if os.environ.get("OPENAI_API_KEY"):
        try:
            _, _, vsm = get_components()
            st.metric("Chunks in store", vsm.count())
        except Exception:
            st.metric("Chunks in store", "—")

# ── Main tabs ──────────────────────────────────────────────────────────────────
tab_chat, tab_review, tab_themes = st.tabs(["💬 Chat with Papers", "📝 Generate Review", "🔍 Explore Themes"])

# ── Tab 1: Chat ────────────────────────────────────────────────────────────────
with tab_chat:
    st.header("💬 Chat with Your Papers")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Sources"):
                    for s in msg["sources"]:
                        st.caption(
                            f"📄 **{s.get('filename', '?')}** — "
                            f"{s.get('title', '')} ({s.get('year', '')}) p.{s.get('page', '?')}"
                        )

    if prompt := st.chat_input("Ask anything about your papers…"):
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("Please enter your OpenAI API key in the sidebar.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving & generating…"):
                    try:
                        _, _, vsm = get_components()
                        qa = get_qa_chain(vsm)
                        result = qa.ask(prompt)
                        answer = result["answer"]
                        sources = result["sources"]
                    except Exception as e:
                        answer = f"⚠️ Error: {e}"
                        sources = []

                st.markdown(answer)
                if sources:
                    with st.expander("📎 Sources"):
                        for s in sources:
                            st.caption(
                                f"📄 **{s.get('filename', '?')}** — "
                                f"{s.get('title', '')} ({s.get('year', '')}) p.{s.get('page', '?')}"
                            )

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ── Tab 2: Generate Review ─────────────────────────────────────────────────────
with tab_review:
    st.header("📝 Generate Literature Review")

    col1, col2 = st.columns([2, 1])
    with col1:
        topic = st.text_input("Research topic", placeholder="e.g. transformer attention mechanisms")
    with col2:
        style = st.selectbox("Style", ["thematic", "chronological", "methodological"])

    extra_queries = st.text_area(
        "Additional retrieval queries (one per line, optional)",
        placeholder="evaluation benchmarks\nstate-of-the-art results",
        height=80,
    )
    max_papers = st.slider("Max context chunks", 5, 30, 15)

    if st.button("✍️ Generate Review", use_container_width=True):
        if not topic:
            st.warning("Please enter a topic.")
        elif not os.environ.get("OPENAI_API_KEY"):
            st.error("Please enter your OpenAI API key in the sidebar.")
        else:
            with st.spinner("Generating your literature review…"):
                try:
                    _, _, vsm = get_components()
                    gen = get_review_generator(vsm)
                    extra = [q.strip() for q in extra_queries.splitlines() if q.strip()]
                    review = gen.generate(
                        topic=topic,
                        style=style,
                        max_papers=max_papers,
                        additional_queries=extra or None,
                    )
                    st.session_state["last_review"] = review
                except Exception as e:
                    st.error(f"Generation failed: {e}")

    if "last_review" in st.session_state:
        st.markdown(st.session_state["last_review"])
        st.download_button(
            "⬇️ Download Review (.md)",
            data=st.session_state["last_review"],
            file_name=f"literature_review_{topic.replace(' ', '_')}.md",
            mime="text/markdown",
        )

# ── Tab 3: Themes ──────────────────────────────────────────────────────────────
with tab_themes:
    st.header("🔍 Explore Research Themes")

    theme_topic = st.text_input("Topic to explore themes for", key="theme_topic")
    n_themes = st.slider("Number of themes", 3, 10, 5)

    if st.button("🧠 Extract Themes"):
        if not theme_topic:
            st.warning("Enter a topic.")
        elif not os.environ.get("OPENAI_API_KEY"):
            st.error("Please enter your OpenAI API key in the sidebar.")
        else:
            with st.spinner("Analysing themes…"):
                try:
                    _, _, vsm = get_components()
                    gen = get_review_generator(vsm)
                    themes = gen.extract_themes(theme_topic, n_themes=n_themes)
                    if themes:
                        st.subheader("Identified Themes")
                        for i, theme in enumerate(themes, 1):
                            st.markdown(f"**{i}.** {theme}")
                    else:
                        st.warning("No themes extracted.")
                except Exception as e:
                    st.error(f"Theme extraction failed: {e}")
