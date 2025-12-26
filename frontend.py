import uuid
import os
from pathlib import Path

import streamlit as st

from src.vector_store import get_retriever
from src.rag import get_rag_chain, get_rag_chain_with_summary
from src import add_files
from pathlib import Path

K_DOCS = 10
UPLOAD_DIR = Path("uploaded_docs")
UPLOAD_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="UC3M RAG Chatbot", page_icon="💬", layout="centered")


@st.cache_resource
def load_chain(k_docs: int, with_summary: bool):
    """Load retriever + chain only once per process."""
    retriever = get_retriever(k_docs=k_docs)
    chain = get_rag_chain_with_summary(retriever) if with_summary else get_rag_chain(retriever)
    return chain


st.title("💬 The ultimate RAG Chatbot")


import re

def format_sources_as_bullets(text: str) -> str:
    """
    Turns:
      Sources: [1] A [2] B [3] C
    into:
      Sources:
      - [1] A
      - [2] B
      - [3] C
    """
    if "Sources:" not in text:
        return text

    head, tail = text.split("Sources:", 1)
    tail = tail.strip()

    # Split at occurrences of [number]
    parts = re.split(r"(?=\[\d+\])", tail)
    parts = [p.strip() for p in parts if p.strip()]

    # If we couldn't split properly, just return as-is
    if len(parts) <= 1:
        return text

    bullet_block = "Sources:\n" + "\n".join([f"- {p}" for p in parts])
    return head.rstrip() + "\n\n" + bullet_block


# -------------------------
# Sidebar: Settings + Upload
# -------------------------
with st.sidebar:
    st.header("OPTIONS")

    with_summary = st.toggle("Auto-summarization", value=False)
    k_docs = st.slider("Number of retrieved documents", 1, 25, K_DOCS)

    if st.button("New chat"):
            st.session_state.clear()
            st.rerun()

    st.divider()

    st.subheader("ADD DOCUMENTS")
    uploaded_files = st.file_uploader(
        "Upload PDF/TXT/MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    # Optional metadata fields (useful for filtering/citations)
    project_name = st.text_input("File name (optional)", value="")
    # section = st.text_input("Section (optional)", value="")

    index_clicked = st.button("Index documents", disabled=not uploaded_files)        

    if index_clicked:
        saved_paths = []
        for uf in uploaded_files:
            save_path = UPLOAD_DIR / uf.name
            with open(save_path, "wb") as f:
                f.write(uf.getbuffer())
            saved_paths.append(str(save_path))

        extra_metadata = {}
        if project_name.strip():
            extra_metadata["project_name"] = project_name.strip()
        # if section.strip():
        #     extra_metadata["section"] = section.strip()

        try:
            print("Indexing documents...")
            print(f"Saved paths: {saved_paths}")
            n_chunks = add_files.upsert_documents_to_chroma(
                paths=saved_paths,
                extra_metadata=extra_metadata if extra_metadata else None
            )

            st.success(f"✅ Indexing completed. Chunks added: {n_chunks}")

            # Clear cached retriever/chain so the new docs are used immediately
            st.toast("File added to upload queue ✅", icon="✅")
            st.cache_resource.clear()
            st.rerun()

        except Exception as e:
            st.error(f"❌ Indexing failed: {e}")


# -------------------------
# Load chain (cached)
# -------------------------
chain = load_chain(k_docs=k_docs, with_summary=with_summary)

# Session id for RunnableWithMessageHistory (src/chat_history.py)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# UI messages (only for rendering)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
user_text = st.chat_input("Type your question...")

if user_text:
    # 1) Render user message
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # 2) Generate response (streaming)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full = ""

        try:
            for chunk in chain.stream(
                {"question": user_text},
                config={"configurable": {"session_id": st.session_state.session_id}},
            ):
                full += str(chunk)
                placeholder.markdown(format_sources_as_bullets(full))

        except Exception as e:
            full = f"❌ Error generating response: {e}"
            placeholder.markdown(full)

    # 3) Save assistant response in UI history
    st.session_state.messages.append({"role": "assistant", "content": full})
