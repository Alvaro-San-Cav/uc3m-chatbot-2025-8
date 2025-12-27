import tempfile
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

import hashlib

def file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

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

    UPLOAD_DIR = Path("uploaded_docs")
    UPLOAD_DIR.mkdir(exist_ok=True)

    # --- state ---
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    if "uploaded_docs" not in st.session_state:
        # Docs that have been indexed (or at least accepted into the uploaded section)
        # list[dict]: {"name": str, "size": int, "saved_path": str, "chunks": int}
        st.session_state.uploaded_docs = []

    st.subheader("ADD DOCUMENTS")

    selected_files = st.file_uploader(
        "Upload PDF/TXT/MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    file_label = st.text_input("File label (optional)", value="")

    index_clicked = st.button("Index documents", disabled=not selected_files)

    if index_clicked and selected_files:
        # 1) Save selected files to disk
        saved_paths = []
        file_ids = []

        for uf in selected_files:
            suffix = Path(uf.name).suffix.lower()  # .pdf, .txt, .md
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uf.getbuffer())
                tmp_path = tmp.name

            saved_paths.append(tmp_path)
            fid = file_sha1(tmp_path)
            file_ids.append(fid)

        extra_metadata = {}
        if file_label.strip():
            extra_metadata["project_name"] = file_label.strip()

        try:
            # 2) Index them
            stats = add_files.upsert_documents_to_chroma(
                paths=saved_paths,
                extra_metadata=extra_metadata if extra_metadata else None
            )

            st.success(f"✅ Indexing completed. Chunks added: {sum(stats.values())}")

            # 3) Move to "Uploaded documents" list (after successful indexing)
            # (we store per-file; chunks returned is total, so we set total or '-' per file)
            for uf, path, fid in zip(selected_files, saved_paths, file_ids):
                st.session_state.uploaded_docs.append({
                    "name": uf.name,
                    "size": uf.size,
                    "saved_path": path,
                    "source_file_id": fid,
                    "chunks": stats.get(path, 0),  # optional: if you later compute per-file chunks, set it here
                })

            # 4) Reset uploader so selected files disappear from Drag&Drop area
            st.session_state.uploader_key += 1

            # 5) Reload retriever/chain so new docs are used
            st.cache_resource.clear()
            st.rerun()

        except Exception as e:
            st.error(f"❌ Indexing failed: {e}")

        finally:
            for p in saved_paths:
                try:
                    os.remove(p)
                except Exception:
                    pass

    st.divider()
    st.markdown("**Uploaded documents**")
    if not st.session_state.uploaded_docs:
        st.caption("No uploaded documents yet.")
    else:
        for i, d in enumerate(st.session_state.uploaded_docs):
            c1, c2 = st.columns([8, 2])
            with c1:
                st.write(f"📄 {d['name']}  *(chunks: {d.get('chunks','-')})*")
            with c2:
                if st.button("🗑️", key=f"del_indexed_{i}", help="Remove from ChromaDB"):
                    try:
                        add_files.delete_document_from_chroma(d["source_file_id"])
                        st.toast("Deleted from index 🗑️", icon="🗑️")

                        st.session_state.uploaded_docs.pop(i)

                        st.cache_resource.clear()
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Delete failed: {e}")

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
