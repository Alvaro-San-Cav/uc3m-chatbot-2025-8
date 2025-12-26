# src/ingest.py
import os
import hashlib
from typing import List, Optional, Dict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader

try:
    from langchain_community.document_loaders import UnstructuredWordDocumentLoader
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

from src.vector_store import get_vectorstore


def _file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_documents(paths: List[str]) -> List[Document]:
    docs: List[Document] = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()

        if ext in [".txt", ".md"]:
            docs.extend(TextLoader(p, encoding="utf-8").load())

        elif ext == ".pdf":
            docs.extend(PyPDFLoader(p).load())

        elif ext in [".docx", ".doc"] and HAS_DOCX:
            docs.extend(UnstructuredWordDocumentLoader(p).load())

        else:
            raise ValueError(f"Formato no soportado o loader no instalado: {p}")

        # añade metadata útil
        file_id = _file_sha1(p)
        for d in docs[-len(docs):]:
            d.metadata.update({
                "source_path": p,
                "source_file_id": file_id,
                "source_name": os.path.basename(p),
            })

    return docs


def split_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def upsert_documents_to_chroma(
    paths: List[str],
    extra_metadata: Optional[Dict[str, str]] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> int:
    """
    Carga archivos, los trocea y los inserta en la Chroma persistida (incremental).
    Devuelve el nº de chunks añadidos.
    """
    vectorstore = get_vectorstore()  # usa CHROMADB_PATH + embeddings del proyecto

    docs = load_documents(paths)

    if extra_metadata:
        for d in docs:
            d.metadata.update(extra_metadata)

    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # IDs estables para evitar duplicados: file_id + page + chunk_index
    ids = []
    for i, c in enumerate(chunks):
        fid = c.metadata.get("source_file_id", "nofile")
        page = c.metadata.get("page", 0)
        ids.append(f"{fid}::p{page}::c{i}")

    vectorstore.add_documents(chunks, ids=ids)

    try:
        vectorstore.persist()
    except Exception:
        pass

    return len(chunks)