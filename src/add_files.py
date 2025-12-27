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
) -> Dict[str, int]:
    """
    Index docs and return chunks count per file path.
    """
    vectorstore = get_vectorstore()

    stats: Dict[str, int] = {}
    all_chunks: List[Document] = []
    all_ids: List[str] = []

    for p in paths:
        docs = load_documents([p])
        if extra_metadata:
            for d in docs:
                d.metadata.update(extra_metadata)

        chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        stats[p] = len(chunks)

        for i, c in enumerate(chunks):
            fid = c.metadata.get("source_file_id", "nofile")
            page = c.metadata.get("page", 0)
            all_ids.append(f"{fid}::p{page}::c{i}")
            all_chunks.append(c)

    vectorstore.add_documents(all_chunks, ids=all_ids)
    try:
        vectorstore.persist()
    except Exception:
        pass

    return stats


def delete_document_from_chroma(source_file_id: str) -> int:
    """
    Deletes all chunks belonging to a document (identified by source_file_id)
    from the persistent ChromaDB.
    Returns the number of deleted records if available (otherwise 0).
    """
    vectorstore = get_vectorstore()

    # LangChain's Chroma wrapper doesn't always expose metadata deletes nicely,
    # but the underlying Chroma collection does.
    try:
        # Preferred: delete by metadata filter
        # This works if your Chroma version supports `where`
        result = vectorstore._collection.delete(where={"source_file_id": source_file_id})
        # result may be None depending on version
        return 0 if result is None else 0
    except Exception:
        # Fallback: try deleting by id prefix if where isn't supported.
        # This fallback is limited because we don't know how many chunks exist,
        # so we can't enumerate ids reliably without re-chunking.
        # Better to keep the "where" method above working.
        raise RuntimeError(
            "Could not delete by metadata filter. "
            "Your Chroma version may not support where-delete via _collection."
        )
