"""RRAG Pipeline - Production Grade (with LCEL)
Requirements:
1. Source citations in every response
2. "No information" when docs don't answer
3. Response in same language as question
4. Auto-summarization extension
"""

from typing import Dict, Generator, Any
from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import BaseMessage

# Import existing dependencies
from src.chat_history import get_session_history
from src.llm_client import call_llm_api, call_llm_api_full

# --- 1. Helper Functions ---

def detect_language(text: str) -> str:
    text_lower = text.lower().split()
    spanish_words = {'qué', 'que', 'cómo', 'como', 'cuál', 'cual', 'cuáles',
            'cuales', 'dónde', 'donde', 'por qué', 'por que', 'el', 'la',
            'los', 'proyecto', 'seguridad', 'requisitos', 'cuánto', 'para'}
    spanish_count = sum(1 for w in spanish_words if w in text_lower)
    return 'spanish' if spanish_count >= 2 else 'english'


def format_sources(docs) -> Dict[str, Any]:
    """Returns dict with formatted context string and source metadata list."""
    if not docs:
        return {"context": "", "sources": [], "has_docs": False}

    formatted_parts = []
    source_list = []

    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        project = meta.get('project_name', 'Unknown')
        inr = meta.get('inr', 'N/A')
        section = meta.get('section', 'N/A')

        formatted_parts.append(f"[Source {i}: {project} ({inr}) - {section}]\n{doc.page_content}\n")
        source_list.append({
            'ref': i,
            'project_name': project,
            'inr': inr,
            'section': section,
            'zone': meta.get('zone'),
            'fuel_type': meta.get('fuel_type')
        })

    return {
        "context": "\n".join(formatted_parts),
        "sources": source_list,
        "has_docs": True
    }


def format_citations(sources: list) -> str:
    if not sources:
        return ""
    lines = ["\n\nSources:"]
    for s in sources[:5]:
        lines.append(f"  [{s['ref']}] {s['project_name']} ({s['inr']}) - {s['section']}")
    return "\n".join(lines)


# --- 2. Prompts ---

SYSTEM_EN = """Answer questions about ERCOT interconnection agreements using ONLY the provided context.

RULES:
- If context doesn't contain the answer, say: "I don't have information about that in the available documents."
- Cite sources using [Source N] format
- Be concise

Context:
{context}"""

SYSTEM_ES = """Responde preguntas sobre acuerdos de interconexión ERCOT usando SOLO el contexto proporcionado.

REGLAS:
- Si el contexto no contiene la respuesta, di: "No tengo información sobre eso en los documentos disponibles."
- Cita fuentes usando formato [Fuente N]
- Sé conciso

Contexto:
{context}"""

# Rephrasing prompt
rephrase_prompt = ChatPromptTemplate.from_messages([
    ("system", "Reformulate the question to be standalone given the chat history. Return only the reformulated question."),
    ("placeholder", "{chat_history}"),
    ("human", "{question}"),
])

# Query decomposition (Optional extension)
decomp_prompt = ChatPromptTemplate.from_messages([
    ("system", "Decompose this ERCOT interconnection question into 2-3 simple sub-queries. Return one per line.\n\nQuestion: {question}"),
    ("human", "{question}")
])


# --- 3. Chain Logic ---

def contextualize_question(input_dict: Dict) -> str:
    """Uses blocking call to reformulate question if history exists."""
    if not input_dict.get("chat_history"):
        return input_dict["question"]

    # We manually invoke the template + blocking LLM here to return a clean string
    prompt_val = rephrase_prompt.invoke(input_dict)
    return call_llm_api_full(prompt_val.to_string())


def generate_rag_response(input_dict: Dict) -> Generator[str, None, None]:
    """
    Core generator that streams the LLM response, citations, and summary.
    This serves as the final node in the LCEL chain.
    """
    question = input_dict["question"]  # This is the (potentially) reformulated question
    retrieval = input_dict["retrieval"]
    history = input_dict["chat_history"]
    with_summary = input_dict["config_summary"]

    # 1. Detect Language
    lang = detect_language(question)

    # 2. Handle No Documents
    if not retrieval["has_docs"]:
        msg = ("No tengo información sobre eso en los documentos disponibles."
               if lang == 'spanish'
               else "I don't have information about that in the available documents.")
        yield msg
        return

    # 3. Construct Prompt
    context = retrieval["context"]
    system_template = SYSTEM_ES if lang == 'spanish' else SYSTEM_EN

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("placeholder", "{chat_history}"),
        ("human", "{question}")
    ])

    prompt_str = prompt_template.invoke({
        "context": context,
        "chat_history": history,
        "question": question
    }).to_string()

    # 4. Stream Main LLM Response
    # call_llm_api is a generator
    for token in call_llm_api(prompt_str):
        yield token

    # 5. Append Citations
    citations = format_citations(retrieval["sources"])
    if citations:
        yield citations

    # 6. Auto-summarization (Blocking call injected into stream)
    if with_summary:
        yield "\n\n--- Summary ---\n"
        summary_prompt = f"Summarize key points from these ERCOT documents in 2-3 sentences:\n{context[:3000]}"
        # Using blocking call here, then yielding result
        summary = call_llm_api_full(summary_prompt)
        yield summary


def get_rag_chain(retriever, with_summary: bool = False):
    """
    Builds a declarative LCEL RAG pipeline.

    Structure:
    1. Contextualize Question (Blocking)
    2. Retrieve & Format (Blocking)
    3. Generate Response (Streaming)
    4. Wrapped in Message History
    """

    # -- Step 1: Retrieval Branch --
    # Fetches documents based on the reformulated question
    retrieval_chain = (
        itemgetter("question")
        | retriever
        | RunnableLambda(format_sources)
    )

    # -- Step 2: Main RAG Logic --
    rag_chain_core = (
        RunnablePassthrough.assign(
            # Reformulate question based on history (if present)
            question=RunnableLambda(contextualize_question)
        )
        | RunnablePassthrough.assign(
            # Fetch docs using the (potentially reformulated) question
            retrieval=retrieval_chain,
            config_summary=lambda _: with_summary
        )
        | RunnableLambda(generate_rag_response)
    )

    # -- Step 3: Wrap with History Management --
    # RunnableWithMessageHistory handles loading history and saving the final aggregated response
    final_chain = RunnableWithMessageHistory(
        rag_chain_core,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return final_chain


def get_rag_chain_with_summary(retriever):
    """RAG chain with auto-summarization enabled."""
    return get_rag_chain(retriever, with_summary=True)
