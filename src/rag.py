"""RRAG Pipeline - Production Grade (with LCEL)
Requirements:
1. Source citations in every response
2. "No information" when docs don't answer
3. Response in same language as question
4. Auto-summarization extension
"""

from typing import Dict, Generator, Any
from operator import itemgetter
from langdetect import detect, LangDetectException

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
    try:
        lang = detect(text)
        return 'spanish' if lang == 'es' else 'english'
    except LangDetectException:
        return 'english'  # Default fallback


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
    for s in sources:
        lines.append(f"  [{s['ref']}] {s['project_name']} ({s['inr']}) - {s['section']}")
    return "\n".join(lines)


def parse_subqueries(llm_output: str) -> list[str]:
    return [line.strip() for line in llm_output.split('\n') if line.strip()]


def format_qa_pairs(qa_pairs: list) -> str:
    formatted_pairs = [f"Question: {q}\nAnswer: {a}" for q, a in qa_pairs]
    return '\n---\n'.join(formatted_pairs)


# --- 2. Prompts ---

SYSTEM_EN = """Answer questions about ERCOT interconnection agreements using ONLY the provided context.

RULES:
- If context doesn't contain the answer, say: "I don't have information about that in the available documents."
- Cite sources using [Source N] format. Only reference sources from the Context section, not from the Related questions and answers one.
- Be concise

Related questions and answers:
{qa_pairs}

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
    ("system", "Decompose this ERCOT interconnection question into 0-2 simple sub-queries. Return the questions ONLY, one per line.\n\nUser question: {question}\n\nOutput sub-queries (0-2):"),
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

# Query decomposition and sequential Q&A
def decomp_and_answer(input_dict: Dict, retriever) -> str:
    """
    Decomposes the main question into subqueries, then answers them sequentially and adds the Q&A pair to the prompt.
    Returns a formatter string of Q&A pairs.
    """
    question = input_dict["question"]

    # Decompose
    decomp_prompt_str = decomp_prompt.invoke({"question": question}).to_string()
    subqueries = parse_subqueries(call_llm_api_full(decomp_prompt_str))

    # Sequential answering
    qa_pairs = []

    for subq in subqueries:
        # Guardrail for ill-formed subqueries
        if len(subq) < 10 or '?' not in subq:
            continue
        docs = retriever.invoke(subq)
        formatted = format_sources(docs)

        previous_qa_str = format_qa_pairs(qa_pairs)

        # Ask LLM
        # system_template = SYSTEM_ES if lang == 'spanish' else SYSTEM_EN
        system_template = SYSTEM_EN

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", "{question}")
        ])

        prompt_str = prompt_template.invoke({
            "qa_pairs": previous_qa_str,
            "context": formatted["context"],
            "question": subq
        }).to_string()

        ans = call_llm_api_full(prompt_str)
        qa_pairs.append((subq, ans))

    return format_qa_pairs(qa_pairs)

# Final chain
def generate_rag_response(input_dict: Dict) -> Generator[str, None, None]:
    """
    Core generator that streams the LLM response, citations, and summary.
    This serves as the final node in the LCEL chain.
    """
    question = input_dict["question"]  # This is the (potentially) reformulated question
    retrieval = input_dict["retrieval"]
    qa_pairs_str = input_dict["qa_pairs"]
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
        "qa_pairs": qa_pairs_str,
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
    2. Decompose and answer sub-queries (Blocking)
    3. Retrieve & Format (Blocking)
    4. Generate Response (Streaming)
    5. Wrapped in Message History
    """

    # -- Step 1: Retrieval Branch --
    # Fetches documents based on the reformulated question
    retrieval_chain = (
        itemgetter("question")
        | retriever
        | RunnableLambda(format_sources)
    )

    # -- Step 2: Decomposition Branch --
    # Generates sub-queries, answers them, and returns Q&A-pairs string
    decomp_chain = RunnableLambda(lambda x: decomp_and_answer(x, retriever))

    # -- Step 3: Main RAG Logic --
    rag_chain_core = (
        RunnablePassthrough.assign(
            # Reformulate question based on history (if present)
            question=RunnableLambda(contextualize_question)
        )
        | RunnablePassthrough.assign(
            # Fetch docs using the (potentially reformulated) question and its
            # sub-queries and answers
            retrieval=retrieval_chain,
            qa_pairs=decomp_chain,
            config_summary=lambda _: with_summary
        )
        | RunnableLambda(generate_rag_response)
    )

    # -- Step 4: Wrap with History Management --
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
