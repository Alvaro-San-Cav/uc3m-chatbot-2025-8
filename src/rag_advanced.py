"""RAG Pipeline - Dual Mode (Flash & Thinking)

Modes:
- FLASH: Fast responses (1-2 LLM calls, ~2-3s)
- THINKING: Deep verification with iterative refinement (5-8 LLM calls, ~20-30s)

Features:
1. Source citations in every response
2. "No information" when docs don't answer
3. Response in same language as question
4. Verbose mode for debugging
"""

from enum import Enum
from typing import Dict, Generator, Any, List, Callable, Tuple
from operator import itemgetter
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import json
from langdetect import detect, LangDetectException

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableGenerator
)
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.chat_history import get_session_history
from src.llm_client import call_llm_api, call_llm_api_full


# --- Enums ---

class RAGMode(Enum):
    FLASH = "flash"
    THINKING = "thinking"


class QuestionType(Enum):
    """Question types for response format customization."""
    YES_NO = "yes_no"                # ¿Tiene NextEra proyectos en Texas? → SÍ/NO + justificación
    COMPARATIVE = "comparative"      # ¿Cómo se comparan los costos de NextEra vs RWE? → Tabla/lista comparativa
    AGGREGATION = "aggregation"      # ¿Cuál es el costo promedio por MW? → Valor agregado + desglose
    FACTUAL = "factual"             # ¿Cuál es el Security Amount del proyecto X? → Dato específico + contexto
    LISTING = "listing"             # ¿Qué proyectos tiene NextEra? → Lista estructurada
    TEMPORAL = "temporal"           # ¿Cómo han cambiado los costos desde 2020? → Análisis temporal/tendencias
    DEFINITIONAL = "definitional"   # ¿Qué es un SGIA? → Definición + contexto del corpus
    GENERAL = "general"             # Pregunta estándar sin formato especial


# =============================================================================
# CONFIGURATION - Adjust these parameters as needed
# =============================================================================

class RAGConfig:
    """Centralized configuration for RAG pipeline parameters."""
    
    # --- Document Retrieval ---
    K_DOCS_DEFAULT = 15          # Default docs to retrieve from vector store
    
    # --- Flash Mode ---
    FLASH_MAX_SOURCES = 8        # Max documents to include in Flash response
    
    # --- Thinking Mode ---
    THINKING_MAX_QUERIES = 4            # Max query variants for expansion (1 original + 3 generated)
    THINKING_RELEVANCE_CHECK_DOCS = 20  # How many docs to check for relevance
    THINKING_MAX_RELEVANT_DOCS = 12     # Max relevant docs to keep for response
    THINKING_MAX_CLAIMS = 6             # Max claims to extract and verify
    THINKING_CONFIDENCE_THRESHOLD = 0.7 # Threshold for triggering refinement
    
    # --- Parallelization ---
    RETRIEVAL_WORKERS = 4        # Parallel workers for multi-query retrieval
    RELEVANCE_WORKERS = 10       # Parallel workers for relevance checking
    VERIFICATION_WORKERS = 6     # Parallel workers for claim verification


# Global config instance
config = RAGConfig()


# --- Precompiled Regex Patterns ---

# Patterns for meta-commentary removal (used in clean_response)
META_COMMENTARY_PATTERNS = [
    re.compile(r"(?i)^Note:.*$"),
    re.compile(r"(?i)^I have removed.*$"),
    re.compile(r"(?i)^I added a qualifier.*$"),
    re.compile(r"(?i)^I removed.*$"),
    re.compile(r"(?i)^This revised response.*$"),
    re.compile(r"(?i)^Based on the provided source documents, here is a revised.*$"),
    re.compile(r"(?i)^Here is a revised response.*$"),
    re.compile(r"(?i).*maintains accuracy with the source.*$"),
    re.compile(r"(?i).*cannot be verified against.*$"),
    re.compile(r"(?i).*as these statements cannot be verified.*$"),
]

# Pattern for duplicate source section removal
SOURCE_SECTION_PATTERN = re.compile(
    r'\n\n(?:Sources?|Fuentes?|References?):?\s*\n(?:\s*\[?\d+\]?[^\n]*\n?)*\s*$',
    re.IGNORECASE
)

# Pattern for consecutive newlines
CONSECUTIVE_NEWLINES_PATTERN = re.compile(r'\n{3,}')


# --- Verbose Logger ---

class VerboseLogger:
    """Handles verbose output for debugging/progress visibility."""
    
    def __init__(self, enabled: bool = True, callback: Callable[[str], None] = None):
        self.enabled = enabled
        self.callback = callback or print
    
    def log(self, step: str, message: str, emoji: str = ""):
        if self.enabled:
            prefix = f"{emoji} " if emoji else ""
            self.callback(f"{prefix}[{step}] {message}")
    
    def step(self, message: str):
        self.log("STEP", message, "🔄")
    
    def success(self, message: str):
        self.log("OK", message, "✓")
    
    def warning(self, message: str):
        self.log("WARN", message, "⚠")
    
    def info(self, message: str):
        self.log("INFO", message, "ℹ")


# Global logger instance (can be replaced)
_verbose_logger = VerboseLogger(enabled=False)


def set_verbose(enabled: bool, callback: Callable[[str], None] = None):
    """Enable/disable verbose mode globally."""
    global _verbose_logger
    _verbose_logger = VerboseLogger(enabled=enabled, callback=callback)


def get_logger() -> VerboseLogger:
    return _verbose_logger


# --- Helper Functions ---

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return 'spanish' if lang == 'es' else 'english'
    except LangDetectException:
        return 'english'


def format_sources(docs, max_sources: int = None) -> Dict[str, Any]:
    """Returns dict with formatted context string and source metadata list.
    
    Args:
        docs: List of documents
        max_sources: Optional limit on number of sources (default: no limit)
    """
    if not docs:
        return {"context": "", "sources": [], "has_docs": False}

    # Apply source limit if specified
    if max_sources:
        docs = docs[:max_sources]

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


# --- Prompts ---

SYSTEM_EN = """You are an expert analyst of ERCOT Standard Generation Interconnection Agreements (SGIAs).
Answer questions using ONLY the provided context from SGIA documents.

DOCUMENT STRUCTURE AWARENESS:
- Article 5: Interconnection Facilities (equipment specs, network upgrades)
- Article 11: Security Amounts (financial deposits/guarantees)
- Annex A: Facility Description (project specs, location)
- Annex B: Detailed Cost Tables (itemized costs per MW, upgrade costs)
- Annex C: Milestone Schedules (construction timelines, deadlines)

TERMINOLOGY:
- ERCOT: Electric Reliability Council of Texas (grid operator)
- PUCT: Public Utility Commission of Texas (regulator)
- INR: Interconnection Request Number (unique project ID)
- FIS: Facilities Study (engineering study before IA)
- Network Upgrades: Grid improvements required for new projects
- Security Amount: Financial deposit required from developer

RESPONSE FORMAT:
1. ALWAYS start your response with: "Based on the researched material, "
2. Then provide a clear categorical answer (YES/NO) if the question asks for confirmation
3. Provide supporting details with source citations
4. When comparing across projects, present data in structured format
5. Include relevant metrics: capacity (MW), costs ($), dates, zones

MULTI-DOCUMENT ANALYSIS:
- If the question requires aggregation (averages, comparisons, trends), synthesize data from ALL relevant sources
- For comparative questions, clearly organize findings by: Developer, Technology (SOL/WIN/OTH/GAS), Time Period, or Zone
- Identify patterns across multiple SGIAs when answering trend questions

RULES:
- If context doesn't contain the answer, say: "I don't have information about that in the available documents."
- Cite sources using [Source N] format for EACH claim
- When data varies across sources, report the range and cite all relevant sources
- Do NOT include meta-commentary about what you removed or changed
- Do NOT hallucinate specific numbers, dates, or names not in the context

Context:
{context}"""

SYSTEM_ES = """Eres un analista experto en Acuerdos Estándar de Interconexión de Generación (SGIAs) de ERCOT.
Responde preguntas usando SOLO el contexto proporcionado de documentos SGIA.
IMPORTANTE: Los documentos están en inglés, pero debes responder en español.

ESTRUCTURA DE DOCUMENTOS:
- Artículo 5: Instalaciones de Interconexión (especificaciones de equipos, mejoras de red)
- Artículo 11: Montos de Garantía (depósitos financieros)
- Anexo A: Descripción de Instalación (especificaciones del proyecto, ubicación)
- Anexo B: Tablas de Costos Detalladas (costos por MW, costos de mejoras)
- Anexo C: Cronogramas de Hitos (plazos de construcción)

TERMINOLOGÍA:
- ERCOT: Electric Reliability Council of Texas (operador de red)
- PUCT: Public Utility Commission of Texas (regulador)
- INR: Interconnection Request Number (ID único del proyecto)
- FIS: Facilities Study (estudio de ingeniería previo)
- Network Upgrades: Mejoras de red requeridas
- Security Amount: Depósito financiero del desarrollador

FORMATO DE RESPUESTA:
1. SIEMPRE comienza tu respuesta con: "Basándome en el material investigado, "
2. Luego proporciona respuesta categórica (SÍ/NO) si la pregunta pide confirmación
3. Proporciona detalles con citas de fuentes
4. Para comparaciones entre proyectos, presenta datos de forma estructurada
5. Incluye métricas relevantes: capacidad (MW), costos ($), fechas, zonas

ANÁLISIS MULTI-DOCUMENTO:
- Si la pregunta requiere agregación (promedios, comparaciones, tendencias), sintetiza datos de TODAS las fuentes relevantes
- Para preguntas comparativas, organiza hallazgos por: Desarrollador, Tecnología (SOL/WIN/OTH/GAS), Período, o Zona
- Identifica patrones entre múltiples SGIAs al responder preguntas de tendencias

REGLAS:
- Si el contexto no contiene la respuesta, di: "No tengo información sobre eso en los documentos disponibles."
- Cita fuentes usando [Fuente N] para CADA afirmación
- Cuando los datos varíen entre fuentes, reporta el rango y cita todas las fuentes
- NO incluyas meta-comentarios sobre lo que eliminaste o cambiaste
- NO inventes números, fechas o nombres específicos que no estén en el contexto

Contexto:
{context}"""

# Domain check prompt (for out-of-scope filtering)
DOMAIN_CHECK_PROMPT = """Is this question related to ANY of the following topics about Texas energy infrastructure?

IN-SCOPE TOPICS:
- ERCOT (Electric Reliability Council of Texas) operations or agreements
- Power grid interconnection agreements (SGIAs, IAs)
- Energy project development (solar SOL, wind WIN, battery storage OTH, gas GAS)
- Specific developers: NextEra Energy, RWE Renewables, and other energy companies
- Transmission service providers (TSPs) and network upgrades
- Security deposits, guarantees, or financial requirements for projects
- Project costs, timelines, milestones, or specifications
- PUCT (Public Utility Commission of Texas) filings
- Interconnection Request Numbers (INRs) or specific project identifiers
- Geographic zones or counties in Texas energy grid

OUT-OF-SCOPE:
- General energy policy not specific to Texas/ERCOT
- Residential electricity rates or consumer questions
- Non-Texas utilities or grid operators

Question: {question}

Answer with ONLY "YES" or "NO":"""

# Rephrasing prompt (for chat history)
REPHRASE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Reformulate the question to be standalone given the chat history. Return only the reformulated question."),
    ("placeholder", "{chat_history}"),
    ("human", "{question}"),
])

# --- Question Type Classification ---

QUESTION_TYPE_PROMPT = """Classify this question about ERCOT SGIAs into ONE of these categories:

QUESTION TYPES:
1. YES_NO - Questions expecting confirmation/denial
   Examples: "Does NextEra have projects in Texas?", "Is the security amount higher than $1M?"

2. COMPARATIVE - Questions comparing entities, projects, or values
   Examples: "How do NextEra costs compare to RWE?", "Which developer has more projects?"

3. AGGREGATION - Questions asking for summaries, averages, totals
   Examples: "What is the average cost per MW?", "How many solar projects are there?"

4. FACTUAL - Questions asking for specific data points
   Examples: "What is the security amount for project X?", "Where is the Brazoria project located?"

5. LISTING - Questions asking for lists or enumerations
   Examples: "What projects does NextEra have?", "List all battery storage projects"

6. TEMPORAL - Questions about changes over time or trends
   Examples: "How have costs changed since 2020?", "What is the timeline for project X?"

7. DEFINITIONAL - Questions asking for explanations of terms/concepts
   Examples: "What is a SGIA?", "What does INR mean?", "Explain network upgrades"

8. GENERAL - Questions that don't fit other categories clearly
   Examples: Open-ended questions, multi-part questions, ambiguous queries

Question: {question}

Answer with ONLY the category name (e.g., "YES_NO" or "COMPARATIVE"):"""

# Response format templates per question type
RESPONSE_FORMAT_TEMPLATES = {
    "YES_NO": {
        "en": """RESPONSE FORMAT FOR YES/NO QUESTION:
1. Start with a clear **YES** or **NO** (bold)
2. Follow with 1-2 sentences of justification citing sources
3. If partial/conditional, state "PARTIALLY" with explanation

Example format:
**YES** - [Brief justification with source citations]""",
        
        "es": """FORMATO PARA PREGUNTA SÍ/NO:
1. Comienza con un claro **SÍ** o **NO** (negrita)
2. Sigue con 1-2 frases de justificación citando fuentes
3. Si es parcial/condicional, indica "PARCIALMENTE" con explicación

Formato ejemplo:
**SÍ** - [Breve justificación con citas de fuentes]"""
    },
    
    "COMPARATIVE": {
        "en": """RESPONSE FORMAT FOR COMPARATIVE QUESTION:
1. Start with a summary statement of the comparison result
2. Present a structured comparison (use table or bullet points):
   | Aspect | Entity A | Entity B |
   |--------|----------|----------|
3. Highlight key differences and cite sources for each data point
4. Conclude with the main takeaway""",
        
        "es": """FORMATO PARA PREGUNTA COMPARATIVA:
1. Comienza con un resumen del resultado de la comparación
2. Presenta una comparación estructurada (tabla o viñetas):
   | Aspecto | Entidad A | Entidad B |
   |---------|-----------|-----------|
3. Destaca diferencias clave y cita fuentes para cada dato
4. Concluye con la conclusión principal"""
    },
    
    "AGGREGATION": {
        "en": """RESPONSE FORMAT FOR AGGREGATION QUESTION:
1. State the aggregate value prominently (bold the number)
2. Show the breakdown/components that led to this value
3. Include the sample size (N=X projects/documents)
4. Note any outliers or important caveats
5. Cite all sources used in the calculation""",
        
        "es": """FORMATO PARA PREGUNTA DE AGREGACIÓN:
1. Indica el valor agregado prominentemente (número en negrita)
2. Muestra el desglose/componentes que llevaron a este valor
3. Incluye el tamaño de muestra (N=X proyectos/documentos)
4. Nota cualquier valor atípico o advertencia importante
5. Cita todas las fuentes usadas en el cálculo"""
    },
    
    "FACTUAL": {
        "en": """RESPONSE FORMAT FOR FACTUAL QUESTION:
1. State the specific fact/data point directly and prominently
2. Provide brief context (project, date, section of document)
3. Cite the exact source
4. If multiple values exist, list all with their sources""",
        
        "es": """FORMATO PARA PREGUNTA FACTUAL:
1. Indica el dato específico directamente y prominentemente
2. Proporciona contexto breve (proyecto, fecha, sección del documento)
3. Cita la fuente exacta
4. Si existen múltiples valores, lista todos con sus fuentes"""
    },
    
    "LISTING": {
        "en": """RESPONSE FORMAT FOR LISTING QUESTION:
1. State the total count first (e.g., "There are X projects:")
2. Present items as a numbered or bulleted list
3. For each item, include key identifiers (name, INR, type)
4. Group by category if applicable (by developer, technology, zone)
5. Cite sources for each item""",
        
        "es": """FORMATO PARA PREGUNTA DE LISTADO:
1. Indica el conteo total primero (ej: "Hay X proyectos:")
2. Presenta elementos como lista numerada o con viñetas
3. Para cada elemento, incluye identificadores clave (nombre, INR, tipo)
4. Agrupa por categoría si aplica (por desarrollador, tecnología, zona)
5. Cita fuentes para cada elemento"""
    },
    
    "TEMPORAL": {
        "en": """RESPONSE FORMAT FOR TEMPORAL QUESTION:
1. State the overall trend or change first
2. Present a timeline or chronological breakdown:
   - 2018-2020: [description]
   - 2021-2022: [description]
   - 2023-2024: [description]
3. Quantify changes where possible (%, absolute values)
4. Cite sources for each time period mentioned""",
        
        "es": """FORMATO PARA PREGUNTA TEMPORAL:
1. Indica la tendencia o cambio general primero
2. Presenta una línea temporal o desglose cronológico:
   - 2018-2020: [descripción]
   - 2021-2022: [descripción]
   - 2023-2024: [descripción]
3. Cuantifica cambios donde sea posible (%, valores absolutos)
4. Cita fuentes para cada período mencionado"""
    },
    
    "DEFINITIONAL": {
        "en": """RESPONSE FORMAT FOR DEFINITIONAL QUESTION:
1. Provide a clear, concise definition first
2. Explain relevance to ERCOT/SGIAs context
3. Give an example from the corpus if available
4. Cite sources if definition comes from documents""",
        
        "es": """FORMATO PARA PREGUNTA DEFINITIONAL:
1. Proporciona una definición clara y concisa primero
2. Explica relevancia en contexto ERCOT/SGIAs
3. Da un ejemplo del corpus si está disponible
4. Cita fuentes si la definición viene de documentos"""
    },
    
    "GENERAL": {
        "en": """RESPONSE FORMAT FOR GENERAL QUESTION:
1. Address the question directly
2. Structure information logically with headers if needed
3. Include relevant data with citations
4. Be comprehensive but concise""",
        
        "es": """FORMATO PARA PREGUNTA GENERAL:
1. Aborda la pregunta directamente
2. Estructura la información lógicamente con encabezados si es necesario
3. Incluye datos relevantes con citas
4. Sé completo pero conciso"""
    }
}

# --- THINKING MODE PROMPTS ---

QUERY_EXPANSION_PROMPT = """You are a search expert for ERCOT Standard Generation Interconnection Agreements (SGIAs).
Given this user question, generate 3 alternative search queries to find relevant information.

GUIDELINES FOR QUERY GENERATION:
1. Use SGIA-specific terminology:
   - Security Amount, Security Deposit, Guarantee → financial requirements
   - Network Upgrades, Transmission Upgrades → grid improvements
   - INR (Interconnection Request Number) → project identifier
   - Capacity (MW), Nameplate Capacity → project size
   - Article 5, Article 11, Annex A/B/C → document sections

2. Consider multiple dimensions:
   - Developer names: NextEra Energy, RWE Renewables, etc.
   - Technology types: solar (SOL), wind (WIN), battery (OTH), gas (GAS)
   - Time periods: 2018-2020, 2021-2022, 2023-2024, 2024-2025
   - Geographic zones and Texas counties

3. Query strategies:
   - If asking about costs → include "cost", "amount", "$", "price"
   - If comparing → include specific entity names for comparison
   - If asking trends → include date ranges or "over time"
   - If asking aggregates → include terms like "average", "total", "typical"

User question: {question}

Return ONLY the 3 queries, one per line, without numbering or explanation:"""

RELEVANCE_CHECK_PROMPT = """Evaluate if this SGIA document excerpt is relevant for answering the question.
Note: The question may be in Spanish but documents are in English. Evaluate semantic relevance.

CONSIDER RELEVANT IF THE DOCUMENT:
- Contains data requested (costs, dates, capacities, companies)
- Is from the same developer, technology type, or time period being asked about
- Contains comparative data useful for aggregation or benchmarking
- Has information from the relevant SGIA section (Article 5/11, Annex A/B/C)

CONSIDER IRRELEVANT IF:
- The document is about a completely different topic
- It lacks any factual data that could contribute to the answer
- It's from a different entity/period than specifically requested

Question: {question}

Document excerpt:
{doc_content}

Answer with ONLY "YES" or "NO":"""

CLAIM_EXTRACTION_PROMPT = """Extract the key factual claims from this response about ERCOT SGIAs.
Focus on extractable, verifiable facts.

TYPES OF CLAIMS TO EXTRACT:
- Numeric claims: costs ($X), capacities (X MW), dates, percentages
- Entity claims: specific developer names, project names, zones, counties
- Comparative claims: "higher than", "average of", "more expensive"
- Existence claims: "project X exists", "NextEra has Y projects"
- Temporal claims: "in 2024", "since 2020", "changed from A to B"

DO NOT extract:
- General knowledge statements ("ERCOT operates the Texas grid")
- Vague qualitative statements without specifics
- Hedged statements ("may", "could", "possibly")

Response: {response}

Claims (one per line, be specific):"""

CLAIM_VERIFICATION_PROMPT = """Verify if this specific claim about ERCOT SGIAs is supported by the provided documents.

Claim: {claim}

VERIFICATION CRITERIA:
- VERIFIED: The exact fact (number, name, date) appears in the documents OR can be directly calculated/inferred from explicit data
- UNVERIFIED: The claim cannot be confirmed - data is missing, documents don't mention it, or information is insufficient
- CONTRADICTED: Documents explicitly state something different (different number, different entity, different date)

For numeric claims:
- Verify the exact number matches or is correctly calculated from source data
- Check if units match (MW, $, months, etc.)

For comparative claims:
- Verify both compared values are in the documents
- Check if the comparison direction is correct

Documents:
{context}

Your answer (VERIFIED/UNVERIFIED/CONTRADICTED):"""

REFINEMENT_PROMPT = """Rewrite this response about ERCOT SGIAs using ONLY verifiable information.

Original response: {original_response}

These claims could NOT be verified:
{unverified_claims}

Source documents:
{context}

REFINEMENT RULES:
- Remove unverified claims entirely - do not hedge them with "possibly" or "may"
- Keep only statements directly supported by sources
- If removing a claim creates an incomplete comparison, remove the entire comparison
- Preserve accurate numeric data with their sources
- Maintain the original response structure where possible
- Cite sources using [Source N] format for each remaining claim

DO NOT:
- Explain what you removed or why
- Add meta-commentary about limitations
- Introduce new information not in sources
- Soften unverified claims instead of removing them

Revised answer:"""

RESPONSE_VALIDATION_PROMPT = """Evaluate if this response is coherent with the question and follows the expected format.

Question: {question}
Question Type: {question_type}
Response: {response}

VALIDATION CRITERIA:

1. COHERENCE CHECK:
   - Does the response directly address what the question is asking?
   - Is the response relevant to the ERCOT/SGIA domain?
   - Are the claims in the response logically connected to the question?

2. FORMAT COMPLIANCE (based on question type):
   - YES_NO: Must start with clear YES/NO/SÍ/NO or PARTIALLY
   - COMPARATIVE: Must include structured comparison (table, side-by-side, or explicit contrast)
   - AGGREGATION: Must include aggregate value with breakdown or sample size
   - FACTUAL: Must provide specific data point with source
   - LISTING: Must present items as list with count
   - TEMPORAL: Must show chronological or trend information
   - DEFINITIONAL: Must provide clear definition
   - GENERAL: Must address the question directly

3. QUALITY CHECK:
   - Are sources cited?
   - Is the response complete (not cut off)?
   - Does it avoid hallucinations or unsupported claims?

Respond with a JSON object:
{{
    "is_coherent": true/false,
    "format_compliant": true/false,
    "issues": ["list of specific issues if any"],
    "suggested_fix": "brief suggestion if issues found, or null if OK"
}}

Your evaluation (JSON only):"""


# --- Domain Filter ---

def is_domain_relevant(question: str) -> bool:
    """Check if question is about ERCOT/energy domain (fast filter for out-of-scope)."""
    logger = get_logger()
    logger.step("Checking if question is in ERCOT domain...")
    
    prompt = DOMAIN_CHECK_PROMPT.format(question=question)
    
    try:
        result = call_llm_api_full(prompt).strip().upper()
    except Exception as e:
        logger.warning(f"Domain check failed: {e}, allowing question to proceed")
        return True  # Default to allowing the question on error
    
    is_relevant = "YES" in result
    if is_relevant:
        logger.success("Question is domain-relevant")
    else:
        logger.info("Question is OUT OF SCOPE (not ERCOT-related)")
    
    return is_relevant


# --- FLASH MODE ---


def classify_question(question: str) -> QuestionType:
    """Classify question type for response format customization."""
    logger = get_logger()
    logger.step("Classifying question type...")
    
    prompt = QUESTION_TYPE_PROMPT.format(question=question)
    
    try:
        result = call_llm_api_full(prompt).strip().upper()
    except Exception as e:
        logger.warning(f"Question classification failed: {e}, defaulting to GENERAL")
        return QuestionType.GENERAL
    
    # Parse result - handle variations
    type_mapping = {
        "YES_NO": QuestionType.YES_NO,
        "YESNO": QuestionType.YES_NO,
        "YES/NO": QuestionType.YES_NO,
        "COMPARATIVE": QuestionType.COMPARATIVE,
        "COMPARISON": QuestionType.COMPARATIVE,
        "AGGREGATION": QuestionType.AGGREGATION,
        "AGGREGATE": QuestionType.AGGREGATION,
        "FACTUAL": QuestionType.FACTUAL,
        "FACT": QuestionType.FACTUAL,
        "LISTING": QuestionType.LISTING,
        "LIST": QuestionType.LISTING,
        "TEMPORAL": QuestionType.TEMPORAL,
        "TIME": QuestionType.TEMPORAL,
        "TREND": QuestionType.TEMPORAL,
        "DEFINITIONAL": QuestionType.DEFINITIONAL,
        "DEFINITION": QuestionType.DEFINITIONAL,
        "GENERAL": QuestionType.GENERAL,
    }
    
    # Try to match the result
    for key, qtype in type_mapping.items():
        if key in result:
            logger.success(f"Question type: {qtype.value}")
            return qtype
    
    # Default fallback
    logger.info("Question type: GENERAL (default)")
    return QuestionType.GENERAL


def get_format_instructions(question_type: QuestionType, lang: str) -> str:
    """Get format instructions for a specific question type and language."""
    lang_key = "es" if lang == "spanish" else "en"
    template = RESPONSE_FORMAT_TEMPLATES.get(question_type.value.upper(), RESPONSE_FORMAT_TEMPLATES["GENERAL"])
    return template.get(lang_key, template["en"])


def validate_response(question: str, question_type: QuestionType, response: str, context: str, lang: str) -> Tuple[str, bool]:
    """Validate response coherence and format compliance. Returns (response, was_fixed)."""
    logger = get_logger()
    logger.step("Validating response coherence and format...")
    
    prompt = RESPONSE_VALIDATION_PROMPT.format(
        question=question,
        question_type=question_type.value.upper(),
        response=response[:2000]  # Limit response length for validation
    )
    
    try:
        result = call_llm_api_full(prompt).strip()
    except Exception as e:
        logger.warning(f"Validation LLM call failed: {e}")
        return response, False
    
    # Try to parse JSON response
    try:
        # Clean up common JSON issues
        if result.startswith("```json"):
            result = result[7:]
        if result.startswith("```"):
            result = result[3:]
        if result.endswith("```"):
            result = result[:-3]
        result = result.strip()
        
        validation = json.loads(result)
        is_coherent = validation.get("is_coherent", True)
        format_compliant = validation.get("format_compliant", True)
        issues = validation.get("issues", [])
        suggested_fix = validation.get("suggested_fix")
        
        if is_coherent and format_compliant:
            logger.success("Response validation: PASSED ✓")
            return response, False
        
        # Log issues
        logger.warning(f"Response validation: ISSUES FOUND")
        for issue in issues:
            logger.info(f"  - {issue}")
        
        # If there are issues, try to fix the response
        if suggested_fix and (not is_coherent or not format_compliant):
            logger.step("Attempting to fix response based on validation feedback...")
            
            fix_prompt = f"""Fix this response to address the following issues:

Original Question: {question}
Question Type: {question_type.value.upper()}
Original Response: {response}

Issues found:
{chr(10).join(f'- {issue}' for issue in issues)}

Suggested fix: {suggested_fix}

Source documents for reference:
{context[:2000]}

RULES:
- Address the specific issues mentioned
- Maintain source citations
- Follow the format expected for {question_type.value.upper()} questions
- Do NOT add meta-commentary about the fix

Fixed response:"""
            
            fixed_response = call_llm_api_full(fix_prompt)
            fixed_response = clean_response(fixed_response)  # Remove duplicate sources
            logger.success("Response fixed based on validation feedback")
            return fixed_response, True
        
        return response, False
        
    except json.JSONDecodeError:
        logger.info("Could not parse validation response, assuming OK")
        return response, False

def contextualize_question(input_dict: Dict) -> str:
    """Uses blocking call to reformulate question if history exists."""
    logger = get_logger()
    
    if not input_dict.get("chat_history"):
        return input_dict["question"]

    logger.step("Reformulating question based on chat history...")
    prompt_val = REPHRASE_PROMPT.invoke(input_dict)
    result = call_llm_api_full(prompt_val.to_string())
    logger.success(f"Reformulated: {result[:50]}...")
    return result


def generate_flash_response(input_dict: Dict) -> Generator[str, None, None]:
    """Flash mode: Direct response generation with minimal processing."""
    logger = get_logger()
    question = input_dict["question"]
    retrieval = input_dict["retrieval"]
    history = input_dict.get("chat_history", [])

    # 1. Detect Language
    lang = detect_language(question)
    logger.info(f"Language detected: {lang}")

    # 2. Handle No Documents
    if not retrieval["has_docs"]:
        msg = ("No tengo información sobre eso en los documentos disponibles."
               if lang == 'spanish'
               else "I don't have information about that in the available documents.")
        yield msg
        return

    # 3. Classify Question Type
    question_type = classify_question(question)
    format_instructions = get_format_instructions(question_type, lang)
    
    # 4. Construct Prompt with format instructions
    context = retrieval["context"]
    system_template = SYSTEM_ES if lang == 'spanish' else SYSTEM_EN
    
    # Inject format instructions into the system template
    enhanced_system = f"{system_template}\n\n{format_instructions}"

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", enhanced_system),
        ("placeholder", "{chat_history}"),
        ("human", "{question}")
    ])

    prompt_str = prompt_template.invoke({
        "context": context,
        "chat_history": history,
        "question": question
    }).to_string()

    logger.step("Generating response...")
    
    # 5. Stream Response
    for token in call_llm_api(prompt_str):
        yield token

    logger.success("Response generated")

    # 6. Append Citations
    citations = format_citations(retrieval["sources"])
    if citations:
        yield citations


# --- THINKING MODE ---

def expand_query(question: str) -> List[str]:
    """Generate multiple query variants for broader retrieval."""
    logger = get_logger()
    logger.step("Expanding query into multiple search variants...")
    
    prompt = QUERY_EXPANSION_PROMPT.format(question=question)
    result = call_llm_api_full(prompt)
    
    queries = [q.strip() for q in result.strip().split('\n') if q.strip()]
    queries = [question] + queries[:3]  # Original + up to 3 variants
    
    logger.success(f"Generated {len(queries)} query variants")
    for i, q in enumerate(queries):
        logger.info(f"  Query {i+1}: {q[:60]}...")
    
    return queries


def multi_retrieve(queries: List[str], retriever) -> List:
    """Retrieve documents for multiple queries in parallel and merge results."""
    logger = get_logger()
    logger.step(f"Retrieving documents for {len(queries)} queries in parallel...")
    
    all_docs = []
    seen_contents = set()
    
    def retrieve_single(query: str) -> List:
        return retriever.invoke(query)
    
    # Parallelize retrieval for all queries
    with ThreadPoolExecutor(max_workers=config.RETRIEVAL_WORKERS) as executor:
        futures = [executor.submit(retrieve_single, q) for q in queries]
        
        for future in as_completed(futures):
            docs = future.result()
            for doc in docs:
                content_hash = hash(doc.page_content[:200])
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    all_docs.append(doc)
    
    logger.success(f"Retrieved {len(all_docs)} unique documents")
    return all_docs


def _check_single_relevance(args: tuple) -> tuple:
    """Check relevance of a single document (helper for parallel execution)."""
    doc, question = args
    prompt = RELEVANCE_CHECK_PROMPT.format(
        question=question,
        doc_content=doc.page_content[:500]
    )
    result = call_llm_api_full(prompt).strip().upper()
    is_relevant = "YES" in result
    return doc, is_relevant


def check_relevance(question: str, docs: List) -> List:
    """Filter documents by LLM-judged relevance (parallelized)."""
    logger = get_logger()
    logger.step("Checking document relevance in parallel...")
    
    max_docs_to_check = config.THINKING_RELEVANCE_CHECK_DOCS
    docs_to_check = docs[:max_docs_to_check]
    
    relevant_docs = []
    
    # Parallelize relevance checks
    with ThreadPoolExecutor(max_workers=config.RELEVANCE_WORKERS) as executor:
        args_list = [(doc, question) for doc in docs_to_check]
        futures = {executor.submit(_check_single_relevance, args): args[0] for args in args_list}
        
        for future in as_completed(futures):
            doc, is_relevant = future.result()
            if is_relevant:
                relevant_docs.append(doc)
    
    logger.success(f"Kept {len(relevant_docs)}/{len(docs_to_check)} relevant documents")
    # Return up to max relevant docs, or fallback if none found
    max_docs = config.THINKING_MAX_RELEVANT_DOCS
    return relevant_docs[:max_docs] if relevant_docs else docs[:max_docs]


def extract_claims(response: str) -> List[str]:
    """Extract factual claims from a response."""
    logger = get_logger()
    logger.step("Extracting claims from response...")
    
    prompt = CLAIM_EXTRACTION_PROMPT.format(response=response)
    result = call_llm_api_full(prompt)
    
    claims = [c.strip() for c in result.strip().split('\n') if c.strip() and len(c.strip()) > 10]
    logger.success(f"Extracted {len(claims)} claims")
    return claims[:config.THINKING_MAX_CLAIMS]


def _verify_single_claim(claim: str, context: str) -> tuple:
    """Verify a single claim (helper for parallel execution)."""
    prompt = CLAIM_VERIFICATION_PROMPT.format(
        claim=claim,
        context=context[:3000]
    )
    result = call_llm_api_full(prompt).strip().upper()
    
    if "VERIFIED" in result:
        status = "VERIFIED"
    elif "CONTRADICTED" in result:
        status = "CONTRADICTED"
    else:
        status = "UNVERIFIED"
    
    return claim, status


def verify_claims(claims: List[str], context: str) -> Dict[str, str]:
    """Verify each claim against the source documents (parallelized)."""
    logger = get_logger()
    logger.step(f"Verifying {len(claims)} claims in parallel...")
    
    results = {}
    
    # Use ThreadPoolExecutor for parallel verification
    with ThreadPoolExecutor(max_workers=config.VERIFICATION_WORKERS) as executor:
        futures = {executor.submit(_verify_single_claim, claim, context): claim for claim in claims}
        
        for future in as_completed(futures):
            claim, status = future.result()
            results[claim] = status
            logger.info(f"  [{status}] {claim[:50]}...")
    
    verified = sum(1 for v in results.values() if v == "VERIFIED")
    logger.success(f"Verification complete: {verified}/{len(claims)} verified")
    
    return results


def clean_response(text: str) -> str:
    """Remove meta-commentary and duplicate source sections from LLM responses."""
    # Remove trailing source section using precompiled pattern
    text = SOURCE_SECTION_PATTERN.sub('', text)
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        is_meta = False
        line_stripped = line.strip()
        for pattern in META_COMMENTARY_PATTERNS:
            if pattern.match(line_stripped):
                is_meta = True
                break
        if not is_meta:
            cleaned_lines.append(line)
    
    # Remove consecutive empty lines using precompiled pattern
    result = '\n'.join(cleaned_lines)
    result = CONSECUTIVE_NEWLINES_PATTERN.sub('\n\n', result)
    return result.strip()


def refine_response(original: str, unverified: List[str], context: str) -> str:
    """Refine response to address unverified claims."""
    logger = get_logger()
    logger.step("Refining response to address unverified claims...")
    
    prompt = REFINEMENT_PROMPT.format(
        original_response=original,
        unverified_claims="\n".join(f"- {c}" for c in unverified),
        context=context[:3000]
    )
    
    refined = call_llm_api_full(prompt)
    # Clean any meta-commentary the LLM may have added
    refined = clean_response(refined)
    logger.success("Response refined")
    return refined


def generate_thinking_response(input_dict: Dict, retriever) -> Generator[str, None, None]:
    """Thinking mode: Deep verification with iterative refinement."""
    logger = get_logger()
    question = input_dict["question"]
    history = input_dict.get("chat_history", [])
    
    logger.info("=" * 50)
    logger.info("THINKING MODE ACTIVATED")
    logger.info("=" * 50)
    
    # 1. Detect Language
    lang = detect_language(question)
    logger.info(f"Language: {lang}")
    
    # 2. Classify Question Type
    question_type = classify_question(question)
    format_instructions = get_format_instructions(question_type, lang)
    
    # 3. Query Expansion
    queries = expand_query(question)
    
    # 4. Multi-Query Retrieval
    all_docs = multi_retrieve(queries, retriever)
    
    if not all_docs:
        msg = ("No tengo información sobre eso en los documentos disponibles."
               if lang == 'spanish'
               else "I don't have information about that in the available documents.")
        yield msg
        return
    
    # 5. Relevance Filtering
    relevant_docs = check_relevance(question, all_docs)
    retrieval = format_sources(relevant_docs)
    context = retrieval["context"]
    
    # 6. Initial Response
    logger.step("Generating initial response...")
    system_template = SYSTEM_ES if lang == 'spanish' else SYSTEM_EN
    
    # Inject format instructions into the system template
    enhanced_system = f"{system_template}\n\n{format_instructions}"
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", enhanced_system),
        ("placeholder", "{chat_history}"),
        ("human", "{question}")
    ])
    
    prompt_str = prompt_template.invoke({
        "context": context,
        "chat_history": history,
        "question": question
    }).to_string()
    
    initial_response = call_llm_api_full(prompt_str)
    initial_response = clean_response(initial_response)  # Remove duplicate sources from LLM
    logger.success("Initial response generated")
    
    # 7. Claim Extraction & Verification
    claims = extract_claims(initial_response)
    
    if claims:
        verification = verify_claims(claims, context)
        
        unverified = [c for c, status in verification.items() if status != "VERIFIED"]
        verified_count = len(claims) - len(unverified)
        confidence = verified_count / len(claims) if claims else 1.0
        
        # 8. Refinement Loop (max 1 iteration)
        if unverified and confidence < 0.7:
            logger.warning(f"Low confidence ({confidence:.0%}), refining response...")
            final_response = refine_response(initial_response, unverified, context)
        else:
            final_response = initial_response
    else:
        final_response = initial_response
        confidence = 0.8  # Default confidence if no claims extracted
    
    # 9. Validate response coherence and format
    final_response, was_fixed = validate_response(
        question=question,
        question_type=question_type,
        response=final_response,
        context=context,
        lang=lang
    )
    
    logger.info("=" * 50)
    logger.success(f"THINKING COMPLETE - Confidence: {confidence:.0%}")
    logger.info("=" * 50)
    
    # 10. Build structured response with thought summary + categorical answer
    confidence_label = "High" if confidence >= 0.8 else "Medium" if confidence >= 0.5 else "Low"
    verified_count = len(claims) - len(unverified) if claims else 0
    question_type_label = question_type.value.replace("_", " ").title()
    validation_status = "Fixed ⚠" if was_fixed else "Passed ✓"
    
    # Thought process summary
    if lang == 'spanish':
        thought_summary = f"""**💭 Proceso de análisis:**
- Tipo de pregunta: {question_type_label}
- Consultas generadas: {len(queries)}
- Documentos recuperados: {len(all_docs)}
- Documentos relevantes: {len(relevant_docs)}
- Claims verificados: {verified_count}/{len(claims) if claims else 0}
- Confianza: {confidence_label} ({confidence:.0%})
- Validación: {"Corregida ⚠" if was_fixed else "Pasada ✓"}

---

**📋 Respuesta:**
"""
    else:
        thought_summary = f"""**💭 Analysis process:**
- Question type: {question_type_label}
- Queries generated: {len(queries)}
- Documents retrieved: {len(all_docs)}
- Relevant documents: {len(relevant_docs)}
- Claims verified: {verified_count}/{len(claims) if claims else 0}
- Confidence: {confidence_label} ({confidence:.0%})
- Validation: {validation_status}

---

**📋 Answer:**
"""
    
    # Stream thought summary + final response
    yield thought_summary
    yield final_response
    
    # 9. Add citations
    citations = format_citations(retrieval["sources"])
    if citations:
        yield citations


# --- Chain Builders ---

def get_flash_chain(retriever, with_history: bool = True):
    """Build Flash mode RAG chain (fast, 1-2 LLM calls)."""
    logger = get_logger()
    logger.info("Building FLASH mode chain")
    
    def flash_with_domain_filter(input_dict: Dict) -> Generator[str, None, None]:
        """Flash generator with domain pre-filter."""
        question = input_dict["question"]
        history = input_dict.get("chat_history", [])
        lang = detect_language(question)
        
        # Domain pre-filter: skip retrieval for out-of-scope questions
        if not is_domain_relevant(question):
            msg = ("Esta pregunta no está relacionada con acuerdos de interconexión ERCOT. "
                   "Solo puedo responder preguntas sobre proyectos de energía, redes eléctricas y ERCOT."
                   if lang == 'spanish'
                   else "This question is not related to ERCOT interconnection agreements. "
                   "I can only answer questions about energy projects, power grids, and ERCOT.")
            yield msg
            return
        
        # Retrieve and format sources
        docs = retriever.invoke(question)
        retrieval = format_sources(docs, max_sources=config.FLASH_MAX_SOURCES)
        
        # Generate response using existing function
        for chunk in generate_flash_response({
            "question": question,
            "retrieval": retrieval,
            "chat_history": history
        }):
            yield chunk
    
    rag_chain_core = (
        RunnablePassthrough.assign(
            question=RunnableLambda(contextualize_question)
        )
        | RunnableLambda(flash_with_domain_filter)
    )
    
    if with_history:
        return RunnableWithMessageHistory(
            rag_chain_core,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )
    return rag_chain_core


def get_thinking_chain(retriever, with_history: bool = True):
    """Build Thinking mode RAG chain (deep verification, 5-8 LLM calls)."""
    logger = get_logger()
    logger.info("Building THINKING mode chain")
    
    def thinking_generator(input_iter):
        """Generator function for RunnableGenerator - yields chunks from thinking response."""
        for input_dict in input_iter:
            # Contextualize question first
            if input_dict.get("chat_history"):
                logger.step("Reformulating question based on chat history...")
                prompt_val = REPHRASE_PROMPT.invoke(input_dict)
                question = call_llm_api_full(prompt_val.to_string())
                logger.success(f"Reformulated: {question[:50]}...")
                input_dict = {**input_dict, "question": question}
            
            # Generate thinking response
            for chunk in generate_thinking_response(input_dict, retriever):
                yield chunk
    
    rag_chain_core = RunnableGenerator(thinking_generator)
    
    if with_history:
        return RunnableWithMessageHistory(
            rag_chain_core,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )
    return rag_chain_core


def get_rag_chain(retriever, mode: RAGMode = RAGMode.FLASH, with_history: bool = True):
    """Get RAG chain with specified mode.
    
    Args:
        retriever: The document retriever
        mode: RAGMode.FLASH or RAGMode.THINKING
        with_history: Whether to include chat history management
    
    Returns:
        The configured RAG chain
    """
    if mode == RAGMode.THINKING:
        return get_thinking_chain(retriever, with_history)
    return get_flash_chain(retriever, with_history)
