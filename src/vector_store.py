"""
Vector Store - Loads ChromaDB from Person B pipeline.
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Load from .env or use default relative path
CHROMADB_PATH = os.getenv("CHROMADB_PATH", "./output/chromadb")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sgia_chunks")


def get_vectorstore(chromadb_path: str = None):
    path = chromadb_path or CHROMADB_PATH

    if not Path(path).exists():
        raise FileNotFoundError(f"ChromaDB not found at: {path}")

    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print(f"Loading ChromaDB at: {path}...")
    vectorstore = Chroma(
        persist_directory=path,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME
    )

    print(f"Loaded {vectorstore._collection.count()} chunks")
    return vectorstore


def get_retriever(k_docs: int = 10, filters: dict = None, chromadb_path: str = None):
    vectorstore = get_vectorstore(chromadb_path)
    search_kwargs = {"k": k_docs}

    if filters:
        if len(filters) == 1:
            key, val = list(filters.items())[0]
            search_kwargs["filter"] = {key: {"$eq": val}}
        else:
            search_kwargs["filter"] = {"$and": [{k: {"$eq": v}} for k, v in filters.items()]}

    return vectorstore.as_retriever(search_kwargs=search_kwargs)


def extract_filters_from_query(query: str) -> dict:
    filters = {}
    q = query.lower()

    def has_match(pattern: str) -> bool:
        return bool(re.search(rf"\b({pattern})\b", q, re.IGNORECASE))

    # Zone
    if has_match(r"coast|coastal|houston"):
        filters['zone'] = 'COAST'
    elif has_match(r"west|western"):
        filters['zone'] = 'WEST'
    elif has_match(r"north|northern"):
        filters['zone'] = 'NORTH'
    elif has_match(r"south|southern"):
        filters['zone'] = 'SOUTH'
    elif has_match(r"panhandle"):
        filters['zone'] = 'PANHANDLE'

    # Fuel Type
    if has_match(r"battery|bateria|baterias|storage|bess"):
        filters['fuel_type'] = 'OTH'
    elif has_match(r"solar|solares|pv|sun"):
        filters['fuel_type'] = 'SOL'
    elif has_match(r"wind|viento|vientos"):
        filters['fuel_type'] = 'WIN'
    elif has_match(r"gas|natural gas"):
        filters['fuel_type'] = 'GAS'

    # Technology
    if has_match(r"pv|solar|solares"):
        filters['technology'] = 'PV'
    elif has_match(r"gas turbine|gt|turbina"):
        filters['technology'] = 'GT'

    # TSP (Transmission Service Provider)
    if has_match(r"centerpoint|cnp|cpnt"):
        filters['tsp_normalized'] = 'CENTERPOINT'
    elif has_match(r"oncor"):
        filters['tsp_normalized'] = 'ONCOR'

    # Parent Company
    if has_match(r"nextera"):
        filters['parent_company'] = 'NEXTERA'
    elif has_match(r"rwe"):
        filters['parent_company'] = 'RWE'
    elif has_match(r"engie"):
        filters['parent_company'] = 'ENGIE'

    # County
    if has_match(r"brazoria"):
        filters['county'] = 'Brazoria'
    elif has_match(r"harris"):
        filters['county'] = 'Harris'
    elif has_match(r"matagorda"):
        filters['county'] = 'Matagorda'
    elif has_match(r"bell"):
        filters['county'] = 'Bell'

    return filters


def similarity_search_with_boost(
    vectorstore, 
    query: str, 
    k: int = 10, 
    boost_factor: float = 0.8,
    k_initial: int = 50
) -> list:
    """
    Performs a soft-filtered search. Instead of excluding docs, it boosts 
    those that match the query's metadata filters.
    """
    filters = extract_filters_from_query(query)
    
    # 1. Wide search
    results = vectorstore.similarity_search_with_score(query, k=k_initial)
    
    # 2. Apply boosting
    boosted_results = []
    for doc, score in results:
        original_score = score
        match_count = 0
        
        # Check metadata matches
        for key, val in filters.items():
            if doc.metadata.get(key) == val:
                match_count += 1
        
        # Apply boost if there's a match (lower distance is better)
        if match_count > 0:
            effective_boost = boost_factor ** match_count
            score = score * effective_boost
            
        boosted_results.append((doc, score, original_score, match_count))
    
    # 3. Re-sort by boosted score
    boosted_results.sort(key=lambda x: x[1])
    
    return boosted_results[:k]


class SmartRetriever:
    """
    A retriever that uses boosted similarity search based on query metadata.
    Compatible with LCEL through the invoke() method.
    """
    
    def __init__(
        self, 
        vectorstore, 
        k: int = 15, 
        boost_factor: float = 0.8,
        k_initial: int = 50
    ):
        self.vectorstore = vectorstore
        self.k = k
        self.boost_factor = boost_factor
        self.k_initial = k_initial
    
    def invoke(self, query: str) -> list:
        """LCEL-compatible invoke method."""
        return self._search(query)
    
    def __call__(self, query: str) -> list:
        """Allow direct calling."""
        return self._search(query)
    
    def _search(self, query: str) -> list:
        """Perform boosted similarity search."""
        boosted_results = similarity_search_with_boost(
            vectorstore=self.vectorstore,
            query=query,
            k=self.k,
            boost_factor=self.boost_factor,
            k_initial=self.k_initial
        )
        # Return only the documents (not scores)
        return [doc for doc, score, orig_score, match_count in boosted_results]
    
    def get_relevant_documents(self, query: str) -> list:
        """LangChain retriever interface compatibility."""
        return self._search(query)


def get_smart_retriever(
    k_docs: int = 15, 
    chromadb_path: str = None,
    boost_factor: float = 0.8,
    k_initial: int = 50
) -> SmartRetriever:
    """
    Creates a SmartRetriever that uses boosted similarity search.
    
    The retriever automatically detects metadata filters from the query
    and boosts matching documents instead of hard-filtering.
    
    Args:
        k_docs: Number of documents to return
        chromadb_path: Path to ChromaDB
        boost_factor: Multiplier for matching docs (lower = more boost, since distance is minimized)
        k_initial: Initial pool size for re-ranking
    
    Returns:
        SmartRetriever instance compatible with LCEL
    """
    vectorstore = get_vectorstore(chromadb_path)
    return SmartRetriever(
        vectorstore=vectorstore,
        k=k_docs,
        boost_factor=boost_factor,
        k_initial=k_initial
    )


def get_hybrid_retriever(
    k_docs: int = 15,
    chromadb_path: str = None,
    use_smart: bool = True,
    boost_factor: float = 0.8
):
    """
    Factory function to get either smart (boosted) or standard retriever.
    
    Args:
        k_docs: Number of documents to return
        chromadb_path: Path to ChromaDB
        use_smart: If True, uses SmartRetriever with boosting
        boost_factor: Boost factor for smart retriever
    
    Returns:
        Retriever instance
    """
    if use_smart:
        return get_smart_retriever(k_docs=k_docs, chromadb_path=chromadb_path, boost_factor=boost_factor)
    else:
        return get_retriever(k_docs=k_docs, chromadb_path=chromadb_path)

