import os
import sys
import random
from typing import List, Optional
from langchain_core.documents import Document

# Add the project root to sys.path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_store import get_vectorstore, extract_filters_from_query, similarity_search_with_boost

def format_metadata(metadata: dict) -> str:
    """Pretty prints metadata dictionary in a vertical list."""
    if not metadata:
        return "  No metadata"
    return "\n".join([f"  - {k}: {v}" for k, v in metadata.items()])

def display_docs(docs: List[Document], extra_info: Optional[List[dict]] = None):
    """Prints documents and their metadata in a highly organized block format."""
    print(f"\n{'='*80}")
    print(f" DISPLAYING {len(docs)} DOCUMENTS")
    print(f"{'='*80}")
    
    for i, doc in enumerate(docs):
        # Header with score info if available
        header_parts = [f"DOCUMENT {i+1}"]
        if extra_info:
            info = extra_info[i]
            if 'original_score' in info:
                header_parts.append(f"BOOSTED: {info['score']:.4f}")
                header_parts.append(f"RAW: {info['original_score']:.4f}")
                if info.get('matches', 0) > 0:
                    header_parts.append(f"MATCHES: {info['matches']}")
            elif 'score' in info:
                header_parts.append(f"SCORE: {info['score']:.4f}")
        
        header = " | ".join(header_parts)
        print(f"\n[ {header} ]")
        print("-" * 40)
        
        # Content formatting
        print("CONTENT:")
        content = doc.page_content.strip()
        # Limit content display to prevent massive scrolls, but show enough context
        lines = content.split('\n')
        preview = "\n".join(lines[:10])
        if len(lines) > 10:
            preview += "\n... (truncated)"
        print(f"  {preview}")
        
        # Metadata formatting
        print("\nMETADATA:")
        print(format_metadata(doc.metadata))
        print("-" * 80)
    
    print(f"{'='*80}\n")

def main():
    try:
        vectorstore = get_vectorstore()
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return

    while True:
        print("\n--- ChromaDB Inspector ---")
        print("1. List 5 random documents")
        print("2. Search by similarity")
        print("3. Exit")
        
        choice = input("\nSelect an option: ").strip()
        
        if choice == '1':
            collection = vectorstore._collection
            count = collection.count()
            print(f"\nTotal chunks in collection: {count}")
            
            if count == 0:
                print("Database is empty.")
                continue

            num_to_fetch = min(5, count)
            print(f"Fetching {num_to_fetch} random documents...")
            
            # Get random indices
            all_ids = collection.get(include=[])['ids']
            random_ids = random.sample(all_ids, num_to_fetch)
            
            sample = collection.get(ids=random_ids)
            
            docs = [Document(page_content=c, metadata=m) for c, m in zip(sample['documents'], sample['metadatas'])]
            display_docs(docs)

        elif choice == '2':
            query = input("Enter search query: ").strip()
            if not query:
                continue
            
            # Extract filters just for user feedback
            filters = extract_filters_from_query(query)
            if filters:
                print(f"Detected Keywords for Boosting: {filters}")
            else:
                print("No metadata keywords detected.")

            k = 5
            print(f"\nPerforming REFINED (Soft) search for top {k} matches...")
            
            # Use the new boosting logic
            boosted_results = similarity_search_with_boost(vectorstore, query, k=k)
            
            docs = [r[0] for r in boosted_results]
            extra_info = [{'score': r[1], 'original_score': r[2], 'matches': r[3]} for r in boosted_results]
            
            display_docs(docs, extra_info)

        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
