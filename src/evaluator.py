"""
RAG Evaluation Script

Metrics:
1. Retrieval: Does it find relevant documents?
2. Faithfulness: Does it only use context (no hallucination)?
3. Answer Relevance: Does it answer the question?
4. Citation: Does it cite sources?
5. Negative Handling: Does it say "no info" for out-of-scope?
6. Latency: Response time
"""

import time
import re
import statistics
from typing import List, Dict, Any

from src.rag import get_rag_chain
from src.vector_store import get_retriever
from src.llm_client import call_llm_api_full

try:
    from bert_score import score as bert_score_fn
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False


class RAGEvaluator:
    def __init__(self, k_docs=10):
        self.k_docs = k_docs
        self.retriever = get_retriever(k_docs=k_docs)
        self.rag_chain = get_rag_chain(self.retriever)
        
        self.refusal_phrases = [
            "no tengo información", "no cuento con información",
            "i don't have information", "documents do not contain",
            "not found in", "no information about"
        ]

    def _judge_faithfulness(self, context: str, response: str) -> float:
        prompt = f"""Context: {context[:2000]}
Response: {response}

Does the response use ONLY information from the context? 
Answer with just 1 (yes) or 0 (no/hallucination)."""
        
        result = call_llm_api_full(prompt).strip()
        match = re.search(r'[01]', result)
        return float(match.group()) if match else 0.0

    def _judge_relevance(self, question: str, response: str) -> float:
        prompt = f"""Question: {question}
Response: {response}

Does the response answer the question?
Answer with just 1 (yes) or 0 (no)."""
        
        result = call_llm_api_full(prompt).strip()
        match = re.search(r'[01]', result)
        return float(match.group()) if match else 0.0

    def _check_citation(self, response: str) -> float:
        return 1.0 if "[source" in response.lower() or "[fuente" in response.lower() else 0.0

    def _check_refusal(self, response: str) -> bool:
        return any(phrase in response.lower() for phrase in self.refusal_phrases)

    def _bert_score(self, prediction: str, reference: str) -> float:
        if not BERT_AVAILABLE or not reference:
            return 0.0
        try:
            P, R, F1 = bert_score_fn([prediction], [reference], lang="en", verbose=False)
            return F1.mean().item()
        except:
            return 0.0

    def evaluate(self, dataset: List[Dict[str, Any]]):
        print(f"\nEvaluating {len(dataset)} test cases...\n")
        
        metrics = {
            "latency": [],
            "faithfulness": [],
            "relevance": [],
            "citation": [],
            "negative_handling": [],
            "bert_score": []
        }

        for i, case in enumerate(dataset):
            question = case["question"]
            is_negative = case.get("is_negative", False)
            reference = case.get("reference_answer", "")
            
            print(f"[{i+1}/{len(dataset)}] {question[:50]}...")
            
            start = time.time()
            
            # Get response
            try:
                config = {"configurable": {"session_id": f"eval_{i}"}}
                response = self.rag_chain.invoke({"question": question}, config=config)
            except Exception as e:
                print(f"  ERROR: {e}")
                response = "ERROR"
            
            latency = time.time() - start
            metrics["latency"].append(latency)
            
            # Get context for faithfulness check
            docs = self.retriever.invoke(question)
            context = "\n".join([d.page_content for d in docs])
            
            if is_negative:
                # Should refuse to answer
                refused = self._check_refusal(response)
                metrics["negative_handling"].append(1.0 if refused else 0.0)
                print(f"  Negative test - Refused: {refused}")
            else:
                # Positive test
                faith = self._judge_faithfulness(context, response)
                rel = self._judge_relevance(question, response)
                cit = self._check_citation(response)
                bert = self._bert_score(response, reference)
                
                metrics["faithfulness"].append(faith)
                metrics["relevance"].append(rel)
                metrics["citation"].append(cit)
                metrics["bert_score"].append(bert)
                
                print(f"  Faith: {faith} | Rel: {rel} | Cit: {cit} | BERT: {bert:.2f}")

        self._report(metrics)

    def _report(self, metrics):
        def avg(lst): 
            return statistics.mean(lst) if lst else 0.0
        
        print("\n" + "="*60)
        print(" EVALUATION REPORT")
        print("="*60)
        print(f"{'Metric':<25} {'Result':<15} {'Target':<10}")
        print("-"*50)
        print(f"{'Faithfulness':<25} {avg(metrics['faithfulness']):.1%}          > 90%")
        print(f"{'Answer Relevance':<25} {avg(metrics['relevance']):.1%}          > 90%")
        print(f"{'Citation Accuracy':<25} {avg(metrics['citation']):.1%}          100%")
        print(f"{'Negative Handling':<25} {avg(metrics['negative_handling']):.1%}          100%")
        print(f"{'BERTScore':<25} {avg(metrics['bert_score']):.2f}           > 0.85")
        print(f"{'Avg Latency':<25} {avg(metrics['latency']):.2f}s          < 5s")
        print("="*60)


# Test dataset
TEST_DATASET = [
    # Positive tests (should answer from documents)
    {
        "question": "What are the security deposit requirements for interconnection?",
        "reference_answer": "The Generator shall provide security as specified in Exhibit A.",
        "is_negative": False
    },
    {
        "question": "What are the milestone requirements in Article 5?",
        "reference_answer": "Article 5 specifies milestones including commercial operation date.",
        "is_negative": False
    },
    {
        "question": "What equipment is required for the Point of Interconnection?",
        "reference_answer": "Equipment requirements are in Exhibit B.",
        "is_negative": False
    },
    {
        "question": "Cuales son los requisitos de seguridad?",
        "reference_answer": "El generador debe proporcionar seguridad segun Exhibit A.",
        "is_negative": False
    },
    
    # Negative tests (should refuse to answer)
    {
        "question": "What is the capital of France?",
        "reference_answer": "",
        "is_negative": True
    },
    {
        "question": "Cual es la receta de la paella?",
        "reference_answer": "",
        "is_negative": True
    }
]


def run_evaluation(k_docs: int = 10, dataset: List[Dict[str, Any]] = None):
    """
    Run RAG evaluation with the default or custom dataset.
    
    Args:
        k_docs: Number of documents to retrieve
        dataset: Custom test dataset (uses TEST_DATASET if None)
    
    Returns:
        RAGEvaluator instance after evaluation
    """
    if dataset is None:
        dataset = TEST_DATASET
    
    print("\n" + "="*60)
    print("📊 RAG EVALUATION")
    print("="*60)
    
    evaluator = RAGEvaluator(k_docs=k_docs)
    evaluator.evaluate(dataset)
    return evaluator


if __name__ == "__main__":
    run_evaluation()
