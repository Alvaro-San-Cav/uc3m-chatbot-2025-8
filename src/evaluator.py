"""
SCRIPT DE EVALUACIÓN ROBUSTA PARA SISTEMAS RAG (Retrieval-Augmented Generation)
-------------------------------------------------------------------------------
Propósito:
    Este script automatiza la validación de la calidad del chatbot UC3M mediante 
    la ejecución de un conjunto de pruebas ("Golden Dataset") y el cálculo 
    de métricas cuantitativas y cualitativas.

Métricas Evaluadas:
    1. Retrieval Recall & MRR: ¿El sistema encuentra los documentos pdf correctos?
    2. Faithfulness (Fidelidad): ¿La respuesta se basa SOLO en el contexto (sin alucinaciones)?
       (Evaluado mediante LLM-as-a-judge)
    3. Answer Relevance: ¿La respuesta contesta a lo que se preguntó?
    4. BERTScore: Similitud semántica con una respuesta de referencia humana.
    5. Citation Accuracy: ¿El sistema cita la fuente correcta (.pdf) en su respuesta?
    6. Negative Handling: Capacidad de decir "No lo sé" cuando la información no existe.
    7. Latencia: Tiempo de respuesta promedio.

Uso:
    Configura tu API KEY en el archivo .env y ejecuta:
    python src/evaluator.py
"""

import time
import json
import re
import statistics
import logging
from typing import List, Dict, Any, Optional

# --- IMPORTACIONES DE TUS MÓDULOS ---
try:
    from src.rag import get_rag_chain
    from src.vector_store import get_retriever
    from src.llm_client import call_llm_api_full
except ImportError as e:
    print(f"Error crítico importando módulos del proyecto: {e}")
    print("Asegúrate de ejecutar esto desde la raíz del proyecto.")
    exit(1)

# --- IMPORTACIÓN DE BERT SCORE ---
try:
    from bert_score import score as bert_score_lib
    BERT_AVAILABLE = True
except ImportError:
    print("ADVERTENCIA: 'bert-score' no está instalado. Ejecuta: pip install bert-score")
    BERT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RobustRAGEvaluator:
    def __init__(self, k_docs=5):
        self.k_docs = k_docs
        self.retriever = get_retriever(k_docs=k_docs)
        self.rag_chain = get_rag_chain(self.retriever)
        
        # Frases para detectar si el modelo admite que no sabe la respuesta (Requisito PDF)
        self.refusal_phrases = [
            "no tengo información", "no cuento con información", 
            "no se menciona", "no aparece en los documentos",
            "no puedo responder", "documents do not contain",
            "lo siento", "disculpa"
        ]

    def _judge_llm(self, prompt: str) -> float:
        """Juez LLM: Extrae un 0 o 1 de la respuesta del LLM."""
        try:
            response = call_llm_api_full(prompt).strip()
            # Busca un número entero o decimal entre 0 y 1 usando Regex
            match = re.search(r'\b(0(\.\d+)?|1(\.0+)?)\b', response)
            if match:
                return float(match.group(1))
            return 0.0
        except Exception as e:
            logger.error(f"Error en Juez LLM: {e}")
            return 0.0

    def _calculate_retrieval_metrics(self, retrieved_docs: List[Any], expected_source: str) -> Dict[str, float]:
        """Calcula Recall@K y MRR (Mean Reciprocal Rank)."""
        if not expected_source: 
            return {"recall": 0.0, "mrr": 0.0}
        
        expected_clean = expected_source.lower().replace(".pdf", "").strip()
        
        for rank, doc in enumerate(retrieved_docs, start=1):
            source_meta = doc.metadata.get('source', '').lower()
            if expected_clean in source_meta:
                return {
                    "recall": 1.0,      # ¡Encontrado!
                    "mrr": 1.0 / rank   # 1.0 si es 1º, 0.5 si es 2º, 0.33 si es 3º...
                }
        
        return {"recall": 0.0, "mrr": 0.0}

    def _calculate_bert_score(self, prediction: str, reference: str) -> float:
        """Calcula similitud semántica (BERTScore)."""
        if not BERT_AVAILABLE or not reference or not prediction:
            return 0.0
        try:
            # lang="es" descarga el modelo multilingüe adecuado
            P, R, F1 = bert_score_lib([prediction], [reference], lang="es", verbose=False)
            return F1.mean().item()
        except Exception as e:
            logger.warning(f"Error cálculo BERTScore: {e}")
            return 0.0

    def _check_citation(self, response: str, expected_source: str) -> float:
        """Verifica si el nombre del archivo fuente está presente en la respuesta."""
        if not expected_source: return 0.0
        # Buscamos el nombre del archivo sin extensión
        clean_source = expected_source.lower().replace(".pdf", "")
        return 1.0 if clean_source in response.lower() else 0.0

    def evaluate_dataset(self, dataset: List[Dict[str, Any]]):
        logger.info(f"Iniciando evaluación completa de {len(dataset)} casos...")
        
        metrics = {
            "latency": [],
            "retrieval_recall": [], # Recall@K
            "retrieval_mrr": [],    # MRR
            "faithfulness": [],     # FactScore (aprox)
            "answer_relevance": [], # Quality
            "bert_score": [],       # Semantic Similarity
            "citation_accuracy": [],# Functional Req
            "negative_handling": [] # Functional Req
        }

        results_detail = []

        for i, case in enumerate(dataset):
            question = case["question"]
            expected_source = case.get("ground_truth_source")
            reference_answer = case.get("reference_answer", "")
            
            logger.info(f"\n--- Caso {i+1}: {question[:50]}... ---")
            
            start_time = time.time()
            full_response = ""
            retrieved_docs = []
            context_text = ""
            
            try:
                # 1. Recuperación (Retrieval)
                retrieved_docs = self.retriever.invoke(question)
                context_text = "\n".join([d.page_content for d in retrieved_docs])
                
                # 2. Generación (Generation)
                # Define config with session_id
                config = {"configurable": {"session_id": f"eval_{i}"}}

                # 2. Generación (Generation)
                result = self.rag_chain.invoke({"question": question}, config=config)
                # Asumiendo que result es un string o un dict con 'answer'
                full_response = result if isinstance(result, str) else result.get('answer', str(result))
                
            except Exception as e:
                logger.error(f"Fallo ejecución Caso {i+1}: {e}")
                full_response = "ERROR DE SISTEMA"

            latency = time.time() - start_time
            metrics["latency"].append(latency)

            # --- LÓGICA DE EVALUACIÓN ---

            # A. PRUEBAS NEGATIVAS (El usuario pregunta algo que no está)
            if expected_source is None:
                is_refusal = 1.0 if any(ph in full_response.lower() for ph in self.refusal_phrases) else 0.0
                metrics["negative_handling"].append(is_refusal)
                metrics["faithfulness"].append(is_refusal) # Si rechaza correctamente, es fiel.
                logger.info(f"   Negative Test -> Correct Refusal: {bool(is_refusal)}")

            # B. PRUEBAS POSITIVAS (Pregunta estándar)
            else:
                # 1. Retrieval Metrics (Recall@K & MRR)
                ret_res = self._calculate_retrieval_metrics(retrieved_docs, expected_source)
                metrics["retrieval_recall"].append(ret_res["recall"])
                metrics["retrieval_mrr"].append(ret_res["mrr"])

                # 2. Faithfulness (LLM Judge)
                prompt_faith = (
                    f"Contexto: {context_text[:2000]}\nRespuesta: {full_response}\n"
                    "Analiza: ¿La respuesta se basa SOLAMENTE en el contexto? Responde 1 (Sí) o 0 (No/Alucinación)."
                )
                faith = self._judge_llm(prompt_faith)
                metrics["faithfulness"].append(faith)

                # 3. Answer Relevance (LLM Judge)
                prompt_rel = (
                    f"Pregunta: {question}\nRespuesta: {full_response}\n"
                    "Analiza: ¿La respuesta contesta a la pregunta? Responde 1 (Sí) o 0 (No)."
                )
                rel = self._judge_llm(prompt_rel)
                metrics["answer_relevance"].append(rel)

                # 4. BERTScore (Comparación con Referencia Humana)
                b_score = self._calculate_bert_score(full_response, reference_answer)
                metrics["bert_score"].append(b_score)

                # 5. Citation Accuracy
                cit_acc = self._check_citation(full_response, expected_source)
                metrics["citation_accuracy"].append(cit_acc)

                logger.info(f"   Recall: {ret_res['recall']} | MRR: {ret_res['mrr']:.2f} | BERT: {b_score:.2f} | Cit: {cit_acc}")

            results_detail.append({
                "q": question, "res": full_response, "ref": reference_answer, "metrics": metrics
            })

        self._print_report(metrics)

    def _print_report(self, metrics):
        def safe_mean(l): return statistics.mean(l) if l else 0.0
        
        print("\n" + "="*65)
        print(" REPORTE DE EVALUACIÓN RAG (Métricas UC3M)")
        print("="*65)
        print(f"{'MÉTRICA':<25} | {'RESULTADO':<10} | {'OBJETIVO':<10}")
        print("-" * 55)
        
        # Retrieval
        print(f"{'Recall@K (Hit Rate)':<25} | {safe_mean(metrics['retrieval_recall']):.2%}      | > 80%")
        print(f"{'MRR (Ranking Quality)':<25} | {safe_mean(metrics['retrieval_mrr']):.2f}        | > 0.7")
        
        # Generation
        print(f"{'Faithfulness (LLM)':<25} | {safe_mean(metrics['faithfulness']):.2%}      | > 90%")
        print(f"{'Answer Relevance (LLM)':<25} | {safe_mean(metrics['answer_relevance']):.2%}      | > 90%")
        print(f"{'BERTScore (Semantic)':<25} | {safe_mean(metrics['bert_score']):.2f}        | > 0.85")
        
        # Functional
        print(f"{'Citation Accuracy':<25} | {safe_mean(metrics['citation_accuracy']):.2%}      | 100%")
        print(f"{'Negative Handling':<25} | {safe_mean(metrics['negative_handling']):.2%}      | 100%")
        print(f"{'Avg Latency':<25} | {safe_mean(metrics['latency']):.2f}s       | < 5s")
        print("="*65)

# --- CONFIGURACIÓN DEL DATASET ---
TEST_DATASET = [
    # --- PRUEBAS POSITIVAS (Recuperación y Generación) ---
    {
        'question': 'What does NLP deal with?',
        'ground_truth_source': 'document4',
        "reference_answer": "No tengo información sobre eso en los documentos."
    },
    {
        'question': 'What is machine learning?',
        'ground_truth_source': 'document3',
        "reference_answer": "No tengo información sobre eso en los documentos."

    },
    # ... Agrega el resto de tus PDFs aquí ...
    
    # --- PRUEBAS NEGATIVAS (Control de Alucinaciones) ---
    {
        "question": "¿Cuál es la receta de la paella valenciana?",
        "ground_truth_source": None,
        "reference_answer": "No tengo información sobre eso en los documentos."
    }
]

if __name__ == "__main__":
    # K=5 para Recall@5
    evaluator = RobustRAGEvaluator(k_docs=5)
    evaluator.evaluate_dataset(TEST_DATASET)
