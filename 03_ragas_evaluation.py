"""
Step 3 — RAGAS Evaluation with Google Gemini
==============================================
TASK:
  1. Run all 50 QA pairs through BOTH prompt versions
  2. Build EvaluationDataset with SingleTurnSample objects
  3. Evaluate with 4 RAGAS metrics
  4. Print a V1 vs V2 comparison table
  5. Save results to data/ragas_report.json

DELIVERABLE: faithfulness >= 0.8 for at least one prompt version

NOTE: This step takes ~20-30 minutes. Start it early!
"""

# Fix Unicode encoding on Windows FIRST
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from dotenv import load_dotenv

# ── 1. Imports ───────────────────────────────────────────────────────────────
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "day22-lab")

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

# ── 2. QA pairs with ground-truth answers ───────────────────────────────────
from qa_pairs import QA_PAIRS

# ── 3. Prompt templates (same as step 2) ────────────────────────────────────
SYSTEM_V1 = """You are a helpful AI assistant. Answer the user's question using ONLY the provided context.
Keep your answer concise (2-4 sentences).
If the context does not contain the answer, say: "I don't have enough information."

Context:
{context}"""

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human", "{question}"),
])

SYSTEM_V2 = """You are an expert AI tutor. Provide a structured, accurate answer.

Instructions:
1. Read the context carefully.
2. Identify the key facts relevant to the question.
3. Write a clear, well-organized answer (3-5 sentences).
4. State explicitly if the context lacks sufficient information.

Context:
{context}"""

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human", "{question}"),
])

PROMPTS = {
    "v1": PROMPT_V1,
    "v2": PROMPT_V2,
}

# ── 4. Setup LLM and Embeddings ─────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)

# For RAGAS evaluation (use same LLM and embeddings)
llm_eval = llm
emb_eval = embeddings

# ── 5. Build vectorstore ────────────────────────────────────────────────────
def build_vectorstore():
    """Load and build FAISS vectorstore."""
    print("[BUILD] Loading knowledge base...")
    text = Path("data/knowledge_base.txt").read_text()
    
    print("[BUILD] Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    
    print("[BUILD] Building FAISS vectorstore...")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    
    return vectorstore


# ── 6. Run RAG and capture outputs + contexts ────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Run the RAG chain for one question.
    
    IMPORTANT: return contexts as a LIST of strings, not a joined string!
    RAGAS needs individual passage strings to compute context_recall.
    
    Returns: {"answer": str, "contexts": list[str]}
    """
    # Retrieve documents
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    ctx_str = "\n\n".join(contexts)
    
    # Run the chain
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": ctx_str,
        "question": question
    })
    
    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Run all 50 QA pairs through the given prompt version.
    Returns a list of dicts with keys: question, reference, answer, contexts.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    prompt = PROMPTS[prompt_version]
    
    results = []
    print(f"\n[RUN] Running 50 questions with prompt {prompt_version}...")
    
    for i, qa in enumerate(QA_PAIRS, 1):
        try:
            out = run_rag(retriever, llm, prompt, qa["question"])
            results.append({
                "question": qa["question"],
                "reference": qa["reference"],
                "answer": out["answer"],
                "contexts": out["contexts"],
            })
            print(f"  [{i:02d}/50] {qa['question'][:60]}")
        except Exception as e:
            print(f"  [{i:02d}/50] [ERROR] {str(e)[:50]}")
            # Add a fallback result
            results.append({
                "question": qa["question"],
                "reference": qa["reference"],
                "answer": "Error generating answer",
                "contexts": ["No context retrieved"],
            })
    
    return results


# ── 7. Build RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list):
    """
    Convert a list of RAG result dicts into a RAGAS EvaluationDataset.
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)


# ── 8. Run RAGAS evaluation ──────────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Evaluate RAG outputs with 4 RAGAS metrics.
    Returns a dict: {metric_name: mean_score}
    """
    print(f"\n[EVAL] Running RAGAS evaluation for prompt {version}...")
    print("   (This may take 10-15 minutes...)\n")
    
    # Create the EvaluationDataset
    dataset = build_ragas_dataset(rag_results)
    
    # Run evaluate
    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=llm_eval,
            embeddings=emb_eval,
        )
        
        # Extract mean scores
        scores = {}
        for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
            raw = result[key]
            # Filter out None values
            valid_scores = [v for v in raw if v is not None]
            if valid_scores:
                scores[key] = float(np.mean(valid_scores))
            else:
                scores[key] = 0.0
        
        # Print scores
        print(f"\n[OK] RAGAS Scores for {version}:")
        for k, v in scores.items():
            star = " [TARGET]" if k == "faithfulness" and v >= 0.8 else ""
            print(f"  {k:30s}: {v:.4f}{star}")
        
        return scores
    
    except Exception as e:
        print(f"[ERROR] During RAGAS evaluation: {e}")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_recall": 0.0,
            "context_precision": 0.0,
        }


# ── 9. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Step 3: RAGAS Evaluation with Google Gemini")
    print("=" * 70)
    
    # Build vectorstore
    vectorstore = build_vectorstore()
    
    # Collect outputs for V1 and V2
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")
    
    # Run RAGAS evaluation on both
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("  Comparison: V1 vs V2")
    print("=" * 70)
    
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1 = v1_scores.get(metric, 0.0)
        s2 = v2_scores.get(metric, 0.0)
        winner = "<- V1 wins" if s1 > s2 else ("<- V2 wins" if s2 > s1 else "<- Tie")
        print(f"  {metric:30s}: V1={s1:.4f}  V2={s2:.4f}  {winner}")
    
    # Check faithfulness target
    best_faith = max(v1_scores.get("faithfulness", 0.0), v2_scores.get("faithfulness", 0.0))
    print("\n" + "=" * 70)
    if best_faith >= 0.8:
        print(f"[OK] Target met: faithfulness = {best_faith:.4f}")
    else:
        print(f"[WARN] Below target ({best_faith:.4f}). Try adjusting chunking or prompts.")
    print("=" * 70)
    
    # Save JSON report
    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
        "best_faithfulness": float(best_faith),
    }
    
    Path("data/ragas_report.json").write_text(json.dumps(report, indent=2))
    print("\n[OK] Saved data/ragas_report.json")


if __name__ == "__main__":
    main()
