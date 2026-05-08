"""
Step 1 — LangSmith-instrumented RAG Pipeline with Google Gemini
================================================================
TASK:
  1. Load dataset, split into chunks, index with FAISS
  2. Build a RAG chain: retriever -> prompt -> LLM -> output parser
  3. Decorate the query function with @traceable so every call is traced
  4. Run all 50 questions -> generates >= 50 LangSmith traces

DELIVERABLE: Open https://smith.langchain.com and confirm traces appear.
"""

# Fix Unicode encoding on Windows FIRST
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
from pathlib import Path
from dotenv import load_dotenv

# ── 1. Environment setup ────────────────────────────────────────────────────
load_dotenv()

# Set LangSmith environment variables BEFORE importing LangChain
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "day22-lab")
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# ── 2. LangChain + LangSmith imports ────────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

# ── 3. LLM and Embeddings ───────────────────────────────────────────────────
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

# ── 4. Build FAISS vector store ─────────────────────────────────────────────
def build_vectorstore():
    """
    Load the knowledge base, split into chunks, embed and index with FAISS.
    """
    print("[BUILD] Loading knowledge base...")
    text = Path("data/knowledge_base.txt").read_text()
    
    print("[BUILD] Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"  Split into {len(chunks)} chunks")
    
    print("[BUILD] Building FAISS vectorstore...")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    print(f"  [OK] Vectorstore ready with {len(chunks)} chunks")
    
    return vectorstore


# ── 5. RAG prompt template ──────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant. Answer the user's question using ONLY the provided context.
If the context does not contain enough information to answer the question, say: "I don't have enough information to answer this question."

Context:
{context}"""),
    ("human", "{question}"),
])


# ── 6. Build the RAG chain ──────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    """
    Build a LangChain RAG chain using LCEL (pipe operator).
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    
    return chain, retriever


# ── 7. Traced query function ────────────────────────────────────────────────
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    """
    Run the RAG chain on a single question.
    The @traceable decorator sends input/output/latency to LangSmith.
    """
    return chain.invoke(question)


# ── 8. Sample questions (50 total) ──────────────────────────────────────────
from qa_pairs import QA_PAIRS

SAMPLE_QUESTIONS = [qa["question"] for qa in QA_PAIRS]


# ── 9. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Step 1: LangSmith RAG Pipeline with Google Gemini")
    print("=" * 70)
    
    # Build the vectorstore
    vectorstore = build_vectorstore()
    
    # Build the RAG chain
    chain, retriever = build_rag_chain(vectorstore)
    
    print("\n[RUN] Running 50 questions through RAG pipeline...\n")
    
    # Loop through all questions
    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        try:
            answer = ask(chain, question)
            print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question[:60]}")
            print(f"       A: {answer[:100]}\n")
        except Exception as e:
            print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] [ERROR] {str(e)[:100]}\n")
    
    print("=" * 70)
    print(f"[OK] {len(SAMPLE_QUESTIONS)} traces sent to LangSmith project '{os.environ['LANGCHAIN_PROJECT']}'")
    print("   Open https://smith.langchain.com to view traces.")
    print("=" * 70)


if __name__ == "__main__":
    main()
