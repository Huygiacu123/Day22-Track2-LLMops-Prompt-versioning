"""
Step 2 — Prompt Hub & A/B Routing with Google Gemini
======================================================
TASK:
  1. Write two distinct system prompts (V1: concise, V2: structured)
  2. Push both to LangSmith Prompt Hub via client.push_prompt()
  3. Pull them back via client.pull_prompt()
  4. Implement deterministic A/B routing: hash(request_id) % 2 → V1 or V2
  5. Run all 50 questions through the router → ≥ 50 more LangSmith traces

DELIVERABLE: 2 named prompts visible in https://smith.langchain.com Prompt Hub
"""

# Fix Unicode encoding on Windows FIRST
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv

# ── 1. Environment / imports ────────────────────────────────────────────────
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "day22-lab")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable

# ── 2. Define two prompt templates ──────────────────────────────────────────
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

# Prompt Hub names
PROMPT_V1_NAME = "day22-rag-prompt-v1-concise"
PROMPT_V2_NAME = "day22-rag-prompt-v2-structured"

# ── 3. Push prompts to LangSmith Prompt Hub ──────────────────────────────────
def push_prompts_to_hub(client):
    """
    Upload both prompt versions to LangSmith Prompt Hub.
    """
    print("\n[PUSH] Pushing prompts to LangSmith Prompt Hub...\n")
    
    try:
        url = client.push_prompt(
            PROMPT_V1_NAME,
            object=PROMPT_V1,
            description="V1 - Concise answers (2-4 sentences)"
        )
        print(f"[OK] Pushed V1 -> {url}")
    except Exception as e:
        print(f"[WARN] V1: {e}")
    
    try:
        url = client.push_prompt(
            PROMPT_V2_NAME,
            object=PROMPT_V2,
            description="V2 - Structured answers (3-5 sentences with reasoning)"
        )
        print(f"[OK] Pushed V2 -> {url}")
    except Exception as e:
        print(f"[WARN] V2: {e}")


# ── 4. Pull prompts from Prompt Hub ─────────────────────────────────────────
def pull_prompts_from_hub(client):
    """
    Download both prompt versions from LangSmith Prompt Hub.
    Fall back to local templates if Hub is unavailable.
    """
    prompts = {}
    
    print("\n[PULL] Pulling prompts from LangSmith Prompt Hub...\n")
    
    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"[OK] Pulled '{PROMPT_V1_NAME}' from Hub")
    except Exception as e:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"[INFO] Using local fallback for '{PROMPT_V1_NAME}': {str(e)[:50]}")
    
    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"[OK] Pulled '{PROMPT_V2_NAME}' from Hub")
    except Exception as e:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"[INFO] Using local fallback for '{PROMPT_V2_NAME}': {str(e)[:50]}")
    
    return prompts


# ── 5. A/B routing — deterministic hash ─────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Route a request to prompt V1 or V2 based on the MD5 hash of request_id.
    
    Rules:
      even hash -> PROMPT_V1_NAME
      odd  hash -> PROMPT_V2_NAME
    
    This is DETERMINISTIC: same request_id always maps to the same version.
    """
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Build vectorstore (reuse from step 1) ────────────────────────────────
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


# ── 7. Traced A/B query function ────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Run the RAG chain using the given prompt version.
    Returns a dict: {"question": ..., "answer": ..., "version": ...}
    """
    # Retrieve docs
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Run the chain
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question
    })
    
    return {"question": question, "answer": answer, "version": version}


# ── 8. Setup LLM and Embeddings ─────────────────────────────────────────────
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


# ── 9. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Step 2: Prompt Hub A/B Routing with Google Gemini")
    print("=" * 70)
    
    # Create LangSmith client
    client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])
    
    # Push both prompts
    push_prompts_to_hub(client)
    
    # Pull both prompts from Hub
    prompts = pull_prompts_from_hub(client)
    
    # Build vectorstore, retriever
    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Load questions
    from qa_pairs import QA_PAIRS
    
    print("\n[RUN] Running 50 questions through A/B routing...\n")
    
    v1_count = 0
    v2_count = 0
    
    # Loop over all 50 questions with A/B routing
    for i, qa in enumerate(QA_PAIRS):
        request_id = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt = prompts[version_key]
        
        try:
            result = ask_ab(retriever, llm, prompt, qa["question"], version_tag)
            print(f"[{i+1:02d}] [prompt-{version_tag}] {qa['question'][:55]}...")
            
            if version_tag == "v1":
                v1_count += 1
            else:
                v2_count += 1
        except Exception as e:
            print(f"[{i+1:02d}] [ERROR] {str(e)[:50]}")
    
    print("\n" + "=" * 70)
    print("[SUMMARY] A/B Routing Summary:")
    print(f"  V1 (concise):    {v1_count} questions")
    print(f"  V2 (structured): {v2_count} questions")
    print(f"  Total:           {v1_count + v2_count} traces sent to LangSmith")
    print("=" * 70)


if __name__ == "__main__":
    main()
