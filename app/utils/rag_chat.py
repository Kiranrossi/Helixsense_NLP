import os
from typing import List, Tuple

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import RatelimitException

load_dotenv()

# --- Config from .env ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY missing in .env")
if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME missing in .env")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing in .env")

# Use a stable Groq chat model
GROQ_MODEL = "llama-3.1-8b-instant"  # update if Groq deprecates this

# --- Global clients & models ---
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# 384‑dim embeddings from all‑MiniLM‑L6‑v2
embedder = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=GROQ_API_KEY)


# ---------- Helper functions ----------
def _embed(text: str) -> List[float]:
    """Return a single 384‑dim embedding for the text."""
    return embedder.encode([text])[0].tolist()


def _search_pinecone(query: str, top_k: int = 4):
    """Vector search over project documentation stored in Pinecone."""
    q_vec = _embed(query)
    result = index.query(
        vector=q_vec,
        top_k=top_k,
        include_metadata=True,
    )
    return result.get("matches", [])


def _web_search_snippet(query: str) -> str:
    """
    Short DuckDuckGo snippet for general knowledge fallback.

    On any error (including rate‑limit), return "" so the chat still works.
    """
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=1):
                return r.get("body", "")[:400]
    except RatelimitException:
        # Hit DuckDuckGo rate limit; skip web snippet gracefully
        return ""
    except Exception:
        # Any other network / parsing issue
        return ""


def _build_user_prompt(
    question: str,
    project_context: str,
    web_context: str = "",
) -> str:
    """Build the content of the user message sent to Groq."""
    parts = [f"User question: {question}"]
    if project_context:
        parts.append("\nProject context:\n" + project_context)
    if web_context:
        parts.append("\nWeb snippet (may be incomplete):\n" + web_context)
    parts.append(
        "\nGive a clear, helpful answer in a natural tone. "
        "Use short paragraphs and simple language. "
        "If an example helps, you can add one."
    )
    return "\n".join(parts)


# ---------- Main entrypoint ----------
def chat_ai(
    question: str,
    history: List[Tuple[str, str]],
    df,
    tfidf_vec,
    tfidf_model,
    setfit_model,
) -> str:
    """
    General chatbot that is also aware of the Helixsense project.

    The signature matches app.app.py, which calls this with keyword args:
    question, history, df, tfidf_vec, tfidf_model, setfit_model.
    Extra arguments are currently unused but kept for compatibility.
    """
    question = question.strip()
    if not question:
        return "Please enter a question."

    lower_q = question.lower()

    # Detect if question is about this project
    project_keywords = [
        "helixsense",
        "facility expense",
        "remarks column",
        "setfit",
        "tf-idf",
        "tfidf",
        "services equipment material",
        "facility classifier",
        "candidate test",
        "assignment",
        "data.xlsx",
    ]
    is_project_question = any(k in lower_q for k in project_keywords)

    project_ctx = ""
    web_ctx = ""

    if is_project_question:
        try:
            matches = _search_pinecone(question, top_k=6)
        except Exception as e:
            matches = []
            project_ctx = f"(Error talking to Pinecone: {e})"

        if matches:
            pieces: List[str] = []
            scores: List[float] = []
            for m in matches:
                meta = m.get("metadata", {}) or {}
                text = meta.get("text", "")
                if text:
                    pieces.append(f"- [{meta.get('section', 'section')}] {text}")
                    scores.append(float(m.get("score", 0.0)))
            project_ctx = "\n".join(pieces)
            avg_score = sum(scores) / len(scores) if scores else 0.0
            if avg_score < 0.65:
                web_ctx = _web_search_snippet(question)
        else:
            web_ctx = _web_search_snippet(question)
    else:
        # Non‑project question: treat as general chat but still allow web
        web_ctx = _web_search_snippet(question)

    system_msg = (
        "You are a friendly, conversational AI assistant. "
        "You explain things clearly in simple language, using short paragraphs "
        "and examples when they help. Avoid sounding like a rigid robot. "
        "You also know about a specific NLP project for Helixsense that "
        "classifies facility expense transactions using the Remarks text. "
        "The project started from an assignment that required data loading, "
        "EDA, designing multiple model approaches, choosing one, training, "
        "evaluating with appropriate metrics, and writing an executive summary. "
        "Use that project and assignment information when it is relevant, "
        "otherwise answer like a normal assistant."
    )

    user_prompt = _build_user_prompt(
        question=question,
        project_context=project_ctx,
        web_context=web_ctx,
    )

    # Build messages from history so the model sees prior turns
    messages = [{"role": "system", "content": system_msg}]
    for role, content in history:
        if role == "user":
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "assistant", "content": content})

    # Append the new user message with context included
    messages.append({"role": "user", "content": user_prompt})

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=600,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error calling Groq API: {e}"
