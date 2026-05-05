
from typing import Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from app.config import settings
from app.vector_store import hybrid_search

# ─── LLM singleton ────────────────────────────────────────────────────────────

_llm: Optional[ChatOllama] = None


def get_llm() -> ChatOllama:
    global _llm
    if _llm is None or getattr(_llm, "model", None) != settings.ollama_model:
        _llm = ChatOllama(
            model=settings.ollama_model,
            temperature=0.0,
        )
    return _llm


# ─── Graph State ──────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    question: str
    top_k: int
    filter_source: Optional[str]
    raw_chunks: list[dict]
    graded_chunks: list[dict]
    answer: str
    hallucination_retries: int
    sources: list[str]


# ─── Node: retrieve ───────────────────────────────────────────────────────────

def retrieve(state: RAGState) -> RAGState:
    chunks = hybrid_search(
        query=state["question"],
        top_k=state.get("top_k", 5),
        filter_source=state.get("filter_source"),
    )
    return {**state, "raw_chunks": chunks}


# ─── Node: grade_documents (BATCHED — 1 LLM call total) ──────────────────────

GRADE_SYSTEM = """You are a relevance grader for a policy Q&A system.
Given a question and numbered document chunks, respond with a comma-separated list of the numbers of ONLY the chunks that are relevant to answering the question.
Example: if chunks 1, 3, 4 are relevant, respond: 1,3,4
If none are relevant, respond: none
Respond with ONLY the numbers or the word 'none'. No other text."""


def grade_documents(state: RAGState) -> RAGState:
    chunks = state["raw_chunks"]
    if not chunks:
        return {**state, "graded_chunks": [], "sources": []}

    llm = get_llm()

    # Build numbered list for batch grading
    numbered = "\n\n".join(
        f"[{i+1}] {c['text'][:300]}" for i, c in enumerate(chunks)
    )

    messages = [
        SystemMessage(content=GRADE_SYSTEM),
        HumanMessage(
            content=f"Question: {state['question']}\n\nChunks:\n{numbered}"
        ),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip().lower()

    if raw == "none":
        graded = []
    else:
        try:
            indices = {int(x.strip()) - 1 for x in raw.split(",") if x.strip().isdigit()}
            graded = [chunks[i] for i in sorted(indices) if 0 <= i < len(chunks)]
        except Exception:
            # Only fall back to all chunks on a parse failure (not on "none")
            graded = chunks

    sources = list({c["source_file"] for c in graded}) if graded else []
    return {**state, "graded_chunks": graded, "sources": sources}


# ─── Node: generate ───────────────────────────────────────────────────────────

GENERATE_SYSTEM = """You are a helpful policy assistant for ACME Corporation.
Answer the employee's question using ONLY the policy excerpts provided below.
- Be concise, clear, and factual.
- If the answer is not found, say "I don't have enough information in the provided policies."
- Do NOT make up information."""


def generate(state: RAGState) -> RAGState:
    if not state["graded_chunks"]:
        return {
            **state,
            "answer": "I don't have enough information in the provided policies.",
        }

    llm = get_llm()
    context = "\n\n---\n\n".join(
        f"[Source: {c['source_file']}]\n{c['text']}"
        for c in state["graded_chunks"]
    )
    messages = [
        SystemMessage(content=GENERATE_SYSTEM),
        HumanMessage(
            content=f"Policy excerpts:\n{context}\n\nQuestion: {state['question']}"
        ),
    ]
    response = llm.invoke(messages)
    return {**state, "answer": response.content.strip()}


# ─── Node: check_hallucination ────────────────────────────────────────────────

HALLUCINATION_SYSTEM = """You are a fact-checker for a policy Q&A system.
Given an answer and source excerpts, respond with ONLY 'yes' (answer is grounded) or 'no' (hallucination detected). No other text."""


def check_hallucination(state: RAGState) -> RAGState:
    # Skip hallucination check if there are no graded chunks (nothing to check against)
    if not state.get("graded_chunks"):
        return state

    if state.get("hallucination_retries", 0) >= 2:
        return state  # stop retrying

    llm = get_llm()
    context = "\n\n".join(c["text"] for c in state["graded_chunks"])
    messages = [
        SystemMessage(content=HALLUCINATION_SYSTEM),
        HumanMessage(
            content=(
                f"Question: {state['question']}\n\n"
                f"Source excerpts:\n{context}\n\n"
                f"Answer: {state['answer']}"
            )
        ),
    ]
    response = llm.invoke(messages)
    verdict = response.content.strip().lower()

    if verdict.startswith("no"):
        return {
            **state,
            "answer": "",
            "hallucination_retries": state.get("hallucination_retries", 0) + 1,
        }
    return state


# ─── Routing ──────────────────────────────────────────────────────────────────

def route_after_check(state: RAGState) -> str:

    if state.get("hallucination_retries", 0) >= 2:
        return END
    return "generate" if not state.get("answer") else END


# ─── Build graph ──────────────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("generate", generate)
    g.add_node("check_hallucination", check_hallucination)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_edge("grade_documents", "generate")
    g.add_edge("generate", "check_hallucination")
    g.add_conditional_edges(
        "check_hallucination",
        route_after_check,
        {"generate": "generate", END: END},
    )
    return g.compile()


rag_graph = _build_graph()


# ─── Public API ───────────────────────────────────────────────────────────────

def run_rag(
    question: str,
    top_k: int = 5,
    filter_source: Optional[str] = None,
) -> dict:
    """Run the full agentic RAG pipeline and return { answer, sources, chunks_used }."""
    initial: RAGState = {
        "question": question,
        "top_k": top_k,
        "filter_source": filter_source,
        "raw_chunks": [],
        "graded_chunks": [],
        "answer": "",
        "hallucination_retries": 0,
        "sources": [],
    }
    final = rag_graph.invoke(initial)
    return {
        "answer":      final["answer"],
        "sources":     final["sources"],
        "chunks_used": len(final["graded_chunks"]),
    }


# ─── Streaming public API ─────────────────────────────────────────────────────

_MAX_STREAM_RETRIES = 2


def stream_rag(
    question: str,
    top_k: int = 5,
    filter_source: Optional[str] = None,
):
    """
    Streaming version of run_rag.

    Yields dicts:
      {"type": "token",  "content": "<text>"}       — one per LLM output token
      {"type": "retry",  "attempt": <int>}           — when hallucination detected
      {"type": "done",   "answer": ...,
                         "sources": [...],
                         "chunks_used": <int>}       — final metadata

    Strategy:
    1. retrieve + grade_documents run synchronously (not worth streaming).
    2. The generate LLM call uses llm.stream() so tokens appear immediately.
    3. After full generation, the hallucination checker runs (one LLM call).
       If it flags a hallucination, the answer is discarded, a "retry" event
       is emitted, and a new streaming generation begins (up to _MAX_STREAM_RETRIES).
    """
    # ── Step 1: retrieve + grade ──────────────────────────────────────────────
    base: RAGState = {
        "question": question,
        "top_k": top_k,
        "filter_source": filter_source,
        "raw_chunks": [],
        "graded_chunks": [],
        "answer": "",
        "hallucination_retries": 0,
        "sources": [],
    }
    state = retrieve(base)
    state = grade_documents(state)

    graded_chunks = state["graded_chunks"]
    sources        = state["sources"]

    # ── No relevant chunks found ───────────────────────────────────────────────
    if not graded_chunks:
        msg = "I don't have enough information in the provided policies."
        yield {"type": "token",  "content": msg}
        yield {"type": "done",   "answer": msg, "sources": [], "chunks_used": 0}
        return

    # ── Pre-build context (reused across retries) ──────────────────────────────
    llm = get_llm()
    context = "\n\n---\n\n".join(
        f"[Source: {c['source_file']}]\n{c['text']}" for c in graded_chunks
    )
    hc_context = "\n\n".join(c["text"] for c in graded_chunks)

    full_answer = ""
    for attempt in range(_MAX_STREAM_RETRIES + 1):
        # ── Step 2: stream generate ────────────────────────────────────────────
        messages = [
            SystemMessage(content=GENERATE_SYSTEM),
            HumanMessage(
                content=f"Policy excerpts:\n{context}\n\nQuestion: {question}"
            ),
        ]
        full_answer = ""
        for chunk in llm.stream(messages):
            token = chunk.content
            full_answer += token
            yield {"type": "token", "content": token}

        # ── Step 3: hallucination check (no streaming needed) ─────────────────
        if attempt < _MAX_STREAM_RETRIES:
            hc_messages = [
                SystemMessage(content=HALLUCINATION_SYSTEM),
                HumanMessage(
                    content=(
                        f"Question: {question}\n\n"
                        f"Source excerpts:\n{hc_context}\n\n"
                        f"Answer: {full_answer}"
                    )
                ),
            ]
            verdict = llm.invoke(hc_messages).content.strip().lower()
            if verdict.startswith("yes"):
                break  # grounded — accept the answer
            # Hallucination detected — retry
            yield {"type": "retry", "attempt": attempt + 1}
        else:
            break  # max retries exhausted — accept whatever we have

    yield {
        "type":        "done",
        "answer":      full_answer,
        "sources":     sources,
        "chunks_used": len(graded_chunks),
    }
