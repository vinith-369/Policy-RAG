Please act as an Expert AI Technical Writer and create a comprehensive, professional "Enterprise AI Case Study Report". 

I am providing you with the exact technical architecture, the agentic reasoning workflow, the mathematical evaluation strategy, and the raw performance metrics of our newly built custom system. Please synthesize this into a polished whitepaper-style report for stakeholders.

---

# 1. Project Context: "Policy RAG"
Our team has developed a fully localized, private Retrieval-Augmented Generation (RAG) Command-Line system. The objective is to allow employees to query highly sensitive corporate documents (HR, IT Security, Leave Policies, etc.) with 100% factual accuracy. 

Because corporate data is sensitive, this application does NOT rely on remote APIs (like OpenAI or Gemini). It runs entirely on the host machine using **Ollama**, ensuring zero data leakage.

# 2. Technology & Architecture Stack
* **LLM Engine:** Local Ollama running `qwen2.5:7b` (a lightweight but highly capable open-weight language model).
* **Agentic Framework:** `LangGraph` (used for node-based, cyclical reasoning and self-reflection workflows, rather than a single-pass chain).
* **Vector Database:** `Qdrant` (In-Memory mode) for hyper-fast vectorized document storage without needing external Docker containers.
* **Embeddings (Dense):** `sentence-transformers/all-MiniLM-L6-v2`. (We purposefully force this PyTorch model to run strictly on the CPU to prevent Apple Metal/MPS unified memory out-of-memory conflicts with the heavy Ollama GPU workload).
* **Embeddings (Sparse):** `rank-bm25` (Python-native Okapi BM25 implementation).

# 3. Agentic Workflow (LangGraph Nodes)
The system uses an advanced "Self-Reflective" agent approach consisting of 5 distinct nodes:
1. **`retrieve` Node:** Uses **Hybrid Search** with **Reciprocal Rank Fusion (RRF)**. It mathematically merges results from Deep Semantic Dense similarity and Precise Lexical Sparse keyword matching.
2. **`grade_documents` Node:** The LLM looks at the retrieved raw document chunks and the user's question, scoring whether the chunks contain relevant answers. If irrelevant, it immediately skips generation to prevent hallucinated guessing.
3. **`generate` Node:** The LLM synthesizes a clean answer STRICTLY grounded in the relevant text chunks.
4. **`check_hallucination` Node:** The LLM reviews its *own* generated answer against the raw text. If it detects a hallucination (invented information), the workflow actively loops back to `generate` to try again.
5. **`check_answer` Node:** A final logical check to ensure the generated answer actually directly addresses the user's initial question.

# 4. Evaluation Strategy & Metrics (LLM-as-a-Judge)
To evaluate the generation pipeline without human bias or rigid keyword-matching, we devised an industry-standard evaluation methodology inspired by frameworks like **Ragas** and **TruLens**:

### Phase A: Synthetic Golden Dataset Generation
To ensure the RAG is tested fairly, we generated a mathematically grounded test dataset:
* We built a script that reads every PDF and TXT file in the `docs/` folder, breaking them into 1,500-character chunks.
* For each chunk, the local `qwen2.5:7b` model wrote realistic employee questions alongside the exact, factual "Expected Answer" based *strictly* on that singular text chunk.
* This process automatically generated 50 isolated "Golden Q&A Pairs" covering all policies.

### Phase B: Automated Evaluation & Semantic Scoring
With the Golden set finalized, we ran the organic evaluation:
* The Agent was fed each question, retrieved its own chunks, went through its self-reflection loop, and output a live answer.
* **The Judge:** Finally, an independent instance of `qwen2.5` acted as an impartial judge. It took the *Golden Expected Answer* and compared it directly against the *Agent's Live Answer*.
* It scored semantic similarity and factual adherence on a strict **1 to 5 scale**:
  * `5` = Perfect match, captures the full expected concept.
  * `4` = Mostly correct, minor missing details or slightly wordy.
  * `3` = Partially correct.
  * `2` = Vague or mostly incorrect.
  * `1` = Completely hallucinated or contradictory to the ground truth.

# 5. Raw Evaluation Results
*(Note to Claude: I have attached a file called `evaluation_results.json`. Please heavily reference the "average_accuracy" score and highlight a few specific examples from the "detailed_results" array in your report to prove the system's effectiveness and its capability to self-correct).*
