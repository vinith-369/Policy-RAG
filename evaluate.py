"""
Custom LLM-as-a-Judge Evaluation script for the Policy RAG.
Runs a test suite of questions and uses Ollama to grade the RAG system's outputs.
"""
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from app.agent import run_rag
from cli import ingest_directory
from app.config import settings

# ─── 1. Test Dataset ─────────────────────────────────────────────────────────
def load_test_cases(filepath: str = "test_cases.json"):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Could not find {filepath}. Please run 'python generate_test_cases.py' first!")
        return []

# ─── 2. Evaluation Judge ──────────────────────────────────────────────────────
def grade_answer(question: str, generated_answer: str, expected_concept: str) -> int:
    """Use Ollama to grade the generated answer against the expected concept (1-5)."""
    judge_llm = ChatOllama(model=settings.ollama_model, temperature=0.0)
    
    prompt = f"""You are an impartial judge evaluating a RAG system.
    Question: {question}
    Expected Concept: {expected_concept}
    Actual RAG Answer: {generated_answer}
    
    Score the 'Actual RAG Answer' from 1 to 5 based on how well it matches the 'Expected Concept':
    5 = Perfect match, captures the full concept.
    4 = Mostly correct, but missing minor details or slightly wordy.
    3 = Partially correct, misses key parts of the concept.
    2 = Mostly incorrect or overly vague.
    1 = Completely wrong, hallucinated, or completely contradicts the expected concept.
    
    Return ONLY the integer number (1, 2, 3, 4, or 5). No other text.
    """
    
    try:
        response = judge_llm.invoke([HumanMessage(content=prompt)])
        score_text = response.content.strip()
        # Extract just the first digit found
        for char in score_text:
            if char.isdigit():
                return int(char)
        return 0
    except Exception as e:
        print(f"Error grading: {e}")
        return 0


# ─── 3. Run Evaluation ────────────────────────────────────────────────────────
def run_evaluation():
    print("🚀 Setting up RAG Evaluation Suite...")
    ingest_directory("docs")
    
    test_cases = load_test_cases()
    if not test_cases:
        return
        
    print("\n📊 Running Test Cases...\n")
    
    total_score = 0
    results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}/{len(test_cases)}: '{test['question']}'")
        
        # Run the RAG pipeline
        rag_output = run_rag(test["question"])
        
        # Grade the output using LLM-as-a-judge
        score = grade_answer(
            question=test["question"],
            generated_answer=rag_output["answer"],
            expected_concept=test["expected_concept"]
        )
        
        total_score += score
        
        print(f"  ↳ Answer: {rag_output['answer']}")
        print(f"  ↳ Sources: {', '.join(rag_output['sources']) if rag_output['sources'] else 'None'}")
        print(f"  ↳ ⭐ Score: {score}/5\n")
        
        results.append({
            "test": test,
            "answer": rag_output["answer"],
            "score": score
        })

    # Summary Calculate
    avg_score = total_score / len(test_cases)
    print("=" * 50)
    print(f"🏆 EVALUATION COMPLETE")
    print(f"📈 Average Accuracy Score: {avg_score:.1f}/5.0")
    print("=" * 50)
    
    report = {
        "average_accuracy": round(avg_score, 2),
        "total_test_cases": len(test_cases),
        "detailed_results": results
    }
    
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print("💾 Full evaluation report saved to evaluation_results.json")


if __name__ == "__main__":
    run_evaluation()
