import json
import random
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import re

from app.config import settings

def extract_text_from_dir(docs_dir: str):
    base = Path(docs_dir)
    if not base.exists():
        return ""
    
    text_content = ""
    for filepath in sorted(list(base.glob("*.txt")) + list(base.glob("*.pdf"))):
        ext = filepath.suffix.lower()
        if ext == ".pdf":
            reader = PdfReader(str(filepath))
            text_content += "\n\n".join(p.extract_text() or "" for p in reader.pages)
        else:
            text_content += filepath.read_text(encoding="utf-8", errors="replace") + "\n\n"
    return text_content


def generate_synthetic_test_cases(target_count: int = 50, output_file: str = "test_cases.json"):
    print("🚀 Gathering text from documents...")
    all_text = extract_text_from_dir("docs")
    
    if not all_text.strip():
        print("❌ No documents found to generate test cases from.")
        return

    # Split into larger chunks so Ollama has good context to write questions from
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    chunks = splitter.split_text(all_text)
    
    # Shuffle chunks to get diverse questions across all policies
    random.shuffle(chunks)
    
    print(f"✅ Found {len(chunks)} document chunks. Starting synthetic generation (this may take a few minutes)...")
    
    generator_llm = ChatOllama(model=settings.ollama_model, temperature=0.7)
    
    test_cases = []
    
    # We will ask for 2 questions per chunk until we hit our target
    prompt_template = """You are an expert HR and Legal auditor.
    Based ONLY on the text chunk below, generate exactly 2 short, realistic questions an employee might ask, and the factual, concise answer based strictly on the text.
    
    Format your response EXACTLY like this with no other text:
    Q: [question 1]
    A: [answer 1]
    ---
    Q: [question 2]
    A: [answer 2]
    
    Text Chunk:
    {context}
    """
    
    for i, chunk in enumerate(chunks):
        if len(test_cases) >= target_count:
            break
            
        print(f"⏳ Processing chunk {i+1}/{len(chunks)}... ({len(test_cases)}/{target_count} generated)")
        try:
            response = generator_llm.invoke([
                SystemMessage(content="You are a data generation bot. Output exactly in the requested format."),
                HumanMessage(content=prompt_template.format(context=chunk))
            ])
            
            # Parse the Q: and A: blocks
            blocks = response.content.split("---")
            for block in blocks:
                q_match = re.search(r"Q:\s*(.+)", block)
                a_match = re.search(r"A:\s*(.+)", block)
                if q_match and a_match:
                    test_cases.append({
                        "question": q_match.group(1).strip(),
                        "expected_concept": a_match.group(1).strip()
                    })
                    
        except Exception as e:
            print(f"  ⚠️ Skipping chunk due to error: {e}")
            
    # Trim to exact target count
    test_cases = test_cases[:target_count]
    
    print(f"\n🎉 Successfully generated {len(test_cases)} test cases!")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, indent=4)
    print(f"💾 Saved to {output_file}")


if __name__ == "__main__":
    generate_synthetic_test_cases(target_count=50)
