from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import requests
import re
import json
import uvicorn

# ------------------------
# 📄 Step 1: PDF Text Extractor
# ------------------------

def download_pdf(url, save_path="temp.pdf"):
    response = requests.get(url)
    with open(save_path, "wb") as f:
        f.write(response.content)
    return save_path

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# ------------------------
# 📄 Step 2: Clause Splitter
# ------------------------

def split_into_clauses(text):
    pattern = r"(?=\n?\s*\d+(\.\d+)+\s)"
    raw_clauses = re.split(pattern, text)
    clauses = []
    for i, clause in enumerate(raw_clauses):
        clean = clause.strip()
        if len(clean) > 30:
            clauses.append({"id": f"Clause-{i+1}", "text": clean})
    return clauses

# ------------------------
# 🧠 Step 3: Semantic Matching
# ------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    return model.encode(texts, convert_to_tensor=True)

def find_best_match(query, clauses):
    query_embedding = embed_texts([query])
    clause_texts = [c["text"] for c in clauses]
    clause_embeddings = embed_texts(clause_texts)

    cosine_scores = util.cos_sim(query_embedding, clause_embeddings)[0]
    top_idx = cosine_scores.argmax().item()
    top_clause = clauses[top_idx]
    score = float(cosine_scores[top_idx])

    decision = "yes" if score > 0.7 else ("maybe" if score > 0.5 else "no")

    return {
        "query": query,
        "decision": decision,
        "justification": f"Matched clause with cosine similarity score of {score:.4f}",
        "matched_clause_id": top_clause["id"],
        "matched_clause_text": top_clause["text"],
        "confidence_score": score
    }

# ------------------------
# (Optional) LLM Intent Parser
# ------------------------

def extract_intent_and_entities(query):
    return {
        "intent": "coverage_question",
        "entities": {
            "topic": query.lower()
        }
    }

# ------------------------
# 🚀 FastAPI Server
# ------------------------

app = FastAPI()

class HackRxRequest(BaseModel):
    pdf_url: str
    query: str

@app.post("/hackrx/run")
async def run_hackrx(req: HackRxRequest):
    try:
        pdf_path = download_pdf(req.pdf_url)
        text = extract_text_from_pdf(pdf_path)
        clauses = split_into_clauses(text)

        if not clauses:
            return {"error": "No valid clauses found in the document."}

        parsed = extract_intent_and_entities(req.query)
        result = find_best_match(req.query, clauses)

        return {
            "input_query": req.query,
            "intent": parsed["intent"],
            "entities": parsed["entities"],
            "response": result
        }

    except Exception as e:
        return {"error": str(e)}

# ------------------------
# Local Dev Run
# ------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
