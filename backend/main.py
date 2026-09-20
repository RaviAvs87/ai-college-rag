from fastapi import FastAPI, HTTPException  # type: ignore[reportMissingImports]
from fastapi.middleware.cors import CORSMiddleware  # type: ignore[reportMissingImports]
from pydantic import BaseModel  # type: ignore[reportMissingImports]
import rag

app = FastAPI(title="AI College RAG")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Ask(BaseModel):
    college_code: str
    question: str

@app.get("/college/{code}")
def college(code: str):
    if not rag.exists(code):
        raise HTTPException(404, "College not found")
    return {"code": code.upper()}

@app.post("/chat")
def chat(body: Ask):
    if not rag.exists(body.college_code):
        raise HTTPException(404, "College not found")
    try:
        text, sources = rag.answer(body.college_code, body.question)
    except Exception as e:
        raise HTTPException(500, f"AI error: {e}")
    return {"answer": text, "sources": sources}

@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}
