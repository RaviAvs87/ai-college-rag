import os, re
from pathlib import Path
import importlib
# Load dotenv dynamically so editors do not report a missing static import
# when the selected Python interpreter does not expose its package metadata.
dotenv = importlib.import_module("dotenv")
load_dotenv = dotenv.load_dotenv

# Load pypdf dynamically so editors do not report a missing static import
# when the selected Python interpreter does not expose its package metadata.
pypdf = importlib.import_module("pypdf")
PdfReader = pypdf.PdfReader

# Load Google GenAI dynamically so editors do not report a missing static import
# when the selected Python interpreter does not expose its package metadata.
genai = importlib.import_module("google.genai")
types = importlib.import_module("google.genai.types")

# Load Chroma dynamically so editors do not report a missing static import
# when the selected Python interpreter does not expose its package metadata.
chromadb = importlib.import_module("chromadb")

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
db = chromadb.PersistentClient(path="chroma_db")
DATA = Path("data")

MAX_DIST = 0.65  # bada karo (0.8) agar sahi sawaalon par bhi "nahi hai" aaye

SYSTEM = (
    "You are an AI College Assistant. Answer ONLY from the provided context, "
    "which comes from official college documents. If the answer is not in the "
    "context, say you could not find it and suggest checking the official notice. "
    "Be short and clear. Use bullet points for fees, dates and lists. "
    "Reply in the same language the student used. "
    "Never mention file names, page numbers or 'the provided document' in your answer."
)


def _name(code):
    return "college_" + re.sub(r"[^A-Za-z0-9]", "_", code.strip().upper())


def embed(texts, task):
    out = []
    for i in range(0, len(texts), 50):
        r = client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts[i:i + 50],
            config=types.EmbedContentConfig(task_type=task),
        )
        out += [e.values for e in r.embeddings]
    return out


def chunk(text, size=900, overlap=150):
    text = re.sub(r"\s+", " ", text).strip()
    parts = (text[i:i + size] for i in range(0, len(text), size - overlap))
    return [p for p in parts if p.strip()]


def ingest(code):
    folder = DATA / code.upper()
    pdfs = list(folder.glob("*.pdf"))
    txts = list(folder.glob("*.txt"))
    if not pdfs and not txts:
        raise FileNotFoundError(f"No PDF or TXT files found in data/{code.upper()}/")
    try:
        db.delete_collection(_name(code))
    except Exception:
        pass
    col = db.create_collection(_name(code), metadata={"hnsw:space": "cosine"})
    docs, metas, ids = [], [], []

    # PDF files
    for pdf in pdfs:
        for pno, page in enumerate(PdfReader(str(pdf)).pages, 1):
            for j, c in enumerate(chunk(page.extract_text() or "")):
                docs.append(c)
                metas.append({"file": pdf.name, "page": pno})
                ids.append(f"{pdf.stem}-{pno}-{j}")

    # TXT files (khaali line se section alag hote hain)
    for txt in txts:
        text = txt.read_text(encoding="utf-8-sig", errors="ignore")
        sections = [s for s in re.split(r"\n\s*\n", text) if s.strip()]
        n = 0
        for s in sections:
            for c in chunk(s):
                docs.append(c)
                metas.append({"file": txt.name, "page": 1})
                ids.append(f"{txt.stem}-txt-{n}")
                n += 1

    if not docs:
        raise ValueError("Files have no readable text (scanned PDFs need OCR).")
    vecs = embed(docs, "RETRIEVAL_DOCUMENT")
    for i in range(0, len(docs), 500):
        col.add(ids=ids[i:i + 500], documents=docs[i:i + 500],
                metadatas=metas[i:i + 500], embeddings=vecs[i:i + 500])
    return len(docs)


def exists(code):
    try:
        db.get_collection(_name(code))
        return True
    except Exception:
        return False


def answer(code, question, history=None):
    col = db.get_collection(_name(code))
    hist = ""
    for h in (history or [])[-6:]:
        hist += f"{h['role']}: {h['text']}\n"

    # follow-up ko poora sawaal bana do, taaki search sahi ho
    if hist:
        rw = client.models.generate_content(
            model=CHAT_MODEL,
            contents=f"Chat so far:\n{hist}\nNew question: {question}\n"
                     "Rewrite the new question as one complete standalone question. "
                     "Output only the question.",
            config=types.GenerateContentConfig(temperature=0),
        )
        search_q = (rw.text or question).strip()
    else:
        search_q = question

    qv = embed([search_q], "RETRIEVAL_QUERY")[0]
    res = col.query(query_embeddings=[qv], n_results=6,
                    include=["documents", "distances"])
    print("distances:", res["distances"][0])  # test ke liye, baad me hata dena

    pairs = [(d, x) for d, x in zip(res["documents"][0], res["distances"][0]) if x < MAX_DIST]
    if not pairs:
        with open("unanswered.txt", "a", encoding="utf-8") as f:
            f.write(question + "\n")
        return "Ye jaankari mere paas nahi hai. Kripya college office se sampark karein.", []

    context = "\n\n".join(d for d, _ in pairs)
    r = client.models.generate_content(
        model=CHAT_MODEL,
        contents=f"Context:\n{context}\n\nChat so far:\n{hist}\nQuestion: {search_q}",
        config=types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0.2),
    )
    return r.text, []