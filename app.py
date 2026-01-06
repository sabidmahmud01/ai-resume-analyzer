from fastapi import FastAPI
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load the model once when the server starts
model = SentenceTransformer("all-MiniLM-L6-v2")


class AnalyzeRequest(BaseModel):
    resume_text: str
    job_text: str


def get_embedding(text: str):
    embedding = model.encode(text)
    return embedding


def calculate_similarity(text1: str, text2: str) -> float:
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)

    similarity = cosine_similarity([emb1], [emb2])[0][0]
    return float(similarity)


def similarity_to_percent(similarity: float) -> float:
    return round(similarity * 100, 2)


@app.get("/")
def root():
    return {"message": "AI Resume Analyzer is running"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    similarity = calculate_similarity(request.resume_text, request.job_text)
    match_percent = similarity_to_percent(similarity)

    return {
        "similarity": similarity,
        "match_percent": match_percent
    }
