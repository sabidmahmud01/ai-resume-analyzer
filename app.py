from fastapi import FastAPI
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re


SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "sql",
    "fastapi", "flask", "django", "node", "express", "react", "next.js",
    "mongodb", "postgresql", "mysql", "redis",
    "docker", "kubernetes", "aws", "gcp", "azure",
    "rest", "graphql",
    "git", "linux",
    "machine learning", "deep learning", "nlp",
    "pandas", "numpy", "scikit-learn", "pytorch", "tensorflow"
]

def extract_skills(text: str) -> list[str]:
    text_lower = text.lower()

    found = set()
    for skill in SKILLS:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found.add(skill)

    return sorted(found)

def compute_missing_skills(job_skills: list[str], resume_skills: list[str]) -> list[str]:
    resume_set = set(s.lower() for s in resume_skills)
    missing = [s for s in job_skills if s.lower() not in resume_set]
    return missing


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

    resume_skills = extract_skills(request.resume_text)
    job_skills = extract_skills(request.job_text)
    missing_skills = compute_missing_skills(job_skills, resume_skills)

    return {
        "similarity": similarity,
        "match_percent": match_percent,
        "resume_skills": resume_skills,
        "job_skills": job_skills,
        "missing_skills": missing_skills
    }

