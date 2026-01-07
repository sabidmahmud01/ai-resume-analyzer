# AI Resume Analyzer

An AI-powered backend service that semantically compares a resume with a job description and identifies skill gaps.

## Features
- Semantic resume–job matching using sentence embeddings
- Cosine similarity–based match score
- Skill extraction from free text
- Missing skill identification
- FastAPI backend with interactive docs

## How It Works
1. Resume and job text are converted into embeddings using a local Sentence Transformer model.
2. Cosine similarity is used to compute a semantic match score.
3. A curated skill list is used to extract skills from both texts.
4. Skills required by the job but missing from the resume are identified.

## Tech Stack
- Python
- FastAPI
- Sentence Transformers (local embeddings)
- scikit-learn
- spaCy

## API Usage

### POST /analyze
**Request**
```json
{
  "resume_text": "Python backend developer with FastAPI, MongoDB, Docker.",
  "job_text": "Looking for a backend engineer with Python, SQL, AWS, and Docker."
}
