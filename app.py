import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

model = SentenceTransformer("all-MiniLM-L6-v2")



load_dotenv()


app = FastAPI()

def get_embedding(text):
    embedding = model.encode(text)
    return embedding



@app.get("/")
def root():
    return {"message": "AI Resume Analyzer is running"}



if __name__ == "__main__":
    emb = get_embedding("I am a Python backend developer")
    print(len(emb))
    print(emb[:5])
