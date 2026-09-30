
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import pickle
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "deployment_artifacts")


# ============================================================
# LOAD ARTIFACTS
# ============================================================

with open(
    os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.pkl"),
    "rb"
) as f:
    tfidf_vectorizer = pickle.load(f)


tfidf_matrix = sparse.load_npz(
    os.path.join(ARTIFACT_DIR, "tfidf_matrix.npz")
)


with open(
    os.path.join(ARTIFACT_DIR, "isbn_to_content_index.pkl"),
    "rb"
) as f:
    isbn_to_content_index = pickle.load(f)


with open(
    os.path.join(ARTIFACT_DIR, "content_index_to_isbn.pkl"),
    "rb"
) as f:
    content_index_to_isbn = pickle.load(f)


with open(
    os.path.join(ARTIFACT_DIR, "user_history.pkl"),
    "rb"
) as f:
    user_history = pickle.load(f)


books = pd.read_csv(
    os.path.join(ARTIFACT_DIR, "books_metadata.csv"),
    dtype={"ISBN": str}
)


with open(
    os.path.join(ARTIFACT_DIR, "config.pkl"),
    "rb"
) as f:
    config = pickle.load(f)


# ============================================================
# CONTENT KNN
# ============================================================

content_knn = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

content_knn.fit(tfidf_matrix)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Book Recommendation API",
    description="Content-Based Book Recommendation System",
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class RecommendationRequest(BaseModel):
    user_id: int
    n: Optional[int] = 10


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Book Recommendation API is running",
        "model": config.get(
            "model_name",
            "Content-Based"
        )
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": config.get(
            "model_name",
            "Content-Based"
        ),
        "books": len(isbn_to_content_index),
        "users": len(user_history)
    }


# ============================================================
# BOOK METADATA
# ============================================================

def get_book_details(isbn):
    row = books[books["ISBN"] == str(isbn)]

    if row.empty:
        return {
            "isbn": str(isbn),
            "title": "Unknown",
            "author": "Unknown",
            "publisher": "Unknown"
        }

    row = row.iloc[0]

    result = {
        "isbn": str(isbn),
        "title": str(row.get("Book-Title", "Unknown")),
        "author": str(row.get("Book-Author", "Unknown")),
        "publisher": str(row.get("Publisher", "Unknown"))
    }

    for col, key in [
        ("Image-URL-S", "image_url_s"),
        ("Image-URL-M", "image_url_m"),
        ("Image-URL-L", "image_url_l")
    ]:
        if col in books.columns:
            value = row.get(col)

            if pd.notna(value):
                result[key] = str(value)

    return result


# ============================================================
# CONTENT RECOMMENDATION
# ============================================================

def recommend_books(history_isbns, n=10):

    history_isbns = [
        str(isbn)
        for isbn in history_isbns
    ]

    valid_indices = []

    for isbn in history_isbns:
        if isbn in isbn_to_content_index:
            valid_indices.append(
                int(isbn_to_content_index[isbn])
            )

    if not valid_indices:
        return []

    # Use recent history
    valid_indices = valid_indices[-5:]

    scores = {}

    for idx in valid_indices:

        distances, neighbors = content_knn.kneighbors(
            tfidf_matrix[idx],
            n_neighbors=min(
                11,
                tfidf_matrix.shape[0]
            )
        )

        for distance, neighbor_idx in zip(
            distances[0],
            neighbors[0]
        ):

            neighbor_idx = int(neighbor_idx)

            isbn = content_index_to_isbn[
                neighbor_idx
            ]

            isbn = str(isbn)

            # Never recommend already seen books
            if isbn in history_isbns:
                continue

            similarity = 1.0 - float(distance)

            if isbn not in scores:
                scores[isbn] = 0.0

            scores[isbn] += similarity

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for isbn, score in ranked[:n]:

        book = get_book_details(isbn)

        book["score"] = round(
            float(score),
            6
        )

        recommendations.append(book)

    return recommendations


# ============================================================
# RECOMMENDATION ENDPOINT
# ============================================================

@app.post("/recommend")
def recommend(request: RecommendationRequest):

    user_id = int(request.user_id)
    n = int(request.n)

    if n < 1 or n > 50:
        raise HTTPException(
            status_code=400,
            detail="n must be between 1 and 50"
        )

    if user_id not in user_history:

        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} not found"
        )

    history = user_history[user_id]

    recommendations = recommend_books(
        history,
        n
    )

    return {
        "user_id": user_id,
        "model": "Content-Based",
        "history_count": len(history),
        "recommendation_count": len(
            recommendations
        ),
        "recommendations": recommendations
    }


# ============================================================
# USER HISTORY ENDPOINT
# ============================================================

@app.get("/users/{user_id}/history")
def get_history(user_id: int):

    if user_id not in user_history:
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} not found"
        )

    history = user_history[user_id]

    books_list = [
        get_book_details(isbn)
        for isbn in history[-20:]
    ]

    return {
        "user_id": user_id,
        "history_count": len(history),
        "history": books_list
    }
