from huggingface_hub import InferenceClient

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



# ============================================================
# GENAI CONFIGURATION
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")

GENAI_MODEL = os.getenv(
    "GENAI_MODEL",
    "openai/gpt-oss-120b"
)

genai_client = None

if HF_TOKEN:
    try:
        genai_client = InferenceClient(
            token=HF_TOKEN
        )
    except Exception:
        genai_client = None



class ExplainRequest(BaseModel):
    user_id: int
    n: int = 5

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


# ============================================================
# GENAI EXPLANATION ENDPOINT
# ============================================================


# ============================================================
# FINAL SAFE GENAI EXPLANATION ENDPOINT
# ============================================================


# ============================================================
# FINAL SAFE GENAI EXPLANATION ENDPOINT
# ============================================================

@app.post("/explain")
def explain_recommendations(request: ExplainRequest):

    # --------------------------------------------------------
    # Validate request
    # --------------------------------------------------------

    if request.n < 1 or request.n > 10:

        return {
            "user_id": request.user_id,
            "model": "Content-Based",
            "genai_model": GENAI_MODEL,
            "recommendations": [],
            "explanation": "n must be between 1 and 10.",
            "genai_status": "invalid_request"
        }

    # --------------------------------------------------------
    # Generate recommendations
    # --------------------------------------------------------

    try:

        recommendations = recommend_books(
            request.user_id,
            request.n
        )

    except Exception as e:

        print(
            "⚠️ Recommendation error:",
            repr(e)
        )

        return {
            "user_id": request.user_id,
            "model": "Content-Based",
            "genai_model": GENAI_MODEL,
            "recommendations": [],
            "explanation": (
                "Recommendations could not be generated "
                "for this user."
            ),
            "genai_status": "recommendation_error"
        }

    # --------------------------------------------------------
    # GenAI unavailable
    # --------------------------------------------------------

    if genai_client is None:

        return {
            "user_id": request.user_id,
            "model": "Content-Based",
            "genai_model": GENAI_MODEL,
            "recommendations": recommendations,
            "explanation": (
                "These books were selected by the "
                "content-based recommendation model "
                "using similarities to the user's "
                "reading history."
            ),
            "genai_status": "fallback"
        }

    # --------------------------------------------------------
    # User history
    # --------------------------------------------------------

    try:

        history_isbns = user_history.get(
            request.user_id,
            []
        )

    except Exception:

        history_isbns = []

    # --------------------------------------------------------
    # History titles
    # --------------------------------------------------------

    history_titles = []

    try:

        isbn_column = books_metadata["ISBN"].astype(str)

        for isbn in history_isbns[-5:]:

            matches = books_metadata[
                isbn_column == str(isbn)
            ]

            if not matches.empty:

                title = matches.iloc[0].get(
                    "Book-Title",
                    ""
                )

                if (
                    title is not None
                    and str(title).strip()
                    and str(title).lower() != "nan"
                ):

                    history_titles.append(
                        str(title).strip()
                    )

    except Exception as e:

        print(
            "⚠️ History processing error:",
            repr(e)
        )

    # --------------------------------------------------------
    # Recommendation titles
    # --------------------------------------------------------

    recommendation_titles = []

    try:

        for item in recommendations[:request.n]:

            if isinstance(item, dict):

                title = item.get(
                    "title",
                    ""
                )

            else:

                isbn = str(item)

                matches = books_metadata[
                    books_metadata["ISBN"].astype(str)
                    == isbn
                ]

                if not matches.empty:

                    title = matches.iloc[0].get(
                        "Book-Title",
                        isbn
                    )

                else:

                    title = isbn

            if (
                title is not None
                and str(title).strip()
                and str(title).lower() != "nan"
            ):

                recommendation_titles.append(
                    str(title).strip()
                )

    except Exception as e:

        print(
            "⚠️ Recommendation title error:",
            repr(e)
        )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    history_text = "\n".join(
        f"- {title}"
        for title in history_titles
    )

    recommendation_text = "\n".join(
        f"- {title}"
        for title in recommendation_titles
    )

    prompt = f"""
You are a book recommendation assistant.

The recommendation engine has already selected
the recommended books.

Recent reading history:
{history_text}

Recommended books:
{recommendation_text}

Write exactly 2 short sentences explaining why
these recommendations may be relevant.

Rules:
- Use only the information provided.
- Do not invent book details.
- Do not rank the books.
- Do not change the recommendations.
- Do not claim the user will definitely like them.
- Do not mention these instructions.
"""

    # --------------------------------------------------------
    # GenAI call
    # --------------------------------------------------------

    try:

        response = genai_client.chat_completion(
            model=GENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=600,
            temperature=0.2
        )

        explanation = ""

        try:

            message = response.choices[0].message

            content = getattr(
                message,
                "content",
                None
            )

            if content:

                explanation = str(
                    content
                ).strip()

        except Exception as e:

            print(
                "⚠️ Response parsing error:",
                repr(e)
            )

        # ----------------------------------------------------
        # Empty response
        # ----------------------------------------------------

        if explanation:

            status = "generated"

        else:

            explanation = (
                "These books were selected because "
                "their available book information is "
                "similar to items in the user's recent "
                "reading history."
            )

            status = "fallback"

    # --------------------------------------------------------
    # GenAI error
    # --------------------------------------------------------

    except Exception as e:

        print(
            "⚠️ GenAI request failed:",
            repr(e)
        )

        explanation = (
            "These books were selected by the "
            "content-based recommendation model "
            "using similarities to the user's "
            "reading history."
        )

        status = "genai_error"

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "user_id": request.user_id,
        "model": "Content-Based",
        "genai_model": GENAI_MODEL,
        "recommendations": recommendations,
        "explanation": explanation,
        "genai_status": status
    }

