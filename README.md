# Book Recommendation System

A machine learning based Book Recommendation System built using
multiple recommendation techniques and deployed using FastAPI.

## Models

- Popularity-Based Recommendation
- Content-Based Recommendation
- Collaborative Filtering
- Singular Value Decomposition (SVD)
- Hybrid Recommendation

## Machine Learning Workflow

Dataset
-> Data Cleaning
-> Feature Engineering
-> Train/Test Split
-> Model Training
-> Leakage-Free Evaluation
-> Model Comparison
-> Content-Based Model Selection
-> Deployment
-> FastAPI API

## Evaluation

The models were evaluated using:

- Precision@10
- Recall@10
- F1@10
- Hit Rate@10

Evaluation setup:

- 2,500 evaluation users
- One relevant book held out per user
- Training-only model fitting
- Top-10 recommendations

## Evaluation Results

| Model | Precision@10 | Recall@10 | F1@10 | Hits |
|---|---:|---:|---:|---:|
| Content-Based | 0.002520 | 0.025200 | 0.004582 | 63 |
| SVD | 0.002160 | 0.021600 | 0.003927 | 54 |
| Hybrid | 0.002080 | 0.020800 | 0.003782 | 52 |
| Collaborative Filtering | 0.001480 | 0.014800 | 0.002691 | 37 |
| Popularity-Based | 0.000040 | 0.000400 | 0.000073 | 1 |

The Content-Based model achieved the highest measured F1@10 in this
specific evaluation experiment and was selected for the current
deployment pipeline.

## Project Structure

book-recommendation-system/

    notebooks/
        Book_Recommendation_System.ipynb

    fastapi_app/
        main.py
        requirements.txt
        start.sh

        deployment_artifacts/
            books_metadata.csv
            config.pkl
            content_index_to_isbn.pkl
            isbn_to_content_index.pkl
            model_info.pkl
            tfidf_matrix.npz
            tfidf_vectorizer.pkl
            user_history.pkl

        tests/

    docs/
    .gitignore
    README.md

## API Endpoints

### Health Check

GET /health

### Recommendations

POST /recommend

Example request:

{
    "user_id": 276747,
    "n": 10
}

### User History

GET /users/{user_id}/history

## Technologies

- Python
- Pandas
- NumPy
- SciPy
- Scikit-learn
- FastAPI
- Uvicorn
- Pydantic
- GitHub
- Render

## Run Locally

Install dependencies:

pip install -r fastapi_app/requirements.txt

Start the API:

uvicorn fastapi_app.main:app --host 0.0.0.0 --port 8000

API documentation:

http://127.0.0.1:8000/docs

## Deployment

The FastAPI backend can be deployed using Render.

Start command:

uvicorn main:app --host 0.0.0.0 --port $PORT

## Author

B.Tech Machine Learning Project

Book Recommendation System
