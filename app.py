"""
Micro Search Engine
===================

This Flask application implements a tiny search engine using TF‑IDF and
cosine similarity.  On startup it reads a CSV dataset of documents,
vectorises the descriptions and stores the resulting TF‑IDF matrix.  When a
user submits a query, the query is vectorised using the same vectoriser and
the top results are returned based on cosine similarity.

To run the application locally:

    python app.py

Then open http://localhost:5000 in your browser.

"""
import math
from pathlib import Path
from typing import List, Dict

import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the dataset from a CSV file and ensure required columns exist.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file containing documents.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'id', 'title', 'description'.
    """
    df = pd.read_csv(csv_path)
    required_columns = {"id", "title", "description"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing)}")
    return df


def build_vectorizer(texts: List[str]) -> (TfidfVectorizer, any):
    """Create a TF‑IDF vectoriser and fit it on the provided texts.

    Parameters
    ----------
    texts : list of str
        List of document descriptions.

    Returns
    -------
    vectorizer : TfidfVectorizer
        Fitted TF‑IDF vectoriser.
    matrix : scipy.sparse matrix
        TF‑IDF matrix of the input texts.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def search_documents(
    query: str,
    vectorizer: TfidfVectorizer,
    matrix,
    df: pd.DataFrame,
    top_k: int = 5,
) -> List[Dict[str, any]]:
    """Search for the most relevant documents given a query.

    Parameters
    ----------
    query : str
        Query string entered by the user.
    vectorizer : TfidfVectorizer
        Previously fitted TF‑IDF vectoriser.
    matrix : scipy.sparse matrix
        Matrix representing the dataset documents in TF‑IDF space.
    df : pd.DataFrame
        DataFrame containing the dataset with 'title' and 'description'.
    top_k : int, default 5
        Number of top results to return.

    Returns
    -------
    list of dict
        List of dictionaries with keys 'title', 'description' and 'score'.
    """
    if not query:
        return []
    # Transform the query into the TF‑IDF space
    query_vec = vectorizer.transform([query])
    # Compute cosine similarity as the dot product of normalised vectors
    # linear_kernel is faster than cosine_similarity because vectors are already L2 normalised by TfidfVectorizer
    cosine_similarities = linear_kernel(query_vec, matrix).flatten()
    # Get indices of the top_k results
    if top_k <= 0:
        top_k = len(cosine_similarities)
    top_indices = cosine_similarities.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        score = float(cosine_similarities[idx])
        if score <= 0:
            continue  # Ignore non‑matching results
        results.append(
            {
                "title": df.iloc[idx]["title"],
                "description": df.iloc[idx]["description"],
                "score": round(score, 4),
            }
        )
    return results


def create_app() -> Flask:
    """Factory function to create and configure the Flask app."""
    app = Flask(__name__)

    # Locate the dataset relative to this file
    data_path = Path(__file__).parent / "data" / "dataset.csv"
    df = load_dataset(data_path)
    vectorizer, matrix = build_vectorizer(df["description"].tolist())

    @app.route("/", methods=["GET"])
    def index():
        """Render the search page and return results if a query is provided."""
        query = request.args.get("q", "").strip()
        results = []
        if query:
            results = search_documents(query, vectorizer, matrix, df)
        return render_template("index.html", query=query, results=results)

    return app


if __name__ == "__main__":
    # Only run the server when executed directly
    app = create_app()
    # Enable debug mode for easier development.  In production you may
    # consider disabling debug mode and using a production WSGI server.
    app.run(host="0.0.0.0", port=5000, debug=True)