from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

DATA_PATH = Path(__file__).parent / "books_preprocessed.csv"

REQUIRED_COLUMNS = [
    "Book_ID",
    "Book_Name",
    "Author",
    "Genre",
    "Description",
    "Rating",
    "Publish_Year",
]

books = pd.DataFrame()
similarity_matrix = None


def load_books():
    """Load and clean the CSV dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing CSV columns: " + ", ".join(missing)
        )

    for col in [
        "Book_ID",
        "Book_Name",
        "Author",
        "Genre",
        "Description",
    ]:
        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["Publish_Year"] = pd.to_numeric(
        df["Publish_Year"], errors="coerce"
    )

    df = df.dropna(subset=["Rating", "Publish_Year"])

    df = df[
        df["Book_ID"].ne("")
        & df["Book_Name"].ne("")
        & df["Author"].ne("")
    ].copy()

    df = df.drop_duplicates(
        subset=["Book_ID"]
    ).reset_index(drop=True)

    if len(df) < 2:
        raise ValueError(
            "At least 2 valid books are required."
        )

    df["Combined_Text"] = (
        df["Book_Name"] + " "
        + df["Author"] + " "
        + df["Genre"] + " "
        + df["Description"]
    ).str.strip()

    return df


def build_model(df):
    """Build TF-IDF vectors and cosine similarities."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2),
        max_features=10000,
    )

    matrix = vectorizer.fit_transform(df["Combined_Text"])

    return cosine_similarity(matrix, dense_output=True)


def serialize_book(row, index=None, score=None):
    """Convert a book to JSON-safe data."""
    result = {
        "Book_ID": str(row["Book_ID"]),
        "Book_Name": str(row["Book_Name"]),
        "Author": str(row["Author"]),
        "Genre": str(row["Genre"]),
        "Description": str(row["Description"]),
        "Rating": float(row["Rating"]),
        "Publish_Year": int(row["Publish_Year"]),
    }

    if index is not None:
        result["_index"] = int(index)

    if score is not None:
        result["Similarity"] = float(score)

    return result


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/books")
def get_books():
    """Return books and dashboard statistics."""
    if books.empty:
        return jsonify({"error": "No books loaded."}), 500

    records = [
        serialize_book(row)
        for _, row in books.iterrows()
    ]

    genres = sorted(
        g for g in books["Genre"].unique() if g.strip()
    )

    return jsonify({
        "books": records,
        "total_books": len(books),
        "total_genres": len(genres),
        "genres": genres,
        "average_rating": round(
            float(books["Rating"].mean()), 2
        ),
    })


@app.get("/api/recommend/<book_id>")
def get_recommendations(book_id):
    """Find similar books using the Python ML model."""
    matches = books.index[
        books["Book_ID"].astype(str) == book_id
    ].tolist()

    if not matches:
        return jsonify({"error": "Book not found."}), 404

    selected_index = matches[0]

    try:
        count = int(request.args.get("count", 5))
        count = max(1, min(count, 10))
        min_rating = float(
            request.args.get("min_rating", 0)
        )

        if not 0 <= min_rating <= 5:
            raise ValueError("Minimum rating must be 0–5.")

    except (ValueError, TypeError):
        return jsonify({"error": "Invalid filter values."}), 400

    genre = request.args.get("genre", "").strip()
    search = request.args.get("search", "").strip().casefold()
    sort_by = request.args.get("sort", "similarity")

    allowed_sorts = {"similarity", "rating", "year", "title"}
    if sort_by not in allowed_sorts:
        sort_by = "similarity"

    candidates = []

    for idx, score in enumerate(similarity_matrix[selected_index]):
        if idx == selected_index:
            continue

        row = books.iloc[idx]

        if float(row["Rating"]) < min_rating:
            continue

        if genre and row["Genre"] != genre:
            continue

        if search:
            searchable = (
                str(row["Book_Name"]) + " "
                + str(row["Author"])
            ).casefold()

            if search not in searchable:
                continue

        candidates.append({
            **serialize_book(row, index=idx, score=score)
        })

    if sort_by == "rating":
        candidates.sort(
            key=lambda b: (-b["Rating"], -b["Similarity"])
        )
    elif sort_by == "year":
        candidates.sort(
            key=lambda b: (-b["Publish_Year"], -b["Similarity"])
        )
    elif sort_by == "title":
        candidates.sort(
            key=lambda b: b["Book_Name"].casefold()
        )
    else:
        candidates.sort(
            key=lambda b: -b["Similarity"]
        )

    return jsonify({
        "selected_book": serialize_book(
            books.iloc[selected_index]
        ),
        "recommendations": candidates[:count],
        "total_matches": len(candidates),
    })


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ready",
        "total_books": len(books),
    })


# Load the CSV and build the model once when the app starts.
try:
    books = load_books()
    similarity_matrix = build_model(books)
    print(f"Loaded {len(books)} books successfully.")
except Exception as exc:
    raise RuntimeError(
        f"Could not initialise the recommendation system: {exc}"
    ) from exc


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )