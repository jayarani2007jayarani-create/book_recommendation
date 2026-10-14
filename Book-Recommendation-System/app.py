import re
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "books_preprocessed.csv"


# ----------------------------
# Load and prepare dataset
# ----------------------------
@st.cache_data
def load_books(csv_path: str) -> pd.DataFrame:
    books = pd.read_csv(csv_path)

    required_columns = [
        "Book_ID", "Book_Name", "Author", "Genre",
        "Description", "Rating", "Publish_Year"
    ]
    missing_columns = [col for col in required_columns if col not in books.columns]
    if missing_columns:
        raise ValueError(
            "Dataset-ல் தேவையான columns இல்லை: " + ", ".join(missing_columns)
        )

    # Clean text fields
    text_columns = ["Book_ID", "Book_Name", "Author", "Genre", "Description"]
    for col in text_columns:
        books[col] = books[col].fillna("").astype(str).str.strip()
        books[col] = books[col].str.replace(r"\s+", " ", regex=True)

    books["Rating"] = pd.to_numeric(books["Rating"], errors="coerce")
    books["Publish_Year"] = pd.to_numeric(
        books["Publish_Year"], errors="coerce"
    )

    # Drop rows missing the minimum information needed by the recommender
    books = books.dropna(subset=["Rating", "Publish_Year"])
    books = books[
        (books["Book_Name"] != "") &
        (books["Author"] != "") &
        (books["Book_ID"] != "")
    ].copy()

    # Build text used by the content-based recommendation model
    books["Combined_Text"] = (
        books["Book_Name"] + " " +
        books["Author"] + " " +
        books["Genre"] + " " +
        books["Description"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    books = books.drop_duplicates(subset=["Book_ID"]).reset_index(drop=True)
    return books


@st.cache_resource
def build_recommendation_model(text_data: tuple):
    """Build TF-IDF vectors and cosine similarity matrix."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2),
        max_features=10000,
    )
    matrix = vectorizer.fit_transform(list(text_data))
    similarity = cosine_similarity(matrix, dense_output=True)
    return vectorizer, similarity


def recommend_books(books: pd.DataFrame, similarity, selected_index: int,
                    number_of_recommendations: int = 5) -> pd.DataFrame:
    scores = list(enumerate(similarity[selected_index]))
    scores = [
        (idx, score) for idx, score in scores
        if idx != selected_index
    ]
    scores.sort(key=lambda item: item[1], reverse=True)

    results = []
    for idx, score in scores:
        row = books.iloc[idx]
        results.append({
            "Book_ID": row["Book_ID"],
            "Book_Name": row["Book_Name"],
            "Author": row["Author"],
            "Genre": row["Genre"],
            "Description": row["Description"],
            "Rating": row["Rating"],
            "Publish_Year": int(row["Publish_Year"]),
            "Similarity": float(score),
            "_index": idx,
        })
        if len(results) >= number_of_recommendations:
            break

    return pd.DataFrame(results)


def safe_rating(value) -> str:
    try:
        return f"{float(value):.2f} / 5"
    except (TypeError, ValueError):
        return "Not available"


# ----------------------------
# Load data and handle errors
# ----------------------------
st.markdown(
    """
    <style>
    .main-title {font-size: 2.4rem; font-weight: 800; margin-bottom: 0.2rem;}
    .subtitle {color: #6b7280; font-size: 1.05rem; margin-bottom: 1.4rem;}
    .book-card {border: 1px solid rgba(128,128,128,.35); border-radius: 14px;
                padding: 16px; margin-bottom: 10px; min-height: 210px;}
    .book-title {font-size: 1.1rem; font-weight: 700;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">📚 Book Recommendation System</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Discover your next favourite book using '
    'content-based Machine Learning.</div>',
    unsafe_allow_html=True,
)

if not DATA_PATH.exists():
    st.error(
        "books_preprocessed.csv file கிடைக்கவில்லை. "
        "அதை app.py இருக்கும் அதே folder-ல் வைக்கவும்."
    )
    st.stop()

try:
    books = load_books(str(DATA_PATH))
except Exception as exc:
    st.error(f"Dataset load செய்ய முடியவில்லை: {exc}")
    st.stop()

if len(books) < 2:
    st.error("Recommendations உருவாக்க குறைந்தது 2 valid books தேவை.")
    st.stop()

try:
    _, similarity_matrix = build_recommendation_model(
        tuple(books["Combined_Text"].tolist())
    )
except ValueError as exc:
    st.error(f"Recommendation model உருவாக்க முடியவில்லை: {exc}")
    st.stop()


# ----------------------------
# Sidebar filters
# ----------------------------
st.sidebar.header("🔎 Search & Filters")
search_text = st.sidebar.text_input(
    "Book name அல்லது author search செய்",
    placeholder="உதா: 1984",
)

all_genres = sorted(
    genre for genre in books["Genre"].dropna().astype(str).unique()
    if genre.strip()
)
selected_genres = st.sidebar.multiselect(
    "Genre தேர்வு செய்",
    options=all_genres,
    default=[],
)

min_rating = float(books["Rating"].min())
max_rating = float(books["Rating"].max())
rating_range = st.sidebar.slider(
    "Rating range",
    min_value=0.0,
    max_value=5.0,
    value=(max(0.0, min_rating), min(5.0, max_rating)),
    step=0.1,
)

recommendation_count = st.sidebar.selectbox(
    "எத்தனை recommendations வேண்டும்?",
    options=[5, 10],
    index=0,
)

st.sidebar.caption(
    "குறிப்பு: Genre மற்றும் rating filters, book தேர்வு மற்றும் "
    "recommendation பட்டியலுக்குப் பயன்படுத்தப்படும்."
)


# ----------------------------
# Main dashboard metrics
# ----------------------------
m1, m2, m3 = st.columns(3)
m1.metric("📚 Total Books", len(books))
m2.metric("🏷️ Genres", books["Genre"].nunique())
m3.metric("⭐ Average Rating", f"{books['Rating'].mean():.2f} / 5")

st.divider()

# Apply filters for browsing/selecting
filtered = books.copy()
if selected_genres:
    filtered = filtered[filtered["Genre"].isin(selected_genres)]

filtered = filtered[
    filtered["Rating"].between(rating_range[0], rating_range[1])
]

if search_text.strip():
    q = re.escape(search_text.strip())
    mask = (
        filtered["Book_Name"].str.contains(q, case=False, na=False, regex=True) |
        filtered["Author"].str.contains(q, case=False, na=False, regex=True)
    )
    filtered = filtered[mask]

if filtered.empty:
    st.warning("இந்த search / filters-க்கு books கிடைக்கவில்லை. Filters மாற்றிப் பாருங்கள்.")
    st.stop()

book_options = {
    f"{row['Book_Name']} — {row['Author']}": int(idx)
    for idx, row in filtered.iterrows()
}

st.subheader("✨ Find Similar Books")
selected_label = st.selectbox(
    "உனக்குப் பிடித்த book-ஐ தேர்வு செய்",
    options=list(book_options.keys()),
)
selected_index = book_options[selected_label]
selected_book = books.loc[selected_index]

left, right = st.columns([1, 2])
with left:
    st.markdown("### 📖 Selected Book")
    st.markdown(f"**{selected_book['Book_Name']}**")
    st.write(f"**Author:** {selected_book['Author']}")
    st.write(f"**Genre:** {selected_book['Genre']}")
    st.write(f"**Rating:** {safe_rating(selected_book['Rating'])}")
    st.write(f"**Published:** {int(selected_book['Publish_Year'])}")
with right:
    st.markdown("### About this book")
    st.write(selected_book["Description"] or "Description available இல்லை.")
    st.info(
        "இந்த system, book title, author, genre மற்றும் description-ஐ "
        "TF-IDF மூலம் process செய்து Cosine Similarity அடிப்படையில் "
        "ஒத்த books-ஐ பரிந்துரைக்கிறது."
    )

if st.button("🔍 Recommend Similar Books", type="primary", use_container_width=True):
    recommendations = recommend_books(
        books, similarity_matrix, selected_index, recommendation_count
    )

    # Respect selected genre and rating filters in the displayed recommendations.
    recommendations = recommendations[
        recommendations["Rating"].between(rating_range[0], rating_range[1])
    ]
    if selected_genres:
        recommendations = recommendations[
            recommendations["Genre"].isin(selected_genres)
        ]

    st.session_state["recommendations"] = recommendations.drop(columns=["_index"], errors="ignore")
    st.session_state["recommended_for"] = selected_book["Book_Name"]

if "recommendations" in st.session_state:
    st.divider()
    st.subheader(f"🎯 Recommended for: {st.session_state.get('recommended_for', '')}")
    recs = st.session_state["recommendations"]

    if recs.empty:
        st.warning(
            "இந்த filters-க்கு similar books கிடைக்கவில்லை. "
            "Genre அல்லது rating filter-ஐ மாற்றி மீண்டும் முயற்சி செய்."
        )
    else:
        cols_per_row = 2
        for start in range(0, len(recs), cols_per_row):
            cols = st.columns(cols_per_row)
            for offset, col in enumerate(cols):
                pos = start + offset
                if pos >= len(recs):
                    break
                book = recs.iloc[pos]
                with col:
                    st.markdown('<div class="book-card">', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="book-title">{pos + 1}. '
                        f'{book["Book_Name"]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.write(f"✍️ **Author:** {book['Author']}")
                    st.write(f"🏷️ **Genre:** {book['Genre']}")
                    st.write(f"⭐ **Rating:** {safe_rating(book['Rating'])}")
                    st.write(f"📅 **Year:** {int(book['Publish_Year'])}")
                    st.write(book["Description"] or "Description available இல்லை.")
                    st.progress(
                        max(0.0, min(1.0, float(book["Similarity"]))),
                        text=f"Text similarity: {float(book['Similarity']):.0%}",
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

        csv_download = recs.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Download recommendations as CSV",
            data=csv_download,
            file_name="recommended_books.csv",
            mime="text/csv",
        )

with st.expander("📚 Browse all books in the dataset"):
    browse = books.copy()
    if selected_genres:
        browse = browse[browse["Genre"].isin(selected_genres)]
    browse = browse[browse["Rating"].between(rating_range[0], rating_range[1])]
    st.dataframe(
        browse[[
            "Book_ID", "Book_Name", "Author", "Genre",
            "Rating", "Publish_Year", "Description"
        ]],
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Academic prototype • Content-based filtering • "
    "Similarity score is text similarity, not a guaranteed user preference."
)
