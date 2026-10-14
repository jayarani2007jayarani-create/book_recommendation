# Book Recommendation System

A content-based book recommendation web app using Python, Pandas, Scikit-learn, TF-IDF, Cosine Similarity, and Streamlit.

## Files
- `app.py` — Streamlit application and recommendation model
- `books_preprocessed.csv` — preprocessed book dataset
- `requirements.txt` — Python dependencies

## Run on Windows
1. Install Python 3.10 or newer and VS Code.
2. Open this folder in VS Code.
3. Open Terminal and run:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

4. Streamlit will show a local URL (usually http://localhost:8501). Open it in your browser.

## How it works
1. Combines book name, author, genre, and description into text.
2. Uses TF-IDF to convert text into numeric vectors.
3. Uses cosine similarity to rank books with similar text.
4. Shows the top matching books with metadata.

## Notes
- The similarity percentage is text similarity, not a verified probability that a user will like a book.
- The app uses the book details present in the supplied dataset. External factual verification is not included.
