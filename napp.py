#   one 
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ✅ Must be first Streamlit command
st.set_page_config(page_title="Book Recommender", layout="centered")

# -------------------------------
# ⚡ Cache for faster loading
# -------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("nartifacts/model.pkl")
    books_name = joblib.load("nartifacts/books_name.pkl")
    final_rating = joblib.load("nartifacts/final_rating.pkl")
    book_pivot = joblib.load("nartifacts/book_pivot.pkl")
    tfidf_matrix = joblib.load("nartifacts/tfidf_matrix.pkl")
    content_sim = joblib.load("nartifacts/content_sim.pkl")
    indices = joblib.load("nartifacts/indices.pkl")
    books = joblib.load("nartifacts/books_df.pkl")
    top_genres = joblib.load("nartifacts/top_genres.pkl")
    return model, books_name, final_rating, book_pivot, tfidf_matrix, content_sim, indices, books, top_genres


model, books_name, final_rating, book_pivot, tfidf_matrix, content_sim, indices, books, top_genres = load_artifacts()

# -------------------------------
# 🌟 Initialize Session State
# -------------------------------
for key, default in {
    "page": "home",
    "selected_books": None,
    "selected_genre": "None",
    "selection_mode": "By Book",
    "keyword_query": "",
    "recommendation_books": [],
    "poster_urls": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# -------------------------------
# 📷 Fetch Poster Using ISBN
# -------------------------------
import os
import urllib.parse

def fetch_poster(book_list):
    """
    Fetches book covers locally if available; otherwise uses placeholder.
    Looks inside ./book_covers/ for saved cover files.
    Does NOT try to download from OpenLibrary.
    """
    poster_urls = []
    base_folder = "book_covers"
    os.makedirs(base_folder, exist_ok=True)

    for title in book_list:
        # Build safe filename (like The_Talented_Mr__Ripley.jpg)
        safe_name = "".join(c if c.isalnum() else "_" for c in title)[:100]
        local_path = os.path.join(base_folder, f"{safe_name}.jpg")

        # If cover exists locally
        if os.path.exists(local_path):
            poster_urls.append(local_path)
        else:
            # Placeholder (with title)
            encoded_title = urllib.parse.quote_plus(f"No Cover: {title.title()[:25]}")
            placeholder_url = f"https://via.placeholder.com/200x300.png?text={encoded_title}"
            poster_urls.append(placeholder_url)
    return poster_urls



# -------------------------------
# 🧠 Hybrid Recommendation Logic
# -------------------------------
def hybrid_recommend(book_name, n=5, genre_filter=None):
    book_name = book_name.strip().lower()
    recommended_books = []

    if book_name in book_pivot.index:
        book_id = np.where(book_pivot.index == book_name)[0][0]
        _, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=n+1)
        recommended_books = [book_pivot.index[i] for i in suggestion[0][1:]]
    elif book_name in indices:
        idx = indices[book_name]
        sim_scores = list(enumerate(content_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+10]
        recommended_books = [books["title"].iloc[i] for i, _ in sim_scores]

    if genre_filter and genre_filter != "None":
        filtered_books = []
        for title in recommended_books:
            genre = books[books["title"] == title]["genres"].values[0]
            if genre_filter.lower() in genre.lower():
                filtered_books.append(title)
        recommended_books = filtered_books[:n]

    poster_url = fetch_poster(recommended_books)
    return recommended_books, poster_url


# -------------------------------
# 🎨 CSS Styling
# -------------------------------
st.markdown("""
<style>
    div.block-container {
        padding-top: 2rem;
        max-width: 1000px;
        margin: auto;
        background: linear-gradient(135deg, #f8f9ff, #eef1ff);
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }

    .header-title {
        text-align: center;
        font-size: 2.4rem;
        color: #5A4FCF;
        margin-bottom: 1rem;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(90,79,207,0.3);
    }

    .recommend-card {
        border: 1px solid rgba(200,200,255,0.4);
        padding: 1rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
        background: rgba(255,255,255,0.9);
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        backdrop-filter: blur(10px);
    }
    .recommend-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 18px rgba(0,0,0,0.1);
    }

    .genre-flex {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
        margin-bottom: 10px;
    }

    .genre-tag {
        display: inline-block;
        background: linear-gradient(135deg, #a18cd1, #fbc2eb);
        padding: 6px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        color: #222;
        font-weight: 600;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        white-space: nowrap;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .genre-tag:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }

    img {
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------
# 🏠 HOME PAGE
# -------------------------------
if st.session_state.page == "home":
    st.markdown("<h1 class='header-title'>📚 Hybrid Book Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("### 🔍 Choose Your Recommendation Type")
    st.session_state.selection_mode = st.radio("", ["By Book", "By Genre", "Search by Keyword"])

    genre_options = sorted(top_genres)

    if st.session_state.selection_mode == "By Book":
        col1, col2 = st.columns(2)
        with col1:
            selected_books = st.selectbox("📖 Select a Book", sorted(books_name))
        with col2:
            selected_genre = st.selectbox("🎭 Filter by Genre (optional)", ["None"] + genre_options)
        st.session_state.selected_books = selected_books
        st.session_state.selected_genre = selected_genre
        st.session_state.keyword_query = ""

    elif st.session_state.selection_mode == "By Genre":
        selected_genre = st.selectbox("🎭 Select a Genre", genre_options)
        st.session_state.selected_books = None
        st.session_state.selected_genre = selected_genre
        st.session_state.keyword_query = ""

    elif st.session_state.selection_mode == "Search by Keyword":
        keyword_query = st.text_input("🔎 Enter author name, title, or keyword")
        st.session_state.keyword_query = keyword_query
        st.session_state.selected_books = None
        st.session_state.selected_genre = "None"

    st.caption("💡 Tip: Filter by genre to make your recommendations more personal!")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🚀 Show Recommendation"):
            st.session_state.page = "recommendation"
    with col2:
        if st.button("🎲 Surprise Me"):
            random_books = books.sort_values(by="rating", ascending=False).sample(5)
            st.session_state.recommendation_books = random_books["title"].tolist()
            st.session_state.poster_urls = fetch_poster(st.session_state.recommendation_books)
            st.session_state.page = "recommendation"


# -------------------------------
# 📖 RECOMMENDATION PAGE
# -------------------------------
elif st.session_state.page == "recommendation":
    st.markdown("<h1 class='header-title'>🎯 Recommended Books for You</h1>", unsafe_allow_html=True)

    if st.session_state.recommendation_books and st.session_state.poster_urls:
        recommendation_books = st.session_state.recommendation_books
        poster_urls = st.session_state.poster_urls
    else:
        if st.session_state.selection_mode == "By Book" and st.session_state.selected_books:
            recommendation_books, poster_urls = hybrid_recommend(
                st.session_state.selected_books,
                genre_filter=st.session_state.selected_genre
            )
        elif st.session_state.selection_mode == "By Genre" and st.session_state.selected_genre:
            genre_books = books[books["genres"].str.contains(st.session_state.selected_genre, case=False)]
            top_books = genre_books.sort_values(by="rating", ascending=False).head(5)
            recommendation_books = top_books["title"].tolist()
            poster_urls = fetch_poster(recommendation_books)
        elif st.session_state.selection_mode == "Search by Keyword" and st.session_state.keyword_query:
            query = st.session_state.keyword_query.lower()
            matched_books = books[
                books["title"].str.lower().str.contains(query) |
                books["author"].str.lower().str.contains(query) |
                books["description"].str.lower().str.contains(query)
            ]
            top_books = matched_books.sort_values(by="rating", ascending=False).head(5)
            recommendation_books = top_books["title"].tolist()
            poster_urls = fetch_poster(recommendation_books)
        else:
            recommendation_books, poster_urls = [], []

        st.session_state.recommendation_books = recommendation_books
        st.session_state.poster_urls = poster_urls

    if recommendation_books:
        for book_title, poster_url in zip(recommendation_books, poster_urls):
            book_info = books[books["title"] == book_title].iloc[0]

            st.markdown("<div class='recommend-card'>", unsafe_allow_html=True)
            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(poster_url, width=250)
                st.markdown(f"<h3 style='color:#5A4FCF; text-align:center;'>📘 {book_title.title()}</h3>", unsafe_allow_html=True)

            with col2:
                # Clean all fields
                author = book_info["author"]
                if isinstance(author, list):
                    author = ", ".join(author)
                else:
                    author = str(author).replace("[", "").replace("]", "").replace("'", "").strip()

                genres = str(book_info["genres"]).replace("[", "").replace("]", "").replace("'", "")
                year = str(book_info.get("year", "Unknown")).replace("[", "").replace("]", "").replace("'", "")
                ratings_count = str(book_info.get("ratingsCount", "N/A")).replace("[", "").replace("]", "").replace("'", "")
                reviews_count = str(book_info.get("reviewsCount", "N/A")).replace("[", "").replace("]", "").replace("'", "")

                st.markdown(f"<b>👩‍💼 Author:</b> {author}", unsafe_allow_html=True)
                st.markdown("<b>🎭 Genre:</b>", unsafe_allow_html=True)
                st.markdown("<div class='genre-flex'>", unsafe_allow_html=True)
                for genre in genres.split(","):
                    st.markdown(f"<span class='genre-tag'>{genre.strip()}</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown(f"<b>📅 Year:</b> {year}", unsafe_allow_html=True)
                st.markdown(f"<b>⭐ Ratings Count:</b> {ratings_count}", unsafe_allow_html=True)
                st.markdown(f"<b>💬 Reviews Count:</b> {reviews_count}", unsafe_allow_html=True)

                rating = book_info.get("rating", 0)
                if pd.notna(rating):
                    stars = "⭐" * int(round(rating))
                    st.markdown(f"<div style='font-size:1.3rem;color:#FFD700;'>{stars}</div>", unsafe_allow_html=True)

                with st.expander("📖 Read Summary"):
                    st.write(book_info["description"])

                if pd.notna(book_info.get("isbn")):
                    st.markdown(f"[🔗 View on Open Library](https://openlibrary.org/isbn/{book_info['isbn']})")

            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("📚 Currently not available — will be available in the near future.")

    if st.button("🔙 Go Back"):
        st.session_state.page = "home"
        st.session_state.selected_books = None
        st.session_state.selected_genre = "None"
        st.session_state.keyword_query = ""
        st.session_state.recommendation_books = []
        st.session_state.poster_urls = []