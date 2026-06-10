# import joblib
# import streamlit as st
# import numpy as np

# st.header("    Book Recommendation System Using Colabrative Filtering Systems ")


# model = joblib.load('artifacts/model.pkl')
# books_name = joblib.load('artifacts/books_name.pkl')
# final_rating = joblib.load('artifacts/final_rating.pkl')
# book_pivot = joblib.load('artifacts/book_pivot.pkl')


# def fetch_poster(suggestion):
#     book_name = []
#     ids_index = []
#     poster_url = []

#     for book_id in suggestion:
#         book_name.append(book_pivot.index[book_id])
    
#     for name in book_name[0]:
#         ids = np.where(final_rating['title'] == name)[0][0]
#         ids_index.append(ids)
    
#     for ids in ids_index:
#         url = final_rating.iloc[ids]['img_url']
#         poster_url.append(url)

#     return poster_url



# def recommend_books(book_name):
#     book_list =[]
#     book_id = np.where(book_pivot.index == book_name)[0][0]
#     distance,suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1),n_neighbors=6)

#     poster_url = fetch_poster(suggestion)

#     for i in range(len(suggestion)):
#         books= book_pivot.index[suggestion[i]]
#         for j in books:
#             book_list.append(j)
#     return book_list, poster_url



# selected_books = st.selectbox(
#     "type or select a book",
#     books_name
#     )


# if st.button('Show Recommendation'):
#     recommendation_books, poster_url = recommend_books(selected_books)
#     col1, col2, col3, col4, col5= st.columns(5)

#     with col1:
#         st.text(recommendation_books[1])
#         st.image(poster_url[1])
    
#     with col2:
#         st.text(recommendation_books[2])
#         st.image(poster_url[2])

#     with col3:
#         st.text(recommendation_books[3])
#         st.image(poster_url[3])

#     with col4:
#         st.text(recommendation_books[4])
#         st.image(poster_url[4])

#     with col5:
#         st.text(recommendation_books[5])
#         st.image(poster_url[5])



import joblib
import streamlit as st
import numpy as np

# Load models and data
model = joblib.load('artifacts/model.pkl')
books_name = joblib.load('artifacts/books_name.pkl')
final_rating = joblib.load('artifacts/final_rating.pkl')
book_pivot = joblib.load('artifacts/book_pivot.pkl')

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "home"
if "selected_books" not in st.session_state:
    st.session_state.selected_books = None

# Function to fetch poster URLs
def fetch_poster(suggestion):
    poster_url = []
    for book_id in suggestion[0][1:]:  # skip the first (input book)
        book_name = book_pivot.index[book_id]
        ids = np.where(final_rating['title'] == book_name)[0][0]
        url = final_rating.iloc[ids]['img_url']
        poster_url.append(url)
    return poster_url

# Function to recommend books
def recommend_books(book_name):
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

    recommended_books = [book_pivot.index[i] for i in suggestion[0][1:]]  # skip the input book
    poster_url = fetch_poster(suggestion)

    return recommended_books, poster_url

# -------------------------------
# PAGE: HOME
# -------------------------------
if st.session_state.page == "home":
    st.header("📚 Book Recommendation System Using Collaborative Filtering")

    selected_books = st.selectbox("Type or select a book", books_name)

    if st.button("Show Recommendation"):
        st.session_state.selected_books = selected_books
        st.session_state.page = "recommendation"

# -------------------------------
# PAGE: RECOMMENDATION
# -------------------------------
elif st.session_state.page == "recommendation" and st.session_state.selected_books:
    st.header("🎯 Recommended Books for You")

    recommendation_books, poster_url = recommend_books(st.session_state.selected_books)

    for i in range(5):
        with st.container():
            st.subheader(f"📖 {recommendation_books[i]}")
            st.image(poster_url[i], use_container_width=True)
            st.markdown("---")

    if st.button("🔙 Go Back"):
        st.session_state.page = "home"
        st.session_state.selected_books = None