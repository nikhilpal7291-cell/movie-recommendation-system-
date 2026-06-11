import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
import time

st.set_page_config(layout="centered", page_title="Movie Recommender")

# --- Load the pickled model components ---
@st.cache_resource
def load_model_components():
    try:
        with open('df.pkl', 'rb') as file:
            df = pickle.load(file)
        with open('tfidf.pkl', 'rb') as file:
            tfidf = pickle.load(file)
        with open('indices.pkl', 'rb') as file:
            indices = pickle.load(file)
        with open('tfidf_matrix.pkl', 'rb') as file:
            tfidf_matrix = pickle.load(file)
        return df, tfidf, indices, tfidf_matrix
    except FileNotFoundError:
        st.error("One or more model files were not found. Please ensure 'df.pkl', 'tfidf.pkl', 'indices.pkl', and 'tfidf_matrix.pkl' are in the same directory.")
        st.stop()

df, tfidf, indices, tfidf_matrix = load_model_components()

# --- Define the recommend function ---
def recommend(title, n=10):
    if title not in indices:
        return pd.Series(['Movie not found. Please check the spelling or try another title.'])

    idx = indices[title]
    sim_score = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_idx = sim_score.argsort()[::-1][1:n+1]
    return df['title'].iloc[similar_idx]

# --- Streamlit UI ---
st.sidebar.title('🎬 About the App')
st.sidebar.info(
    "This is a simple Movie Recommender System built using TF-IDF and cosine similarity. "
    "Enter a movie title, and it will suggest similar movies from its database."
)

st.title('🍿 Movie Recommender System')
st.markdown("### Discover your next favorite movie!")

# Input for movie title
# Using st.selectbox for a more controlled input, pre-filling with popular movies
all_movie_titles = df['title'].tolist()
movie_title = st.selectbox(
    'Select a movie or type to search:',
    options=[''] + sorted(list(set(all_movie_titles))),
    index=0,  # Default to empty
    help='Start typing a movie title or select from the dropdown.'
)

if st.button('Get Recommendations'):
    if movie_title:
        with st.spinner('Finding recommendations...'):
            time.sleep(1) # Simulate processing time
            recommendations = recommend(movie_title)

        if recommendations.iloc[0] == 'Movie not found. Please check the spelling or try another title.':
            st.warning(recommendations.iloc[0])
        else:
            st.subheader(f'Movies similar to "{movie_title}":')
            for i, rec_movie in enumerate(recommendations):
                st.success(f"**{i+1}.** {rec_movie}")
    else:
        st.warning('Please select or enter a movie title.')

st.markdown("""
--- 
<small>*Powered by TF-IDF and Cosine Similarity*</small>
""", unsafe_allow_html=True)
