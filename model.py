from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and preprocess data
def load_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')  # Replace with the path to your dataset
    credits = pd.read_csv('tmdb_5000_credits.csv')
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)

    def process_feature(feature):
        return feature.lower().replace(" ", "")

    movies['genres'] = movies['genres'].apply(process_feature)
    movies['keywords'] = movies['keywords'].apply(process_feature)
    movies['cast'] = movies['cast'].apply(process_feature)
    movies['crew'] = movies['crew'].apply(process_feature)
    movies['tags'] = (
        movies['overview'] + " " + movies['genres'] + " " +
        movies['keywords'] + " " + movies['cast'] + " " + movies['crew']
    )
    new_df = movies[['movie_id', 'title', 'tags']]
    return new_df

# Build similarity matrix
def build_similarity_matrix(new_df):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return similarity

# Get movie recommendations
def get_recommendations(movie_title, new_df, similarity):
    try:
        movie_index = new_df[new_df['title'] == movie_title].index[0]
    except IndexError:
        return ["Movie not found. Please try another title."]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = [new_df.iloc[i[0]].title for i in movie_list]
    return recommended_movies

# Route to display the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and show recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form['movie_title']
    new_df = load_data()
    similarity = build_similarity_matrix(new_df)
    recommended_movies = get_recommendations(movie_title, new_df, similarity)
    return render_template('index.html', movie_title=movie_title, recommendations=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)
