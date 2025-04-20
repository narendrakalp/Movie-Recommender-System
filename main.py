import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load datasets
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge datasets on title
movies = movies.merge(credits, on='title')

# Select relevant columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Drop NaN values to avoid errors
movies.dropna(inplace=True)


# Function to extract names from a list of dictionaries
def convert(text):
    try:
        return [i['name'] for i in ast.literal_eval(text)]
    except (ValueError, SyntaxError):
        return []  # Return an empty list if conversion fails


# Function to extract top 3 actors
def convert_top3(text):
    try:
        return [i['name'] for i in ast.literal_eval(text)[:3]]
    except (ValueError, SyntaxError):
        return []


# Function to fetch director's name
def fetch_director(text):
    try:
        return [i['name'] for i in ast.literal_eval(text) if i['job'] == 'Director']
    except (ValueError, SyntaxError):
        return []


# Apply functions to respective columns
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_top3)
movies['crew'] = movies['crew'].apply(fetch_director)


# Remove spaces in names for better text processing
def collapse(L):
    return [i.replace(" ", "") for i in L]


movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)

# Convert overview into a list of words
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Create a 'tags' column by combining all processed text features
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Keep only essential columns and explicitly create a copy
new = movies[['movie_id', 'title', 'tags']].copy()

# Convert tags list into a single string
new['tags'] = new['tags'].apply(lambda x: " ".join(x))

# Feature extraction using CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new['tags']).toarray()

# Compute cosine similarity
similarity = cosine_similarity(vector)


# Recommendation function
def recommend(movie):
    if movie not in new['title'].values:
        print("Movie not found in dataset.")
        return

    index = new[new['title'] == movie].index[0]
    distances = sorted(enumerate(similarity[index]), reverse=True, key=lambda x: x[1])

    print("\nTop 5 Recommended Movies:")
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)


# Example usage
recommend('Gandhi')

# Save processed data and similarity matrix
pickle.dump(new, open('movie_list.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
