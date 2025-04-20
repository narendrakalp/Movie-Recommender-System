import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

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

# Save processed data and similarity matrix
pickle.dump(new, open('movie_list.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

# Recommendation function
def recommend(movie):
    if movie not in new['title'].values:
        messagebox.showerror("Error", "Movie not found in dataset.")
        return

    index = new[new['title'] == movie].index[0]
    distances = sorted(enumerate(similarity[index]), reverse=True, key=lambda x: x[1])

    result = ""
    for i in distances[1:6]:
        result += f"- {new.iloc[i[0]].title}\n\n"  # Adding extra line break for spacing

    # Display results in the GUI itself
    recommendation_label.config(text=result)

# GUI setup
def create_gui():
    global recommendation_label
    root = tk.Tk()
    root.title("Movie Recommendation System")
    root.geometry("800x600")
    root.config(bg="#2c3e50")  # Dark background for a modern feel

    # Set the font styles
    font_title = ("Arial", 24, "bold")
    font_label = ("Arial", 18)
    font_recommendations = ("Arial", 14, "italic")

    def on_select(event):
        selected_movie = movie_combobox.get()
        recommend(selected_movie)

    # Title Label
    title_label = tk.Label(root, text="Movie Recommendation System", font=font_title, fg="#ecf0f1", bg="#2c3e50")
    title_label.pack(pady=25)

    # Subtitle Label
    subtitle_label = tk.Label(root, text="Select a Movie to Get Recommendations", font=font_label, fg="#ecf0f1", bg="#2c3e50")
    subtitle_label.pack(pady=15)

    # Create the dropdown (combobox) with movie titles, increase font size
    movie_combobox = ttk.Combobox(root, values=new['title'].tolist(), width=50, font=("Arial", 14))
    movie_combobox.pack(pady=15)

    # Create a frame for the recommendation result
    recommendation_frame = tk.Frame(root, bg="#34495e", padx=20, pady=15)
    recommendation_frame.pack(pady=25, fill="both", expand=True)

    # Label to display recommendations
    recommendation_label = tk.Label(recommendation_frame, text="", justify="left", font=font_recommendations, fg="#ecf0f1", bg="#34495e")
    recommendation_label.pack(pady=10)

    # Bind the combobox selection event to the on_select function
    movie_combobox.bind("<<ComboboxSelected>>", on_select)

    # Run the GUI
    root.mainloop()

# Run the GUI
create_gui()
