import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import zipfile
import os

# Loading the movies and ratings datasets
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Merging datasets
movie_ratings = pd.merge(ratings, movies, on='movieId')


user_movie_ratings = movie_ratings.pivot_table(index='userId', columns='title', values='rating')
user_movie_ratings.fillna(0, inplace=True)

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_movie_ratings)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_ratings.index, columns=user_movie_ratings.index)

# Below is the function i hvae created to get recommendations
def get_recommendations(user_id, num_recommendations=5):
    similarity_scores = user_similarity_df[user_id]
    similar_users_ratings = user_movie_ratings.mul(similarity_scores, axis=0)
    recommendation_scores = similar_users_ratings.sum(axis=0) / similarity_scores.sum()
    user_ratings = user_movie_ratings.loc[user_id]
    recommendation_scores = recommendation_scores[user_ratings == 0]
    top_recommendations = recommendation_scores.sort_values(ascending=False).head(num_recommendations)
    return top_recommendations

# Testing the recommendation system
user_id = 1  # Example user ID
recommendations = get_recommendations(user_id)

# Printing recommendations in a readable format
print(f"Top {len(recommendations)} movie recommendations for user {user_id}:\n")
for i, (title, score) in enumerate(recommendations.items(), start=1):
    print(f"{i}. {title} - Score: {score:.2f}")
