import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import uuid
import os
import ast
import warnings

warnings.filterwarnings('ignore')

# Load the movies dataset
@st.cache_data
def load_data():
    return pd.read_csv('movies.csv')

movies_df = load_data()

# First Layer: User Segmentation Layer
class UserSegmentLayer:
    def __init__(self):
        self.questions = [
            "How often do you watch movies? (1-5): ",
            "Rate your interest in action movies (1-5): ",
            "Rate your interest in comedy movies (1-5): ",
            "Rate your interest in drama movies (1-5): ",
            "Rate your interest in sci-fi movies (1-5): ",
            "Rate your interest in horror movies (1-5): ",
            "How important is the movie's budget to you? (1-5): ",
            "How much do you care about movie ratings? (1-5): ",
            "Do you prefer movies in English? (1-5): ",
            "How important is the movie's cast to you? (1-5): ",
            "Do you prefer recent movies or classics? (1-5, 1 for classics, 5 for recent): ",
            "How much do you care about movie duration? (1-5): ",
            "Rate your interest in foreign language films (1-5): ",
            "How important are special effects to you? (1-5): ",
            "Do you prefer watching movies in theaters or at home? (1-5, 1 for home, 5 for theaters): "
        ]
        self.scaler = StandardScaler()

    def get_user_answers(self):
        answers = []
        for question in self.questions:
            answer = st.slider(question, 1, 5)
            answers.append(answer)
        return answers

    def segment_user(self, answers):
        avg_score = sum(answers) / len(answers)
        if avg_score < 2:
            return 0
        elif avg_score < 3:
            return 1
        elif avg_score < 4:
            return 2
        else:
            return 3

    def generate_user_id(self):
        return str(uuid.uuid4())

    def store_user_data(self, user_id, answers, segment):
        user_data = {
            'user_id': user_id,
            'answers': answers,
            'segment': int(segment)
        }
        os.makedirs('user_data', exist_ok=True)
        with open(f'user_data/{user_id}.pkl', 'wb') as f:
            pickle.dump(user_data, f)

    def process_user(self):
        answers = self.get_user_answers()
        segment = self.segment_user(answers)
        user_id = self.generate_user_id()
        self.store_user_data(user_id, answers, segment)
        return user_id

# Second Layer: Recommendation Layer
class RecommendationLayer:
    def __init__(self, movies_df):
        self.movies_df = movies_df
        self.prepare_data()
        self.model = RandomForestRegressor()
        self.train_model()

    def prepare_data(self):
        def parse_genres(genre_str):
            try:
                genres = ast.literal_eval(genre_str)
                if isinstance(genres, list) and all(isinstance(g, dict) for g in genres):
                    return [g['name'] for g in genres]
                elif isinstance(genres, list) and all(isinstance(g, str) for g in genres):
                    return genres
                elif isinstance(genres, str):
                    return [genres]
                else:
                    return []
            except:
                return []

        self.movies_df['genres'] = self.movies_df['genres'].fillna('[]').apply(parse_genres)

        self.genre_list = ['Action', 'Comedy', 'Drama', 'Science Fiction', 'Horror']
        for genre in self.genre_list:
            self.movies_df[f'genre_{genre}'] = self.movies_df['genres'].apply(lambda x: 1 if genre in x else 0)

        self.movies_df['release_year'] = pd.to_datetime(self.movies_df['release_date'], errors='coerce').dt.year

        self.movies_df['is_english'] = (self.movies_df['original_language'] == 'en').astype(int)

        numerical_features = ['budget', 'popularity', 'release_year', 'runtime', 'vote_average', 'vote_count']
        for feature in numerical_features:
            self.movies_df[feature] = (self.movies_df[feature] - self.movies_df[feature].min()) / (self.movies_df[feature].max() - self.movies_df[feature].min())

        self.features = [col for col in self.movies_df.columns if col.startswith('genre_')] + ['budget', 'popularity', 'release_year', 'runtime', 'vote_average', 'vote_count', 'is_english']

        self.X = self.movies_df[self.features].fillna(0)
        self.y = self.movies_df['vote_average']

    def train_model(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        self.model.fit(self.X_train, self.y_train)

        y_train_pred = self.model.predict(self.X_train)
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)

        y_test_pred = self.model.predict(self.X_test)
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)

        st.write(f"Training MSE: {train_mse:.4f}")
        st.write(f"Training R^2 score: {train_r2:.4f}")
        st.write(f"Testing MSE: {test_mse:.4f}")
        st.write(f"Testing R^2 score: {test_r2:.4f}")

    def get_user_segment(self, user_id):
        with open(f'user_data/{user_id}.pkl', 'rb') as f:
            user_data = pickle.load(f)
        return user_data['segment'], user_data['answers']

    def recommend_movies(self, user_id, top_n=10):
        _, user_answers = self.get_user_segment(user_id)

        user_profile = {
            'watch_frequency': user_answers[0],
            'genre_preferences': user_answers[1:6],
            'budget_importance': user_answers[6],
            'rating_importance': user_answers[7],
            'english_preference': user_answers[8],
            'cast_importance': user_answers[9],
            'recency_preference': user_answers[10],
            'duration_importance': user_answers[11],
            'foreign_preference': user_answers[12],
            'effects_importance': user_answers[13],
            'theater_preference': user_answers[14]
        }

        self.movies_df['user_score'] = 0

        for i, genre in enumerate(self.genre_list):
            self.movies_df['user_score'] += self.movies_df[f'genre_{genre}'] * user_profile['genre_preferences'][i]

        self.movies_df['user_score'] += self.movies_df['budget'] * user_profile['budget_importance']
        self.movies_df['user_score'] += self.movies_df['vote_average'] * user_profile['rating_importance']
        self.movies_df['user_score'] += self.movies_df['is_english'] * user_profile['english_preference']
        self.movies_df['user_score'] += self.movies_df['popularity'] * user_profile['cast_importance']
        self.movies_df['user_score'] += self.movies_df['release_year'] * user_profile['recency_preference']
        self.movies_df['user_score'] += self.movies_df['runtime'] * user_profile['duration_importance']
        self.movies_df['user_score'] += (1 - self.movies_df['is_english']) * user_profile['foreign_preference']

        self.movies_df['user_score'] = (self.movies_df['user_score'] - self.movies_df['user_score'].min()) / (self.movies_df['user_score'].max() - self.movies_df['user_score'].min())

        self.movies_df['final_score'] = 0.7 * self.movies_df['user_score'] + 0.3 * self.movies_df['vote_average']

        recommended_movies = self.movies_df.sort_values('final_score', ascending=False).head(top_n)
        return recommended_movies[['title', 'final_score', 'genres', 'vote_average']]

# Streamlit app
def main():
    st.title("Movie Recommendation System")

    # Display dataset overview
    st.header("Dataset Overview")
    st.subheader("First few rows of the dataset:")
    st.dataframe(movies_df.head())

    st.subheader("Last rows of the dataset:")
    st.dataframe(movies_df.tail())

    st.subheader("Columns in the dataset:")
    st.write(movies_df.columns.tolist())

    # Instantiate user segmentation and recommendation layers
    user_segment_layer = UserSegmentLayer()
    recommendation_layer = RecommendationLayer(movies_df)

    st.header("User Preferences")
    user_id = user_segment_layer.process_user()

    if st.button("Get Recommendations"):
        st.write(f"User ID: {user_id}")
        recommendations = recommendation_layer.recommend_movies(user_id)
        st.header("Top 10 Recommended Movies:")
        st.dataframe(recommendations)

    # Option to add another user
    if st.button("Add Another User"):
        st.rerun()

    st.write("Thank you for using the movie recommendation system!")

if __name__ == "__main__":
    main()