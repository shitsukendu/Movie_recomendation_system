#!/usr/bin/env python
# coding: utf-8

# <div align="center">
# 
# # MOVIE RECOMMENDATION SYSTEM
# </div>

# ## 🎬 Introduction
# 
# A Movie Recommendation System is a machine learning application designed to suggest movies to users based on their interests, preferences, and past interactions. With the rapid growth of online streaming platforms and digital entertainment services, recommendation systems have become an essential tool for improving user experience and helping users discover relevant content efficiently.
# 
# In this project, a movie recommendation system is developed using the MovieLens dataset containing information about movies, genres, user ratings, and timestamps. The system analyzes movie features and user behavior to generate personalized movie recommendations.
# 
# The project mainly focuses on Content-Based Filtering, where movies are recommended based on similarities in genres and movie attributes. Cosine Similarity and Count Vectorization techniques are used to measure the similarity between movies. Additionally, exploratory data analysis (EDA) and visualizations are performed to better understand rating patterns, popular genres, and user activity.
# 
# The main objectives of this project are:
# 
# * To analyze movie and rating datasets
# * To perform data preprocessing and visualization
# * To build a movie recommendation model
# * To recommend similar movies based on user input
# * To understand recommendation system techniques used in real-world applications
# 
# This project demonstrates how machine learning and data analysis techniques can be applied to create intelligent recommendation systems similar to those used by platforms like Netflix, Amazon Prime Video, and Disney+.
# 

# ## 📚 Import Libraries

# In[19]:


# Data Manipulation
import pandas as pd
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Text Processing
from sklearn.feature_extraction.text import CountVectorizer

# Similarity Calculation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Ignore Warnings
import warnings
warnings.filterwarnings('ignore')


# | Library           | Purpose                           |
# | ----------------- | --------------------------------- |
# | pandas            | Data loading and manipulation     |
# | numpy             | Numerical calculations            |
# | matplotlib        | Basic plotting and charts         |
# | seaborn           | Advanced data visualization       |
# | CountVectorizer   | Convert text data into vectors    |
# | cosine_similarity | Measure similarity between movies |
# | train_test_split  | Split data for training/testing   |
# | WordCloud         | Generate movie title word clouds  |
# | warnings          | Hide unnecessary warnings         |
# 

# ## 📂 Load Dataset

# In[20]:


df_movies = pd.read_csv("movies.csv")
df_rating = pd.read_csv("ratings.csv")


# In[21]:


df_movies.head()


# In[22]:


df_rating.head()


# In[23]:


print("Movies Dataset Shape :", df_movies.shape)


# In[24]:


print("Ratings Dataset Shape :", df_rating.shape)


# In[25]:


df_movies.info()


# In[26]:


df_rating.info()


# In[27]:


df_movies.describe().T


# In[28]:


df_rating.describe().T


# ## 🧹 Data Preprocessing

# ### 📌 Check Missing Values

# In[29]:


# Missing values in Movies Dataset
df_movies.isnull().sum()


# In[30]:


# Missing values in Ratings Dataset
df_rating.isnull().sum()


# ### 📌 Check Duplicate Values

# In[31]:


# Duplicate values in Movies Dataset
df_movies.duplicated().sum()


# In[32]:


# Duplicate values in Ratings Dataset
df_rating.duplicated().sum()


# ## 📌 Handle Missing Values

# In[33]:


# Fill missing genres with empty string
df_movies['genres'] = df_movies['genres'].fillna('')


# In[34]:


# Merge datasets
df = pd.merge(df_rating, df_movies, on='movieId')

# Display merged dataset
df.head()


# In[35]:


print("Merged Dataset Shape :", df.shape)


# In[36]:


df.columns


# ## 📊 Exploratory Data Analysis (EDA)

# Exploratory Data Analysis (EDA) is used to understand the dataset through statistics and visualizations. 
# 
# * Popular movies
# * Rating patterns
# * Genre distribution
# * User activity
# * Trends in the dataset
# 
# EDA is an important step before building the recommendation system

# ## 🎥 Top Rated Movies

# In[37]:


# Average rating of each movie
top_rated = df.groupby('title')['rating'].mean().sort_values(ascending=False)

# Convert into dataframe
top_rated = pd.DataFrame(top_rated)

# Display top 10 rated movies
top_rated.head(10)


# ## ⭐ Most Popular Movies

# In[38]:


# Count ratings for each movie
popular_movies = df.groupby('title')['rating'].count().sort_values(ascending=False)

# Convert into dataframe
popular_movies = pd.DataFrame(popular_movies)

# Display top 10 popular movies
popular_movies.head(10)


# ## 📈 Rating Distribution

# In[39]:


plt.figure(figsize=(8,5))

plt.hist(df['rating'])

plt.xlabel("Ratings")
plt.ylabel("Number of Ratings")
plt.title("Distribution of Movie Ratings")

plt.show()


# ## 🎭 Genre Analysis

# In[40]:


# Split genres
genres = df_movies['genres'].str.split('|').explode()

# Count genres
genre_count = genres.value_counts()

# Display genre counts
genre_count


# In[41]:


plt.figure(figsize=(10,6))

genre_count.plot(kind='bar')

plt.xlabel("Genres")
plt.ylabel("Count")
plt.title("Movie Genre Distribution")

plt.show()


# 👤 Most Active Users

# In[42]:


# Extract year
df_movies['year'] = df_movies['title'].str.extract(r'\((\d{4})\)')

# Display first rows
df_movies.head()


# 🎬 Movies Released Per Year

# In[43]:


# Extract year
df_movies['year'] = df_movies['title'].str.extract(r'\((\d{4})\)')

# Display first rows
df_movies.head()


# 📉 Movies Released Trend

# In[44]:


movies_per_year = df_movies['year'].value_counts().sort_index()

plt.figure(figsize=(12,5))

movies_per_year.plot()

plt.xlabel("Year")
plt.ylabel("Number of Movies")
plt.title("Movies Released Per Year")

plt.show()


# ## 🤖 Building the Movie Recommendation System

# In[45]:


movies_data = df_movies[['movieId', 'title', 'genres']]

movies_data.head()


# In[46]:


movies_data['genres'] = movies_data['genres'].fillna('')


# In[47]:


# Initialize CountVectorizer
cv = CountVectorizer(tokenizer=lambda x: x.split('|'))

# Convert genres into matrix
count_matrix = cv.fit_transform(movies_data['genres'])

print(count_matrix)


# In[48]:


print(count_matrix.shape)


# In[49]:


tfidf = TfidfVectorizer(token_pattern=r'[^|]+')

tfidf_matrix = tfidf.fit_transform(df_movies['genres'])


# In[ ]:


model = NearestNeighbors( metric='cosine', algorithm='brute')

model.fit(tfidf_matrix)


# In[51]:


def recommend(movie_name):

    movie_name = movie_name.lower()

    # Find movie index
    idx = df_movies[
        df_movies['title'].str.lower() == movie_name
    ].index

    if len(idx) == 0:
        print("Movie not found!")
        return

    idx = idx[0]

    # Find nearest neighbors
    distances, indices = model.kneighbors(
        tfidf_matrix[idx],
        n_neighbors=11
    )

    print(f"\n🎬 Recommended Movies for '{df_movies.iloc[idx]['title']}'\n")

    for i in range(1, len(indices[0])):

        movie_idx = indices[0][i]

        print(df_movies.iloc[movie_idx]['title'])


# In[52]:


recommend("Toy Story (1995)")


# In[58]:


# ==========================================
# USER-MOVIE RATINGS HEATMAP
# ==========================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st 

# Reload data after kernel restart
df_rating = pd.read_csv("ratings.csv", dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

# Filter to top 30 users and top 30 movies BEFORE pivot
top_movies = df_rating['movieId'].value_counts().head(30).index
top_users  = df_rating['userId'].value_counts().head(30).index

filtered = df_rating[
    df_rating['movieId'].isin(top_movies) &
    df_rating['userId'].isin(top_users)
]

# Create User-Movie Matrix
movie_matrix = filtered.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
)

# Select smaller portion for visualization
heatmap_data = movie_matrix.iloc[:30, :30]

# Create Heatmap
fig=plt.figure(figsize=(15,10))
sns.heatmap(
    heatmap_data,
    cmap='coolwarm',
    linewidths=0.5,
    linecolor='white',
    cbar=True
)

# Titles and Labels
plt.title("User-Movie Ratings Heatmap", fontsize=20)
plt.xlabel("Movie IDs", fontsize=14)
plt.ylabel("User IDs", fontsize=14)

st.pyplot(fig)


# In[ ]:




