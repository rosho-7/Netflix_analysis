import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.neighbors import NearestNeighbors

@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['premiere'] = pd.to_datetime(data['premiere'], errors='coerce')
    return data


file_path = 'netflix.csv'
netflix_data = load_data(file_path)


st.sidebar.title("Navigation")
pages = [
    "Home",
    "Descriptive Analysis",
    "Visual Analysis",
    "Advanced Analysis"
]
page = st.sidebar.radio("Go to", pages)


st.sidebar.header('Filter Movies')
st.sidebar.info("Use these filters to narrow down the movies displayed in the analysis.")
selected_genres = st.sidebar.multiselect('Genres', netflix_data['genre'].dropna().unique(), help="Select one or more genres of the movies.")
selected_languages = st.sidebar.multiselect('Languages', netflix_data['language'].dropna().unique(), help="Select one or more languages of the movies.")
selected_years = st.sidebar.multiselect('Years', netflix_data['year'].dropna().unique(), help="Select one or more release years of the movies.")
score_range = st.sidebar.slider('IMDB Score Range', 0.0, 10.0, (0.0, 10.0), help="Select the range of IMDB scores for the movies.")
runtime_range = st.sidebar.slider('Runtime Range', 0, int(netflix_data['runtime'].dropna().max()), (0, int(netflix_data['runtime'].dropna().max())), help="Select the range of runtime (in minutes) for the movies.")


filtered_data = netflix_data[
    ((netflix_data['genre'].isin(selected_genres)) | (len(selected_genres) == 0)) &
    ((netflix_data['language'].isin(selected_languages)) | (len(selected_languages) == 0)) &
    ((netflix_data['year'].isin(selected_years)) | (len(selected_years) == 0)) &
    (netflix_data['imdb_score'] >= score_range[0]) &
    (netflix_data['imdb_score'] <= score_range[1]) &
    (netflix_data['runtime'] >= runtime_range[0]) &
    (netflix_data['runtime'] <= runtime_range[1])
]


if filtered_data.empty:
    st.sidebar.warning("No movies match the selected criteria. Please adjust your filters.")


def home():
    st.title('Netflix Movies Data Analysis')
    st.write("Welcome to the Netflix Movies Data Analysis dashboard. Use the navigation bar to explore different analyses.")
    st.header('Dataset Overview')
    st.write(filtered_data.head())

def descriptive_analysis():
    st.header('Summary Statistics')
    st.write(filtered_data.describe())
    
    st.header('Genre Analysis')
    genre_counts = filtered_data['genre'].value_counts()
    st.bar_chart(genre_counts)
    
    #st.header('Language Analysis')
    #language_counts = filtered_data['language'].value_counts()
    #st.bar_chart(language_counts)
    
    st.header('Language Analysis')
    language_counts = filtered_data['language'].value_counts()
    plt.figure(figsize=(10, 6))
    language_counts.plot(kind='bar')
    plt.title('Movies by Language')
    plt.xlabel('Language')
    plt.ylabel('Number of Movies')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(plt)
    
    st.header('Top Movies Analysis')
    top_movies = filtered_data.sort_values(by='imdb_score', ascending=False).head(10)
    st.write(top_movies[['title', 'imdb_score']])
    
    st.header('Genre vs Language Analysis')
    genre_language_pivot = pd.pivot_table(filtered_data, values='title', index='genre', columns='language', aggfunc='count', fill_value=0)
    st.write(genre_language_pivot)

def visual_analysis():
    st.header('IMDB Score Analysis')
    st.markdown("The histogram below shows the distribution of IMDB scores for the filtered movies.")
    plt.figure(figsize=(10, 6))
    sns.histplot(filtered_data['imdb_score'], bins=10, kde=True)
    plt.title('Distribution of IMDB Scores')
    plt.xlabel('IMDB Score')
    plt.ylabel('Frequency')
    st.pyplot(plt)
    
    st.header('Correlation Analysis')
    st.markdown("The scatter plot below shows the correlation between runtime and IMDB score for the filtered movies.")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='runtime', y='imdb_score', data=filtered_data)
    plt.title('Runtime vs IMDB Score')
    plt.xlabel('Runtime (minutes)')
    plt.ylabel('IMDB Score')
    st.pyplot(plt)
    
    st.header('Runtime Analysis')
    st.markdown("The histogram below shows the distribution of movie runtimes for the filtered movies.")
    plt.figure(figsize=(10, 6))
    sns.histplot(filtered_data['runtime'], bins=10, kde=True)
    plt.title('Distribution of Movie Runtimes')
    plt.xlabel('Runtime (minutes)')
    plt.ylabel('Frequency')
    st.pyplot(plt)
    
    st.header('Premiere Date Analysis')
    st.markdown("The bar chart below shows the number of movies released each month for the filtered movies.")
    filtered_data['month'] = filtered_data['premiere'].dt.month
    month_counts = filtered_data['month'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    month_counts.plot(kind='bar')
    plt.title('Number of Movies Released Each Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Movies')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(plt)

    st.header('Heatmap of Correlation Matrix')
    st.markdown("The heatmap below shows the correlation matrix for numerical features.")
    plt.figure(figsize=(10, 6))
    sns.heatmap(filtered_data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    st.pyplot(plt)
    
    st.header('Bar Plot of Average Runtime by Language')
    st.markdown("The bar plot below shows the average runtime of movies for each language.")
    avg_runtime_language = filtered_data.groupby('language')['runtime'].mean().sort_values()
    plt.figure(figsize=(10, 6))
    avg_runtime_language.plot(kind='bar')
    plt.title('Average Runtime by Language')
    plt.xlabel('Language')
    plt.ylabel('Average Runtime (minutes)')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(plt)
    
    st.header('Line Plot of Number of Movies Released Each Year')
    st.markdown("The line plot below shows the number of movies released each year.")
    movies_per_year = filtered_data['premiere'].dt.year.value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    movies_per_year.plot(kind='line')
    plt.title('Number of Movies Released Each Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Movies')
    st.pyplot(plt)

def advanced_analysis():
    st.header('Clustering Analysis')
    st.markdown("This section uses K-Means clustering to group movies into clusters based on runtime and IMDB score.")
    numeric_data = filtered_data[['runtime', 'imdb_score']].dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    filtered_data['Cluster'] = clusters

    fig = px.scatter(filtered_data, x='runtime', y='imdb_score', color='Cluster', title='Movie Clusters')
    st.plotly_chart(fig)
    
    st.header('Time-Series Analysis')
    st.markdown("The line chart below shows the average IMDB score over the years for the filtered movies.")
    filtered_data['premiere_year'] = filtered_data['premiere'].dt.year
    imdb_trend = filtered_data.groupby('premiere_year')['imdb_score'].mean().reset_index()

    fig = px.line(imdb_trend, x='premiere_year', y='imdb_score', title='Average IMDB Score Over Years')
    st.plotly_chart(fig)
    
    st.header('Group-By Analysis')
    st.markdown("Select a column to group by and see the average values for each group.")
    group_by_column = st.selectbox('Select column to group by', ['genre', 'language', 'year', 'Cluster'])
    grouped_data = filtered_data.groupby(group_by_column).mean()
    st.write(grouped_data)
    
    st.header('Movie Recommendation System')
    st.markdown("Select a movie to get recommendations for similar movies based on runtime and IMDB score.")
    selected_movie = st.selectbox('Select a movie for recommendations', filtered_data['title'].unique())

    
    selected_movie_features = filtered_data[filtered_data['title'] == selected_movie][['imdb_score', 'runtime']].values

    # NearestNeighbors for finding similar movies
    nn = NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
    nn.fit(filtered_data[['imdb_score', 'runtime']].dropna())
    distances, indices = nn.kneighbors(selected_movie_features)

    #recommended movies
    recommended_indices = indices[0][indices[0] != filtered_data[filtered_data['title'] == selected_movie].index[0]]
    recommended_movies = filtered_data.iloc[recommended_indices]
    st.write('Recommended Movies:')
    st.write(recommended_movies[['title', 'imdb_score', 'runtime']])
    st.markdown("These recommendations are based on the closest matches in terms of runtime and IMDB score.")

if page == "Home":
    home()
elif page == "Descriptive Analysis":
    descriptive_analysis()
elif page == "Visual Analysis":
    visual_analysis()
elif page == "Advanced Analysis":
    advanced_analysis()
