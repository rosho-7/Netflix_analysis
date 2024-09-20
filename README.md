# Netflix Movies Data Analysis Dashboard

This Streamlit app provides an interactive data analysis dashboard for Netflix movies. Users can explore descriptive statistics, visual analyses, and advanced analyses such as clustering and movie recommendations based on various filters.

## Features

- **Descriptive Analysis**: Summary statistics, genre and language analysis, top movies, and genre vs. language analysis.
- **Visual Analysis**: IMDB score distribution, correlation analysis, runtime distribution, premiere date analysis, heatmap of correlation matrix, average runtime by language, and movie release trends.
- **Advanced Analysis**: Clustering analysis, time-series analysis, group-by analysis, and a movie recommendation system.

## Requirements

- Python 3.7 or above
- Streamlit
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Plotly

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/netflix-data-analysis.git
    cd netflix-data-analysis
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Place the `netflix.csv` file in the same directory as the Streamlit app script.

2. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

3. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).

## File Structure

- `app.py`: The main Streamlit app script.
- `netflix.csv`: The dataset containing Netflix movies data.
- `requirements.txt`: A file listing all the required Python packages.
- `README.md`: This readme file.

## Data Source

The `netflix.csv` dataset should contain the following columns:
- `title`: The title of the movie.
- `genre`: The genre of the movie.
- `language`: The language of the movie.
- `year`: The release year of the movie.
- `premiere`: The premiere date of the movie.
- `runtime`: The runtime of the movie (in minutes).
- `imdb_score`: The IMDB score of the movie.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.

## License

This project is licensed under the MIT License.
