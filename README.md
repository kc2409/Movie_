# Movie Recommender App

A full-stack movie recommender application with a React frontend and Flask backend. Provides personalized movie recommendations based on user input.

## Features
- Personalized movie recommendations
- Responsive design for mobile and desktop

## Tech Stack

- **Frontend:** React.js
- **Backend:** Flask (Python)
- **API calls:** REST API
- **Testing:** Postman (for API testing)
- **Build Tools:** Webpack, Babel 
- **Version Control:** Git 


### Backend
Recommender Model
The Movie Recommender application employs a content-based recommendation system to suggest movies based on their textual descriptions. Below is an overview of the methodology and process used in the system:

1. Data Preprocessing
Data Loading and Cleaning:

The model starts by loading movie metadata from a CSV file. The dataset includes various attributes of movies, but for this model, irrelevant columns are removed to focus on 'title', 'tagline', 'overview', and 'popularity'.
Missing values are handled by dropping rows with missing critical information.
The 'tagline' and 'overview' fields are combined into a single 'description' field to provide a comprehensive textual representation of each movie.
Popularity Encoding:

The 'popularity' attribute is transformed into numerical values using label encoding, allowing for the movies to be sorted by their popularity.
2. Text Normalization
Normalization Process:

Text normalization is performed to standardize the text data. This includes converting all text to lowercase, removing special characters and whitespaces, and correcting contractions (e.g., converting "don't" to "do not").
The text is then tokenized into individual words, and common stopwords (e.g., "the", "and") are filtered out to focus on meaningful words.
The resulting tokens are reassembled into a cleaned-up textual representation of the movie descriptions.
3. Feature Extraction
TF-IDF Vectorization:

The Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer is used to convert the normalized text descriptions into numerical feature vectors. This technique captures the importance of each word in the context of the movie descriptions and across the entire dataset.
The TF-IDF matrix represents each movie description in a high-dimensional space, where similar descriptions have similar vectors.
4. Similarity Computation
Cosine Similarity:

To determine the similarity between movies, cosine similarity is computed based on the TF-IDF vectors. This measure calculates the cosine of the angle between vectors, providing a similarity score between -1 and 1, where 1 indicates identical vectors.
A similarity matrix is created to hold the similarity scores between all pairs of movies.
5. Recommendation Function
Generating Recommendations:

The recommendation function identifies similar movies based on the input movie title. It finds the index of the given movie in the dataset and retrieves the similarity scores for that movie.
Movies are ranked by their similarity scores, and the top recommendations are selected based on their similarity to the input movie.
The function returns a list of movie titles that are most similar to the input movie, helping users discover movies of interest based on their preferences.
Example Usage
To generate recommendations, the user simply provides a movie title, and the system returns a list of similar movies. For instance, if 'Blade Runner' is provided as the input, the model will suggest other movies that are textually similar to 'Blade Runner'.
## API

### Endpoint: `/movie`

- **Method:** GET
- **Description:** Retrieves movie recommendations based on the provided movie title.
- **Query Parameter:** `title` (string) - The title of the movie for which recommendations are requested.

**Responses:**
- **200 OK:** Returns a JSON array of recommended movies.
- **400 Bad Request:** Returned if no movie title is provided.
- **404 Not Found:** Returned if the provided movie title is not found in the dataset.
## FRONTEND
![Screenshot of the app](1ss.png)

<video width="560" controls>
  <source src="assets/demo-video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
