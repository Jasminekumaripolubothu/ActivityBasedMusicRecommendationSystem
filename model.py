import pandas as pd
from sklearn.preprocessing import MinMaxScaler

FEATURES = ['energy', 'tempo', 'danceability', 'valence', 'popularity',
            'loudness', 'acousticness', 'instrumentalness']

def load_and_preprocess_data(filename):
    # Load the dataset
    data = pd.read_csv(filename)
    
    # Drop rows with missing values
    data = data.dropna()
    
    # Remove duplicates based on track name and artist
    data = data.drop_duplicates(subset=['track_name', 'artists'])
    
    # Normalize the feature columns
    scaler = MinMaxScaler()
    data[FEATURES] = scaler.fit_transform(data[FEATURES])
    
    return data

def train_knn_model(data):
    from sklearn.neighbors import NearestNeighbors
    
    knn = NearestNeighbors(n_neighbors=10, metric='cosine')
    knn.fit(data[FEATURES])
    return knn

def recommend_music(activity, knn_model, data):
    activity_features = {
        'running': {'energy': 0.9, 'tempo': 140, 'danceability': 0.85, 'valence': 0.75,
                    'popularity': 0.7, 'loudness': -5, 'acousticness': 0.1, 'instrumentalness': 0.2},
        'reading': {'energy': 0.2, 'tempo': 60, 'danceability': 0.2, 'valence': 0.3, 
                    'popularity': 0.5, 'loudness': -20, 'acousticness': 0.6, 'instrumentalness': 0.8},
        'studying': {'energy': 0.4, 'tempo': 90, 'danceability': 0.4, 'valence': 0.6, 
                    'popularity': 0.6, 'loudness': -10, 'acousticness': 0.3, 'instrumentalness': 0.5},
        'working_out': {'energy': 0.95, 'tempo': 135, 'danceability': 0.9, 'valence': 0.8, 
                    'popularity': 0.8, 'loudness': -4, 'acousticness': 0.1, 'instrumentalness': 0.1},
        'party': {'energy': 1.0, 'tempo': 120, 'danceability': 0.95, 'valence': 0.85, 
                    'popularity': 0.9, 'loudness': -3, 'acousticness': 0.2, 'instrumentalness': 0.1},
        'relaxing': {'energy': 0.3, 'tempo': 50, 'danceability': 0.3, 'valence': 0.7, 
                    'popularity': 0.4, 'loudness': -15, 'acousticness': 0.8, 'instrumentalness': 0.9},
        'driving': {'energy': 0.6, 'tempo': 100, 'danceability': 0.5, 'valence': 0.7, 
                    'popularity': 0.6, 'loudness': -8, 'acousticness': 0.4, 'instrumentalness': 0.2},
        'cooking': {'energy': 0.5, 'tempo': 70, 'danceability': 0.6, 'valence': 0.75, 
                    'popularity': 0.7, 'loudness': -12, 'acousticness': 0.5, 'instrumentalness': 0.4},
        'meditating': {'energy': 0.1, 'tempo': 30, 'danceability': 0.1, 'valence': 0.9, 
                    'popularity': 0.3, 'loudness': -25, 'acousticness': 0.9, 'instrumentalness': 1.0},
        'cleaning': {'energy': 0.7, 'tempo': 110, 'danceability': 0.7, 'valence': 0.65, 
                    'popularity': 0.5, 'loudness': -10, 'acousticness': 0.6, 'instrumentalness': 0.3},
        'traveling': {'energy': 0.75, 'tempo': 110, 'danceability': 0.75, 'valence': 0.8, 
                    'popularity': 0.6, 'loudness': -7, 'acousticness': 0.4, 'instrumentalness': 0.3},
        # Add more activities here...
    }

    if activity not in activity_features:
        return []

    query = pd.DataFrame([activity_features[activity]])
    distances, indices = knn_model.kneighbors(query[FEATURES])
    recommended_tracks = data.iloc[indices[0]]

    # Optional: Shuffle the recommended tracks to avoid repetition
    recommended_tracks = recommended_tracks.sample(frac=1).reset_index(drop=True)

    return recommended_tracks[['track_name', 'artists', 'album_name']].to_dict(orient='records')
