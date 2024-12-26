import pickle
from model import train_knn_model,load_and_preprocess_data

# Load and preprocess data
data = load_and_preprocess_data('tracks.csv')

# Train the KNN model
knn_model = train_knn_model(data)

# Save the trained model to a .pkl file
with open('knn_model.pkl', 'wb') as file:
    pickle.dump(knn_model, file)

print("Model saved as knn_model.pkl")
