from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from model import recommend_music, load_and_preprocess_data
import pickle

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Load and preprocess data
data = load_and_preprocess_data('tracks.csv')

# Load the model (only once) - this prevents repetitive loading
def load_model():
    with open('knn_model.pkl', 'rb') as model_file:
        knn_model = pickle.load(model_file)
    return knn_model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendation():
    activity = request.json.get('activity')
    knn_model = load_model()  # Load model each time a request is made
    
    recommendations = recommend_music(activity, knn_model, data)
    
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
