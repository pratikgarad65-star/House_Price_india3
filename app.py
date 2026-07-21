import os
import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# --- Load Model Safely ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'linear_model.pkl')
model = None

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully!")
    else:
        print(f"❌ Error: {MODEL_PATH} not found.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# --- Feature Names (Exact order expected by model) ---
FEATURE_NAMES = [
    'number of bedrooms',
    'number of bathrooms',
    'living area',
    'lot area',
    'number of floors',
    'waterfront present',
    'number of views',
    'condition of the house',
    'grade of the house',
    'Area of the house(excluding basement)',
    'Area of the basement',
    'Built Year',
    'Renovation Year',
    'lot_area_renov',
    'Number of schools nearby',
    'Distance from the airport'
]

@app.route('/', methods=['GET'])
def index():
    """Renders the main input page."""
    try:
        return render_template('index.html')
    except Exception as e:
        return f"<h3>Template Error: Make sure 'index.html' is inside a folder named 'templates'. Details: {e}</h3>", 500

@app.route('/predict', methods=['POST'])
def predict():
    """Handles price predictions."""
    if model is None:
        return render_template('index.html', error="Model file not found or failed to load on the server.")

    try:
        # Extract features safely
        input_data = {}
        for feature in FEATURE_NAMES:
            raw_val = request.form.get(feature, 0)
            input_data[feature] = [float(raw_val)]

        # Convert input into a pandas DataFrame matching feature names
        df_input = pd.DataFrame(input_data)

        # Make prediction
        prediction = model.predict(df_input)[0]
        formatted_price = f"${max(0, prediction):,.2f}"

        return render_template('index.html', prediction_text=formatted_price)

    except Exception as e:
        return render_template('index.html', error=f"Prediction error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
