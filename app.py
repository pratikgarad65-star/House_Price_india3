import os
import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the pickled linear regression model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'linear_model.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Feature names in exact order expected by the model
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
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return render_template('index.html', error="Model not loaded on server.")

    try:
        input_data = {}
        for feature in FEATURE_NAMES:
            raw_val = request.form.get(feature, 0)
            input_data[feature] = [float(raw_val)]

        # Convert to DataFrame to retain feature names
        df_input = pd.DataFrame(input_data)

        # Make prediction
        prediction = model.predict(df_input)[0]
        formatted_price = f"${max(0, prediction):,.2f}"

        return render_template('index.html', prediction_text=formatted_price)

    except Exception as e:
        return render_template('index.html', error=f"Prediction error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
