import os
import pickle
import pandas as pd
from flask import Flask, render_template, request, render_template_string

app = Flask(__name__)

# Model loading
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'linear_model.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

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
    try:
        return render_template('index.html')
    except Exception as e:
        return f"<h2>Template Error: Make sure index.html is inside a 'templates' folder. Details: {e}</h2>", 500

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return render_template('index.html', error="Model failed to load on server.")

    try:
        input_data = {}
        for feature in FEATURE_NAMES:
            raw_val = request.form.get(feature, 0)
            input_data[feature] = [float(raw_val)]

        df_input = pd.DataFrame(input_data)
        prediction = model.predict(df_input)[0]
        formatted_price = f"${max(0, prediction):,.2f}"

        return render_template('index.html', prediction_text=formatted_price)

    except Exception as e:
        return render_template('index.html', error=f"Prediction error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
