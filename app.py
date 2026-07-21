import os
import pickle
import pandas as pd
from flask import Flask, render_template, request, render_template_string

app = Flask(__name__)

# --- Load the Model ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'linear_model.pkl')
model = None

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully!")
    else:
        print(f"❌ Error: {MODEL_PATH} file not found.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# --- Features expected by your linear model ---
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

# --- Fallback HTML template if templates/index.html is missing ---
FALLBACK_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>House Price Predictor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light p-4">
    <div class="container bg-white p-5 rounded shadow" style="max-width: 600px;">
        <h2 class="mb-4 text-center">🏡 House Price Predictor</h2>
        
        {% if prediction_text %}
        <div class="alert alert-success text-center">
            <h3>Estimated Price: {{ prediction_text }}</h3>
        </div>
        {% endif %}

        {% if error %}
        <div class="alert alert-danger text-center">{{ error }}</div>
        {% endif %}

        <form action="/predict" method="POST">
            {% for feature in features %}
            <div class="mb-3">
                <label class="form-label text-capitalize">{{ feature }}</label>
                <input type="number" step="any" class="form-control" name="{{ feature }}" required value="1">
            </div>
            {% endfor %}
            <button type="submit" class="btn btn-primary w-100 mt-3">Predict Price</button>
        </form>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    """Renders main page safely."""
    try:
        return render_template('index.html', features=FEATURE_NAMES)
    except Exception:
        # Fallback if templates/index.html isn't committed properly to GitHub
        return render_template_string(FALLBACK_TEMPLATE, features=FEATURE_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles inference request."""
    if model is None:
        error_msg = "Model failed to load. Please check server logs for linear_model.pkl."
        try:
            return render_template('index.html', features=FEATURE_NAMES, error=error_msg)
        except Exception:
            return render_template_string(FALLBACK_TEMPLATE, features=FEATURE_NAMES, error=error_msg)

    try:
        input_data = {}
        for feature in FEATURE_NAMES:
            raw_val = request.form.get(feature, 0)
            input_data[feature] = [float(raw_val)]

        # Convert to pandas DataFrame matching original feature order
        df_input = pd.DataFrame(input_data)

        # Make prediction
        prediction = model.predict(df_input)[0]
        formatted_price = f"${max(0, prediction):,.2f}"

        try:
            return render_template('index.html', features=FEATURE_NAMES, prediction_text=formatted_price)
        except Exception:
            return render_template_string(FALLBACK_TEMPLATE, features=FEATURE_NAMES, prediction_text=formatted_price)

    except Exception as e:
        error_msg = f"Prediction Error: {str(e)}"
        try:
            return render_template('index.html', features=FEATURE_NAMES, error=error_msg)
        except Exception:
            return render_template_string(FALLBACK_TEMPLATE, features=FEATURE_NAMES, error=error_msg)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
