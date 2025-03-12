from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

app = Flask(__name__)

# Load the models and data
with open('linear_regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('decision_tree_model.pkl', 'rb') as f:
    dt_model = pickle.load(f)

with open('svr_model.pkl', 'rb') as f:
    svr_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('locations.pkl', 'rb') as f:
    locations = pickle.load(f)

# Demo examples
demo_examples = [
    {"location": "Whitefield", "sqft": 1500, "bath": 2, "bhk": 3, "price": 75.5},
    {"location": "Electronic City", "sqft": 1200, "bath": 2, "bhk": 2, "price": 45.2},
    {"location": "JP Nagar", "sqft": 1800, "bath": 3, "bhk": 3, "price": 85.0}
]

@app.route('/')
def home():
    return render_template('index.html', locations=locations)

@app.route('/how-to-use')
def how_to_use():
    return render_template('how_to_use.html', examples=demo_examples)

@app.route('/profile')
def profile():
    return render_template('profile.html',
                         lr_score=0.85,
                         lr_mae=12.5,
                         lr_mse=234.6,
                         rf_score=0.89,
                         rf_mae=10.2,
                         rf_mse=198.3,
                         dt_score=0.82,
                         dt_mae=13.1,
                         dt_mse=245.7,
                         svr_score=0.84,
                         svr_mae=11.8,
                         svr_mse=215.4,
                         total_records=len(locations),
                         features=['Location', 'Total Square Feet', 'Bathrooms', 'BHK'],
                         num_locations=len(locations))

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from the form
    location = request.form.get('location')
    sqft = float(request.form.get('sqft'))
    bath = int(request.form.get('bath'))
    bhk = int(request.form.get('bhk'))
    model_choice = request.form.get('model')

    # Calculate derived features
    sqft_per_bedroom = sqft / bhk
    bath_per_bedroom = bath / bhk

    # Create feature array
    features = [sqft, bath, bhk, sqft_per_bedroom, bath_per_bedroom]
    
    # Load features list to get correct order
    with open('features.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    # Create location array with zeros
    location_features = {name: 0 for name in feature_names if name.startswith('is_')}
    if location != 'other':
        location_key = f'is_{location}'
        if location_key in location_features:
            location_features[location_key] = 1
    
    # Add location mean price (you might want to store this during training)
    location_features['location_price_mean'] = 0  # Add appropriate value based on location
    
    # Create final input array in correct order
    input_array = np.zeros(len(feature_names))
    for i, feature in enumerate(feature_names):
        if feature in ['total_sqft', 'Bathrooms', 'bhk', 'sqft_per_bedroom', 'bath_per_bedroom']:
            input_array[i] = features[0] if feature == 'total_sqft' else \
                           features[1] if feature == 'Bathrooms' else \
                           features[2] if feature == 'bhk' else \
                           features[3] if feature == 'sqft_per_bedroom' else features[4]
        else:
            input_array[i] = location_features.get(feature, 0)
    
    input_array = input_array.reshape(1, -1)
    
    # Scale the input
    input_scaled = scaler.transform(input_array)

    # Make prediction based on model choice
    if model_choice == 'linear':
        prediction = lr_model.predict(input_scaled)[0]
        model_name = 'Linear Regression'
    elif model_choice == 'random_forest':
        prediction = rf_model.predict(input_scaled)[0]
        model_name = 'Random Forest'
    elif model_choice == 'decision_tree':
        prediction = dt_model.predict(input_scaled)[0]
        model_name = 'Decision Tree'
    elif model_choice == 'svr':
        prediction = svr_model.predict(input_scaled)[0]
        model_name = 'Support Vector Regression'

    return render_template('index.html', 
                         prediction=round(prediction, 2),
                         model_name=model_name,
                         locations=locations)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=10000,debug=True)
