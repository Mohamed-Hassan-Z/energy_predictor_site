import os
import joblib
import numpy as np
from flask import Flask, request, render_template
import pandas as pd
# --- Configuration ---
app = Flask(__name__)

# --- Load Models and Scaler ---
# Make sure these files are in the same directory as app.py
try:
    # Load the scaler
    scaler = joblib.load('scaler.pkl')

    # Load the two trained XGBoost models
    heating_model = joblib.load('xgboost_heating_load_model.pkl')
    cooling_model = joblib.load('xgboost_cooling_load_model.pkl')
    FEATURE_NAMES = [
    "Relative_Compactness", "Surface_Area", "Wall_Area", "Roof_Area",
    "Overall_Height", "Orientation", "Glazing_Area", "Glazing_Area_Distribution"
]

    print("Models and Scaler loaded successfully.")

except Exception as e:
    print(f"Error loading models or scaler: {e}")
    # Exit if models cannot be loaded, the app won't function
    exit()

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def home():
    """Renders the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the form."""
    
    # 1. Collect Input Data (remains the same)
    try:
        input_data = [
            float(request.form['Relative_Compactness']),
            float(request.form['Surface_Area']),
            float(request.form['Wall_Area']),
            float(request.form['Roof_Area']),
            float(request.form['Overall_Height']),
            float(request.form['Orientation']),
            float(request.form['Glazing_Area']),
            float(request.form['Glazing_Area_Distribution'])
        ]

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error processing input: {e}. Please ensure all fields are valid numbers.", error=True)

    # 2. Convert to NumPy Array (or Pandas DataFrame from previous fix)
    # We will use the Pandas DataFrame fix as it eliminates the UserWarning
    features_df = pd.DataFrame([input_data], columns=FEATURE_NAMES) 
    
    # 3. ***CRITICAL CHANGE: DO NOT SCALE THE FEATURES***
    # The XGBoost models were trained on UNscaled data, so we use the raw DataFrame/Array.
    
    # 4. Make Predictions using the UNscaled features_df
    # You may need to convert the DataFrame back to a NumPy array if the XGBoost model throws a warning/error on the DF. 
    # Let's use the underlying NumPy array for maximum compatibility.
    unscaled_features_array = features_df.values 
    
    heating_pred = heating_model.predict(unscaled_features_array)[0]
    cooling_pred = cooling_model.predict(unscaled_features_array)[0]
    
    # 5. Format Output
    output = {
        'heating': f'{heating_pred:.2f}',
        'cooling': f'{cooling_pred:.2f}'
    }

    # 6. Render the result back to the user
    return render_template(
        'index.html', 
        heating_result=output['heating'],
        cooling_result=output['cooling'],
        input_values=input_data # Optional: sends inputs back to form
    )

# --- Run the App ---
if __name__ == "__main__":
    # Ensure debug is set to False for production
    app.run(debug=True)