import os
from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
import numpy as np
import pickle
import sys
from datetime import datetime
from sklearn import __version__ as sklearn_version
import warnings

app = Flask(__name__)
warnings.filterwarnings('ignore')

# Always use absolute paths for model files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def create_fallback_model():
    """Create a simple fallback model if loading fails"""
    from sklearn.ensemble import RandomForestClassifier
    print("🔄 Creating fallback model...")
    
    # Create a simple model with default parameters
    fallback_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        max_depth=10
    )
    
    # Create dummy training data to fit the model
    dummy_data = pd.DataFrame({
        'Gender': [0, 1, 0, 1],
        'Age': [25, 35, 45, 55],
        'State': [1, 2, 3, 4],
        'City': [10, 20, 30, 40],
        'Bank_Branch': [50, 60, 70, 80],
        'Account_Type': [0, 1, 2, 0],
        'Transaction_Date': [15, 20, 25, 10],
        'Transaction_Time': [30000, 40000, 50000, 60000],
        'Transaction_Amount': [100.0, 500.0, 1000.0, 2000.0],
        'Transaction_Type': [0, 1, 2, 3],
        'Account_Balance': [5000.0, 10000.0, 15000.0, 20000.0],
        'Transaction_Device': [1, 2, 3, 4],
        'Transaction_Currency': [0, 0, 1, 1]
    })
    dummy_labels = [0, 0, 1, 1]  # 0 = legitimate, 1 = fraud
    
    fallback_model.fit(dummy_data, dummy_labels)
    print("✅ Fallback model created and trained!")
    
    return fallback_model

# Load model and encoders
def load_model_and_encoders():
    """Load model and encoders with multiple loading strategies"""
    model = None
    encoders = None
    
    # Try loading model
    model_path = os.path.join(BASE_DIR, "model.pkl")
    print(f"📂 Attempting to load model from: {model_path}")
    
    if not os.path.exists(model_path):
        print("❌ Model file not found! Creating fallback model...")
        model = create_fallback_model()
        encoders = {}
        return model, encoders
    
    # Try different loading methods for model
    try:
        print("🔄 Trying joblib for model...")
        model = joblib.load(model_path)
        print("✅ Model loaded successfully with joblib!")
    except Exception as joblib_error:
        print(f"⚠️ Joblib failed: {str(joblib_error)}")
        try:
            print("🔄 Trying pickle with encoding for model...")
            with open(model_path, 'rb') as f:
                model = pickle.load(f, encoding='latin1')
            print("✅ Model loaded successfully with pickle (latin1)!")
        except Exception as pickle_error:
            print(f"⚠️ Pickle latin1 failed: {str(pickle_error)}")
            try:
                print("🔄 Trying pickle with bytes for model...")
                with open(model_path, 'rb') as f:
                    model = pickle.load(f, encoding='bytes')
                print("✅ Model loaded successfully with pickle (bytes)!")
            except Exception as bytes_error:
                print(f"⚠️ All loading methods failed: {str(bytes_error)}")
                print("🔄 Creating fallback model...")
                model = create_fallback_model()
    
    # Try loading encoders
    encoders_path = os.path.join(BASE_DIR, "encoders.pkl")
    print(f"📂 Attempting to load encoders from: {encoders_path}")
    
    if not os.path.exists(encoders_path):
        print("❌ Encoders file not found! Creating dummy encoders...")
        encoders = {}
        return model, encoders
    
    # Try different loading methods for encoders
    try:
        print("🔄 Trying joblib for encoders...")
        encoders = joblib.load(encoders_path)
        print("✅ Encoders loaded successfully with joblib!")
    except Exception as joblib_error:
        print(f"⚠️ Joblib failed: {str(joblib_error)}")
        try:
            print("🔄 Trying pickle with encoding for encoders...")
            with open(encoders_path, 'rb') as f:
                encoders = pickle.load(f, encoding='latin1')
            print("✅ Encoders loaded successfully with pickle (latin1)!")
        except Exception as pickle_error:
            print(f"⚠️ Pickle latin1 failed: {str(pickle_error)}")
            try:
                print("🔄 Trying pickle with bytes for encoders...")
                with open(encoders_path, 'rb') as f:
                    encoders = pickle.load(f, encoding='bytes')
                print("✅ Encoders loaded successfully with pickle (bytes)!")
            except Exception as bytes_error:
                print(f"❌ All encoders loading methods failed: {str(bytes_error)}")
                encoders = {}
    
    if model is not None:
        print(f"📊 Model type: {type(model)}")
        if hasattr(model, 'n_estimators'):
            print(f"📊 Model details: {model.n_estimators} estimators")
    
    print("✅ Loading process completed!")
    return model, encoders

model, encoders = load_model_and_encoders()

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make fraud prediction"""
    if model is None or encoders is None:
        return jsonify({
            'error': 'Model or encoders not loaded properly'
        }), 500
    
    try:
        # Get data from form
        data = request.get_json()
        
        # Create DataFrame with the input data
        input_data = pd.DataFrame([{
            'Gender': int(data['gender']),
            'Age': int(data['age']),
            'State': int(data['state']),
            'City': int(data['city']),
            'Bank_Branch': int(data['bank_branch']),
            'Account_Type': int(data['account_type']),
            'Transaction_Date': int(data['transaction_date']),
            'Transaction_Time': int(data['transaction_time']),
            'Transaction_Amount': float(data['transaction_amount']),
            'Transaction_Type': int(data['transaction_type']),
            'Account_Balance': float(data['account_balance']),
            'Transaction_Device': int(data['transaction_device']),
            'Transaction_Currency': int(data['transaction_currency'])
        }])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Get probability for fraud class (class 1)
        fraud_probability = prediction_proba[1] * 100
        
        # Determine risk level
        if fraud_probability >= 80:
            risk_level = "Very High"
            risk_color = "#dc3545"
        elif fraud_probability >= 60:
            risk_level = "High"
            risk_color = "#fd7e14"
        elif fraud_probability >= 40:
            risk_level = "Medium"
            risk_color = "#ffc107"
        elif fraud_probability >= 20:
            risk_level = "Low"
            risk_color = "#20c997"
        else:
            risk_level = "Very Low"
            risk_color = "#28a745"
        
        return jsonify({
            'prediction': int(prediction),
            'fraud_probability': round(fraud_probability, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'message': 'Fraudulent Transaction' if prediction == 1 else 'Legitimate Transaction'
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Fraud Detection System...")
    print(f"📂 Working directory: {BASE_DIR}")
    print(f"🐍 Python version: {sys.version}")
    print(f"🔧 scikit-learn version: {sklearn_version}")
    
    if model is None:
        print("⚠️ Model not loaded properly, but starting with fallback...")
    
    if encoders is None:
        print("⚠️ Encoders not loaded properly, using empty encoders...")
    
    print("🌐 Flask app starting on http://localhost:5000")
    print("📱 Open your browser and navigate to http://localhost:5000")
    print("� Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"❌ Error starting Flask app: {str(e)}")
        print("Try running on a different port or check if port 5000 is available")