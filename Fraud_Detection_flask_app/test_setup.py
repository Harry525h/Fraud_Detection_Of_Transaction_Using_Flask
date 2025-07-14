"""
Quick test script to verify model loading and Flask setup
"""

import os
import sys

def test_setup():
    print("🧪 Testing Fraud Detection Setup")
    print("=" * 40)
    
    # Test 1: Check files exist
    print("\n📁 Checking files...")
    required_files = ['model.pkl', 'encoders.pkl', 'app.py', 'templates/index.html']
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} - Found")
        else:
            print(f"❌ {file} - Missing")
    
    # Test 2: Test imports
    print("\n📦 Testing imports...")
    try:
        import flask
        print(f"✅ Flask - {flask.__version__}")
    except ImportError:
        print("❌ Flask - Not installed")
    
    try:
        import sklearn
        print(f"✅ scikit-learn - {sklearn.__version__}")
    except ImportError:
        print("❌ scikit-learn - Not installed")
    
    try:
        import pandas
        print(f"✅ pandas - {pandas.__version__}")
    except ImportError:
        print("❌ pandas - Not installed")
    
    try:
        import joblib
        print(f"✅ joblib - {joblib.__version__}")
    except ImportError:
        print("❌ joblib - Not installed")
    
    # Test 3: Test model loading
    print("\n🤖 Testing model loading...")
    try:
        import joblib
        model = joblib.load('model.pkl')
        print(f"✅ Model loaded - {type(model)}")
        
        encoders = joblib.load('encoders.pkl')
        print(f"✅ Encoders loaded - {type(encoders)}")
        
        # Test prediction
        import pandas as pd
        test_data = pd.DataFrame([{
            'Gender': 1, 'Age': 35, 'State': 15, 'City': 127,
            'Bank_Branch': 127, 'Account_Type': 2, 'Transaction_Date': 22,
            'Transaction_Time': 52151, 'Transaction_Amount': 1000.0,
            'Transaction_Type': 3, 'Account_Balance': 50000.0,
            'Transaction_Device': 17, 'Transaction_Currency': 0
        }])
        
        prediction = model.predict(test_data)[0]
        probability = model.predict_proba(test_data)[0]
        
        print(f"✅ Test prediction - {prediction} (prob: {probability[1]:.3f})")
        
    except Exception as e:
        print(f"❌ Model loading failed - {str(e)}")
    
    print("\n" + "=" * 40)
    print("🚀 Setup test completed!")
    print("\nIf all tests passed, you can run:")
    print("   python app.py")
    print("\nThen open your browser to:")
    print("   http://localhost:5000")

if __name__ == "__main__":
    test_setup()
