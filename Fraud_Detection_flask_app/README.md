# Fraud Detection System

An interactive web application for real-time fraud detection using machine learning.

## Features

- **Real-time Analysis**: Instant fraud detection with probability scores
- **Interactive Interface**: User-friendly web interface with Bootstrap styling
- **Machine Learning**: Random Forest classifier trained on transaction data
- **Risk Assessment**: 5-level risk classification (Very Low to Very High)
- **Visual Results**: Progress bars and color-coded risk indicators

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python app.py
   ```

3. **Or use the batch file (Windows):**
   ```bash
   start_app.bat
   ```

## Usage

1. Open your browser and go to `http://localhost:5000`
2. Fill in the transaction details:
   - Personal Information (Gender, Age, Location)
   - Account Information (Bank Branch, Account Type, Balance)
   - Transaction Information (Amount, Type, Device, etc.)
3. Click "Analyze Transaction"
4. View the fraud probability and risk assessment

## Input Fields

### Personal Information
- **Gender**: Female or Male
- **Age**: Age of the account holder (18-100)
- **State**: Indian states (Andhra Pradesh to Ladakh)
- **City**: Major Indian cities (Mumbai, Delhi, Bangalore, etc.)

### Account Information
- **Bank Branch**: Indian bank branches (SBI, HDFC, ICICI, etc.)
- **Account Type**: Savings, Current, Credit Card, Fixed Deposit, Salary, Joint
- **Account Balance**: Current account balance in Indian Rupees (₹)

### Transaction Information
- **Transaction Amount**: Amount being transacted in Indian Rupees (₹)
- **Transaction Type**: ATM Withdrawal, Bank Deposit, Fund Transfer, UPI Payment, etc.
- **Transaction Device**: ATM, POS, Mobile Apps, UPI platforms, Internet Banking, etc.
- **Transaction Date**: Day of month (1-31)
- **Transaction Time**: Hourly time slots (24-hour format)
- **Currency**: INR (default), USD, EUR, GBP, AED, SGD, CAD, AUD

## API Endpoints

### GET /
Returns the main web interface

### POST /predict
**Request Body:**
```json
{
    "gender": "1",
    "age": "35",
    "state": "15",
    "city": "127",
    "bank_branch": "127",
    "account_type": "2",
    "transaction_date": "22",
    "transaction_time": "52151",
    "transaction_amount": "32415.45",
    "transaction_type": "3",
    "account_balance": "74557.27",
    "transaction_device": "17",
    "transaction_currency": "0"
}
```

**Response:**
```json
{
    "prediction": 0,
    "fraud_probability": 15.75,
    "risk_level": "Very Low",
    "risk_color": "#28a745",
    "message": "Legitimate Transaction"
}
```

## Risk Levels

- **Very Low** (0-20%): Green (#28a745)
- **Low** (20-40%): Teal (#20c997)
- **Medium** (40-60%): Yellow (#ffc107)
- **High** (60-80%): Orange (#fd7e14)
- **Very High** (80-100%): Red (#dc3545)

## Testing

Run the API test script:
```bash
python test_api.py
```

## Files Structure

```
fraud_flask_app/
├── app.py                 # Main Flask application
├── model.pkl             # Trained Random Forest model
├── encoders.pkl          # Label encoders for categorical features
├── requirements.txt      # Python dependencies
├── start_app.bat        # Windows startup script
├── test_api.py          # API testing script
├── templates/
│   └── index.html       # Web interface template
├── static/
│   └── style.css        # Custom styling
└── README.md           # This file
```

## Model Information

- **Algorithm**: Random Forest Classifier
- **Features**: 13 input features
- **Training Data**: Bank transaction fraud dataset
- **Performance**: Optimized for fraud detection accuracy

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Machine Learning**: scikit-learn, pandas, numpy
- **Icons**: Font Awesome

## Browser Support

- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge

## Troubleshooting

### Model Loading Issues
- Ensure `model.pkl` and `encoders.pkl` are in the same directory as `app.py`
- Check Python package versions match requirements.txt

### Connection Issues
- Verify Flask is running on port 5000
- Check firewall settings
- Ensure no other application is using port 5000

### Prediction Errors
- Validate all input fields are filled correctly
- Check input value ranges match the specified limits
- Ensure numeric fields contain valid numbers

## License

This project is for educational and demonstration purposes.
