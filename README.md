# Credit Card Fraud Detection

A machine learning project that classifies credit card transactions as fraudulent or non-fraudulent.

## Project Overview
This project uses machine learning to detect fraudulent credit card transactions. It includes a trained model (`CreditCardClassifier.pkl`) and supporting code for prediction.

## Features
- Preprocessed dataset for training
- Machine learning model for classification
- Python script for making predictions
- Dependencies listed in `requirements.txt`

## Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/CreditCardClassifier.git
   cd CreditCardClassifier
   ```
2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. Run the classifier script:
   ```sh
   python CreditCard.py
   ```
2. Modify `CreditCard.py` to load new data and make predictions using `CreditCardClassifier.pkl`.

## File Structure
- `CreditCard.py` - Main script for classification
- `CreditCardClassifier.pkl` - Trained ML model
- `requirements.txt` - Required dependencies
- `.gitignore` - Files to be ignored in Git
- `README.md` - Project documentation

## Model Training (Optional)
To retrain the model, modify `CreditCard.py` to include:
- Data loading
- Feature engineering
- Model training and saving

## Contributing
Feel free to submit issues or pull requests.

## License
[MIT License](LICENSE)
