# AI-Based SMS Fraud Detection System (ML Layer)

This directory contains the machine learning code for training and exporting a lightweight SMS fraud detection model for Android deployment.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the SMS Spam dataset (UCI) and place it in `data/sms_spam.csv`.

## Usage

- `train.py`: Preprocesses data, trains models (Naive Bayes, XGBoost), evaluates, and saves the best model.
- `export_tflite.py`: Converts the best model to TensorFlow Lite format for Android.

## Outputs
- `fraud_detector.tflite`: TFLite model for Android.
- `vectorizer.pkl`: Saved vectorizer for text preprocessing on device.

---

For end-to-end integration with the Android app, see the main project plan. 