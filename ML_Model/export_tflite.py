import joblib
import os
import numpy as np
import tensorflow as tf

MODEL_PATH = 'best_model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'
TFLITE_PATH = 'fraud_detector.tflite'

# Load model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Check if model is Keras (for direct TFLite export)
if hasattr(model, 'save'):
    # Keras model
    model.save('keras_model.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f'TFLite model exported to {TFLITE_PATH}')
else:
    print('Direct TFLite export is only supported for Keras models.')
    print('To use TFLite on Android, retrain your model using Keras/TensorFlow.')
    print('For scikit-learn/XGBoost, consider reimplementing the model in Keras or use ONNX as an intermediate step.') 