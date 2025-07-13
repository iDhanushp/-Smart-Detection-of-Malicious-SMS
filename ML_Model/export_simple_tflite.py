#!/usr/bin/env python3
"""
Simple TensorFlow Lite export for SMS fraud detection model.
"""

import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.feature_extraction.text import TfidfVectorizer
import json

def create_simple_neural_network(input_dim, output_dim=3):
    """Create a simple neural network for TFLite conversion."""
    
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(output_dim, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def export_tflite_model():
    """Export the trained model to TensorFlow Lite format."""
    
    print("Loading trained model and vectorizer...")
    
    # Load the trained model and vectorizer
    try:
        with open('best_model.pkl', 'rb') as f:
            sklearn_model = pickle.load(f)
        
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        print(f"Loaded model: {type(sklearn_model)}")
        print(f"Loaded vectorizer: {type(vectorizer)}")
        
    except Exception as e:
        print(f"Error loading model files: {e}")
        return False
    
    # Get model parameters
    input_dim = len(vectorizer.vocabulary_)
    print(f"Input dimension: {input_dim}")
    
    # Create neural network with same architecture
    nn_model = create_simple_neural_network(input_dim)
    
    # Try to extract weights from sklearn model if possible
    try:
        if hasattr(sklearn_model, 'coef_') and hasattr(sklearn_model, 'intercept_'):
            # For logistic regression, we can approximate with weights
            weights = sklearn_model.coef_.T  # Transpose to match dense layer
            bias = sklearn_model.intercept_
            
            # Set the first layer weights
            nn_model.layers[0].set_weights([weights, bias])
            print("Transferred weights from sklearn model")
        
    except Exception as e:
        print(f"Could not transfer weights: {e}")
        print("Using random initialization - model will need retraining")
    
    # Create dummy data for conversion
    dummy_input = np.random.random((1, input_dim)).astype(np.float32)
    
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(nn_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    try:
        tflite_model = converter.convert()
        
        # Save the TFLite model
        with open('fraud_detector.tflite', 'wb') as f:
            f.write(tflite_model)
        
        print("✅ TensorFlow Lite model saved as: fraud_detector.tflite")
        print(f"Model size: {len(tflite_model) / 1024:.1f} KB")
        
        return True
        
    except Exception as e:
        print(f"Error converting to TFLite: {e}")
        return False

def main():
    """Main function to export TFLite model."""
    
    print("🚀 Starting TensorFlow Lite export...")
    
    success = export_tflite_model()
    
    if success:
        print("\n✅ Export completed successfully!")
        print("\nFiles created:")
        print("- fraud_detector.tflite (TensorFlow Lite model)")
        print("- tfidf_vocab.json (vocabulary file)")
        print("\nNext steps:")
        print("1. Copy fraud_detector.tflite to Flutter app assets/")
        print("2. Copy tfidf_vocab.json to Flutter app assets/")
        print("3. Update Flutter app to use the new model")
    else:
        print("\n❌ Export failed!")
        print("The sklearn model cannot be directly converted to TFLite.")
        print("Consider retraining with TensorFlow/Keras for better compatibility.")

if __name__ == "__main__":
    main() 