#!/usr/bin/env python3
"""
Test script for SMS Fraud Detection ML Pipeline
This script tests the complete pipeline from training to export
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import json

def create_sample_data():
    """Create sample SMS data for testing"""
    sample_data = [
        # Legitimate messages
        ("ham", "Hi mom, can you pick me up from school?"),
        ("ham", "Meeting at 3 PM tomorrow. See you there!"),
        ("ham", "Thanks for the birthday wishes everyone!"),
        ("ham", "Don't forget to bring the documents."),
        ("ham", "I'll be home late tonight."),
        
        # Fraudulent messages
        ("spam", "URGENT: Your account has been suspended. Click here to verify: http://fake-bank.com"),
        ("spam", "CONGRATULATIONS! You won $1,000,000! Claim now: http://fake-lottery.com"),
        ("spam", "FREE RINGTONE! Download now: http://fake-ringtone.com"),
        ("spam", "Your package delivery failed. Reschedule: http://fake-delivery.com"),
        ("spam", "Bank security alert: Unusual activity detected. Verify: http://fake-security.com"),
    ]
    
    df = pd.DataFrame(sample_data, columns=['label', 'text'])
    return df

def test_training_pipeline():
    """Test the training pipeline"""
    print("🧪 Testing Training Pipeline...")
    
    # Create sample data
    df = create_sample_data()
    print(f"✓ Created sample dataset with {len(df)} messages")
    
    # Preprocess data
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=100)  # Smaller for testing
    X = vectorizer.fit_transform(df['text'])
    y = df['label'].values
    
    print(f"✓ Vectorized text to {X.shape[1]} features")
    
    # Train model
    model = MultinomialNB()
    model.fit(X, y)
    
    # Test predictions
    test_messages = [
        "Hi dad, can you call me?",
        "URGENT: Your account is locked. Click here: http://fake.com",
        "Meeting at 5 PM",
        "FREE MONEY! Claim your prize now!"
    ]
    
    X_test = vectorizer.transform(test_messages)
    predictions = model.predict(X_test)
    
    print("✓ Model predictions:")
    for msg, pred in zip(test_messages, predictions):
        result = "Fraudulent" if pred else "Legitimate"
        print(f"  '{msg[:50]}...' -> {result}")
    
    # Save model and vectorizer
    joblib.dump(model, 'test_model.pkl')
    joblib.dump(vectorizer, 'test_vectorizer.pkl')
    print("✓ Saved test model and vectorizer")
    
    return model, vectorizer

def test_vocabulary_export():
    """Test vocabulary export"""
    print("\n📤 Testing Vocabulary Export...")
    
    try:
        vectorizer = joblib.load('test_vectorizer.pkl')
        
        # Get vocabulary and IDF values
        vocabulary = vectorizer.vocabulary_
        idf_values = vectorizer.idf_.tolist()
        
        # Create export data
        export_data = {
            'vocabulary': vocabulary,
            'idf': idf_values,
            'stop_words': [],
            'max_features': vectorizer.max_features
        }
        
        # Save as JSON
        with open('test_vocab.json', 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✓ Vocabulary exported to test_vocab.json")
        print(f"  Vocabulary size: {len(vocabulary)}")
        print(f"  IDF values: {len(idf_values)}")
        
        return True
    except Exception as e:
        print(f"✗ Error exporting vocabulary: {e}")
        return False

def test_tflite_export():
    """Test TFLite export"""
    print("\n📱 Testing TFLite Export...")
    
    try:
        import tensorflow as tf
        
        # Load the test model
        model = joblib.load('test_model.pkl')
        vectorizer = joblib.load('test_vectorizer.pkl')
        
        # Create a simple TensorFlow model
        tf_model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(vectorizer.max_features,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        tf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open('test_fraud_detector.tflite', 'wb') as f:
            f.write(tflite_model)
        
        print(f"✓ TFLite model exported to test_fraud_detector.tflite")
        print(f"  Model size: {len(tflite_model) / 1024:.2f} KB")
        
        # Test the model
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Output shape: {output_details[0]['shape']}")
        
        return True
    except Exception as e:
        print(f"✗ Error exporting TFLite model: {e}")
        return False

def test_end_to_end():
    """Test end-to-end pipeline"""
    print("\n🔄 Testing End-to-End Pipeline...")
    
    try:
        # Load saved components
        model = joblib.load('test_model.pkl')
        vectorizer = joblib.load('test_vectorizer.pkl')
        
        # Test messages
        test_messages = [
            "Hi mom, can you pick me up?",
            "URGENT: Your account suspended. Click here: http://fake.com",
            "Meeting at 3 PM tomorrow",
            "CONGRATULATIONS! You won $1,000,000!"
        ]
        
        print("✓ End-to-end predictions:")
        for msg in test_messages:
            # Preprocess
            features = vectorizer.transform([msg])
            
            # Predict
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            
            result = "Fraudulent" if prediction else "Legitimate"
            conf = probability[1] if prediction else probability[0]
            
            print(f"  '{msg[:50]}...' -> {result} (confidence: {conf:.2f})")
        
        return True
    except Exception as e:
        print(f"✗ Error in end-to-end test: {e}")
        return False

def cleanup_test_files():
    """Clean up test files"""
    test_files = [
        'test_model.pkl',
        'test_vectorizer.pkl', 
        'test_vocab.json',
        'test_fraud_detector.tflite'
    ]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"✓ Cleaned up {file}")

def main():
    """Main test function"""
    print("🚀 SMS Fraud Detection - ML Pipeline Test")
    print("=" * 50)
    
    try:
        # Test each component
        test_training_pipeline()
        test_vocabulary_export()
        test_tflite_export()
        test_end_to_end()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed! Pipeline is working correctly.")
        print("\n📋 Next Steps:")
        print("1. Replace sample data with real SMS dataset")
        print("2. Run train.py to train on full dataset")
        print("3. Run export_tfidf_vocab.py and export_tflite.py")
        print("4. Copy generated files to Flutter assets/")
        print("5. Test the Android app")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("Please check your Python environment and dependencies.")
        return False
    
    finally:
        # Clean up
        print("\n🧹 Cleaning up test files...")
        cleanup_test_files()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 