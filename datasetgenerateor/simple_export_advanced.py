#!/usr/bin/env python3
"""
Simple export script for advanced fraud detector using scikit-learn -> TFLite conversion
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Try to import TensorFlow with fallbacks
try:
    import tensorflow as tf
    print("✅ TensorFlow imported successfully")
except ImportError:
    print("❌ TensorFlow not available")
    exit(1)

class SimpleBehavioralExporter:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def create_sample_data(self):
        """Create sample training data"""
        print("🏗️ Creating sample training data...")
        
        # Sample data with 30 features
        np.random.seed(42)
        n_samples = 1000
        
        # Generate feature data
        X = np.random.randn(n_samples, 30)
        
        # Create labels (0: LEGIT, 1: SPAM, 2: FRAUD)
        y = np.random.randint(0, 3, n_samples)
        
        # Add some pattern to make it learnable
        # FRAUD samples (label 2) have higher urgency/fear features
        fraud_mask = y == 2
        X[fraud_mask, 0] += 2  # urgency_immediate
        X[fraud_mask, 2] += 2  # fear_tactics
        
        # SPAM samples (label 1) have higher reward features
        spam_mask = y == 1
        X[spam_mask, 4] += 2  # reward_money
        X[spam_mask, 5] += 2  # reward_prizes
        
        # LEGIT samples (label 0) have lower risk scores
        legit_mask = y == 0
        X[legit_mask, 27] -= 1  # fraud_score
        X[legit_mask, 28] -= 1  # spam_score
        X[legit_mask, 29] += 1  # legit_score
        
        return X, y
    
    def train_model(self, X, y):
        """Train the model"""
        print("🤖 Training RandomForest model...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=50,  # Smaller for faster conversion
            max_depth=8,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✅ Model trained with {accuracy:.3f} accuracy")
        
        return X_test, y_test
    
    def convert_to_tflite(self, X_test):
        """Convert model to TensorFlow Lite using representative dataset"""
        print("📱 Converting to TensorFlow Lite...")
        
        try:
            # Create a simple neural network that mimics the RandomForest
            input_shape = X_test.shape[1]
            
            # Build a simple model
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(input_shape,)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(3, activation='softmax')
            ])
            
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            
            # Generate synthetic training data to train the TF model
            n_synthetic = 5000
            X_synthetic = np.random.randn(n_synthetic, input_shape)
            X_synthetic = self.scaler.transform(X_synthetic)
            y_synthetic = self.model.predict(X_synthetic)
            
            # Train the TF model to mimic sklearn model
            model.fit(X_synthetic, y_synthetic, epochs=20, verbose=0)
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Use representative dataset
            def representative_data_gen():
                for i in range(100):
                    yield [X_test[i:i+1].astype(np.float32)]
            
            converter.representative_dataset = representative_data_gen
            tflite_model = converter.convert()
            
            # Save the model
            tflite_path = "../sms_fraud_detectore_app/assets/advanced_fraud_detector.tflite"
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"✅ TFLite model saved: {tflite_path}")
            print(f"📏 Model size: {len(tflite_model) / 1024:.1f} KB")
            
            # Save model configuration
            config = {
                "feature_count": 30,
                "model_type": "advanced_behavioral",
                "classes": ["LEGITIMATE", "SPAM", "FRAUD"],
                "scaler_mean": self.scaler.mean_.tolist(),
                "scaler_scale": self.scaler.scale_.tolist()
            }
            
            config_path = "../sms_fraud_detectore_app/assets/behavioral_model_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"✅ Configuration saved: {config_path}")
            
            return tflite_path
            
        except Exception as e:
            print(f"❌ Error converting to TFLite: {e}")
            raise

def main():
    print("🚀 SIMPLE ADVANCED BEHAVIORAL SMS DETECTOR - TFLITE EXPORT")
    print("=" * 80)
    
    exporter = SimpleBehavioralExporter()
    
    # Create sample data
    X, y = exporter.create_sample_data()
    
    # Train model
    X_test, y_test = exporter.train_model(X, y)
    
    # Convert to TFLite
    tflite_path = exporter.convert_to_tflite(X_test)
    
    print("\n🎉 Export completed successfully!")
    print(f"📱 TFLite model: {tflite_path}")

if __name__ == "__main__":
    main()
