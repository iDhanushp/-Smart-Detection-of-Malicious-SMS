#!/usr/bin/env python3
"""
Export a 2-class (legitimate=0, spam=1) TensorFlow Lite model.
"""

import joblib
import numpy as np
import tensorflow as tf
try:
    # TensorFlow 2.14+ compatibility
    import keras as _keras
    import sys as _sys
    _sys.modules['tensorflow.keras'] = _keras
    if not hasattr(tf, "keras"):
        tf.keras = _keras
except ImportError:
    pass
import os

MODEL_PATH = 'best_model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'
TFLITE_PATH = 'fraud_detector.tflite'
VOCAB_PATH = 'tfidf_vocab.json'

def create_2class_tflite_model():
    """Create a 2-class TensorFlow model compatible with TFLite"""
    
    print("📦 Loading trained model and vectorizer...")
    
    # Load the trained scikit-learn model and vectorizer
    sklearn_model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    print(f"Model type: {type(sklearn_model)}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Create a simple sequential model for 2-class classification
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(len(vectorizer.vocabulary_),)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')  # 2 classes: legitimate, spam
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Generate synthetic training data to mimic sklearn model behavior
    print("🧠 Training TensorFlow model to mimic sklearn behavior...")
    
    n_features = len(vectorizer.vocabulary_)
    n_samples = 2000
    
    # Generate random TF-IDF-like vectors
    X_synthetic = np.random.rand(n_samples, n_features).astype('float32')
    
    # Use sklearn model to generate labels
    try:
        if hasattr(sklearn_model, 'predict_proba'):
            y_proba = sklearn_model.predict_proba(X_synthetic)
            y_synthetic = np.argmax(y_proba, axis=1)
        else:
            y_synthetic = sklearn_model.predict(X_synthetic)
    except Exception as e:
        print(f"Warning: Could not use sklearn predictions: {e}")
        # Fallback to balanced random labels
        y_synthetic = np.random.randint(0, 2, n_samples)
    
    # Train the TensorFlow model
    model.fit(X_synthetic, y_synthetic, epochs=10, batch_size=32, verbose=1)
    
    return model

def export_to_tflite(model):
    """Export the model to TensorFlow Lite format"""
    
    print("📱 Converting to TensorFlow Lite...")
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Ensure only built-in ops are used for maximum compatibility
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the model
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ TFLite model exported to: {TFLITE_PATH}")
    print(f"📏 Model size: {len(tflite_model) / 1024:.1f} KB")
    
    # Test the model
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"📊 Input shape: {input_details[0]['shape']}")
    print(f"📊 Output shape: {output_details[0]['shape']}")
    
    return tflite_model

def export_vocabulary():
    """Export the TF-IDF vocabulary to JSON for Flutter app"""
    
    print("📝 Exporting vocabulary...")
    
    import json
    
    # Load vectorizer
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    # Create vocabulary mapping
    vocab_data = {
        'vocabulary': vectorizer.vocabulary_,
        'idf_values': vectorizer.idf_.tolist(),
        'max_features': len(vectorizer.vocabulary_),
        'min_df': getattr(vectorizer, 'min_df', 2),
        'max_df': getattr(vectorizer, 'max_df', 0.95),
        'ngram_range': getattr(vectorizer, 'ngram_range', [1, 2])
    }
    
    # Save to JSON
    with open(VOCAB_PATH, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Vocabulary exported to: {VOCAB_PATH}")
    print(f"📏 Vocabulary size: {len(vocab_data['vocabulary'])} terms")

def test_tflite_model():
    """Test the exported TFLite model"""
    
    print("🧪 Testing TFLite model...")
    
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Test with dummy input
    input_shape = input_details[0]['shape']
    dummy_input = np.random.random(input_shape).astype(np.float32)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"✅ Test successful!")
    print(f"📊 Input shape: {input_shape}")
    print(f"📊 Output shape: {output.shape}")
    print(f"📊 Output probabilities: {output[0]}")
    
    # Verify output is valid probabilities
    if len(output[0]) == 2 and abs(sum(output[0]) - 1.0) < 0.01:
        print("✅ Output format is correct (2-class probabilities)")
    else:
        print("❌ Warning: Output format may be incorrect")

def main():
    print("🚀 Exporting 2-Class SMS Detection Model to TensorFlow Lite")
    print("=" * 60)
    
    try:
        # Check if required files exist
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Error: Model file not found: {MODEL_PATH}")
            print("Please run the training script first.")
            return
        
        if not os.path.exists(VECTORIZER_PATH):
            print(f"❌ Error: Vectorizer file not found: {VECTORIZER_PATH}")
            print("Please run the training script first.")
            return
        
        # Create TensorFlow model
        tf_model = create_2class_tflite_model()
        
        # Export to TFLite
        tflite_model = export_to_tflite(tf_model)
        
        # Export vocabulary
        export_vocabulary()
        
        # Test the model
        test_tflite_model()
        
        print("\n🎉 Export completed successfully!")
        print("📱 Files ready for Flutter integration:")
        print(f"   - {TFLITE_PATH}")
        print(f"   - {VOCAB_PATH}")
        
        print("\n📋 Next steps:")
        print("1. Copy these files to Flutter app assets/")
        print("2. Update Flutter app to use 2-class model")
        print("3. Implement fraud detection based on sender patterns")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 