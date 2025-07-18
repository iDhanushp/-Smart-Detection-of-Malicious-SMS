#!/usr/bin/env python3
"""
Export TensorFlow Lite model using FULL real SMS dataset (28K+ messages)
This script processes all your real SMS data, not just the 100-message sample.
"""

import pandas as pd
import numpy as np
import json
import os
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from datetime import datetime

class FullDatasetBehavioralAnalyzer:
    def __init__(self):
        self.fraud_patterns = {
            'urgency': [
                r'\b(urgent|immediate|asap|now|quick|hurry|fast|limited time|expires?|deadline|last chance)\b',
                r'\b(act now|don\'t wait|time running out|offer ends|final|last day)\b'
            ],
            'fear': [
                r'\b(suspended|blocked|closed|frozen|terminated|cancelled|expired|deactivated)\b',
                r'\b(fraud|scam|security|verify|confirm|update|warning|alert|risk)\b',
                r'\b(account.*(?:suspended|blocked|closed)|card.*(?:blocked|suspended))\b'
            ],
            'reward': [
                r'\b(?:rs\.?|₹|inr)\s*\d+(?:,\d{3})*(?:\.\d{2})?\b',
                r'\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:rs\.?|₹|rupees?|inr)\b',
                r'\b(free|cashback|reward|bonus|gift|prize|win|won|earn|save|discount|off)\b',
                r'\b(\d+%\s*off|\d+\s*percent\s*off|flat\s*\d+|upto\s*\d+)\b'
            ],
            'authority': [
                r'\b(bank|rbi|government|police|court|legal|official|authorized|verified)\b',
                r'\b(sbi|hdfc|icici|axis|kotak|pnb|canara|union|bob)\b',
                r'\b(paytm|gpay|phonepe|amazon|flipkart|ola|uber|zomato)\b'
            ],
            'action': [
                r'\b(click|tap|call|dial|visit|download|install|reply|send|forward)\b',
                r'\b(http[s]?://|www\.|\.com|\.in|bit\.ly|tinyurl)\b',
                r'\b(sms.*to|call.*on|dial.*\d+|click.*link|visit.*site)\b'
            ]
        }
        
    def calculate_behavioral_scores(self, text):
        """Calculate 5-dimensional behavioral scores for SMS text"""
        text_lower = str(text).lower()
        scores = {}
        
        for category, patterns in self.fraud_patterns.items():
            score = 0
            word_count = len(text_lower.split())
            
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                score += matches
            
            # Normalize by word count to get proportion
            scores[f'{category}_score'] = score / max(word_count, 1) if word_count > 0 else 0
            
        return scores

def load_full_sms_dataset():
    """Load all real SMS data from your CSV files"""
    print("Loading full SMS dataset...")
    
    base_path = Path(r"d:\code\Smart Detection of Malicious SMS\datasetgenerateor\sms data set")
    all_data = []
    
    # Load phone export files
    phone_files = [
        "phone_sms_export_2025-07-13T14-41-31.344697.csv",
        "phone_sms_export_2025-07-13T14-59-37.079178.csv", 
        "phone_sms_export_2025-07-14T09-30-54.278524.csv"
    ]
    
    for filename in phone_files:
        filepath = base_path / filename
        if filepath.exists():
            print(f"Loading {filename}...")
            try:
                df = pd.read_csv(filepath)
                print(f"  - Loaded {len(df)} messages from {filename}")
                
                # Standardize column names
                if 'body' in df.columns and 'text' not in df.columns:
                    df['text'] = df['body']
                if 'address' in df.columns and 'sender' not in df.columns:
                    df['sender'] = df['address']
                
                # Add source info
                df['source_file'] = filename
                all_data.append(df)
                
            except Exception as e:
                print(f"  - Error loading {filename}: {e}")
    
    # Load spam dataset
    spam_file = base_path / "sms_spam.csv"
    if spam_file.exists():
        print(f"Loading sms_spam.csv...")
        try:
            df_spam = pd.read_csv(spam_file)
            print(f"  - Loaded {len(df_spam)} messages from sms_spam.csv")
            
            # This file likely has different format - adapt as needed
            if 'v2' in df_spam.columns and 'text' not in df_spam.columns:
                df_spam['text'] = df_spam['v2']
            if 'v1' in df_spam.columns and 'classification' not in df_spam.columns:
                df_spam['classification'] = df_spam['v1'].map({'ham': 'LEGIT', 'spam': 'FRAUD'})
            
            df_spam['source_file'] = 'sms_spam.csv'
            all_data.append(df_spam)
            
        except Exception as e:
            print(f"  - Error loading sms_spam.csv: {e}")
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal combined dataset: {len(combined_df)} messages")
        return combined_df
    else:
        raise Exception("No data files could be loaded!")

def auto_classify_messages(df, analyzer):
    """Apply automatic classification based on behavioral patterns"""
    print("Applying behavioral analysis and auto-classification...")
    
    classifications = []
    confidences = []
    behavioral_data = []
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"  Processing message {idx}/{len(df)}")
            
        text = str(row.get('text', ''))
        sender = str(row.get('sender', ''))
        
        # Calculate behavioral scores
        scores = analyzer.calculate_behavioral_scores(text)
        behavioral_data.append(scores)
        
        # Enhanced 3-class auto-classification logic (LEGITIMATE/SPAM/FRAUD)
        total_score = sum(scores.values())
        
        # Check for FRAUD patterns (security threats, phishing, account takeover)
        if 'kotak.com/KBANKT/Fraud' in text or 'Not you' in text:
            classification = 'FRAUD'
            confidence = 0.8
        elif any(pattern in text.lower() for pattern in ['suspended', 'blocked', 'verify account', 'click here immediately']):
            classification = 'FRAUD' 
            confidence = 0.7
        elif scores['fear_score'] > 0.2 and scores['urgency_score'] > 0.1:  # High fear + urgency = fraud
            classification = 'FRAUD'
            confidence = 0.6
        
        # Check for SPAM patterns (promotional, marketing, commercial)
        elif any(spam_sender in sender.upper() for spam_sender in ['MGLAMM', 'FLPKRT', 'MYNTRA', 'AMZNSM', 'BEYOUN', 'SNITCH', 'DOMINO']):
            classification = 'SPAM'
            confidence = 0.7
        elif 'offer' in text.lower() and scores['reward_score'] > 0.05:
            classification = 'SPAM'
            confidence = 0.6
        elif any(spam_word in text.lower() for spam_word in ['sale', 'discount', 'flat', '%', 'off', 'buy now', 'limited time']):
            classification = 'SPAM'
            confidence = 0.6
        
        # LEGITIMATE patterns (banking, OTP, service notifications)
        elif any(legit in sender.lower() for legit in ['airtel', 'jio', 'vodafone', 'kotakb', 'sbi', 'hdfc', 'icici']):
            classification = 'LEGITIMATE'
            confidence = 0.6
        elif 'otp' in text.lower() or 'verification code' in text.lower():
            classification = 'LEGITIMATE'
            confidence = 0.7
        
        # Default classification based on overall behavioral score
        elif total_score > 0.3:  # High behavioral score suggests suspicious
            classification = 'FRAUD'
            confidence = 0.5
        elif scores['reward_score'] > 0.1:  # High reward score suggests promotional
            classification = 'SPAM'
            confidence = 0.5
        else:
            classification = 'LEGITIMATE'
            confidence = 0.5
            
        classifications.append(classification)
        confidences.append(confidence)
    
    # Add behavioral scores and classifications to dataframe
    behavioral_df = pd.DataFrame(behavioral_data)
    for col in behavioral_df.columns:
        df[col] = behavioral_df[col]
    
    df['classification'] = classifications
    df['confidence'] = confidences
    
    print(f"3-Class Classification complete:")
    print(f"  - FRAUD: {sum(1 for c in classifications if c == 'FRAUD')} messages")
    print(f"  - SPAM: {sum(1 for c in classifications if c == 'SPAM')} messages") 
    print(f"  - LEGITIMATE: {sum(1 for c in classifications if c == 'LEGITIMATE')} messages")
    
    return df

def export_tflite_model(df):
    """Export TensorFlow Lite model from full dataset"""
    print("\nPreparing data for TensorFlow Lite export...")
    
    # Prepare features
    feature_columns = ['urgency_score', 'fear_score', 'reward_score', 'authority_score', 'action_score']
    X_behavioral = df[feature_columns].fillna(0).values
    
    # Text features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    X_text = vectorizer.fit_transform(df['text'].fillna('').astype(str)).toarray()
    
    # Combine features
    X = np.hstack([X_behavioral, X_text])
    
    # Prepare labels for 3-class classification (LEGITIMATE=0, SPAM=1, FRAUD=2)
    label_mapping = {'LEGITIMATE': 0, 'SPAM': 1, 'FRAUD': 2}
    y = df['classification'].map(label_mapping).values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"3-Class label distribution:")
    for class_name, class_id in label_mapping.items():
        count = np.sum(y == class_id)
        percentage = (count / len(y)) * 100
        print(f"  {class_name}: {count} messages ({percentage:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    
    # Create and train 3-class neural network model
    print("\nTraining 3-class neural network model...")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')  # 3 classes: LEGITIMATE, SPAM, FRAUD
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',  # For integer labels
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=20,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate model
    print("\nModel evaluation:")
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Predictions for 3-class classification report
    y_pred = np.argmax(model.predict(X_test_scaled), axis=1)
    class_names = ['LEGITIMATE', 'SPAM', 'FRAUD']
    print("\n3-Class Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Convert to TensorFlow Lite
    print("\nConverting to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Fixed representative dataset function for quantization
    def representative_dataset_gen():
        for i in range(100):
            sample = X_train_scaled[i:i+1].astype(np.float32)  # Single sample with batch dimension
            yield [sample]  # Wrap in list for TensorFlow Lite
    
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    
    tflite_model = converter.convert()
    
    # Save model
    model_filename = 'full_dataset_3class_fraud_detector.tflite'
    with open(model_filename, 'wb') as f:
        f.write(tflite_model)
    
    model_size = len(tflite_model) / 1024  # Size in KB
    print(f"\nTensorFlow Lite model saved: {model_filename}")
    print(f"Model size: {model_size:.1f} KB")
    
    # Save model configuration
    config = {
        'model_filename': model_filename,
        'feature_columns': feature_columns,
        'vectorizer_vocab_size': len(vectorizer.vocabulary_),
        'total_features': X.shape[1],
        'behavioral_features': len(feature_columns),
        'text_features': X_text.shape[1],
        'training_samples': X_train.shape[0],
        'test_accuracy': float(test_accuracy),
        'dataset_size': len(df),
        'fraud_samples': int(np.sum(y == 2)),
        'spam_samples': int(np.sum(y == 1)), 
        'legitimate_samples': int(np.sum(y == 0)),
        'class_distribution': {
            'LEGITIMATE': int(np.sum(y == 0)),
            'SPAM': int(np.sum(y == 1)),
            'FRAUD': int(np.sum(y == 2))
        },
        'created_timestamp': datetime.now().isoformat()
    }
    
    config_filename = 'full_dataset_3class_model_config.json'
    with open(config_filename, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Model configuration saved: {config_filename}")
    
    # Save preprocessing components for 3-class model
    import joblib
    joblib.dump(scaler, 'full_dataset_3class_scaler.pkl')
    joblib.dump(vectorizer, 'full_dataset_3class_vectorizer.pkl')
    print("Preprocessing components saved: scaler and vectorizer")
    
    return model_filename, config

def main():
    """Main execution function"""
    print("=== Full Dataset 3-Class TensorFlow Lite Export ===")
    print(f"Processing your complete 28K+ SMS dataset with 3-class system...")
    print(f"Classes: LEGITIMATE, SPAM, FRAUD")
    print(f"Started at: {datetime.now()}\n")
    
    try:
        # Initialize behavioral analyzer
        analyzer = FullDatasetBehavioralAnalyzer()
        
        # Load full dataset
        df = load_full_sms_dataset()
        
        # Apply behavioral analysis and classification
        df_analyzed = auto_classify_messages(df, analyzer)
        
        # Save analyzed dataset
        output_filename = f'full_analyzed_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df_analyzed.to_csv(output_filename, index=False)
        print(f"\nFull analyzed dataset saved: {output_filename}")
        
        # Export TensorFlow Lite model
        model_filename, config = export_tflite_model(df_analyzed)
        
        print(f"\n=== SUCCESS ===")
        print(f"✅ Processed {len(df_analyzed)} real SMS messages")
        print(f"✅ Generated 3-class TensorFlow Lite model: {model_filename}")
        print(f"✅ Model size: {os.path.getsize(model_filename) / 1024:.1f} KB")
        print(f"✅ Test accuracy: {config['test_accuracy']:.4f}")
        print(f"✅ Class distribution:")
        for class_name, count in config['class_distribution'].items():
            percentage = (count / config['dataset_size']) * 100
            print(f"   - {class_name}: {count} messages ({percentage:.1f}%)")
        print(f"✅ Training completed at: {datetime.now()}")
        
        print(f"\n=== Next Steps ===")
        print(f"1. Copy {model_filename} to your Flutter app's assets folder")
        print(f"2. Update your Flutter code to use this new 3-class model")
        print(f"3. The model now properly detects SPAM + FRAUD + LEGITIMATE!")
        print(f"4. Trained on {config['dataset_size']} real messages with proper 3-class distribution")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
