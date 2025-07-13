#!/usr/bin/env python3
"""
Train a 2-class SMS fraud detection model using the labeled dataset.
Maps: Legitimate=0, Spam=1 (fraud messages are mapped to spam)
"""

import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import re

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths
LABELED_DATA_PATH = '../datasetgenerateor/final_labeled_sms.csv'
VECTORIZER_PATH = 'vectorizer.pkl'
MODEL_PATH = 'best_model.pkl'

def clean_text(text):
    """Clean and preprocess text."""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print("🚀 Training 2-Class SMS Detection Model")
    print("=" * 50)
    
    # 1. Load labeled data
    print("📊 Loading labeled dataset...")
    try:
        df = pd.read_csv(LABELED_DATA_PATH)
        print(f"Total messages loaded: {len(df):,}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find labeled data at {LABELED_DATA_PATH}")
        print("Please run the labeling pipeline first.")
        return
    
    # Check required columns
    if 'body' not in df.columns or 'predicted_label' not in df.columns:
        print("❌ Error: Required columns 'body' and 'predicted_label' not found")
        return
    
    # 2. Filter high confidence predictions
    if 'confidence' in df.columns:
        print(f"📈 Filtering high confidence predictions (≥0.8)...")
        high_conf_df = df[df['confidence'] >= 0.8].copy()
        print(f"High confidence messages: {len(high_conf_df):,} ({len(high_conf_df)/len(df)*100:.1f}%)")
        df = high_conf_df
    
    # 3. Map labels to 2-class system
    print("🔄 Mapping labels to 2-class system...")
    
    # Map fraud to spam (as requested)
    label_mapping = {
        'legit': 0,
        'spam': 1,
        'fraud': 1  # Map fraud to spam
    }
    
    df['binary_label'] = df['predicted_label'].map(label_mapping)
    
    # Remove any unmapped labels
    df = df.dropna(subset=['binary_label'])
    df['binary_label'] = df['binary_label'].astype(int)
    
    print(f"Final label distribution:")
    label_counts = df['binary_label'].value_counts().sort_index()
    print(f"  Legitimate (0): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"  Spam (1): {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/len(df)*100:.1f}%)")
    
    # 4. Clean and preprocess text
    print("🧹 Preprocessing text...")
    
    # Remove empty messages
    df = df.dropna(subset=['body'])
    df = df[df['body'].str.strip() != '']
    
    # Clean text
    df['cleaned_text'] = df['body'].apply(clean_text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    # Add SMS-specific stop words
    sms_stop_words = {'u', 'ur', 'urs', 'im', 'ive', 'ill', 'id', 'dont', 'cant', 'wont'}
    stop_words.update(sms_stop_words)
    
    df['cleaned_text'] = df['cleaned_text'].apply(
        lambda x: ' '.join([w for w in x.split() if w not in stop_words and len(w) > 1])
    )
    
    # Remove very short messages (less than 3 words)
    df = df[df['cleaned_text'].str.split().str.len() >= 3]
    
    print(f"Messages after preprocessing: {len(df):,}")
    
    # 5. Split data
    print("📊 Splitting data...")
    X = df['cleaned_text']
    y = df['binary_label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train):,} messages")
    print(f"Test set: {len(X_test):,} messages")
    
    # 6. Create TF-IDF features
    print("🔤 Creating TF-IDF features...")
    
    vectorizer = TfidfVectorizer(
        max_features=3000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),  # Include bigrams for better context
        stop_words='english'
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"Feature matrix shape: {X_train_tfidf.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_):,}")
    
    # 7. Train models
    print("🤖 Training models...")
    
    models = {
        'MultinomialNB': MultinomialNB(alpha=1.0),
        'XGBoost': XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        
        # Train model
        model.fit(X_train_tfidf, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_tfidf)
        y_proba = model.predict_proba(X_test_tfidf)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        print(f"✅ {name} Accuracy: {accuracy:.4f}")
        print(f"Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Spam']))
        print(f"Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    
    # 8. Select best model and save
    print("\n💾 Saving best model...")
    
    best_name = max(results, key=lambda k: results[k]['accuracy'])
    best_model = results[best_name]['model']
    best_accuracy = results[best_name]['accuracy']
    
    print(f"🏆 Best model: {best_name} (Accuracy: {best_accuracy:.4f})")
    
    # Save model and vectorizer
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    print(f"✅ Model saved to: {MODEL_PATH}")
    print(f"✅ Vectorizer saved to: {VECTORIZER_PATH}")
    
    # 9. Test with sample messages
    print("\n🧪 Testing with sample messages:")
    print("=" * 50)
    
    test_messages = [
        "Your OTP is 123456. Valid for 10 minutes.",
        "URGENT: Your account has been suspended. Verify now at suspicious-link.com",
        "Congratulations! You've won $1000. Click here to claim your prize!",
        "Your appointment is confirmed for tomorrow at 2 PM",
        "Limited time offer! 50% off all items. Shop now!",
        "Your bank account has been compromised. Call +91234567890 immediately"
    ]
    
    for i, message in enumerate(test_messages, 1):
        # Preprocess
        clean_msg = clean_text(message)
        clean_msg = ' '.join([w for w in clean_msg.split() if w not in stop_words and len(w) > 1])
        
        # Predict
        msg_tfidf = vectorizer.transform([clean_msg])
        prediction = best_model.predict(msg_tfidf)[0]
        probabilities = best_model.predict_proba(msg_tfidf)[0]
        
        # Format output
        label = "Spam" if prediction == 1 else "Legitimate"
        confidence = probabilities[prediction]
        
        print(f"\n{i}. Message: {message}")
        print(f"   Prediction: {label} (confidence: {confidence:.3f})")
        print(f"   Probabilities: Legit={probabilities[0]:.3f}, Spam={probabilities[1]:.3f}")
    
    print(f"\n🎉 Training completed successfully!")
    print(f"📱 Ready for TensorFlow Lite export and Flutter integration.")
    
    # Print summary for documentation
    print(f"\n📋 SUMMARY:")
    print(f"   Training Data: {len(df):,} messages")
    print(f"   Model Type: {best_name}")
    print(f"   Accuracy: {best_accuracy:.4f}")
    print(f"   Classes: 0=Legitimate, 1=Spam")
    print(f"   Note: Fraud detection will be handled in Flutter app based on sender patterns")

if __name__ == "__main__":
    main() 