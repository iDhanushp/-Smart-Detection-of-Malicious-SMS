#!/usr/bin/env python3
"""
Train a basic SMS classifier using labeled data.
"""

import pandas as pd
import numpy as np
import argparse
import os
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

class SMSClassifier:
    def __init__(self, min_df=2, max_features=5000, max_iter=1000):
        self.vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)  # Use unigrams and bigrams
        )
        self.classifier = LogisticRegression(
            max_iter=max_iter,
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        )
        self.label_encoder = LabelEncoder()
        self.is_trained = False
    
    def preprocess_text(self, text):
        """Basic text preprocessing."""
        if pd.isna(text):
            return ""
        return str(text).lower().strip()
    
    def train(self, X_train, y_train):
        """Train the classifier."""
        print("Preprocessing text...")
        X_train_clean = [self.preprocess_text(text) for text in X_train]
        
        print("Vectorizing text...")
        X_train_vec = self.vectorizer.fit_transform(X_train_clean)
        
        print("Encoding labels...")
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        print("Training classifier...")
        self.classifier.fit(X_train_vec, y_train_encoded)
        self.is_trained = True
        
        print(f"Training completed!")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print(f"Classes: {list(self.label_encoder.classes_)}")
    
    def predict(self, X_test):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_test_clean = [self.preprocess_text(text) for text in X_test]
        X_test_vec = self.vectorizer.transform(X_test_clean)
        
        predictions_encoded = self.classifier.predict(X_test_vec)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        return predictions
    
    def predict_proba(self, X_test):
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_test_clean = [self.preprocess_text(text) for text in X_test]
        X_test_vec = self.vectorizer.transform(X_test_clean)
        
        probas = self.classifier.predict_proba(X_test_vec)
        return probas
    
    def evaluate(self, X_test, y_test):
        """Evaluate the classifier."""
        predictions = self.predict(X_test)
        
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        
        return predictions
    
    def save_model(self, filepath):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'label_encoder': self.label_encoder
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.label_encoder = model_data['label_encoder']
        self.is_trained = True
        
        print(f"Model loaded from: {filepath}")

def load_labeled_data(filepath):
    """Load labeled SMS data."""
    df = pd.read_csv(filepath)
    
    # Check required columns
    if 'body' not in df.columns:
        raise ValueError("CSV must contain 'body' column")
    
    if 'label' not in df.columns:
        raise ValueError("CSV must contain 'label' column")
    
    # Remove rows with empty labels or bodies
    df = df.dropna(subset=['body', 'label'])
    df = df[df['body'].str.strip() != '']
    df = df[df['label'].str.strip() != '']
    
    # Normalize labels
    df['label'] = df['label'].str.lower().str.strip()
    
    # Filter valid labels
    valid_labels = ['legit', 'spam', 'fraud']
    df = df[df['label'].isin(valid_labels)]
    
    print(f"Loaded {len(df)} labeled messages")
    print(f"Label distribution:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
    
    return df['body'].tolist(), df['label'].tolist()

def main():
    parser = argparse.ArgumentParser(description='Train SMS classifier')
    parser.add_argument('labeled_data', help='CSV file with labeled SMS messages')
    parser.add_argument('-o', '--output', default='sms_classifier.pkl',
                       help='Output model file (default: sms_classifier.pkl)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--min-df', type=int, default=2,
                       help='Minimum document frequency for TF-IDF (default: 2)')
    parser.add_argument('--max-features', type=int, default=5000,
                       help='Maximum features for TF-IDF (default: 5000)')
    parser.add_argument('--cross-validation', action='store_true',
                       help='Perform cross-validation')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.labeled_data):
        print(f"Error: Labeled data file '{args.labeled_data}' not found!")
        return
    
    try:
        # Load data
        print(f"Loading labeled data from: {args.labeled_data}")
        X, y = load_labeled_data(args.labeled_data)
        
        # Split data
        print(f"\nSplitting data (test size: {args.test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Initialize and train classifier
        classifier = SMSClassifier(
            min_df=args.min_df,
            max_features=args.max_features
        )
        
        classifier.train(X_train, y_train)
        
        # Evaluate
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        classifier.evaluate(X_test, y_test)
        
        # Cross-validation
        if args.cross_validation:
            print("\n" + "="*50)
            print("CROSS-VALIDATION RESULTS")
            print("="*50)
            
            # Prepare data for cross-validation
            X_clean = [classifier.preprocess_text(text) for text in X]
            X_vec = classifier.vectorizer.transform(X_clean)
            y_encoded = classifier.label_encoder.transform(y)
            
            cv_scores = cross_val_score(
                classifier.classifier, X_vec, y_encoded, cv=5, scoring='accuracy'
            )
            
            print(f"Cross-validation scores: {cv_scores}")
            print(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Save model
        classifier.save_model(args.output)
        
        # Save predictions on test set for analysis
        predictions = classifier.predict(X_test)
        probas = classifier.predict_proba(X_test)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'message': X_test,
            'true_label': y_test,
            'predicted_label': predictions,
            'confidence': probas.max(axis=1)
        })
        
        # Add probability columns for each class
        for i, class_name in enumerate(classifier.label_encoder.classes_):
            results_df[f'prob_{class_name}'] = probas[:, i]
        
        results_file = args.output.replace('.pkl', '_results.csv')
        results_df.to_csv(results_file, index=False)
        print(f"Test results saved to: {results_file}")
        
        print("\n✅ Training completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 