#!/usr/bin/env python3
"""
Use trained classifier to label remaining SMS data.
"""

import pandas as pd
import numpy as np
import argparse
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

class SMSClassifier:
    def __init__(self):
        self.vectorizer = None
        self.classifier = None
        self.label_encoder = None
        self.is_trained = False
    
    def preprocess_text(self, text):
        """Basic text preprocessing."""
        if pd.isna(text):
            return ""
        return str(text).lower().strip()
    
    def load_model(self, filepath):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.label_encoder = model_data['label_encoder']
        self.is_trained = True
        
        print(f"Model loaded from: {filepath}")
        print(f"Classes: {list(self.label_encoder.classes_)}")
    
    def predict(self, X_test):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be loaded before making predictions")
        
        X_test_clean = [self.preprocess_text(text) for text in X_test]
        X_test_vec = self.vectorizer.transform(X_test_clean)
        
        predictions_encoded = self.classifier.predict(X_test_vec)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        return predictions
    
    def predict_proba(self, X_test):
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be loaded before making predictions")
        
        X_test_clean = [self.preprocess_text(text) for text in X_test]
        X_test_vec = self.vectorizer.transform(X_test_clean)
        
        probas = self.classifier.predict_proba(X_test_vec)
        return probas

def main():
    parser = argparse.ArgumentParser(description='Label remaining SMS data using trained classifier')
    parser.add_argument('input_file', help='Input CSV file with SMS messages')
    parser.add_argument('model_file', help='Trained model file (.pkl)')
    parser.add_argument('-o', '--output', default='fully_labeled.csv',
                       help='Output CSV file (default: fully_labeled.csv)')
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                       help='Confidence threshold for review (default: 0.7)')
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found!")
        return
    
    if not os.path.exists(args.model_file):
        print(f"Error: Model file '{args.model_file}' not found!")
        return
    
    print(f"Loading SMS data from: {args.input_file}")
    
    try:
        # Load data
        df = pd.read_csv(args.input_file)
        print(f"Total messages loaded: {len(df)}")
        
        # Check required columns
        if 'body' not in df.columns:
            print("Error: CSV must contain 'body' column")
            return
        
        # Remove rows with empty bodies
        df = df.dropna(subset=['body'])
        df = df[df['body'].str.strip() != '']
        print(f"Messages after removing empty bodies: {len(df)}")
        
        # Load classifier
        classifier = SMSClassifier()
        classifier.load_model(args.model_file)
        
        # Make predictions
        print("Making predictions...")
        predictions = classifier.predict(df['body'])
        probas = classifier.predict_proba(df['body'])
        
        # Add predictions to dataframe
        df['predicted_label'] = predictions
        df['confidence'] = probas.max(axis=1)
        
        # Add probability columns for each class
        for i, class_name in enumerate(classifier.label_encoder.classes_):
            df[f'prob_{class_name}'] = probas[:, i]
        
        # Save full results
        df.to_csv(args.output, index=False)
        print(f"Fully labeled dataset saved to: {args.output}")
        
        # Print summary
        label_counts = df['predicted_label'].value_counts()
        print(f"\nLabel distribution:")
        for label, count in label_counts.items():
            print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
        
        # High confidence predictions
        high_confidence = df[df['confidence'] >= args.confidence_threshold]
        print(f"\nHigh confidence predictions (>= {args.confidence_threshold}): {len(high_confidence)} ({len(high_confidence)/len(df)*100:.1f}%)")
        
        # Low confidence predictions that need review
        low_confidence = df[df['confidence'] < args.confidence_threshold]
        print(f"Low confidence predictions (< {args.confidence_threshold}): {len(low_confidence)} ({len(low_confidence)/len(df)*100:.1f}%)")
        
        if len(low_confidence) > 0:
            review_file = args.output.replace('.csv', '_review.csv')
            low_confidence.to_csv(review_file, index=False)
            print(f"Review file saved to: {review_file}")
        
        # Average confidence by class
        print(f"\nAverage confidence by class:")
        for label in label_counts.index:
            avg_conf = df[df['predicted_label'] == label]['confidence'].mean()
            print(f"  {label}: {avg_conf:.3f}")
        
        print("\n✅ Labeling completed successfully!")
        print(f"\nNext steps:")
        print(f"1. Review the low confidence predictions in {review_file if len(low_confidence) > 0 else 'N/A'}")
        print(f"2. Correct any mistakes and add to training data")
        print(f"3. Retrain the model with expanded dataset")
        print(f"4. Repeat until satisfied with quality")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 