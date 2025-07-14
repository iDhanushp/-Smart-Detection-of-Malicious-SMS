#!/usr/bin/env python3
"""
Enhanced Training Script with Semantic Analysis and Behavioral Features
Replaces the simple keyword-based approach with advanced ML techniques
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import datetime
import json
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add the advanced features directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'advanced_features'))

try:
    from semantic_detector import SemanticFraudDetector
    SEMANTIC_AVAILABLE = True
except ImportError:
    print("Warning: Semantic detector not available. Install sentence-transformers to enable.")
    SEMANTIC_AVAILABLE = False

# Add the dataset generator directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasetgenerateor'))

try:
    from enhanced_behavioral_labeler import BehavioralSMSLabeler
    BEHAVIORAL_AVAILABLE = True
except ImportError:
    print("Warning: Behavioral labeler not available.")
    BEHAVIORAL_AVAILABLE = False

class EnhancedSMSTrainer:
    def __init__(self, use_semantic=True, use_behavioral=True):
        """Initialize enhanced SMS trainer"""
        self.use_semantic = use_semantic and SEMANTIC_AVAILABLE
        self.use_behavioral = use_behavioral and BEHAVIORAL_AVAILABLE
        
        if self.use_semantic:
            self.semantic_detector = SemanticFraudDetector()
        
        if self.use_behavioral:
            self.behavioral_labeler = BehavioralSMSLabeler()
        
        self.models = {}
        self.feature_scaler = StandardScaler()
        
    def load_and_prepare_data(self, data_path: str, text_column: str = 'body', 
                             label_column: str = 'label', sender_column: str = 'address'):
        """Load and prepare data with enhanced features"""
        
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Clean data
        df = df.dropna(subset=[text_column])
        df[text_column] = df[text_column].astype(str)
        
        if label_column not in df.columns:
            print(f"Label column '{label_column}' not found. Auto-labeling with behavioral analysis...")
            if self.use_behavioral:
                labeled_df = self.behavioral_labeler.process_dataset(df, text_column, sender_column)
                # Convert string labels to numeric
                label_map = {'LEGIT': 0, 'SPAM': 1, 'FRAUD': 1}  # Treat fraud as spam for binary classification
                df['label'] = labeled_df['classification'].map(label_map)
                df = df.dropna(subset=['label'])
            else:
                raise ValueError(f"Label column '{label_column}' not found and behavioral labeling not available")
        
        texts = df[text_column].tolist()
        labels = df[label_column].tolist()
        
        print(f"Loaded {len(texts)} messages")
        print(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
        
        return texts, labels, df
    
    def extract_enhanced_features(self, texts: list) -> np.ndarray:
        """Extract comprehensive features combining multiple approaches"""
        all_features = []
        
        # 1. Semantic features (if available)
        if self.use_semantic:
            print("Extracting semantic embeddings...")
            semantic_features = self.semantic_detector.extract_combined_features(texts)
            all_features.append(semantic_features)
        
        # 2. Behavioral features
        if self.use_behavioral:
            print("Extracting behavioral features...")
            behavioral_features = []
            for text in texts:
                signals = self.behavioral_labeler.analyze_behavioral_signals(text)
                structural = self.behavioral_labeler.analyze_structural_features(text)
                # Combine all features into a single vector
                feature_vector = list(signals.values()) + list(structural.values())
                behavioral_features.append(feature_vector)
            
            behavioral_features = np.array(behavioral_features)
            all_features.append(behavioral_features)
        
        # Combine all feature types
        if all_features:
            combined_features = np.hstack(all_features)
        else:
            # Fallback to basic TF-IDF if no enhanced features available
            from sklearn.feature_extraction.text import TfidfVectorizer
            print("Using fallback TF-IDF features...")
            vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
            combined_features = vectorizer.fit_transform(texts).toarray()
        
        return combined_features
    
    def train_ensemble_model(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train ensemble of models for robust prediction"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Define base models
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
            ),
            'xgboost': None,  # Will be added if available
            'lightgbm': None   # Will be added if available
        }
        
        # Add XGBoost if available
        try:
            from xgboost import XGBClassifier
            models['xgboost'] = XGBClassifier(
                n_estimators=100, max_depth=6, random_state=42, eval_metric='logloss'
            )
        except ImportError:
            print("XGBoost not available, skipping...")
        
        # Add LightGBM if available
        try:
            from lightgbm import LGBMClassifier
            models['lightgbm'] = LGBMClassifier(
                n_estimators=100, max_depth=6, random_state=42, verbose=-1
            )
        except ImportError:
            print("LightGBM not available, skipping...")
        
        # Filter out None models
        models = {k: v for k, v in models.items() if v is not None}
        
        # Train individual models
        trained_models = {}
        model_scores = {}
        
        print(f"Training {len(models)} base models...")
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            model_scores[name] = accuracy
            trained_models[name] = model
            
            print(f"{name} accuracy: {accuracy:.4f}")
        
        # Create ensemble
        if len(trained_models) > 1:
            print("Creating ensemble model...")
            ensemble_models = [(name, model) for name, model in trained_models.items()]
            ensemble = VotingClassifier(estimators=ensemble_models, voting='soft')
            ensemble.fit(X_train_scaled, y_train)
            
            # Evaluate ensemble
            y_pred_ensemble = ensemble.predict(X_test_scaled)
            ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
            print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
            
            trained_models['ensemble'] = ensemble
            model_scores['ensemble'] = ensemble_accuracy
        
        # Select best model
        best_model_name = max(model_scores, key=model_scores.get)
        best_model = trained_models[best_model_name]
        
        print(f"Best model: {best_model_name} (accuracy: {model_scores[best_model_name]:.4f})")
        
        # Detailed evaluation of best model
        y_pred_best = best_model.predict(X_test_scaled)
        
        print(f"\nDetailed evaluation of {best_model_name}:")
        print("Classification Report:")
        print(classification_report(y_test, y_pred_best))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_best))
        
        return {
            'best_model': best_model,
            'best_model_name': best_model_name,
            'all_models': trained_models,
            'scores': model_scores,
            'test_accuracy': model_scores[best_model_name],
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_pred': y_pred_best
        }
    
    def save_model(self, model_data: dict, output_dir: str):
        """Save trained model and metadata"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main model
        model_path = os.path.join(output_dir, 'enhanced_fraud_detector.pkl')
        model_to_save = {
            'model': model_data['best_model'],
            'scaler': self.feature_scaler,
            'model_name': model_data['best_model_name'],
            'test_accuracy': model_data['test_accuracy'],
            'use_semantic': self.use_semantic,
            'use_behavioral': self.use_behavioral
        }
        
        joblib.dump(model_to_save, model_path)
        print(f"Model saved to {model_path}")
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'enhanced_sms_fraud_detector',
            'best_model': model_data['best_model_name'],
            'test_accuracy': float(model_data['test_accuracy']),
            'all_scores': {k: float(v) for k, v in model_data['scores'].items()},
            'features': {
                'semantic_enabled': self.use_semantic,
                'behavioral_enabled': self.use_behavioral
            }
        }
        
        metadata_path = os.path.join(output_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {metadata_path}")
        
        return model_path, metadata_path
    
    def export_for_mobile(self, model_data: dict, output_dir: str):
        """Export model for mobile deployment"""
        print("Exporting model for mobile deployment...")
        
        # For now, save feature extraction logic and model
        # TODO: Implement TensorFlow Lite conversion for semantic features
        mobile_export = {
            'lightweight_model': model_data['all_models'].get('random_forest', model_data['best_model']),
            'feature_scaler': self.feature_scaler,
            'behavioral_patterns': self.behavioral_labeler.__dict__ if self.use_behavioral else None,
            'model_type': 'enhanced_mobile_compatible'
        }
        
        mobile_path = os.path.join(output_dir, 'mobile_fraud_detector.pkl')
        joblib.dump(mobile_export, mobile_path)
        print(f"Mobile-compatible model saved to {mobile_path}")
        
        return mobile_path

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Enhanced SMS Fraud Detection Training')
    parser.add_argument('--data', required=True, help='Path to SMS dataset CSV')
    parser.add_argument('--output', default='./enhanced_models', help='Output directory for trained models')
    parser.add_argument('--text-column', default='body', help='Text column name')
    parser.add_argument('--label-column', default='label', help='Label column name')
    parser.add_argument('--sender-column', default='address', help='Sender column name')
    parser.add_argument('--no-semantic', action='store_true', help='Disable semantic features')
    parser.add_argument('--no-behavioral', action='store_true', help='Disable behavioral features')
    
    args = parser.parse_args()
    
    # Initialize trainer
    use_semantic = not args.no_semantic
    use_behavioral = not args.no_behavioral
    
    trainer = EnhancedSMSTrainer(use_semantic=use_semantic, use_behavioral=use_behavioral)
    
    # Load and prepare data
    texts, labels, df = trainer.load_and_prepare_data(
        args.data, args.text_column, args.label_column, args.sender_column
    )
    
    # Extract enhanced features
    print("Extracting enhanced features...")
    X = trainer.extract_enhanced_features(texts)
    y = np.array(labels)
    
    print(f"Feature matrix shape: {X.shape}")
    
    # Train ensemble model
    print("Training ensemble model...")
    model_data = trainer.train_ensemble_model(X, y)
    
    # Save models
    print("Saving models...")
    model_path, metadata_path = trainer.save_model(model_data, args.output)
    mobile_path = trainer.export_for_mobile(model_data, args.output)
    
    print(f"\nTraining completed successfully!")
    print(f"Best model: {model_data['best_model_name']}")
    print(f"Test accuracy: {model_data['test_accuracy']:.4f}")
    print(f"Models saved to: {args.output}")

if __name__ == "__main__":
    main()
