#!/usr/bin/env python3
"""
Advanced SMS Fraud Detection Training with Real Data
Combines Kaggle spam dataset with user's real SMS data for balanced training
"""

import pandas as pd
import numpy as np
import re
from semantic_detector import SemanticFraudDetector
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_and_prepare_data():
    """Load and combine all available datasets"""
    print("Loading datasets...")
    
    # 1. Load Kaggle SMS spam dataset
    kaggle_df = pd.read_csv('../../../datasetgenerateor/sms data set/sms_spam.csv')
    print(f"Kaggle dataset: {len(kaggle_df)} messages")
    print(f"  - Ham (legitimate): {len(kaggle_df[kaggle_df.label=='ham'])}")
    print(f"  - Spam: {len(kaggle_df[kaggle_df.label=='spam'])}")
    
    # 2. Load your real SMS data (all considered legitimate since from your phone)
    real_sms_files = [
        '../../../datasetgenerateor/sms data set/phone_sms_export_2025-07-13T14-41-31.344697.csv',
        '../../../datasetgenerateor/sms data set/phone_sms_export_2025-07-13T14-59-37.079178.csv', 
        '../../../datasetgenerateor/sms data set/phone_sms_export_2025-07-14T09-30-54.278524.csv'
    ]
    
    real_sms_data = []
    total_real_messages = 0
    
    for file_path in real_sms_files:
        try:
            df = pd.read_csv(file_path)
            # Filter out personal messages (from phone numbers) and keep only service messages
            service_messages = df[
                (df['address'].str.contains(r'^[A-Z]{2,6}-', na=False)) |  # Service codes like AX-KOTAKB
                (df['address'].str.len() <= 8) |  # Short codes
                (df['address'].str.contains(r'^[A-Z]+$', na=False))  # All caps service names
            ].copy()
            
            service_messages['label'] = 'ham'  # All service messages are legitimate
            service_messages['text'] = service_messages['body']
            real_sms_data.append(service_messages[['text', 'label']])
            total_real_messages += len(service_messages)
            print(f"  - {file_path.split('/')[-1]}: {len(service_messages)} service messages")
        except Exception as e:
            print(f"  - Error loading {file_path}: {e}")
    
    print(f"Total real service messages: {total_real_messages}")
    
    # 3. Combine all datasets
    all_datasets = [kaggle_df[['text', 'label']]]
    if real_sms_data:
        all_datasets.extend(real_sms_data)
    
    combined_df = pd.concat(all_datasets, ignore_index=True)
    
    # 4. Clean and preprocess
    combined_df['text'] = combined_df['text'].astype(str)
    combined_df = combined_df[combined_df['text'].str.len() > 10]  # Remove very short messages
    combined_df = combined_df.drop_duplicates(subset=['text'])  # Remove duplicates
    
    # 5. Convert labels to binary
    combined_df['binary_label'] = combined_df['label'].map({'ham': 0, 'spam': 1})
    
    print(f"\n=== FINAL COMBINED DATASET ===")
    print(f"Total messages: {len(combined_df)}")
    print(f"Legitimate (ham): {len(combined_df[combined_df.binary_label==0])}")
    print(f"Spam/Fraud: {len(combined_df[combined_df.binary_label==1])}")
    print(f"Balance ratio: {len(combined_df[combined_df.binary_label==1])/len(combined_df[combined_df.binary_label==0]):.2f}")
    
    return combined_df

def analyze_message_patterns(df):
    """Analyze patterns in legitimate vs spam messages"""
    print(f"\n=== MESSAGE PATTERN ANALYSIS ===")
    
    # Separate ham and spam
    ham_messages = df[df.binary_label == 0]['text'].tolist()
    spam_messages = df[df.binary_label == 1]['text'].tolist()
    
    print(f"\nLegitimate Message Examples:")
    for i, msg in enumerate(ham_messages[:5]):
        print(f"  {i+1}. {msg[:80]}...")
    
    print(f"\nSpam Message Examples:")
    for i, msg in enumerate(spam_messages[:5]):
        print(f"  {i+1}. {msg[:80]}...")
    
    # Analyze common patterns
    detector = SemanticFraudDetector()
    
    # Sample analysis
    print(f"\n=== BEHAVIORAL PATTERN ANALYSIS ===")
    sample_ham = ham_messages[:100]
    sample_spam = spam_messages[:100] if len(spam_messages) >= 100 else spam_messages
    
    ham_features = [detector.extract_behavioral_features(msg) for msg in sample_ham]
    spam_features = [detector.extract_behavioral_features(msg) for msg in sample_spam]
    
    # Average feature scores
    ham_avg = {key: np.mean([f[key] for f in ham_features]) for key in ham_features[0].keys()}
    spam_avg = {key: np.mean([f[key] for f in spam_features]) for key in spam_features[0].keys()}
    
    print(f"\nKey Behavioral Differences:")
    key_features = ['urgency_score', 'fear_score', 'reward_score', 'authority_score', 'action_score']
    for feature in key_features:
        print(f"  {feature}:")
        print(f"    Ham avg: {ham_avg[feature]:.3f}")
        print(f"    Spam avg: {spam_avg[feature]:.3f}")
        print(f"    Difference: {spam_avg[feature] - ham_avg[feature]:.3f}")

def train_enhanced_model(df):
    """Train the enhanced semantic fraud detector"""
    print(f"\n=== TRAINING ENHANCED MODEL ===")
    
    # Prepare data
    texts = df['text'].tolist()
    labels = df['binary_label'].tolist()
    
    # Initialize detector
    detector = SemanticFraudDetector()
    
    # Train model
    results = detector.train(texts, labels)
    
    # Save trained model
    model_path = '../enhanced_semantic_model.pkl'
    detector.save_model(model_path)
    
    return detector, results

def test_with_real_examples(detector):
    """Test the model with real-world examples"""
    print(f"\n=== TESTING WITH REAL EXAMPLES ===")
    
    # Real examples from your SMS data
    test_messages = [
        # Legitimate bank/service messages
        "Rs.2.00 spent via Kotak Debit Card XX5673 at AMAZONAWSESC on 12/07/2025. Avl bal Rs.15566.11 Not you?Tap https://kotak.com/KBANKT/Fraud",
        "Your Apple Account code is: 490687. Do not share it with anyone.",
        "An amount of INR 1,500.00 has been DEBITED to your account XXXXX05840 on 12/07/2025. Total Avail.bal INR 8.03.Dial 1930 to report cyber fraud - Canara Bank",
        
        # Promotional but legitimate
        "Get up to Rs. 150 cashback now!  Just recharge via Airtel Thanks App https://i.airtel.in/1_upto150_cashback",
        "Your weekend watch is ready! This Xstream Weekend Watch, 'Mitti - Ek Nayi Pehchaan' is FREE for Airtel users! Watch it Saturday & Sunday.",
        
        # Potential spam/fraud patterns
        "BILLS DUE ? Get EASY credit of Rs.1,03,000 at Rs.4849/month with QUICK disbursal! Apply now: a1.Freo.in/x0WKK0l T&C -Freo",
        "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010",
        "Your account has been suspended. Click here immediately to verify: http://fake-bank.com"
    ]
    
    expected_labels = [
        "LEGITIMATE",  # Bank transaction
        "LEGITIMATE",  # Apple OTP
        "LEGITIMATE",  # Bank debit alert
        "LEGITIMATE",  # Airtel cashback
        "LEGITIMATE",  # Airtel promo
        "SPAM",        # Loan offer
        "FRAUD",       # Prize scam
        "FRAUD"        # Account suspension
    ]
    
    print(f"\nDetailed Analysis Results:")
    print("=" * 80)
    
    correct_predictions = 0
    for i, (message, expected) in enumerate(zip(test_messages, expected_labels)):
        analysis = detector.analyze_message(message)
        prediction = analysis['prediction']
        confidence = analysis['fraud_probability']
        
        is_correct = prediction == expected
        correct_predictions += is_correct
        
        print(f"\nMessage {i+1}: {message[:60]}...")
        print(f"Expected: {expected} | Predicted: {prediction} | {'✓' if is_correct else '✗'}")
        print(f"Confidence: {confidence:.2f} | Urgency: {analysis['urgency_level']} | Fear: {analysis['fear_level']}")
        if analysis['risk_factors']:
            print(f"Risk Factors: {', '.join(analysis['risk_factors'][:3])}")
        print("-" * 80)
    
    accuracy = correct_predictions / len(test_messages)
    print(f"\nReal-world Test Accuracy: {accuracy:.1%} ({correct_predictions}/{len(test_messages)})")
    
    return accuracy

def main():
    """Main training and testing pipeline"""
    print("🚀 ENHANCED SMS FRAUD DETECTION TRAINING")
    print("=" * 60)
    
    # Step 1: Load and prepare data
    df = load_and_prepare_data()
    
    # Step 2: Analyze patterns
    analyze_message_patterns(df)
    
    # Step 3: Train model
    detector, training_results = train_enhanced_model(df)
    
    # Step 4: Test with real examples
    real_world_accuracy = test_with_real_examples(detector)
    
    # Step 5: Summary
    print(f"\n🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"📊 Training Accuracy: {training_results['accuracy']:.1%}")
    print(f"🌍 Real-world Test Accuracy: {real_world_accuracy:.1%}")
    print(f"💾 Model saved to: enhanced_semantic_model.pkl")
    print(f"📈 Total training messages: {len(df)}")
    print(f"🔒 Enhanced behavioral analysis active")
    
    return detector

if __name__ == "__main__":
    main()
