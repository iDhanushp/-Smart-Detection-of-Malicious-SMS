#!/usr/bin/env python3
"""
Comprehensive Testing with Large SMS Dataset
Tests 3-class detector with extensive real SMS data
"""

import pandas as pd
import numpy as np
from three_class_detector import ThreeClassSMSDetector
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

def load_comprehensive_dataset():
    """Load and analyze comprehensive SMS dataset"""
    print("📊 LOADING COMPREHENSIVE SMS DATASET")
    print("=" * 60)
    
    # Load all available SMS datasets
    datasets = [
        "../../datasetgenerateor/sms data set/phone_sms_export_2025-07-13T14-41-31.344697.csv",
        "../../datasetgenerateor/sms data set/phone_sms_export_2025-07-13T14-59-37.079178.csv", 
        "../../datasetgenerateor/sms data set/phone_sms_export_2025-07-14T09-30-54.278524.csv",
        "../../datasetgenerateor/sms data set/sms_spam.csv"
    ]
    
    all_messages = []
    dataset_info = {}
    
    for dataset_path in datasets:
        try:
            df = pd.read_csv(dataset_path)
            
            if 'sms_spam.csv' in dataset_path:
                # Kaggle dataset
                messages = df[['text', 'label']].copy()
                messages['source'] = 'kaggle'
                messages['sender'] = 'unknown'
            else:
                # Your real SMS data
                messages = df[['body', 'address']].copy()
                messages.rename(columns={'body': 'text', 'address': 'sender'}, inplace=True)
                messages['label'] = 'unknown'  # We'll classify these
                messages['source'] = dataset_path.split('/')[-1]
            
            # Clean data
            messages = messages.dropna(subset=['text'])
            messages = messages[messages['text'].str.len() > 10]
            
            all_messages.append(messages)
            dataset_info[dataset_path] = len(messages)
            
            print(f"✅ {dataset_path.split('/')[-1]}: {len(messages)} messages")
            
        except Exception as e:
            print(f"❌ Error loading {dataset_path}: {e}")
    
    # Combine all datasets
    combined_df = pd.concat(all_messages, ignore_index=True)
    
    print(f"\n📈 DATASET SUMMARY:")
    print(f"Total messages: {len(combined_df)}")
    print(f"Sources: {combined_df['source'].value_counts().to_dict()}")
    
    return combined_df, dataset_info

def analyze_sender_patterns(df):
    """Analyze sender patterns in the dataset"""
    print(f"\n🔍 SENDER PATTERN ANALYSIS")
    print("=" * 40)
    
    sender_patterns = {
        'phone_numbers': 0,
        'service_codes': 0,
        'short_codes': 0,
        'alphanumeric': 0,
        'unknown': 0
    }
    
    for sender in df['sender'].fillna('unknown'):
        sender_str = str(sender)
        
        if sender_str == 'unknown':
            sender_patterns['unknown'] += 1
        elif re.match(r'^\+?\d{10,}$', sender_str):
            sender_patterns['phone_numbers'] += 1
        elif re.match(r'^[A-Z]{2,6}-', sender_str):
            sender_patterns['service_codes'] += 1
        elif re.match(r'^\d{4,6}$', sender_str):
            sender_patterns['short_codes'] += 1
        else:
            sender_patterns['alphanumeric'] += 1
    
    print("Sender Distribution:")
    for pattern, count in sender_patterns.items():
        percentage = (count / len(df)) * 100
        print(f"  {pattern}: {count} ({percentage:.1f}%)")
    
    return sender_patterns

def test_with_large_sample(detector, df, sample_size=1000):
    """Test detector with large sample of real SMS data"""
    print(f"\n🧪 TESTING WITH {sample_size} REAL SMS MESSAGES")
    print("=" * 60)
    
    # Take a diverse sample
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    results = []
    classification_counts = {'LEGITIMATE': 0, 'SPAM': 0, 'FRAUD': 0}
    confidence_scores = {'LEGITIMATE': [], 'SPAM': [], 'FRAUD': []}
    
    print("Processing messages...")
    for idx, row in sample_df.iterrows():
        text = row['text']
        sender = row['sender']
        source = row['source']
        
        try:
            analysis = detector.predict_3class(text)
            prediction = analysis['prediction']
            confidence = analysis['confidence']
            
            classification_counts[prediction] += 1
            confidence_scores[prediction].append(confidence)
            
            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'sender': sender,
                'source': source,
                'prediction': prediction,
                'confidence': confidence,
                'fraud_prob': analysis['probabilities']['FRAUD'],
                'spam_prob': analysis['probabilities']['SPAM'],
                'legit_prob': analysis['probabilities']['LEGITIMATE'],
                'reasoning': '; '.join(analysis['reasoning']) if analysis['reasoning'] else '',
                'risk_factors': '; '.join(analysis['risk_factors']) if analysis['risk_factors'] else ''
            })
            
            if len(results) % 100 == 0:
                print(f"  Processed {len(results)} messages...")
                
        except Exception as e:
            print(f"Error processing message: {e}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Analysis
    print(f"\n📊 CLASSIFICATION RESULTS:")
    print("=" * 40)
    for class_name, count in classification_counts.items():
        percentage = (count / len(results)) * 100
        avg_confidence = np.mean(confidence_scores[class_name]) if confidence_scores[class_name] else 0
        print(f"{class_name}: {count} ({percentage:.1f}%) - Avg Confidence: {avg_confidence:.2f}")
    
    return results_df, classification_counts

def analyze_fraud_messages(results_df):
    """Detailed analysis of messages classified as fraud"""
    fraud_messages = results_df[results_df['prediction'] == 'FRAUD']
    
    if len(fraud_messages) == 0:
        print("\n✅ No messages classified as FRAUD - System appears conservative")
        return
    
    print(f"\n🚨 DETAILED FRAUD ANALYSIS ({len(fraud_messages)} messages)")
    print("=" * 60)
    
    # Show top fraud messages by confidence
    top_fraud = fraud_messages.nlargest(10, 'confidence')
    
    for idx, row in top_fraud.iterrows():
        print(f"\nFRAUD MESSAGE {idx+1}:")
        print(f"Text: {row['text']}")
        print(f"Sender: {row['sender']}")
        print(f"Confidence: {row['confidence']:.2f}")
        print(f"Reasoning: {row['reasoning']}")
        if row['risk_factors']:
            print(f"Risk Factors: {row['risk_factors']}")
        print("-" * 40)

def analyze_spam_messages(results_df):
    """Detailed analysis of messages classified as spam"""
    spam_messages = results_df[results_df['prediction'] == 'SPAM']
    
    if len(spam_messages) == 0:
        print("\n✅ No messages classified as SPAM")
        return
    
    print(f"\n📢 DETAILED SPAM ANALYSIS ({len(spam_messages)} messages)")
    print("=" * 60)
    
    # Show top spam messages by confidence
    top_spam = spam_messages.nlargest(10, 'confidence')
    
    for idx, row in top_spam.iterrows():
        print(f"\nSPAM MESSAGE {idx+1}:")
        print(f"Text: {row['text']}")
        print(f"Sender: {row['sender']}")
        print(f"Confidence: {row['confidence']:.2f}")
        print(f"Reasoning: {row['reasoning']}")
        print("-" * 40)

def manual_validation_sample(results_df):
    """Show sample for manual validation"""
    print(f"\n👁️ MANUAL VALIDATION SAMPLE")
    print("=" * 50)
    
    # Get diverse sample for manual review
    validation_sample = []
    
    # High confidence fraud
    fraud_high = results_df[(results_df['prediction'] == 'FRAUD') & (results_df['confidence'] > 0.8)]
    if len(fraud_high) > 0:
        validation_sample.extend(fraud_high.head(3).to_dict('records'))
    
    # High confidence spam
    spam_high = results_df[(results_df['prediction'] == 'SPAM') & (results_df['confidence'] > 0.7)]
    if len(spam_high) > 0:
        validation_sample.extend(spam_high.head(3).to_dict('records'))
    
    # Borderline cases (medium confidence)
    borderline = results_df[(results_df['confidence'] >= 0.4) & (results_df['confidence'] <= 0.6)]
    if len(borderline) > 0:
        validation_sample.extend(borderline.head(4).to_dict('records'))
    
    print("Please review these classifications:")
    print("=" * 80)
    
    for i, msg in enumerate(validation_sample, 1):
        print(f"\nSample {i}:")
        print(f"Message: {msg['text']}")
        print(f"Sender: {msg['sender']}")
        print(f"Classification: {msg['prediction']} (Confidence: {msg['confidence']:.2f})")
        print(f"Probabilities: FRAUD={msg['fraud_prob']:.2f}, SPAM={msg['spam_prob']:.2f}, LEGIT={msg['legit_prob']:.2f}")
        if msg['reasoning']:
            print(f"Reasoning: {msg['reasoning']}")
        print("-" * 80)

def comprehensive_test():
    """Run comprehensive test with large dataset"""
    print("🚀 COMPREHENSIVE SMS FRAUD DETECTION TEST")
    print("=" * 60)
    
    # Load detector
    print("Loading 3-class detector...")
    detector = ThreeClassSMSDetector()
    
    # Train with current data first
    print("Training detector...")
    detector.train_3class()
    
    # Load comprehensive dataset
    df, dataset_info = load_comprehensive_dataset()
    
    # Analyze sender patterns
    sender_patterns = analyze_sender_patterns(df)
    
    # Test with different sample sizes
    sample_sizes = [500, 1000, 2000]
    
    all_results = {}
    
    for sample_size in sample_sizes:
        if sample_size <= len(df):
            print(f"\n{'='*20} TESTING WITH {sample_size} MESSAGES {'='*20}")
            
            results_df, classification_counts = test_with_large_sample(detector, df, sample_size)
            all_results[sample_size] = {
                'results_df': results_df,
                'classification_counts': classification_counts
            }
            
            # Save results
            output_file = f"comprehensive_test_results_{sample_size}.csv"
            results_df.to_csv(output_file, index=False)
            print(f"Results saved to: {output_file}")
    
    # Detailed analysis with largest sample
    largest_sample = max(all_results.keys())
    results_df = all_results[largest_sample]['results_df']
    
    # Analyze fraud and spam messages
    analyze_fraud_messages(results_df)
    analyze_spam_messages(results_df)
    
    # Manual validation sample
    manual_validation_sample(results_df)
    
    # Summary statistics
    print(f"\n📈 COMPREHENSIVE TEST SUMMARY")
    print("=" * 50)
    print(f"Total datasets tested: {len(dataset_info)}")
    print(f"Total messages available: {len(df)}")
    print(f"Largest sample tested: {largest_sample} messages")
    
    final_results = all_results[largest_sample]['classification_counts']
    total_tested = sum(final_results.values())
    
    print(f"\nFinal Classification Distribution ({total_tested} messages):")
    for class_name, count in final_results.items():
        percentage = (count / total_tested) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    # Risk assessment
    fraud_rate = (final_results['FRAUD'] / total_tested) * 100
    spam_rate = (final_results['SPAM'] / total_tested) * 100
    
    print(f"\nRisk Assessment:")
    print(f"  Fraud Detection Rate: {fraud_rate:.2f}%")
    print(f"  Spam Detection Rate: {spam_rate:.2f}%")
    print(f"  Legitimate Rate: {((final_results['LEGITIMATE'] / total_tested) * 100):.2f}%")
    
    if fraud_rate > 5:
        print("⚠️  HIGH FRAUD DETECTION - Manual review recommended")
    elif fraud_rate > 1:
        print("⚠️  MODERATE FRAUD DETECTION - Monitor closely")
    else:
        print("✅ LOW FRAUD DETECTION - System appears conservative")
    
    return all_results

if __name__ == "__main__":
    comprehensive_test()
