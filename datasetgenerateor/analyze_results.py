#!/usr/bin/env python3
"""
Analyze the final labeled SMS dataset results.
"""

import pandas as pd

def analyze_results(filename):
    """Analyze the labeled dataset results."""
    
    print(f"Loading results from: {filename}")
    df = pd.read_csv(filename)
    
    print(f"\n📊 FINAL DATASET ANALYSIS")
    print("=" * 50)
    
    print(f"Total messages: {len(df):,}")
    
    print(f"\nLabel distribution:")
    label_counts = df['predicted_label'].value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count:,} ({count/len(df)*100:.1f}%)")
    
    print(f"\nConfidence Analysis:")
    high_conf = df[df['confidence'] >= 0.8]
    medium_conf = df[(df['confidence'] >= 0.6) & (df['confidence'] < 0.8)]
    low_conf = df[df['confidence'] < 0.6]
    
    print(f"  High confidence (>=0.8): {len(high_conf):,} ({len(high_conf)/len(df)*100:.1f}%)")
    print(f"  Medium confidence (0.6-0.8): {len(medium_conf):,} ({len(medium_conf)/len(df)*100:.1f}%)")
    print(f"  Low confidence (<0.6): {len(low_conf):,} ({len(low_conf)/len(df)*100:.1f}%)")
    
    print(f"\nAverage confidence by class:")
    for label in label_counts.index:
        avg_conf = df[df['predicted_label'] == label]['confidence'].mean()
        print(f"  {label}: {avg_conf:.3f}")
    
    print(f"\nOverall average confidence: {df['confidence'].mean():.3f}")
    
    # Check if review file exists
    review_file = filename.replace('.csv', '_review.csv')
    try:
        review_df = pd.read_csv(review_file)
        print(f"\nReview file: {review_file}")
        print(f"Messages needing review: {len(review_df):,}")
    except FileNotFoundError:
        print(f"\nNo review file found.")
    
    print(f"\n✅ Analysis complete!")
    return df

if __name__ == "__main__":
    analyze_results('final_labeled_sms.csv') 