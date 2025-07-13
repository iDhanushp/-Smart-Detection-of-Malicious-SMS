#!/usr/bin/env python3
"""
Fix the auto-labeled data by copying predicted_label to label column.
"""

import pandas as pd
import sys

def fix_labels(input_file, output_file):
    """Copy predicted_label to label column for training."""
    
    # Load the data
    df = pd.read_csv(input_file)
    
    # Copy predicted_label to label column
    df['label'] = df['predicted_label']
    
    # Save the fixed data
    df.to_csv(output_file, index=False)
    
    print(f"Fixed labels in {input_file}")
    print(f"Saved to {output_file}")
    
    # Print label distribution
    label_counts = df['label'].value_counts()
    print(f"\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_labels.py input_file output_file")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    fix_labels(input_file, output_file) 