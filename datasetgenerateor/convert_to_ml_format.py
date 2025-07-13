#!/usr/bin/env python3
"""
Convert labeled SMS dataset to ML_Model training format.
Creates legit.txt, spam.txt, and fraud.txt files for training.
"""

import pandas as pd
import os
import argparse

def convert_to_ml_format(input_file, output_dir="../ML_Model/data", min_confidence=0.8):
    """
    Convert labeled SMS dataset to ML training format.
    
    Args:
        input_file: Path to labeled CSV file
        output_dir: Directory to save the text files
        min_confidence: Minimum confidence threshold to include messages
    """
    
    print(f"Loading labeled data from: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"Total messages: {len(df):,}")
    
    # Filter by confidence if specified
    if min_confidence > 0:
        high_conf_df = df[df['confidence'] >= min_confidence]
        print(f"High confidence messages (>={min_confidence}): {len(high_conf_df):,}")
        df = high_conf_df
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by label
    label_counts = df['predicted_label'].value_counts()
    print(f"\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Save each class to separate files
    for label in ['legit', 'spam', 'fraud']:
        label_df = df[df['predicted_label'] == label]
        
        if len(label_df) > 0:
            # Extract just the message bodies
            messages = label_df['body'].tolist()
            
            # Clean messages (remove newlines, strip whitespace)
            cleaned_messages = []
            for msg in messages:
                if pd.notna(msg) and str(msg).strip():
                    cleaned_msg = str(msg).strip().replace('\n', ' ').replace('\r', ' ')
                    cleaned_messages.append(cleaned_msg)
            
            # Save to file
            output_file = os.path.join(output_dir, f"{label}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                for msg in cleaned_messages:
                    f.write(msg + '\n')
            
            print(f"Saved {len(cleaned_messages):,} {label} messages to: {output_file}")
        else:
            print(f"No {label} messages found!")
    
    print(f"\n✅ Conversion complete!")
    print(f"Files saved to: {output_dir}")
    
    # Create a summary file
    summary_file = os.path.join(output_dir, "dataset_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"SMS Dataset Summary\n")
        f.write(f"==================\n\n")
        f.write(f"Total messages processed: {len(df):,}\n")
        f.write(f"Minimum confidence threshold: {min_confidence}\n\n")
        f.write(f"Label distribution:\n")
        for label, count in label_counts.items():
            f.write(f"  {label}: {count:,} ({count/len(df)*100:.1f}%)\n")
        f.write(f"\nAverage confidence: {df['confidence'].mean():.3f}\n")
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Summary saved to: {summary_file}")
    
    return label_counts

def main():
    parser = argparse.ArgumentParser(description='Convert labeled SMS dataset to ML training format')
    parser.add_argument('input_file', help='Input CSV file with labeled SMS messages')
    parser.add_argument('-o', '--output-dir', default='../ML_Model/data',
                       help='Output directory for text files (default: ../ML_Model/data)')
    parser.add_argument('--min-confidence', type=float, default=0.8,
                       help='Minimum confidence threshold (default: 0.8)')
    parser.add_argument('--include-all', action='store_true',
                       help='Include all messages regardless of confidence')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found!")
        return
    
    min_conf = 0.0 if args.include_all else args.min_confidence
    
    try:
        convert_to_ml_format(args.input_file, args.output_dir, min_conf)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 