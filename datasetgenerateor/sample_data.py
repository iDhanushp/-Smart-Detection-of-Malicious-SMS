#!/usr/bin/env python3
"""
Sample random SMS messages from a CSV file for manual labeling.
"""

import pandas as pd
import argparse
import os

def sample_sms_data(input_file, output_file, sample_size=1000, random_state=42):
    """
    Sample random SMS messages from input CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        sample_size (int): Number of messages to sample
        random_state (int): Random seed for reproducibility
    """
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        return False
    
    print(f"Loading SMS data from: {input_file}")
    
    try:
        # Load the full dataset
        df = pd.read_csv(input_file)
        print(f"Total messages loaded: {len(df)}")
        
        # Check required columns
        required_cols = ['body']  # At minimum we need message body
        if not all(col in df.columns for col in required_cols):
            print(f"Error: Required columns {required_cols} not found in CSV!")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        # Remove any rows with empty/null message bodies
        df = df.dropna(subset=['body'])
        df = df[df['body'].str.strip() != '']
        print(f"Messages after removing empty bodies: {len(df)}")
        
        # Sample the data
        if len(df) < sample_size:
            print(f"Warning: Dataset has only {len(df)} messages, using all of them.")
            sample = df.copy()
        else:
            sample = df.sample(n=sample_size, random_state=random_state)
        
        # Add empty label column for manual labeling
        sample['label'] = ''
        
        # Reorder columns to put label at the end
        cols = [col for col in sample.columns if col != 'label'] + ['label']
        sample = sample[cols]
        
        # Save the sample
        sample.to_csv(output_file, index=False)
        print(f"Sample saved to: {output_file}")
        print(f"Sample size: {len(sample)} messages")
        print(f"\nNext steps:")
        print(f"1. Open {output_file} in Excel or Google Sheets")
        print(f"2. Fill in the 'label' column with: legit, spam, or fraud")
        print(f"3. Save the file when done")
        
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Sample SMS messages for labeling')
    parser.add_argument('input_file', help='Input CSV file with SMS messages')
    parser.add_argument('-o', '--output', default='sms_sample_for_labeling.csv', 
                       help='Output CSV file (default: sms_sample_for_labeling.csv)')
    parser.add_argument('-n', '--sample-size', type=int, default=1000,
                       help='Number of messages to sample (default: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    success = sample_sms_data(
        input_file=args.input_file,
        output_file=args.output,
        sample_size=args.sample_size,
        random_state=args.seed
    )
    
    if success:
        print("\n✅ Sampling completed successfully!")
    else:
        print("\n❌ Sampling failed!")

if __name__ == "__main__":
    main() 