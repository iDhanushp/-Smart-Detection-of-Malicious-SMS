import pandas as pd
import os

# Paths
input_file = os.path.join('data_set', 'sms+spam+collection', 'SMSSpamCollection')
output_file = os.path.join('data', 'sms_spam.csv')

# Read the tab-separated file
print(f"Reading data from {input_file}...")
df = pd.read_csv(input_file, sep='\t', header=None, names=['label', 'text'])

# Display basic info
print(f"Dataset shape: {df.shape}")
print(f"Label distribution:\n{df['label'].value_counts()}")
print(f"Label distribution (%):\n{df['label'].value_counts(normalize=True) * 100}")

# Save as CSV
print(f"Saving to {output_file}...")
df.to_csv(output_file, index=False, encoding='utf-8')

print("Data preparation completed!")
print(f"Output file: {output_file}")
print(f"Total samples: {len(df)}")
print(f"Ham messages: {len(df[df['label'] == 'ham'])}")
print(f"Spam messages: {len(df[df['label'] == 'spam'])}") 