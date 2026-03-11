"""dry_run_labeler.py — test new label_message() on 117 known samples without training"""
import sys, re, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

# Execute only the labeling portion of train_real_model.py
src = open('train_real_model.py', encoding='utf-8').read()
cut = src.index('# ─── load & label data')
exec(compile(src[:cut], 'train_real_model.py', 'exec'))

df = pd.read_csv('fraud_samples.csv')
corr = pd.read_csv('fraud_samples_corrected.csv')

# apply new labeler
df['new_auto'] = df.apply(lambda r: label_message(r['sender'], r['body']), axis=1)
df = df.merge(corr[['body','corrected_label','correction_note']], on='body', how='left')

print('=== New auto-labeler distribution on 117 samples ===')
print(df['new_auto'].value_counts().to_string())
print()

# Compare new auto vs human-corrected
df['match'] = df['new_auto'] == df['corrected_label']
accuracy = df['match'].mean()
print(f'Agreement with human labels: {df["match"].sum()}/117  ({accuracy:.1%})')
print()

wrong = df[~df['match']]
print(f'=== {len(wrong)} still wrong after new rules ===')
for _, r in wrong.iterrows():
    print(f"  auto={r['new_auto']:5s} human={r['corrected_label']:5s} | [{str(r['sender'])[:18]:18s}] {str(r['body'])[:70]}")
