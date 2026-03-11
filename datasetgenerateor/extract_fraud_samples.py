"""
extract_fraud_samples.py
========================
Extracts all FRAUD-labeled SMS from the v3.0 training dataset
using the same label_message() pipeline as train_real_model.py.

Output: fraud_samples.csv  (same folder as this script)
Columns: sender, body, label, fraud_reason
"""

import os, re, warnings, sys
import pandas as pd
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# ── locate data ───────────────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'sms data set')
OUT_CSV  = os.path.join(HERE, 'fraud_samples.csv')

PHONE_FILES = [
    'phone_sms_export_2025-07-13T14-41-31.344697.csv',
    'phone_sms_export_2025-07-13T14-59-37.079178.csv',
    'phone_sms_export_2025-07-14T09-30-54.278524.csv',
]
SPAM_FILE = 'sms_spam.csv'

# ── import label_message from train_real_model ───────────────────────────────
# We exec the source up to (but not including) the data-loading block so we
# get all the helper functions and label_message without triggering training.
_src_path = os.path.join(HERE, 'train_real_model.py')
with open(_src_path, encoding='utf-8') as _f:
    _src = _f.read()

# The data-loading block starts with the first "# ─── load" or "df = " outside a function.
# Cut off just before the data-loading block so functions are imported
# but no training/loading code runs.
_cutoff = _src.find('\n# ─── load & label data')
if _cutoff == -1:
    _cutoff = _src.find("\nif __name__")
if _cutoff == -1:
    _cutoff = len(_src)

# Patch __file__ so path joins inside the imported code don't break
_ns = {'__file__': _src_path, '__name__': 'extract_helper'}
exec(compile(_src[:_cutoff], _src_path, 'exec'), _ns)
label_message = _ns['label_message']
print("✅ label_message imported from train_real_model.py")

# ── load phone CSVs ───────────────────────────────────────────────────────────
phone_frames = []
for fname in PHONE_FILES:
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        print(f"  ⚠️  Not found, skipping: {fname}")
        continue
    df = pd.read_csv(path)
    # normalise column names
    df.columns = [c.lower().strip() for c in df.columns]
    if 'body' not in df.columns and 'text' in df.columns:
        df.rename(columns={'text': 'body'}, inplace=True)
    if 'address' not in df.columns and 'sender' in df.columns:
        df.rename(columns={'sender': 'address'}, inplace=True)
    df = df[['address', 'body']].dropna()
    df['source'] = fname
    phone_frames.append(df)
    print(f"  📂 {fname}: {len(df)} rows")

phone_df = pd.concat(phone_frames, ignore_index=True)
phone_df.drop_duplicates(subset='body', inplace=True)
print(f"  → {len(phone_df)} unique phone messages after dedup")

# ── load sms_spam.csv (ham / spam) ───────────────────────────────────────────
spam_path = os.path.join(DATA_DIR, SPAM_FILE)
spam_df = pd.read_csv(spam_path, encoding='latin-1')
spam_df.columns = [c.lower().strip() for c in spam_df.columns]

# normalise: expect columns v1(label) v2(text) or label/text/message
if 'v1' in spam_df.columns:
    spam_df.rename(columns={'v1': 'ham_label', 'v2': 'body'}, inplace=True)
elif 'label' in spam_df.columns:
    spam_df.rename(columns={'label': 'ham_label', 'message' if 'message' in spam_df.columns else 'text': 'body'}, inplace=True)

spam_df['address'] = spam_df['ham_label'].map({'ham': 'UNKNOWN', 'spam': 'UNKNOWN'})
spam_df['source'] = SPAM_FILE
spam_df = spam_df[['address', 'body', 'source']].dropna()
spam_df.drop_duplicates(subset='body', inplace=True)
print(f"  📂 {SPAM_FILE}: {len(spam_df)} rows")

# ── combine & dedup ───────────────────────────────────────────────────────────
all_df = pd.concat([phone_df, spam_df], ignore_index=True)
before = len(all_df)
all_df.drop_duplicates(subset='body', inplace=True)
print(f"\n📊 Combined: {before} → {len(all_df)} after cross-dataset dedup")

# ── apply label_message ───────────────────────────────────────────────────────
print("\n🏷️  Labeling (this may take a minute)…")
labels = []
for _, row in all_df.iterrows():
    lbl = label_message(str(row['address']), str(row['body']))
    labels.append(lbl)

all_df['label'] = labels

counts = all_df['label'].value_counts()
print(f"\n📊 Label distribution:")
for lbl, n in counts.items():
    print(f"   {lbl}: {n}")

# ── extract FRAUD rows ────────────────────────────────────────────────────────
fraud_df = all_df[all_df['label'] == 'fraud'].copy()
fraud_df.rename(columns={'address': 'sender'}, inplace=True)
fraud_df = fraud_df[['sender', 'body', 'label', 'source']]
fraud_df.reset_index(drop=True, inplace=True)

print(f"\n🚨 FRAUD samples: {len(fraud_df)}")
print(f"\nSender type breakdown:")
fraud_df['sender_type'] = fraud_df['sender'].apply(
    lambda s: 'phone' if str(s).startswith('+') and str(s).replace('+','').isdigit() else 'service'
)
print(fraud_df['sender_type'].value_counts().to_string())

# ── save ──────────────────────────────────────────────────────────────────────
fraud_df.to_csv(OUT_CSV, index=False, encoding='utf-8')
print(f"\n✅ Saved → {OUT_CSV}")
print(fraud_df[['sender', 'body', 'sender_type']].head(10).to_string())
