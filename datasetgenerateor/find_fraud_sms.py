import pandas as pd, re

base = r'd:\code\Smart Detection of Malicious SMS\datasetgenerateor\sms data set'
files = [
    base + r'\phone_sms_export_2025-07-13T14-41-31.344697.csv',
    base + r'\phone_sms_export_2025-07-13T14-59-37.079178.csv',
    base + r'\phone_sms_export_2025-07-14T09-30-54.278524.csv',
]
frames = []
for f in files:
    try:
        df = pd.read_csv(f)
        frames.append(df)
        print(f"Loaded {len(df)} rows from {f.split(chr(92))[-1]}")
    except Exception as e:
        print(f"Error: {e}")

df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=['body'])
df['body'] = df['body'].fillna('')
df['address'] = df['address'].fillna('')
print(f"\nTotal unique messages: {len(df)}")

fraud_rules = [
    ('account_threat',    r'\b(suspended|blocked|deactivated|terminated)\b.{0,60}\b(account|card|upi|wallet)\b'),
    ('credential_harvest',r'\b(verify|confirm|update)\b.{0,40}\b(otp|pin|cvv|password|card number|account number)\b'),
    ('legal_threat',      r'\b(legal action|court|arrest|police|penalty|fine|lawsuit|fir)\b'),
    ('kyc_fraud',         r'\b(kyc|aadhaar|aadhar|pan)\b.{0,40}\b(update|expire|verify|link|complete)\b'),
    ('fraud_alert_phish', r'\b(unauthorized|suspicious|unusual)\b.{0,40}\b(transaction|activity|login|access)\b'),
    ('phishing_link',     r'(http[s]?://(?!www\.zomato|www\.swiggy|www\.flipkart|www\.amazon|www\.irctc)[^\s]{10,})'),
    ('impersonation',     r'\b(income.?tax|irdai|sebi|rbi.?official|trai|government.?of.?india)\b.{0,60}\b(verify|update|link|action|notice)\b'),
    ('prize_fraud',       r'\b(won|winner|selected|lucky|congratulations)\b.{0,60}\b(rs\.?|₹|\d{4,}|lakh|crore|cash|prize)\b'),
    ('data_steal',        r'\b(share|provide|send|enter)\b.{0,30}\b(otp|pin|password|cvv|card|account|aadhaar|pan)\b'),
]

fraud_rows = []
for _, row in df.iterrows():
    t = str(row['body']).lower()
    s = str(row['address'])
    for reason, pattern in fraud_rules:
        if re.search(pattern, t, re.IGNORECASE | re.DOTALL):
            fraud_rows.append({
                'address': s,
                'reason': reason,
                'body': row['body']
            })
            break  # one label per message

print(f"\nFound {len(fraud_rows)} FRAUD SMS:\n" + "="*80)
for i, r in enumerate(fraud_rows):
    print(f"\n[{i+1}] SENDER: {r['address']}  |  RULE: {r['reason']}")
    print(f"     {r['body'][:200]}")

# Summary by rule
from collections import Counter
counts = Counter(r['reason'] for r in fraud_rows)
print("\n\n=== BY RULE ===")
for rule, cnt in counts.most_common():
    print(f"  {rule}: {cnt}")
