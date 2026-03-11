"""
relabel_fraud_samples.py
Row indices verified against actual fraud_samples.csv content on 2025-07-18.
Applies corrected labels to all 117 rows.
"""
import pandas as pd

df = pd.read_csv('fraud_samples.csv')

# Row-index -> (corrected_label, note)
# All rows originally labeled 'fraud'; only rows listed here get changed.
# Unlisted rows stay 'fraud'.
corrections = {
    # SPAM  ────────────────────────────────────────────────────────────────────
    0:  ('spam',  'Airtel Warning SPAM - investment strategy promo, no bank impersonation'),
    1:  ('spam',  'Airtel Warning SPAM - investment strategy promo'),
    3:  ('spam',  'Airtel Warning SPAM - investment strategy promo'),
    4:  ('spam',  'investment strategy promo'),
    5:  ('spam',  'Airtel Warning SPAM - investment strategy promo'),
    6:  ('spam',  'Airtel Warning SPAM - investment strategy promo'),
    8:  ('spam',  'Airtel Warning SPAM - stock portfolio tips'),
    9:  ('spam',  'Airtel Warning SPAM - stock investment promo'),
    10: ('spam',  'Pi Network crypto referral / MLM spam'),
    20: ('spam',  'generic stock tip spam, tinyurl'),
    21: ('spam',  'generic stock tip spam, goo.su'),
    22: ('spam',  'generic stock tip spam, dub.sh'),
    23: ('spam',  'generic stock tip spam, lihi.cc'),
    24: ('spam',  'generic stock tip spam, goo.su'),
    25: ('spam',  'generic stock tip spam, dub.sh'),
    26: ('spam',  'generic stock tip spam, dub.sh'),
    27: ('spam',  'generic stock tip spam, bit.ly'),
    28: ('spam',  'generic stock tip spam, s.id'),
    52: ('spam',  'weight loss product spam - FDA/GMP/ISO fake claims'),
    53: ('spam',  'university Navratri festival event promotion'),
    56: ('spam',  'stock ratings spam, urlokyin.com'),
    69: ('spam',  'rummy app signup bonus promo, sm9.me'),
    70: ('spam',  'rummy app signup bonus promo'),
    88: ('spam',  'rummy app signup bonus promo, lv0.me'),
    89: ('spam',  'unsolicited MBA education spam, wa.me'),
    92: ('spam',  'rummy app signup bonus promo'),
    96: ('spam',  'work-from-home paper plate manufacturing scam'),
    97: ('spam',  'prize-bait - Kent RO Purifier free gift'),
    98: ('spam',  'casino/rummy welcome bonus signup promo'),
    99: ('spam',  'Nojoto social app referral spam'),
    103:('spam',  'work-from-home LED/paper plate manufacturing scam'),
    104:('spam',  'work-from-home manufacturing scam'),
    105:('spam',  'rummy account balance promo, epq9.com'),
    106:('spam',  'earn second income Rs99990/month'),
    107:('spam',  'rummy welcome bonus signup, va.pcb3.in'),
    108:('spam',  'rummy win promo, l0n.me'),
    109:('spam',  'rummy welcome bonus promo'),
    110:('spam',  'earn second income Rs45954/month spam'),
    111:('spam',  'unsolicited car insurance solicitation, zszs.in'),
    112:('spam',  'rummy win promo, l0n.me'),
    113:('spam',  'rummy win promo'),
    114:('spam',  'N95 mask spam, aynl.in'),
    115:('spam',  'work-from-home LED/CFL manufacturing scam'),

    # LEGIT ────────────────────────────────────────────────────────────────────
    7:  ('legit', 'genuine PVR Cinema cashback SMS post-transaction'),
    16: ('legit', 'genuine Amazon OTP code - amazon.in'),
    36: ('legit', 'genuine UTI MF SIP transaction confirmation'),
    37: ('legit', 'genuine UTI MF distributor SIP recommendation, utimf.com'),
    38: ('legit', 'genuine UTI MF distributor SIP recommendation, utimf.com'),
    39: ('legit', 'genuine Nippon India MF switch request confirmation'),
    40: ('legit', 'genuine Angel One IPO allotment result - Concord Biotech'),
    41: ('legit', 'genuine Angel One IPO allotment result - SBFC Finance'),
    42: ('legit', 'genuine Angel One IPO allotment result - Netweb Technologies'),
    44: ('legit', 'genuine Angel One IPO allotment result - Utkarsh SFB'),
    45: ('legit', 'genuine Angel One IPO allotment result - Cyient DLM'),
    46: ('legit', 'genuine Splitwise group invite - splitwise.com'),
    47: ('legit', 'IDBI Bank anti-phishing security awareness message'),
    48: ('legit', 'genuine Angel One IPO allotment result - Dharmaj Crop Guard'),
    49: ('legit', 'personal GPS location share via Google Maps'),
    50: ('legit', 'personal GPS location share via Google Maps'),
    51: ('legit', 'genuine Jupiter Bank debit card security notification'),
    54: ('legit', 'personal GPS location share via Google Maps'),
    55: ('legit', 'genuine Bajaj Finance EMI card blocked notification, bflcomm.in'),
    62: ('legit', 'official BBMP govt election duty order, bbmpgov.in'),
    63: ('legit', 'official BBMP govt election duty order, bbmpgov.in'),
    64: ('legit', 'genuine WhatsApp deep link invite - whatsapp.com'),
    71: ('legit', 'genuine Airtel recharge cashback promo - i.airtel.in'),
    72: ('legit', 'Airtel missed-call notification service - i.airtel.in'),
    73: ('legit', 'Airtel missed-call notification service - i.airtel.in'),
    74: ('legit', 'Airtel missed-call notification service - i.airtel.in'),
    75: ('legit', 'Airtel missed-call notification service - i.airtel.in'),
    76: ('legit', 'Airtel missed-call notification service - i.airtel.in'),
    77: ('legit', 'Airtel missed-call notification service - i.airtel.in'),
    78: ('legit', 'Airtel missed-call notification service - i.airtel.in'),
    79: ('legit', 'Airtel missed-call notification service - i.airtel.in'),
    80: ('legit', 'Airtel missed-call notification service - i.airtel.in'),
    81: ('legit', 'Airtel missed-call notification service - i.airtel.in'),
    82: ('legit', 'Airtel missed-call notification service - i.airtel.in'),
    83: ('legit', 'Airtel missed-call notification service - i.airtel.in'),
    84: ('legit', 'Airtel missed-call notification service - i.airtel.in'),
    85: ('legit', 'Airtel missed-call notification service - i.airtel.in'),
    86: ('legit', 'Airtel missed-call notification service - i.airtel.in'),
    87: ('legit', 'Airtel missed-call notification service - i.airtel.in'),
   100: ('legit', 'genuine school admission - East West School of Business Management'),
   101: ('legit', 'genuine college admission - Indira Nagar Evening College B.COM'),
   102: ('legit', 'genuine college admission - Sindhi College'),

    # FRAUD (verified - keep) ─────────────────────────────────────────────────
    2:  ('fraud', 'impersonates RBI - RBI-backed arbitrage plan, linstok.com'),
    11: ('fraud', 'impersonates HSBC Securities - fake Rs320450 returns'),
    12: ('fraud', 'impersonates HSBC Securities - fake Rs294700 returns'),
    13: ('fraud', 'impersonates HSBC Securities - fake Rs291220 returns'),
    14: ('fraud', 'impersonates HSBC Securities - fake Rs305760 profits'),
    15: ('fraud', 'impersonates Barclays - fake stock market guidance'),
    17: ('fraud', 'impersonates Barclays - fake Rs355520 returns'),
    18: ('fraud', 'fake investment advisory impersonating INDIRA'),
    19: ('fraud', 'fake investment advisory impersonating INDIRA Rs52000'),
    29: ('fraud', 'CP-MPOKKT account BLOCKED + Video-KYC phishing, ct3.io'),
    30: ('fraud', 'CP-MPOKKT account BLOCKED + Video-KYC phishing'),
    31: ('fraud', 'CP-MPOKKT account BLOCKED + Video-KYC phishing'),
    32: ('fraud', 'CP-MPOKKT account BLOCKED + Video-KYC phishing'),
    33: ('fraud', 'CP-MPOKKT account BLOCKED + Video-KYC phishing'),
    34: ('fraud', 'CP-MPOKKT account BLOCKED + Video-KYC phishing'),
    35: ('fraud', 'CP-MPOKKT account BLOCKED + Video-KYC phishing'),
    43: ('fraud', 'fake MSME govt loan scheme Rs25L - epq9.com phishing'),
    57: ('fraud', 'impersonates RBI - Rs10K to Rs50L story, mktasset.com'),
    58: ('fraud', 'impersonates Barclays - stock update phishing'),
    59: ('fraud', 'impersonates HSBC Securities - fake Rs258760 returns'),
    60: ('fraud', 'fake investment advisory impersonating INDIRA Rs292870'),
    61: ('fraud', 'SBI YONO phishing - obfuscated S.B,I Y0N0, rb.gy link'),
    65: ('fraud', 'fake BHIM transfer Rs98560 - Withdraw Cash NOW, 9ko6.com'),
    66: ('fraud', 'fake BHIM transfer Rs89699 - Withdraw Cash NOW'),
    67: ('fraud', 'fake BHIM credit Rs82850 - GET Cash NOW, 4ps2.com'),
    68: ('fraud', 'fake rummy account credit Rs93472 - Withdraw N0W, 8jo5.com'),
    90: ('fraud', 'impersonates HDFC Bank - unsolicited loan from phone number + wa.me'),
    91: ('fraud', 'impersonates Axis Bank - unsolicited loan from phone number + wa.me'),
    93: ('fraud', 'impersonates Axis Bank credit card - epq9.com phishing'),
    94: ('fraud', 'fake government Bima Yojana insurance scheme, vb.pcb3.in'),
    95: ('fraud', 'GOLDIFY fake trading platform - wap.goldifyapp.com + wa.me'),
   116: ('fraud', 'data-harvest phishing - confirm details a2fn.com'),
}

df['corrected_label'] = df.index.map(
    lambda i: corrections[i][0] if i in corrections else df.loc[i, 'label']
)
df['correction_note'] = df.index.map(
    lambda i: corrections[i][1] if i in corrections else 'unchanged'
)

# Summary
dist = df['corrected_label'].value_counts()
print('Corrected label distribution:')
print(dist.to_string())
print()
fraud_n = dist.get('fraud', 0)
spam_n  = dist.get('spam',  0)
legit_n = dist.get('legit', 0)
print(f'  stayed fraud :  {fraud_n}')
print(f'  fraud -> spam:  {spam_n}')
print(f'  fraud -> legit: {legit_n}')
print(f'  total rows:     {len(df)}')

assert len(df) == 117, f'Expected 117 rows, got {len(df)}'
assert dist.sum() == 117

df.to_csv('fraud_samples_corrected.csv', index=False)
print('\nSaved -> fraud_samples_corrected.csv')

# Verification printout
print(f'\n=== FRAUD rows kept ({fraud_n}) ===')
for i, row in df[df['corrected_label'] == 'fraud'].iterrows():
    print(f'  [{i:3d}] {row["correction_note"]}')
    print(f'        {str(row["body"])[:88]}')

print(f'\n=== SPAM rows ({spam_n}) ===')
for i, row in df[df['corrected_label'] == 'spam'].iterrows():
    print(f'  [{i:3d}] {row["correction_note"]}')

print(f'\n=== LEGIT rows ({legit_n}) ===')
for i, row in df[df['corrected_label'] == 'legit'].iterrows():
    print(f'  [{i:3d}] {row["correction_note"]}')
