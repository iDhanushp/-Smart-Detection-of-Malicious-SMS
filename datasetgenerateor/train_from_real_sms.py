#!/usr/bin/env python3
"""
Train a proper SMS fraud detector from REAL phone SMS data.

Pipeline:
  1. Load all 17k+ real phone SMS (address + body) + sms_spam.csv
  2. Auto-label using smart Indian DLT sender rules + content rules
  3. Extract the EXACT 30 features used by advanced_fraud_detector.dart
  4. Train a deep neural network with proper regularisation
  5. Export TFLite + behavioral_model_config.json to the Flutter assets folder

Classes:  0 = LEGITIMATE   1 = SPAM   2 = FRAUD
"""

import re, json, math, os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"d:\code\Smart Detection of Malicious SMS\datasetgenerateor\sms data set")
OUT_DIR    = Path(r"d:\code\Smart Detection of Malicious SMS\sms_fraud_detectore_app\assets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TFLITE_PATH = OUT_DIR / "advanced_fraud_detector.tflite"
CONFIG_PATH = OUT_DIR / "behavioral_model_config.json"

# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION  (mirrors advanced_fraud_detector.dart)
# ─────────────────────────────────────────────────────────────

_URGENCY_KW  = ['urgent','immediately','asap','expire','deadline',
                 'limited time','act now','hurry','last chance',
                 'expire today','expires soon','time running out']
_FEAR_KW     = ['suspended','blocked','terminated','legal action',
                 'penalty','fine','arrest','court','lawsuit',
                 'closed','cancelled','frozen','unauthorized']
_REWARD_KW   = ['congratulations','winner','won','prize','cash',
                 'reward','lottery','jackpot','free','gift',
                 'bonus','cashback','refund','lakh','crore']
_ACTION_KW   = ['click','call','reply','text','visit','download',
                 'verify','confirm','update','provide','share',
                 'enter','submit','activate','redeem']

def _score(t, kws):
    h = sum(1 for k in kws if k in t)
    return min(h / len(kws), 1.0)

def _time_pressure(t):
    return any(k in t for k in ['expire','deadline','limited time','hurry','asap'])

def _loss_threats(t):
    return any(k in t for k in ['lose','loss','miss out','forfeit','penalty'])

def _money_rewards(t):
    return any(k in t for k in ['₹','lakh','crore','cash','money','amount'])

def _upper_ratio(t):
    if not t: return 0.0
    upper = sum(1 for c in t if c.isupper())
    return upper / len(t)

def _digit_ratio(t):
    if not t: return 0.0
    return sum(1 for c in t if c.isdigit()) / len(t)

_SP = set('!@#$%^&*()_+-=[]{}|;:,.<>?')
def _special_ratio(t):
    if not t: return 0.0
    return sum(1 for c in t if c in _SP) / len(t)

def _caps_words(t):
    return sum(1 for w in t.split() if len(w) > 2 and w == w.upper())

def _has_url(t):
    return bool(re.search(r'https?://|www\.|\.com|\.in|\.org', t))

def _has_phone(t):
    return bool(re.search(r'\b\d{10,}\b', t))

def _is_phone(s):
    return bool(s.startswith('+') and re.match(r'^\+\d{10,}$', s))

def _is_service(s):
    return (not _is_phone(s)) and ('-' in s or len(s) <= 6)

def _impersonates_bank(t, s):
    bank_kw = ['bank','sbi','hdfc','icici','axis','rbi']
    return any(k in t for k in bank_kw) and _is_phone(s)

def _impersonates_gov(t, s):
    gov_kw = ['government','ministry','department','income tax','aadhaar']
    return any(k in t for k in gov_kw) and _is_phone(s)

def _requests_data(t):
    return any(k in t for k in ['otp','pin','password','cvv',
                                  'card number','account number'])

def _fraud_risk(t, s):
    score = 0.0
    if _money_rewards(t):        score += 0.3
    if _requests_data(t):        score += 0.4
    if _impersonates_bank(t, s): score += 0.4
    return min(score, 1.0)

def _spam_risk(t):
    score = 0.0
    if _has_url(t):                    score += 0.2
    if _score(t, _REWARD_KW) > 0.3:   score += 0.3
    return min(score, 1.0)

def _legit_score(t, s):
    score = 0.0
    if 'otp' in t or re.search(r'\b\d{4,6}\b', t): score += 0.3
    if len(t) < 160 and not _has_url(t):            score += 0.2
    if re.match(r'^[A-Z]{2}-', s):                  score += 0.5   # Indian DLT code
    if not _is_phone(s) and len(s) <= 6:            score += 0.3
    return min(score, 1.0)

FEATURE_NAMES = [
    'urgency_immediate','urgency_time_pressure',
    'fear_account_threats','fear_loss_threats',
    'reward_money','reward_prizes',
    'authority_financial','authority_government',
    'action_data_harvesting','action_immediate',
    'total_urgency','total_fear','total_reward','total_authority','total_action',
    'length_normalized','word_count_normalized',
    'uppercase_ratio','digit_ratio','special_char_ratio',
    'exclamation_count_normalized','caps_words_normalized',
    'has_url','has_phone',
    'sender_is_phone','sender_is_service','sender_length_normalized',
    'fraud_risk_score','spam_risk_score','legit_score',
]

def extract_features(body: str, sender: str) -> list:
    t = str(body).lower()
    s = str(sender).strip()

    urg_imm  = _score(t, _URGENCY_KW)
    urg_time = 1.0 if _time_pressure(t) else 0.0
    fear_acc = _score(t, _FEAR_KW)
    fear_los = 1.0 if _loss_threats(t) else 0.0
    rew_mon  = 1.0 if _money_rewards(t) else 0.0
    rew_pri  = _score(t, _REWARD_KW)
    auth_fin = 1.0 if _impersonates_bank(t, s) else 0.0
    auth_gov = 1.0 if _impersonates_gov(t, s) else 0.0
    act_dat  = 1.0 if _requests_data(t) else 0.0
    act_imm  = _score(t, _ACTION_KW)

    excl_cnt = min((t.count('!')) / 5.0, 1.0)
    caps_w   = min(_caps_words(body) / 10.0, 1.0)   # use original for case

    return [
        urg_imm, urg_time,
        fear_acc, fear_los,
        rew_mon, rew_pri,
        auth_fin, auth_gov,
        act_dat, act_imm,
        urg_imm + urg_time,       # total_urgency
        fear_acc + fear_los,      # total_fear
        rew_mon + rew_pri,        # total_reward
        auth_fin + auth_gov,      # total_authority
        act_dat + act_imm,        # total_action
        min(len(body) / 500.0, 1.0),                        # length_normalized
        min(len(body.split()) / 100.0, 1.0),                # word_count_normalized
        _upper_ratio(body),
        _digit_ratio(body),
        _special_ratio(body),
        excl_cnt,
        caps_w,
        1.0 if _has_url(t) else 0.0,
        1.0 if _has_phone(t) else 0.0,
        1.0 if _is_phone(s) else 0.0,
        1.0 if _is_service(s) else 0.0,
        min(len(s) / 20.0, 1.0),                            # sender_length_normalized
        _fraud_risk(t, s),
        _spam_risk(t),
        _legit_score(t, s),
    ]


# ─────────────────────────────────────────────────────────────
# AUTO-LABELLER  (returns 0 / 1 / 2  or  None = skip)
# ─────────────────────────────────────────────────────────────

# Indian DLT sender prefixes that are always legitimate services
_LEGIT_SERVICE_CODES = re.compile(
    r'^[A-Z]{2}-('
    r'AIRTEL|AIRTL|AIRINF|AIRMCA|AIRDOT|ARWINF|ARWGOV|ARWTEL|'
    r'CANBNK|KOTAKB|HDFCBN|SBIINB|SBIUPI|SBIBNK|SBICRM|SBICRD|IDBIBK|'
    r'IDFCFB|AXISBK|ICICIB|PNBHFL|BSNLSC|IOBCHN|'
    r'NBHOME|NVMAWM|NVMHOM|DLHVRY|INDANE|RRLACC|TRAIND|'
    r'NSESMS|CDSLTX|BNKBZR|EDMART|BPILCS|PFINCO|AAKASH|'
    r'PHONPE|PAYTMB|GPAYB|'
    r'CBSSBI|BIKEDK|AIROAM|CRDLIA|ANUMTI|BLAPPL|'
    r'IDFCBK|RBISAY|'
    r')',
    re.IGNORECASE
)

# Known promotional/spam sender codes
_SPAM_SERVICE_CODES = re.compile(
    r'^[A-Z]{2}-('
    r'MGLAMM|MYNTRA|AMZNSM|FLPKRT|BEYOUN|SNITCH|DOMINO|'
    r'SWIGGY|ZOMATO|BNANZA|KUKUFM|GALINS|MARUTI|TRUVAL|'
    r'BAJAJF|SHPRKT|SHIPRT|LITLBX|BAJFIN|WOOHOO|PASBAZ|'
    r'AIROAM|VIDAEV|OBENEV|EXCITE|PWCRSJ|'
    r')',
    re.IGNORECASE
)

# Patterns indicating fraud (phishing, impersonation)
_FRAUD_BODY_RE = re.compile(
    r'(account.*(?:suspended|blocked|closed|frozen)|'
    r'(?:suspended|blocked|frozen).*account|'
    r'(?:verify|update|confirm).*(?:otp|pin|password|cvv|card|account)|'
    r'(?:click|tap|visit).*(?:link|here|now).*(?:immediately|urgent|expire)|'
    r'(?:legal action|arrest|court|penalty).*(?:\d+ hours?|\d+ days?)|'
    r'kyc.*(?:expire|update|verify)|'
    r'(?:deactivat|suspend).*aadhaar|'
    r'income.?tax.*(?:notice|alert|penalty)|'
    r'(?:win|won|prize|reward).*(?:claim|call|click))',
    re.IGNORECASE
)

# Strong spam indicators
_SPAM_BODY_RE = re.compile(
    r'(\d+%\s*off|flat\s*\d+%?|upto\s*\d+%?|'
    r'(?:mega|super|grand|big)\s*sale|'
    r'buy\s*(?:\d+\s*get|now)|'
    r'(?:limited|exclusive)\s*(?:offer|deal|discount)|'
    r'shop\s*(?:now|today)|'
    r'free\s*(?:delivery|shipping|gift)|'
    r'(?:cashback|reward)\s*on\s*(?:every|all)|'
    r'congratulations.*(?:won|winner|prize|lucky)|'
    r'you\s*(?:have\s*)?won|'
    r'lucky\s*(?:winner|draw|customer))',
    re.IGNORECASE
)

# Strong OTP / transaction patterns → always legitimate
_LEGIT_BODY_RE = re.compile(
    r'(\botp\b.*\d{4,8}|'
    r'\d{4,8}.*\botp\b|'
    r'one.?time.?password.*\d{4,8}|'
    r'\d{4,8}.*one.?time.?password|'
    r'verification\s+code[:\s]+\d{4,8}|'
    r'(?:debit|credit|debited|credited)\s*(?:rs\.?|₹|inr)?\s*\d+|'
    r'(?:rs\.?|₹)\s*\d+\s*(?:debited|credited|transferred|paid)|'
    r'transaction\s+(?:id|ref|no)[:\s#]+\w+|'
    r'your\s+(?:bill|invoice|statement|balance|account\s+balance)\s+(?:is|for)|'
    r'pnr\s+\d+|'
    r'booking\s+(?:confirmed|id|no))',
    re.IGNORECASE
)

def auto_label(body: str, sender: str) -> int | None:
    """
    Returns:
        0 = LEGITIMATE
        1 = SPAM
        2 = FRAUD
        None = uncertain (will be soft-labelled later)
    """
    s = str(sender).strip()
    t = str(body)
    tl = t.lower()

    # ── Sender-based hard rules ─────────────────────────────
    if _is_phone(s):
        # Unknown phone number messages are suspicious if they ask for action
        if _FRAUD_BODY_RE.search(t):
            return 2   # FRAUD
        if _SPAM_BODY_RE.search(t):
            return 1   # SPAM
        # Regular contact messages
        if len(t) < 200 and not _has_url(tl):
            return 0   # probably personal/legit
        return None    # uncertain

    if re.match(r'^[A-Z]{2}-', s):
        # Indian DLT registered sender
        if _LEGIT_SERVICE_CODES.match(s):
            if not _FRAUD_BODY_RE.search(t):
                return 0   # LEGITIMATE service message

        if _SPAM_SERVICE_CODES.match(s):
            return 1   # SPAM promo

        # Generic DLT code — use body content
        if _FRAUD_BODY_RE.search(t):
            return 2
        if _LEGIT_BODY_RE.search(t):
            return 0
        if _SPAM_BODY_RE.search(t):
            return 1
        # Short DLT message with no red flags → legitimate
        if len(t) < 250 and not _has_url(tl):
            return 0
        return None   # uncertain

    # Short service code (e.g. IRCTC, BESCOM, SBI, HDFC)
    if len(s) <= 8 and s.isalnum():
        if _LEGIT_BODY_RE.search(t):
            return 0
        if _SPAM_BODY_RE.search(t):
            return 1
        if _FRAUD_BODY_RE.search(t):
            return 2
        return 0   # assume service message

    return None   # unknown sender format


def soft_label(body: str, sender: str) -> int:
    """Fallback labelling for uncertain messages using feature scores."""
    t = body.lower()
    feats = extract_features(body, sender)
    fraud_risk = feats[27]
    spam_risk  = feats[28]
    legit_sc   = feats[29]
    total_fear = feats[11]
    total_urg  = feats[10]

    # Score-based decision
    if fraud_risk > 0.5 or (total_fear > 0.5 and total_urg > 0.3):
        return 2
    if spam_risk > 0.3 or feats[5] > 0.3:   # reward_prizes
        return 1
    if legit_sc > 0.4:
        return 0
    # Default: low-confidence legitimate
    return 0


# ─────────────────────────────────────────────────────────────
# LOAD AND LABEL DATA
# ─────────────────────────────────────────────────────────────

def _add_synthetic_fraud_spam(records):
    """
    Add high-quality synthetic FRAUD and SPAM examples drawn from
    real-world Indian SMS phishing / spam patterns.
    Each message is labelled with high confidence.
    We add ~600 FRAUD + ~400 SPAM to give the model enough signal.
    """
    import random
    random.seed(42)

    # ── FRAUD templates (Indian phishing / account-takeover patterns) ────────
    fraud_templates = [
        # Account threat + data harvest
        ("Your {bank} account has been SUSPENDED due to suspicious activity. "
         "Update KYC immediately: {url}  Reply STOP to ignore at your risk.", "INCOMETAX"),
        ("URGENT: Debit card ending {last4} is BLOCKED. To unblock call {phone} "
         "or share OTP to our agent. {bank} Security Team.", "+91{phone}"),
        ("Dear Customer, your {bank} Net Banking has been DISABLED. "
         "Verify your account within 24 hours to avoid permanent closure: {url}", "SBIFRAUD"),
        ("Final Warning: Your {bank} account will be CLOSED in 2 hours due to "
         "incomplete KYC. Provide Aadhaar + OTP now to keep account active.", "+91{phone}"),
        ("Income Tax Department: You have unpaid tax of ₹{amount}. "
         "Pay immediately at {url} or face legal action. Ref: IT/{ref}", "+91{phone}"),
        ("Aadhaar Deactivation Notice: Your Aadhaar {aadhaar} will be deactivated "
         "in 48 hours. Update linked mobile: {url}  UIDAI Helpline", "+91{phone}"),
        ("Your EPFO pension linked to {phone} is on HOLD due to inactive KYC. "
         "Update immediately at {url} or lose benefits.", "EPFOFRAUD"),
        ("RBI Notice: Your bank account will be FROZEN in 6 hours due to "
         "money laundering alert. Contact: {phone} immediately.", "+91{phone}"),
        # Prize / lottery fraud
        ("CONGRATULATIONS! You have WON ₹{prize} in KBC Lucky Draw. "
         "Call KBC Helpline {phone} to claim. Lottery Ref: KBC{ref}.", "+91{phone}"),
        ("You are selected as the LUCKY WINNER of ₹{prize} from Reliance Jio "
         "Lucky Subscriber Scheme. Send your account details to claim.", "+91{phone}"),
        ("Amazon Customer Survey Winner! You have been selected to receive "
         "FREE iPhone 15. Click {url} to claim within 1 hour.", "+91{phone}"),
        ("Dear Jio User, you have won ₹{prize} cash prize! "
         "Share your bank account number and IFSC to receive funds.", "+91{phone}"),
        # Loan fraud
        ("Instant Loan Approved! ₹{amount} personal loan at 0% interest! "
         "No documents needed. WhatsApp {phone} to process.", "+91{phone}"),
        ("Pre-approved loan of ₹{amount} for you! "
         "Repay in easy EMIs. Click {url} to disburse in 10 minutes.", "+91{phone}"),
        # OTP phishing
        ("Your {bank} OTP is being misused. To STOP unauthorised transaction "
         "of ₹{amount} share the OTP sent to you with our agent {phone}.", "+91{phone}"),
        ("Security Alert {bank}: An OTP has been generated for ₹{amount} transaction. "
         "If not done by you, call IMMEDIATELY {phone} and share OTP.", "+91{phone}"),
        # Investment fraud
        ("Earn ₹{amount} daily guaranteed! Join our stock market group. "
         "WhatsApp {phone}. Limited slots only!", "+91{phone}"),
        ("SEBI Registered Advisor Tips: 100% accurate trading signals. "
         "Today's tip: BUY XYZ. Join free: {url}  Earn ₹{amount}/day", "+91{phone}"),
        # Fake delivery
        ("Your package could not be delivered. Update delivery address: {url} "
         "within 24 hrs or parcel will be returned. Tracking: {ref}", "+91{phone}"),
        ("DTDC Courier: Your parcel {ref} is held at customs. "
         "Pay ₹{amount} clearance fee at {url} to release.", "+91{phone}"),
    ]

    banks   = ['SBI', 'HDFC', 'ICICI', 'Axis Bank', 'Kotak Mahindra', 'PNB', 'Canara Bank']
    prizes  = ['50,000', '1,00,000', '5,00,000', '25,000', '2,50,000']
    amounts = ['15,000', '50,000', '1,00,000', '5,00,000', '25,000', '75,000']
    urls    = ['bit.ly/3xFraud', 'sbi-verify.net', 'kycupdate.in', 'income-tax-notice.com',
               'hdfc-kyc-update.co', 'uidai-verify.info']

    def fill(tmpl):
        return tmpl.format(
            bank   = random.choice(banks),
            last4  = random.randint(1000, 9999),
            phone  = f"9{random.randint(100000000, 999999999)}",
            url    = random.choice(urls),
            amount = random.choice(amounts),
            prize  = random.choice(prizes),
            ref    = f"{random.randint(100000, 999999)}",
            aadhaar= f"XXXX-XXXX-{random.randint(1000, 9999)}",
        )

    for _ in range(30):  # 30 variations per template = 600 FRAUD
        for tmpl, sender_tmpl in fraud_templates:
            body   = fill(tmpl)
            sender = fill(sender_tmpl)
            records.append({'body': body, 'sender': sender, 'label': 2, 'source': 'synth_fraud'})

    # ── SPAM templates ───────────────────────────────────────────────────────
    spam_templates = [
        ("MEGA SALE {discount}% OFF on all products! Shop now: {url} "
         "Use code {code} for extra {extra}% off. Hurry, ends tonight!", "VM-MGLAMM"),
        ("Exclusive offer for you! Buy any {product} & get {discount}% cashback. "
         "Valid till {date}. Shop at {url}", "AD-MYNTRA"),
        ("Flash Sale! Minimum {discount}% OFF + Free delivery. "
         "{product} starts at just ₹{price}. Order now: {url}", "AX-FLPKRT"),
        ("You have {points} reward points worth ₹{price}. "
         "Redeem before expiry on {date}. Shop at {url}", "VM-AMAZON"),
        ("Big Billion Days! Upto {discount}% off on Electronics, Fashion & more. "
         "Download app for early access: {url}", "AD-FLPKRT"),
        ("Earn ₹{price} daily from home! No investment. "
         "Work 2 hours & earn guaranteed. Join: {url}", "+91{phone}"),
        ("Today only: Flat ₹{price} OFF on first order. "
         "Use code {code}. Order food at {url}", "VM-SWIGGY"),
        ("Congratulations! You have {points} Paytm cashback credits. "
         "Redeem now: {url}  Expires in 24 hrs", "AD-PAYTMB"),
    ]

    products  = ['Smartphone', 'Laptop', 'Shoes', 'Kurta', 'Saree', 'Watch', 'Headphones']
    discounts = ['40', '50', '60', '70', '30', '80']
    prices    = ['99', '199', '299', '499', '999', '149']
    codes     = ['SAVE50', 'FLASH40', 'NEW30', 'EXTRA20', 'DEAL60']
    dates     = ['31 March', '15 March', '20 March', 'Sunday midnight', 'tonight']
    points    = ['500', '1000', '250', '750', '1500']

    def fill_spam(tmpl):
        return tmpl.format(
            discount = random.choice(discounts),
            url      = f"bit.ly/{random.randint(10000,99999)}",
            code     = random.choice(codes),
            extra    = random.randint(5, 20),
            product  = random.choice(products),
            date     = random.choice(dates),
            price    = random.choice(prices),
            points   = random.choice(points),
            phone    = f"9{random.randint(100000000, 999999999)}",
        )

    for _ in range(50):  # 50 variations × 8 templates = 400 SPAM
        for tmpl, sender in spam_templates:
            body = fill_spam(tmpl)
            if '{phone}' in sender:
                sender = sender.format(phone=f"9{random.randint(100000000, 999999999)}")
            records.append({'body': body, 'sender': sender, 'label': 1, 'source': 'synth_spam'})

    fraud_added = sum(1 for r in records if r['label'] == 2 and r['source'] == 'synth_fraud')
    spam_added  = sum(1 for r in records if r['label'] == 1 and r['source'] == 'synth_spam')
    print(f"  Synthetic FRAUD added: {fraud_added}")
    print(f"  Synthetic SPAM  added: {spam_added}")


def load_data():
    records = []

    # --- Real phone SMS ---
    phone_files = [
        "phone_sms_export_2025-07-13T14-41-31.344697.csv",
        "phone_sms_export_2025-07-13T14-59-37.079178.csv",
        "phone_sms_export_2025-07-14T09-30-54.278524.csv",
    ]
    phone_dfs = []
    for fname in phone_files:
        p = DATA_DIR / fname
        if p.exists():
            df = pd.read_csv(p)
            print(f"  Loaded {len(df):>6} rows from {fname}")
            phone_dfs.append(df)

    combined = pd.concat(phone_dfs, ignore_index=True)
    combined.drop_duplicates(subset=['body'], inplace=True)
    combined.dropna(subset=['body'], inplace=True)
    print(f"  {len(combined):>6} unique real SMS messages")

    labeled = uncertain = 0
    for _, row in combined.iterrows():
        body   = str(row['body'])
        sender = str(row.get('address', ''))
        lbl    = auto_label(body, sender)
        if lbl is None:
            lbl = soft_label(body, sender)
            uncertain += 1
        else:
            labeled += 1
        records.append({'body': body, 'sender': sender, 'label': lbl, 'source': 'phone'})

    print(f"  Hard-labeled: {labeled}  |  Soft-labeled: {uncertain}")

    # --- sms_spam.csv (English benchmark) ---
    spam_csv = DATA_DIR / "sms_spam.csv"
    if spam_csv.exists():
        df_spam = pd.read_csv(spam_csv)
        # columns: label (ham/spam), text
        for _, row in df_spam.iterrows():
            lbl_str = str(row.get('label', row.get('v1', 'ham'))).lower().strip()
            body    = str(row.get('text', row.get('v2', '')))
            lbl = 1 if lbl_str == 'spam' else 0   # no 'fraud' in this dataset
            records.append({'body': body, 'sender': 'SMSSPAM', 'label': lbl, 'source': 'spam_csv'})
        print(f"  Loaded {len(df_spam):>6} rows from sms_spam.csv")

    df = pd.DataFrame(records)
    print(f"\n  TOTAL: {len(df)} messages")
    print(f"  LEGITIMATE : {(df.label==0).sum()}")
    print(f"  SPAM       : {(df.label==1).sum()}")
    print(f"  FRAUD      : {(df.label==2).sum()}")
    return df


# ─────────────────────────────────────────────────────────────
# BUILD FEATURE MATRIX
# ─────────────────────────────────────────────────────────────

def build_features(df):
    print("\nExtracting features...")
    X = []
    for i, row in df.iterrows():
        if i % 2000 == 0:
            print(f"  {i}/{len(df)}")
        X.append(extract_features(row['body'], row['sender']))
    return np.array(X, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────

def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.40),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.35),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.15),
        keras.layers.Dense(3, activation='softmax'),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def train(X_train, y_train, X_val, y_val):
    model = build_model(X_train.shape[1])
    model.summary()

    # Compute class weights to handle imbalance
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"\nClass weights: {class_weight}")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                           patience=7, min_lr=1e-5, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=64,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    return model, history


# ─────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────

def export_tflite(model, scaler):
    print("\nExporting TFLite model...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    TFLITE_PATH.write_bytes(tflite_bytes)
    print(f"  Saved {TFLITE_PATH}  ({len(tflite_bytes)/1024:.1f} KB)")

    config = {
        'feature_names':  FEATURE_NAMES,
        'feature_count':  len(FEATURE_NAMES),
        'scaler_mean':    scaler.mean_.tolist(),
        'scaler_scale':   scaler.scale_.tolist(),
        'classes':        ['LEGITIMATE', 'SPAM', 'FRAUD'],
        'model_version':  '4.0.0',
        'export_date':    pd.Timestamp.now().isoformat(),
        'training_notes': 'Trained on 17k+ real Indian phone SMS + sms_spam.csv benchmark',
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    print(f"  Saved {CONFIG_PATH}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  SMS FRAUD DETECTOR — Training from real data")
    print("=" * 70)

    # 1. Load & label
    print("\n[1/5] Loading data...")
    df = load_data()

    # 2. Extract features
    print("\n[2/5] Feature extraction...")
    X = build_features(df)
    y = df['label'].values.astype(np.int32)

    # 3. Scale
    print("\n[3/5] Scaling...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Split
    print("\n[4/5] Training...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=42, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    print(f"  Train: {len(X_tr)}  Val: {len(X_val)}  Test: {len(X_test)}")

    model, _ = train(X_tr, y_tr, X_val, y_val)

    # 5. Evaluate
    print("\n[5/5] Evaluation on held-out test set:")
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(classification_report(y_test, y_pred,
                                 target_names=['LEGITIMATE', 'SPAM', 'FRAUD']))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 6. Export
    export_tflite(model, scaler)

    print("\n✅  Done!  New model exported to:")
    print(f"   {TFLITE_PATH}")
    print(f"   {CONFIG_PATH}")
    print("\nNow rebuild the Flutter app with:  flutter run -d <device>")


if __name__ == '__main__':
    main()
