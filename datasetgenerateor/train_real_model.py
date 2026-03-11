"""
train_real_model.py
===================
Trains a 30-feature behavioral fraud detector on real Indian SMS data.

Pipeline
--------
1. Load 3 phone export CSVs  +  sms_spam.csv (English spam/ham reference)
2. Deduplicate on body text
3. Auto-label with sender-first rules:
      LEGIT  → known DLT service senders (AX-/AD- + known brand)
      SPAM   → known promo senders (Swiggy, Myntra, Rummy apps, etc.)
      FRAUD  → phone-number sender + scam content
              + DLT sender + job-scam / phishing content
4. Extract the EXACT 30 features that Dart computes in _extractFeatures()
5. Train Dense(128→64→32→3) with class_weight='balanced'
6. Export:
      ../../sms_fraud_detectore_app/assets/advanced_fraud_detector.tflite
      ../../sms_fraud_detectore_app/assets/behavioral_model_config.json
"""

import re, math, json, os, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
warnings.filterwarnings('ignore')

# ─── paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), 'sms data set')
ASSETS_DIR = os.path.join(os.path.dirname(__file__),
                          '..', 'sms_fraud_detectore_app', 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)

TFLITE_OUT = os.path.join(ASSETS_DIR, 'advanced_fraud_detector.tflite')
CONFIG_OUT  = os.path.join(ASSETS_DIR, 'behavioral_model_config.json')

# ─── keyword banks (IDENTICAL to Dart) ────────────────────────────────────────
URGENCY_KW = [
    'urgent','immediately','asap','expire','deadline',
    'limited time','act now','hurry','last chance',
    'expire today','expires soon','time running out'
]
FEAR_KW = [
    'suspended','blocked','terminated','legal action',
    'penalty','fine','arrest','court','lawsuit',
    'closed','cancelled','frozen','unauthorized'
]
REWARD_KW = [
    'congratulations','winner','won','prize','cash',
    'reward','lottery','jackpot','free','gift',
    'bonus','cashback','refund','lakh','crore'
]
ACTION_KW = [
    'click','call','reply','text','visit','download',
    'verify','confirm','update','provide','share',
    'enter','submit','activate','redeem'
]

# ─── 30-feature extractor (mirrors Dart exactly) ──────────────────────────────
def _score(t, kws):
    return min(sum(1 for k in kws if k in t) / len(kws), 1.0)

def _time_pressure(t):
    return any(k in t for k in ['expire','deadline','limited time','hurry','asap'])

def _loss_threats(t):
    return any(k in t for k in ['lose','loss','miss out','forfeit','penalty'])

def _money_rewards(t):
    return any(k in t for k in ['₹','lakh','crore','cash','money','amount'])

def _has_url(t):
    return bool(re.search(r'https?://|www\.|\.com|\.in|\.org', t))

def _has_phone(t):
    return bool(re.search(r'\b\d{10,}\b', t))

def _is_phone_sender(s):
    """sender.startswith('+') and only digits after — same as Dart"""
    return bool(re.match(r'^\+\d{10,}$', s))

def _is_service_sender(s):
    """len<=6 or contains '-' and not a phone number"""
    return not _is_phone_sender(s) and ('-' in s or len(s) <= 6)

def _upper_ratio(t):
    if not t: return 0.0
    return sum(1 for c in t if c.isupper()) / len(t)

def _digit_ratio(t):
    if not t: return 0.0
    return sum(1 for c in t if c.isdigit()) / len(t)

def _special_ratio(t):
    if not t: return 0.0
    sp = set('!@#$%^&*()_+-=[]{}|;:,.<>?')
    return sum(1 for c in t if c in sp) / len(t)

def _caps_words(t):
    return sum(1 for w in t.split() if len(w) > 2 and w == w.upper())

def _impersonates_bank(t, s):
    banks = ['bank','sbi','hdfc','icici','axis','rbi']
    return any(b in t for b in banks) and _is_phone_sender(s)

def _impersonates_gov(t, s):
    govt = ['government','ministry','department','income tax','aadhaar']
    return any(g in t for g in govt) and _is_phone_sender(s)

def _requests_data(t):
    return any(k in t for k in ['otp','pin','password','cvv','card number','account number'])

def _fraud_risk(t, s):
    return min(
        (_money_rewards(t)        * 0.3) +
        (_requests_data(t)        * 0.4) +
        (_impersonates_bank(t, s) * 0.4),
        1.0)

def _spam_risk(t):
    return min(
        (_has_url(t) * 0.2) +
        (0.3 if _score(t, REWARD_KW) > 0.3 else 0),
        1.0)

def _legit_score(t, s):
    score = 0.0
    if 'otp' in t or re.search(r'\b\d{4,6}\b', t): score += 0.3
    if len(t) < 160 and not _has_url(t):            score += 0.2
    if re.match(r'^[A-Z]{2}-', s):                  score += 0.5
    if not _is_phone_sender(s) and len(s) <= 6:     score += 0.3
    return min(score, 1.0)

def extract_features(body: str, sender: str):
    """Returns list of 30 floats — IDENTICAL order to Dart _extractFeatures()"""
    t = body.lower()
    s = sender.upper()   # Dart lowercases for matching but the regex ^[A-Z]{2}- checks the original

    # For the legit_score regex, Dart uses the original sender casing
    # We pass original sender to _legit_score and _is_phone_sender
    s_orig = sender

    urg_imm  = _score(t, URGENCY_KW)
    urg_time = 1.0 if _time_pressure(t) else 0.0
    fear_acc = _score(t, FEAR_KW)
    fear_loss= 1.0 if _loss_threats(t) else 0.0
    rew_money= 1.0 if _money_rewards(t) else 0.0
    rew_prize= _score(t, REWARD_KW)
    auth_fin = 1.0 if _impersonates_bank(t, s_orig) else 0.0
    auth_gov = 1.0 if _impersonates_gov(t, s_orig) else 0.0
    act_data = 1.0 if _requests_data(t) else 0.0
    act_imm  = _score(t, ACTION_KW)

    total_urgency   = urg_imm + urg_time
    total_fear      = fear_acc + fear_loss
    total_reward    = rew_money + rew_prize
    total_authority = auth_fin + auth_gov
    total_action    = act_data + act_imm

    length_norm     = min(len(body) / 500.0, 1.0)
    word_count_norm = min(len(body.split()) / 100.0, 1.0)
    upper_ratio     = _upper_ratio(body)
    digit_ratio     = _digit_ratio(body)
    special_ratio   = _special_ratio(body)
    excl_count      = min(body.count('!') / 5.0, 1.0)
    caps_words_norm = min(_caps_words(body) / 10.0, 1.0)

    has_url_f   = 1.0 if _has_url(body) else 0.0
    has_phone_f = 1.0 if _has_phone(body) else 0.0
    sender_is_phone   = 1.0 if _is_phone_sender(s_orig) else 0.0
    sender_is_service = 1.0 if _is_service_sender(s_orig) else 0.0
    sender_len_norm   = min(len(s_orig) / 20.0, 1.0)

    fraud_score = _fraud_risk(t, s_orig)
    spam_score  = _spam_risk(t)
    legit_score = _legit_score(t, s_orig)   # uses original sender for regex

    return [
        urg_imm, urg_time,
        fear_acc, fear_loss,
        rew_money, rew_prize,
        auth_fin, auth_gov,
        act_data, act_imm,
        total_urgency, total_fear, total_reward, total_authority, total_action,
        length_norm, word_count_norm, upper_ratio, digit_ratio, special_ratio,
        excl_count, caps_words_norm,
        has_url_f, has_phone_f,
        sender_is_phone, sender_is_service, sender_len_norm,
        fraud_score, spam_score, legit_score,
    ]

# ─── sender classification rules ─────────────────────────────────────────────

# Known legitimate Indian DLT service sender suffixes
LEGIT_SENDER_BRANDS = {
    # Telecoms
    'AIRTEL','ARWINF','ARWNFO','ARINFO','650022','650025','650002','602607','602604',
    'JIOVOC','JIOCIN','VODAFO','VODAFN','BSNL','MTNNL','IDEA',
    # Banks & Finance
    'SBIINB','SBIPAY','HDFCBK','ICICIB','AXISBK','CANBNK','KOTAKA','KOTAKB',
    'PNBSMS','BOIIND','UNIBAN','SYNDBK','IOBSMS','CBSSMS','YESBNK','IDBIBN',
    'PAYTMB','IPAYTM','PHONPE','GPAYSB','BHIMUPI','RAZPAY','MOBIKW',
    'BAJAJF','BFDLTS','BFDLPS','INDFIN','CAPFST',
    'LICIOF','EXIDLI','SBIINR','ICICIL','HDFCLI',
    # Govt & Regulatory
    'MYGOVT','GOVRAS','KAEOFF','MOBKAR','DIPRUK','RBISAY','TRAI','UIDAI',
    'GOIEMS','NSDLEG','INCOTX','EPFINR','ESICIN',
    # Ecommerce & services
    'FLPKRT','AMAZON','AMZNIN','MYNTRA','ZOMATO','SWIGGY','887700',
    'IRCTCI','IRCTC','MAKEMY','CLEARR','GOIBOB','IXIGO',
    'DUNZOO','PORTER','BPCLTD','BPCLIN','HPCLIN','IOCLTD',
    # Telehealth / edu
    'PRACTO','APOLLO','FORTIS','NARAYN',
    'DSCENG','ISBRPG','DSUEDU','LOHITS','JKSHAH','PREUNI','MSRSBG',
    # Crypto/fintech (legit)
    'CRED', 'NAVI',
}

# Sender codes known to be SPAM (promotional, gambling, rummy)
SPAM_SENDER_BRANDS = {
    'JERUMY','JLRUMY','JLRMMY','JUNRMY','JRUMMY','JGRUMY','JLERMY',
    'GRRUMY','GARUMY','CLRWEB',
    'PLYWIN','MGLAMM','CRCOIN','SMKKBM',
    'MOPVCW','MPSGWN','FRNDEL','JAINMM','AGIEDU','MOBELC','GOPAUM',
    'ENTERP','ISBRPG','BAPUSE','GETFRE','KOTAKA',  # some kotaka entries are spam
}

# DLT sender prefixes that are almost always legitimate
LEGIT_PREFIXES = {'AX','AD','VM','VK','TX','JD','JM','JK','BG','BW','BP',
                  'BH','BT','BZ','CP','QP','TM','VD','VK','VN','TP'}


def label_message(address: str, body: str) -> str:
    """
    Returns 'legit', 'spam', or 'fraud'.
    Strategy: sender-first, then content rules for ambiguous cases.
    """
    s    = address.strip()
    t    = body.lower()
    s_up = s.upper()

    # ── Rule 1: phone-number sender → likely fraud if scam content ──────────
    if _is_phone_sender(s):
        # Known scam patterns from personal numbers
        scam_content = [
            r'(h0me|home).{0,20}(job|j0b)',
            r'(earn|income|salary).{0,30}(day|daily|month)',
            r'(rs\.?\s*\d{3,}|₹\s*\d{3,}).{0,30}(day|daily|hour)',
            r'wa\.me/\d+',
            r'(won|winner|selected|lucky).{0,60}(rs\.?|₹|\d{4,}|lakh|crore)',
            r'(congratulations|congrats).{0,60}(rummy|bonus|cash|prize)',
            r'(register|download).{0,30}(rummy|game|play)',
            r'http[s]?://(?!www\.(amazon|flipkart|irctc|paytm|phonepe|google))',
        ]
        for pat in scam_content:
            if re.search(pat, t):
                return 'fraud'
        # If phone sender with URL → suspicious spam
        if _has_url(t):
            return 'spam'
        return 'legit'  # personal SMS between people

    # ── Rule 2: extract brand code from DLT sender (e.g. AX-AIRTEL → AIRTEL) ─
    brand = ''
    m = re.match(r'^[A-Z]{2}-(.+)$', s_up)
    if m:
        brand = m.group(1).replace('-', '').replace('S', '').strip()
        # Strip trailing single char qualifiers like -S, -P
        brand = re.sub(r'-[A-Z]$', '', m.group(1))
    else:
        brand = s_up

    # ── Rule 3: known LEGIT brand ────────────────────────────────────────────
    for legit in LEGIT_SENDER_BRANDS:
        if legit in s_up:
            # Even legit senders can have fraud-injected content
            # Only override if body is extremely suspicious
            fraud_override = [
                r'(verify|update).{0,30}(aadhaar|kyc|pan).{0,30}(expire|suspend)',
                r'(your account|card).{0,20}(suspended|blocked).{0,40}(click|link|call)',
            ]
            for pat in fraud_override:
                if re.search(pat, t):
                    return 'fraud'
            return 'legit'

    # ── Rule 4: known SPAM brand ─────────────────────────────────────────────
    for sp in SPAM_SENDER_BRANDS:
        if sp in s_up:
            return 'spam'

    # ── Rule 5: DLT prefix heuristics ────────────────────────────────────────
    prefix = s_up[:2] if len(s_up) >= 2 else ''

    # Job scam DLT senders (h0me J0B wa.me pattern)
    if re.search(r'(h0me|home).{0,20}(j0b|job)', t) and re.search(r'wa\.me/', t):
        return 'fraud'

    # Rummy / gambling promotional
    if re.search(r'(rummy|poker|fantasy|casino|bet|satta)', t) and _has_url(t):
        return 'spam'

    # Generic promo pattern
    promo_patterns = [
        r'(offer|deal|discount|sale|off).{0,30}(shop|buy|order)',
        r'(download|install).{0,20}(app|apk)',
        r'cashback.{0,20}(order|pay|recharge)',
    ]
    for pat in promo_patterns:
        if re.search(pat, t) and _has_url(t):
            return 'spam'

    # ── Rule 6: content-only fraud patterns ──────────────────────────────────
    fraud_patterns = [
        (r'(suspended|blocked|deactivated).{0,60}(account|card|upi)', 'account_threat'),
        (r'(kyc|aadhaar|aadhar|pan).{0,40}(expire|update|verify)', 'kyc_fraud'),
        (r'(legal action|court|arrest|fir|police).{0,40}(file|case|action)', 'legal_threat'),
        (r'(unauthorized|suspicious).{0,40}(transaction|login|access)', 'fraud_alert'),
        (r'(income.?tax|irdai|sebi).{0,60}(verify|update|notice)', 'impersonation'),
    ]
    for pat, _ in fraud_patterns:
        if re.search(pat, t):
            return 'fraud'

    # ── Rule 7: DLT sender (has dash) — default LEGIT if we reach here ───────
    if '-' in s or len(s) <= 6:
        return 'legit'

    return 'legit'


# ─── load & label data ────────────────────────────────────────────────────────

print("📂 Loading phone SMS exports...")
phone_dfs = []
for fname in os.listdir(DATA_DIR):
    if fname.startswith('phone_sms_export') and fname.endswith('.csv'):
        df = pd.read_csv(os.path.join(DATA_DIR, fname), dtype=str).fillna('')
        # columns: id, address, body, date
        df = df[['address','body']].rename(columns={'address':'sender','body':'body'})
        phone_dfs.append(df)

phone_df = pd.concat(phone_dfs, ignore_index=True)
phone_df = phone_df.drop_duplicates(subset='body')
print(f"   Phone SMS: {len(phone_df):,} unique messages")

print("📂 Loading sms_spam.csv (English ham/spam reference)...")
spam_df = pd.read_csv(os.path.join(DATA_DIR, 'sms_spam.csv'), dtype=str).fillna('')
# columns: label, text
spam_df = spam_df[['label','text']].rename(columns={'text':'body'})
spam_df['sender'] = 'UNKNOWN'
# Map existing labels: ham→legit, spam→spam
spam_df['label'] = spam_df['label'].map({'ham':'legit','spam':'spam'})
spam_df = spam_df[spam_df['label'].notna()]
print(f"   Spam CSV: {len(spam_df):,} messages  ({spam_df['label'].value_counts().to_dict()})")

# Auto-label phone SMS
print("\n🏷️  Auto-labeling phone SMS...")
phone_df['label'] = phone_df.apply(
    lambda r: label_message(r['sender'], r['body']), axis=1)

print(f"   Label distribution:\n{phone_df['label'].value_counts()}")

# ── combine ────────────────────────────────────────────────────────────────────
combined = pd.concat([
    phone_df[['sender','body','label']],
    spam_df[['sender','body','label']],
], ignore_index=True).drop_duplicates(subset='body')

print(f"\n📊 Combined dataset: {len(combined):,} messages")
print(combined['label'].value_counts())

# ── extract features ───────────────────────────────────────────────────────────
print("\n⚙️  Extracting 30 features...")
X_list = []
y_list = []
label_map = {'legit': 0, 'spam': 1, 'fraud': 2}

for _, row in combined.iterrows():
    if row['label'] not in label_map:
        continue
    feats = extract_features(row['body'], row['sender'])
    X_list.append(feats)
    y_list.append(label_map[row['label']])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)
print(f"   X shape: {X.shape}  |  classes: {np.bincount(y)}")

# ── scale ─────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── train / val split ─────────────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n🔀 Train: {len(X_train):,}  Val: {len(X_val):,}")

# ── class weights ─────────────────────────────────────────────────────────────
from sklearn.utils.class_weight import compute_class_weight
cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: cw[i] for i in range(len(cw))}
print(f"   Class weights: {class_weights}")

# ── model ─────────────────────────────────────────────────────────────────────
print("\n🧠 Building model...")
# NOTE: BatchNormalization is intentionally removed.
# When TF 2.16+ MLIR converter folds BN into Dense it emits FULLY_CONNECTED
# op version 12, which tflite_flutter 0.11.0 (TFLite 2.14 runtime) cannot run.
# L2 regularisation + lower LR compensates for the missing BN.
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(30,)),
    tf.keras.layers.Dense(128, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax'),
], name='sms_fraud_detector')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.summary()

# ── train ─────────────────────────────────────────────────────────────────────
print("\n🏋️  Training...")
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, verbose=1),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)

# ── evaluate ──────────────────────────────────────────────────────────────────
print("\n📈 Evaluation on validation set:")
y_pred = np.argmax(model.predict(X_val), axis=1)
print(classification_report(y_val, y_pred,
      target_names=['LEGIT','SPAM','FRAUD']))
print("Confusion matrix:")
print(confusion_matrix(y_val, y_pred))

# ── export TFLite ─────────────────────────────────────────────────────────────
print("\n📦 Exporting TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Do NOT use Optimize.DEFAULT — it triggers FULLY_CONNECTED op v12 (TF 2.16+)
# which is not supported by tflite_flutter 0.11.0 (bundles TFLite 2.14).
# Float32 export produces FC op v9 or lower, compatible with all TFLite >= 2.0.
tflite_model = converter.convert()

with open(TFLITE_OUT, 'wb') as f:
    f.write(tflite_model)
print(f"   ✅ Saved: {TFLITE_OUT}  ({len(tflite_model)/1024:.1f} KB)")

# ── export scaler config ───────────────────────────────────────────────────────
feature_names = [
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
    'fraud_score','spam_score','legit_score',
]

config = {
    'feature_names':  feature_names,
    'scaler_mean':    scaler.mean_.tolist(),
    'scaler_scale':   scaler.scale_.tolist(),
    'label_map':      {'LEGITIMATE': 0, 'SPAM': 1, 'FRAUD': 2},
    'num_features':   30,
    'model_version':  '3.0-real-data',
    'trained_on':     f"{len(X):,} real Indian SMS",
}

with open(CONFIG_OUT, 'w') as f:
    json.dump(config, f, indent=2)
print(f"   ✅ Saved: {CONFIG_OUT}")

# ── quick sanity check ────────────────────────────────────────────────────────
print("\n🔬 Sanity check on known examples:")
tests = [
    ("AD-SBIINB", "Your OTP for SBI login is 847291. Valid for 5 minutes. Do NOT share.",  "LEGIT"),
    ("AX-HDFCBK", "Dear Customer, Rs.5000 debited from your a/c ending 4521 on 10-Mar.", "LEGIT"),
    ("VM-887700", "Order your fave food on Swiggy! Get up to 60% off http://m.swig.gy/n/abc", "SPAM"),
    ("VK-JERUMY", "Free Rs.8850 Welcome Bonus on Junglee Rummy. Win now: http://gmg.im/f0HX", "SPAM"),
    ("+919876543210", "Dear, You are selected for h0me based J0B. Earn 20000/day wa.me/91987", "FRAUD"),
    ("+918001234567", "Your account is SUSPENDED! Update KYC now or your card will be blocked click http://bit.ly/scam", "FRAUD"),
    ("JD-SMKKBM",  "Billno Your Applicati0n approved. Get 7,000/Day. https://wa.me/916397437664", "FRAUD"),
]

interp = tf.lite.Interpreter(model_path=TFLITE_OUT)
interp.allocate_tensors()
inp_idx = interp.get_input_details()[0]['index']
out_idx = interp.get_output_details()[0]['index']
names = ['LEGIT','SPAM','FRAUD']

all_pass = True
for sender, body, expected in tests:
    feats  = extract_features(body, sender)
    normed = ((np.array(feats) - scaler.mean_) / scaler.scale_).astype(np.float32)
    interp.set_tensor(inp_idx, [normed])
    interp.invoke()
    probs  = interp.get_tensor(out_idx)[0]
    pred   = names[np.argmax(probs)]
    ok     = "✅" if pred == expected else "❌"
    if pred != expected: all_pass = False
    print(f"  {ok}  [{pred:5s}]  L={probs[0]:.2f} S={probs[1]:.2f} F={probs[2]:.2f}  {sender[:12]:12s}  {body[:55]}")

print()
if all_pass:
    print("🎉 All sanity checks passed! Model is ready.")
else:
    print("⚠️  Some sanity checks failed — review labeling or add more examples.")

print(f"\n✅ Done. Model at:\n   {TFLITE_OUT}\n   {CONFIG_OUT}")
