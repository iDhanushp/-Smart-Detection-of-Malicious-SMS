# Project Documentation — Smart Detection of Malicious SMS

**Version**: 4.2.0  
**Last updated**: March 2026  
**Status**: Production-ready debug build installed on SM E135F (Android 14)

---

## 1. System Overview

The app is an on-device Android SMS fraud detector built with Flutter. It reads up to 500 recent
SMS messages, classifies each as LEGIT / SPAM / FRAUD using a TFLite neural network, then applies
a deterministic rule engine to produce a human-readable reason tag. All processing is local — no
network calls, no cloud API.

### Key Design Decisions

| Decision | Reason |
|----------|--------|
| No BatchNormalization in model | TF 2.17 MLIR folds BN into Dense → FC op v12; tflite_flutter 0.11.0 only supports FC up to v11 |
| float32 export (no Optimize.DEFAULT) | INT8 quantization also triggers FC v12; float32 keeps FC v9 |
| Rule engine always runs | TFLite may fail to init; rules provide baseline classification with no TFLite dependency |
| Contacts permission optional | Android 14 users sometimes deny contacts; blocking sync on this was dropping all SMS reads |
| 500 message cap | Prevents ANR on large inboxes; yields between chunks |

---

## 2. Model History

### Model v3.0 (superseded — commit `37ee98c`)

Trained on 22,615 messages (17,447 real Indian SMS + 5,572 sms_spam.csv). Fraud training examples:
43 (auto-labeled). Class weight FRAUD: 64×. Val accuracy: 93.9%. FRAUD precision: low (0.30) due
to severe class imbalance — only 23 fraud samples in val set.

---

## 2a. Model v4.0 (baseline — commits `cb05ea4`, `c1114a0`)

### Training Data

| Source | Messages |
|--------|----------|
| Phone CSV (real Indian SMS from device) | 17,447 (after within-source dedup) |
| sms_spam.csv (Kaggle SMS spam dataset) | 5,572 |
| **fraud_master.csv** (synthetic + manually curated fraud) | **3,899** |
| Cross-dataset duplicates removed | −447 |
| **Total** | **26,471** |

Class distribution after injection: LEGIT 81.0%, FRAUD 14.7%, SPAM 4.3%.
Class weights (auto-computed, `balanced`): `{LEGIT: 0.412, SPAM: 7.757, FRAUD: 2.263}`.

> **Key improvement**: FRAUD class weight dropped from ~177× to **2.26×** — class balance
> is nearly restored without artificial upweighting, eliminating the main source of
> false positives from the previous model.

### Architecture

Unchanged from v3.0:

```
InputLayer(30)
Dense(128, relu, L2=1e-4)
Dropout(0.3)
Dense(64, relu, L2=1e-4)
Dropout(0.2)
Dense(32, relu)
Dense(3, softmax)
```

No BatchNormalization — see Key Design Decisions above.

### Training Results

Best epoch: 25, val_accuracy: **0.9488**

```
              precision    recall  f1-score   support

       LEGIT       0.99      0.95      0.97      4288
        SPAM       0.46      0.85      0.59       227
       FRAUD       0.98      1.00      0.99       780

    accuracy                           0.95      5295
   macro avg       0.81      0.93      0.85      5295
weighted avg       0.97      0.95      0.96      5295
```

Confusion matrix:
```
[[4053  227    8]
 [  27  193    7]
 [   0    2  778]]
```

### Exported Model

| Property | Value |
|----------|-------|
| File | assets/advanced_fraud_detector.tflite |
| Size | 58.8 KB |
| Format | float32 (no quantization) |
| FC op version | v9 |
| Input shape | [1, 30] float32 |
| Output shape | [1, 3] float32 |

---

## 2c. Model v4.1 (superseded — commit `d06ecb1`)

v4.1 keeps the same architecture and dataset composition as v4.0, but updates the behavioral
scoring logic to reduce borderline LEGIT/SPAM errors seen in the 117-row human-reviewed set.

### What changed

1. **LEGIT score strengthened for trusted service-link SMS**
   - Added trusted URL bonus: `+0.5`
   - Added missed-call pattern bonus (`missed.{0,5}call`): `+0.3`
2. **SPAM score strengthened for gambling promos**
   - Added/raised gambling pattern contribution (`rummy|poker|casino|bet|satta|fantasy.*league`): `+0.4`
3. **Trusted URL allowlist expanded**
   - `TRUSTED_URL_PATTERNS` expanded from 14 → 52 entries to align Python training and Dart inference behavior.

### v4.1 validation summary

| Metric | v4.0 | v4.1 |
|--------|------|------|
| Auto-label validation accuracy | 0.949 | 0.945 |
| Human-label accuracy (117 reviewed SMS) | 0.667 | **0.744** |
| Auto vs human gap | 0.282 | **0.202** |

Interpretation: v4.1 trades a small drop in auto-label validation for a meaningful gain on
human-reviewed ground truth, which is the target objective for real-world inbox performance.

---

## 2d. Model v4.2 (current)

v4.2 keeps the same architecture and scoring logic as v4.1 but injects a new
`spam_master.csv` of 3,000 synthetic SPAM rows (25 Indian promotional categories) to close the
SPAM recall gap that persisted at 85% through v4.0–v4.1.

### What changed

1. **New file `generate_spam_samples.py`** — mirror of `generate_fraud_samples.py` for SPAM;
   produces 3,000 unique rows across 25 categories.
2. **`spam_master.csv` injected in `train_real_model.py`** — added after the `fraud_master.csv`
   injection block; label forced to `'spam'`, bypassing the auto-labeler.
3. **Combined dataset grows**: 26,471 → **29,471 messages** (+3,000 SPAM rows)
   — SPAM count in training: 484 (real) + 3,000 (synthetic via `spam_master.csv`) = **4,137**

### v4.2 validation summary

| Metric | v4.0 | v4.1 | v4.2 |
|--------|------|------|------|
| Training messages | 26,471 | 26,471 | **29,471** |
| Auto-label validation accuracy | 0.949 | 0.945 | **0.976** |
| SPAM recall | 0.85 | ~0.85 | **0.913** |
| FRAUD recall | 1.00 | 1.00 | **0.990** |
| Human-label accuracy (117 reviewed SMS) | 0.667 | 0.744 | **0.761** |
| Auto-vs-human gap | 0.282 | 0.202 | **0.215** |

Interpretation: v4.2 achieves the primary goal — SPAM recall above 90% (91.3%) — with a
small additional gain in human-label accuracy. The slight widening of the auto-vs-human gap
(0.202 → 0.215) is expected: the 3,000 injected synthetic rows are designed to look like real
promotional SMS, which raises auto-label accuracy but does not directly target borderline cases
in the human-reviewed set.

### v4.2 confusion matrix (auto-label validation)

```
[[4228   55    5]    ← LEGIT  : 4228 correct, 55 → SPAM, 5 → FRAUD
 [  58  755   14]   ← SPAM   : 755 correct (91.3% recall), 58 → LEGIT
 [   0    8  772]]  ← FRAUD  : 772 correct (99.0% recall)
```

### Sanity checks (v4.2)

| Message | Expected | v4.1 | v4.2 | 
|---------|----------|------|------|
| Bank OTP | LEGIT | ✅ | ✅ |
| Bank debit alert | LEGIT | ✅ | ✅ |
| Swiggy promo URL | SPAM | ❌ | ✅ |
| Rummy bonus offer | SPAM | ✅ | ✅ |
| WFH job scam from +91 | FRAUD | ❌ | ❌ (known) |
| Fake DLT loan approval | FRAUD | ❌ | ❌ (known) |

---

## 2b. Synthetic Fraud Data Generation Pipeline

To address the chronic shortage of labelled fraud examples (43 auto-detected rows in v3.0), a
two-layer data augmentation strategy was implemented.

### Layer 1 — Manual curation (`fraud_training_samples.csv`, 49 rows)

The 43 auto-detected fraud messages from real device SMS were supplemented with 6 hand-crafted
examples derived from real screenshots of Indian fraud SMS:

| Rows | Category | Example sender |
|------|----------|----------------|
| 44–46 | Fake delivery/courier scam (lihi.cc links) | `+919405412628`, `+918293633513`, `+919343389377` |
| 47–49 | Fake SBI bank credit from +91 phone numbers | `+917894561230`, `+918765432109`, `+917012345678` |

### Layer 2 — Synthetic generation (`generate_fraud_samples.py`, 3,850 rows)

A dedicated generator script was written with 25 fraud categories covering the full spectrum of
Indian SMS fraud patterns. Each category uses randomised helper functions:

| Helper | Purpose |
|--------|---------|
| `rphone()` | Random `+91XXXXXXXXXX` sender |
| `dlt_sender()` | Fraud-style DLT sender IDs (CP-MPOKKT, JD-SMKKBM, etc.) |
| `ra(lo, hi, step)` | Random rupee amount |
| `rurl(domains=None)` | Random shortener URL (bit.ly, lihi.cc, rb.gy, tinyurl, s.id) |
| `rphish()` | Random phishing domain (sbi-secure.in, kyc-verify.net, etc.) |
| `rwa()` | Random WhatsApp link (`wa.me/91XXXXXXXXXX`) |
| `rmobile_no()` | 10-digit mobile number string |

**25 categories and row counts:**

| Category | Rows | Category | Rows | Category | Rows |
|----------|------|----------|------|----------|------|
| fake_broker | 300 | kyc_phish | 220 | bhim_upi_fake | 220 |
| fake_bank_credit | 200 | fake_delivery | 200 | wfh_job_scam | 200 |
| lottery_prize | 180 | fake_loan | 180 | fake_ecom | 150 |
| crypto_scam | 150 | utility_threat | 150 | aadhaar_pan_fraud | 150 |
| credit_card_fraud | 150 | otp_theft | 150 | fake_govt_scheme | 150 |
| fake_job_offer | 150 | rbi_govt_scheme | 150 | income_tax_fraud | 120 |
| sim_expiry_fraud | 100 | insurance_fraud | 100 | dlt_kyc_phish | 100 |
| emi_threat | 100 | police_impersonation | 100 | sbi_yono_phish | 100 |
| real_estate_fraud | 80 | | | **TOTAL** | **3,850** |

Output: `fraud_synthetic_4000.csv` — columns: `address, body, label, category`. Rows are shuffled
with `random.seed(42)` for reproducibility.

### Layer 3 — Merge into `fraud_master.csv` (3,899 unique rows)

`fraud_training_samples.csv` (49 rows, columns: `address, body, label`) and
`fraud_synthetic_4000.csv` (3,850 rows, 4 columns) are merged by keeping only the three shared
columns and deduplicating on `body`:

```python
keep = ['address', 'body', 'label']
merged = pd.concat([seed[keep], synth[keep]], ignore_index=True)
merged = merged.drop_duplicates(subset='body')   # 3,899 unique rows
merged.to_csv('fraud_master.csv', index=False)
```

### Injection into training (`train_real_model.py`)

`fraud_master.csv` is loaded **after** the auto-labeling step and injected directly into the
combined DataFrame before feature extraction. Because the label column is forced to `'fraud'` on
load, these rows bypass `label_message()` entirely — the model learns from human-designed fraud
patterns rather than the same heuristics used at runtime:

```python
fm = pd.read_csv('fraud_master.csv').rename(columns={'address': 'sender'})
fm['label'] = 'fraud'   # guaranteed — bypass auto-labeler
combined = pd.concat([phone_df, spam_df, fm], ignore_index=True).drop_duplicates(subset='body')
```

> **v4.2 update**: a second injection block for `spam_master.csv` was added immediately after
> this block. See Section 2e for the full SPAM generation pipeline.

---

## 2e. Synthetic SPAM Data Generation Pipeline

To mirror the fraud injection strategy, a dedicated SPAM generator was written that covers real
Indian promotional SMS patterns across 25 categories.

### `generate_spam_samples.py`

Key helpers (same naming convention as `generate_fraud_samples.py`):

| Helper | Purpose |
|--------|---------|
| `rphone()` | Random `+91XXXXXXXXXX` sender |
| `dlt_sender(*brands)` | Promotional DLT sender IDs (e.g. `AD-JERUMY`, `JD-PAYTMM`) |
| `ramount(lo, hi, step)` | Random INR amount |
| `rpercent(lo, hi)` | Random percentage string |
| `rshort(domains)` | Shortener URL (bit.ly, rb.gy, tinyurl, s.id) |
| `rbrand_url(*domains)` | Branded domain URL (e.g. `https://swiggy.com/...`) |
| `rcode()` | Random promo/referral code |

**25 categories and row counts:**

| Category | Rows | Category | Rows | Category | Rows |
|----------|------|----------|------|----------|------|
| rummy_bonus | 220 | casino_vip | 140 | poker_tourney | 120 |
| fantasy_sports | 180 | cashback_wallet | 180 | food_delivery_offer | 160 |
| ecommerce_sale | 180 | instant_loan_offer | 150 | credit_card_offer | 110 |
| bnpl_offer | 110 | insurance_marketing | 90 | travel_flash_sale | 110 |
| movie_ticket_offer | 90 | telecom_recharge | 120 | data_pack_offer | 100 |
| app_install_campaign | 100 | referral_program | 90 | festival_sale | 120 |
| beauty_fashion_sale | 100 | grocery_offer | 100 | edtech_promo | 90 |
| real_estate_lead | 90 | auto_service_offer | 80 | wellness_offer | 80 |
| local_event_promo | 90 | | | **TOTAL** | **3,000** |

**Outputs:**
- `spam_synthetic_3000.csv` — columns: `address, body, label=spam, category` (analysis/audit)
- `spam_master.csv` — columns: `address, body, label=spam` (training-ready, 3 columns)

### Injection into training (`train_real_model.py`)

The `spam_master.csv` is loaded after the `fraud_master.csv` injection block and appended to the
combined DataFrame before feature extraction. The label is forced to `'spam'` on load:

```python
SPAM_MASTER = os.path.join(os.path.dirname(__file__), 'spam_master.csv')
if os.path.exists(SPAM_MASTER):
    sm = pd.read_csv(SPAM_MASTER, dtype=str).fillna('')
    sm = sm.rename(columns={'address': 'sender'})
    sm['label'] = 'spam'   # all rows confirmed promotional — bypass auto-labeler
else:
    sm = pd.DataFrame(columns=['sender','body','label'])

combined = pd.concat([
    phone_df[['sender','body','label']],
    spam_df[['sender','body','label']],
    fm[['sender','body','label']],
    sm[['sender','body','label']],    # ← v4.2 NEW
], ignore_index=True).drop_duplicates(subset='body')
```

---

## 3. 30 Behavioral Features

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | urgency_immediate | Keywords: immediate, urgent, right now |
| 1 | urgency_time_pressure | Keywords: within 24 hours, deadline, expire |
| 2 | fear_account_threats | Keywords: blocked, suspended, deactivated |
| 3 | fear_loss_threats | Keywords: penalty, fine, arrested |
| 4 | reward_money | Currency amounts, Rs, rupee |
| 5 | reward_prizes | Won, winner, prize, lottery |
| 6 | authority_financial | Bank, HDFC, SBI, RBI |
| 7 | authority_government | Income tax, IRDAI, SEBI, government |
| 8 | action_data_harvesting | Share OTP, provide card, send password |
| 9 | action_immediate | Click here, call now, respond immediately |
| 10 | total_urgency | Sum of urgency signals |
| 11 | total_fear | Sum of fear signals |
| 12 | total_reward | Sum of reward signals |
| 13 | total_authority | Sum of authority signals |
| 14 | total_action | Sum of action signals |
| 15 | length_normalized | Message character count / 1000 |
| 16 | word_count_normalized | Word count / 100 |
| 17 | uppercase_ratio | Uppercase chars / total chars |
| 18 | digit_ratio | Digit chars / total chars |
| 19 | special_char_ratio | Special chars / total chars |
| 20 | exclamation_count_normalized | Exclamation marks / 10 |
| 21 | caps_words_normalized | ALL-CAPS words / 10 |
| 22 | has_url | 1 if URL present else 0 |
| 23 | has_phone | 1 if phone number present else 0 |
| 24 | sender_is_phone | 1 if sender is numeric else 0 |
| 25 | sender_is_service | 1 if sender matches service header pattern else 0 |
| 26 | sender_length_normalized | len(sender) / 20 |
| 27 | fraud_score | Composite fraud keyword score |
| 28 | spam_score | Composite spam keyword score |
| 29 | legit_score | Composite legit keyword score |

---

## 4. 12 Reason Rules

Applied in priority order. First match wins.

| Priority | Tag | Condition |
|----------|-----|-----------|
| 1 | `account_threat` | (suspended/blocked/deactivated) AND (account/card/upi) |
| 2 | `kyc_fraud` | (kyc/aadhaar/pan) AND (update/verify/complete) |
| 3 | `legal_threat` | court/arrest/police/fir present |
| 4 | `fraud_alert` | unauthorized AND (transaction/activity) |
| 5 | `impersonation` | (income tax/irdai/sebi) AND (verify/action required) |
| 6 | `credential_harvest` | (verify/confirm) AND (otp/pin/cvv) |
| 7 | `data_steal` | (share/provide) AND (otp/card/password) BUT NOT "do not share" / "never share" |
| 8 | `prize_fraud` | (won/winner/lucky) AND (Rs/prize/cash) |
| 9 | `job_scam` | (job/earn/daily) AND (wa.me/telegram OR phone-number sender) |
| 10 | `phishing_link` | Any URL detected in body |
| 11 | `suspicious` | FRAUD class, no rule matched |
| 12 | `promotional` | SPAM class, no rule matched |

Rules run on ALL messages (LEGIT included). Only SPAM/FRAUD messages show the badge in the UI.
The Threat Breakdown panel counts all tagged messages regardless of classification.

---

## 5. Flutter App — Key Files

### lib/advanced_fraud_detector.dart

classify() flow:
1. Extract 30 features from body + sender
2. Normalize with StandardScaler (mean/scale from behavioral_model_config.json)
3. Run TFLite inference (in try/catch — failure is non-fatal)
4. ALWAYS call _detectReason() after the try/catch block
5. Return ClassificationOutput(result, reason)

Critical note: _detectReason() MUST be called outside the try/catch so it runs even when TFLite fails.

### lib/sms_log_state.dart

syncDeviceSms() flow:
1. Request SMS permission (contacts optional)
2. Attempt detector init — if fails, log WARNING but continue
3. Load up to 500 most recent SMS
4. Classify each with yield every 10 messages (prevents ANR)
5. Build _log list, notify listeners

### lib/sms_permission_helper.dart

requestAll() returns sms.isGranted only.
Contacts permission is still requested but its result is ignored.

---

## 6. Assets

Only two asset files are used:

| File | Size | Purpose |
|------|------|---------|
| assets/advanced_fraud_detector.tflite | 58.8 KB | Neural network model |
| assets/behavioral_model_config.json | ~2 KB | scaler_mean and scaler_scale arrays (30 values each) |

---

## 7. Root Causes Fixed (Build History)

### FC op v12 incompatibility
- **Symptom**: E/tflite: Didn't find op for builtin opcode 'FULLY_CONNECTED' version '12'
- **Cause**: TF 2.17 MLIR with BatchNormalization folds BN into Dense, emitting FC op v12
- **Fix**: Removed BatchNormalization + removed Optimize.DEFAULT from converter

### Sync aborts on TFLite failure
- **Symptom**: No messages read when model fails to load
- **Cause**: syncDeviceSms() had `if (!_detectorOk) { return; }`
- **Fix**: Warning logged, sync continues with rule-only classification

### Contacts permission blocking sync
- **Symptom**: No messages on Android 14 when contacts denied
- **Cause**: requestAll() returned `sms.isGranted && contacts.isGranted`
- **Fix**: Returns `sms.isGranted` only

### Threat Breakdown panel empty
- **Symptom**: Panel shows 0 counts even after sync
- **Cause**: classify() catch block returned without calling _detectReason() → reason = null
- **Fix**: _detectReason() moved outside try/catch, always executes

---

## 8. Build & Deployment

```bash
# Debug APK
cd sms_fraud_detectore_app
flutter pub get
flutter build apk --debug
adb install build/app/outputs/flutter-apk/app-debug.apk

# Retrain model (Windows)
D:\venvs\sms_fraud_detector_env\Scripts\activate
cd datasetgenerateor
python train_real_model.py
```

---

## 9. Known Limitations

| Limitation | Notes |
|------------|-------|
| ~~FRAUD precision 0.30~~ → **0.98 (v4.0)** | Resolved by injecting 3,899 fraud rows from `fraud_master.csv`; FRAUD class weight reduced from 177× to 2.26× |
| Human-label accuracy gap | Improved in v4.2: human-label acc 76.1% (auto-label val acc 97.6%), gap ~21.5 pp. Continue adding manually-reviewed borderline LEGIT/SPAM samples |
| ~~SPAM recall 85% (v4.0)~~ → **91.3% (v4.2)** | Resolved by injecting 3,000 synthetic SPAM rows from `spam_master.csv` across 25 promotional categories |
| phishing_link still broad by design | Runtime guard now suppresses LEGIT badges and trusted-domain allowlist is expanded, but URL-heavy service inboxes can still dominate Threat Breakdown counts |
| 500 message cap | Older messages beyond 500 are never seen |
| Synthetic fraud coverage | Synthetic rows cover 25 categories but may not generalise to novel fraud patterns not represented in the generator |

### 9.1 Immediate Mitigation Plan — Human Evaluation Expansion

To address reviewer concerns around the 117-sample human set, evaluation is being upgraded to a
larger **human-gold benchmark (target: 500-1000+ messages)** collected from real device exports.

**Objective**
- Reduce variance in human-label metrics and make claims statistically defensible.
- Quantify uncertainty via confidence intervals, not point accuracy alone.
- Separate model-quality issues from small-sample noise.

**Operational plan**
1. Generate a stratified review pack from phone exports:
   - Script: `datasetgenerateor/create_human_eval_pack.py`
   - Example: `python create_human_eval_pack.py --size 800 --seed 42`
2. Human review all rows in `human_eval_pack.csv` by filling `corrected_label` with
   `legit|spam|fraud` (plus reviewer metadata).
3. Evaluate deployed TFLite model on the reviewed gold set:
   - Script: `datasetgenerateor/evaluate_human_eval.py`
   - Example: `python evaluate_human_eval.py --gold human_eval_pack.csv`
4. Publish alongside accuracy:
   - Per-class precision/recall/F1
   - Confusion matrix
   - 95% CI for human-label accuracy
   - Error bucket analysis from `human_eval_errors.csv`

**Acceptance gate for next release**
- Human-gold sample size: `N >= 500` (preferred `N >= 1000`)
- Reported with 95% CI and per-class metrics
- Hard-case slice explicitly included (phone-sender fraud mimicry, URL-heavy service SMS,
  gambling promos, OTP/bank alerts)

---

## 10. Methodology

### 10.1 Problem Statement

SMS fraud in India targets users through impersonation of banks, regulators, and government bodies.
Attackers exploit the trust users place in service-header SMS (DLT senders such as `AD-SBIINB`,
`VM-HDFCBK`) by spoofing or hijacking similar-looking sender IDs, or by sending phishing links and
account-suspension threats from registered promo IDs. Personal phone numbers are also used for
job-scam and prize-fraud campaigns.

The goal is to classify any incoming SMS into one of three mutually exclusive categories:
- **LEGIT** — transactional/informational: OTPs, bank alerts, delivery updates, personal messages
- **SPAM** — unsolicited promotional: rummy/gambling apps, cashback offers, marketing blasts
- **FRAUD** — socially engineered attack: account threats, phishing, credential harvesting, impersonation

The solution must run **entirely on-device** with no network dependency, fit within ~100 KB, and
produce inference in under 50 ms on a mid-range Android phone (Android 7+, API 23+).

---

### 10.2 Data Collection

Two complementary sources were combined to balance Indian context with class diversity:

#### 10.2.1 Dedicated Data Collection App (`sms_extractor`)

Real-device SMS collection is handled by a **separate Flutter utility app** in
`sms_extractor/` (not by the detection app). This separation keeps data collection,
dataset curation, and inference deployment decoupled.

**Why a separate app was built**
- Isolates one-time export permissions from the production detector app.
- Enables repeatable dataset snapshots during model iterations.
- Keeps training data generation independent from runtime classification logic.

**Collector workflow**
1. Request `Permission.sms` and `Permission.manageExternalStorage`.
2. Read inbox messages via `flutter_sms_inbox` (`SmsQueryKind.inbox`).
3. Normalize message body (`\n` replaced with space) and date to ISO-8601.
4. Write CSV with header `id,address,body,date`.
5. Save file under `SMSExports/phone_sms_export_<timestamp>.csv` in device storage.

**Source implementation**
- Export logic: `sms_extractor/lib/sms_exporter.dart`
- Minimal UI trigger: `sms_extractor/lib/main.dart`

**Output contract used by training pipeline**
- Required columns: `id`, `address`, `body`, `date`
- Loader mapping in training: `address -> sender`, `body -> body`
- Dedup key: message `body`

This collector produced the phone CSV exports used in model training and evaluation
through v4.x.

**Source 1 — Real Indian SMS (phone CSV exports)**
Three CSV files exported directly from a physical Android device (SM E135F, India) using the
`sms_extractor` Flutter utility included in this project. Each CSV contains columns:
`id, address, body, date`. After within-source deduplication on `body`, this yielded 17,447 unique
messages dominated by real DLT service senders (Airtel, SBI, HDFC, government, e-commerce).

**Source 2 — sms_spam.csv (UCI/Kaggle SMS Spam Collection)**
5,572 English-language messages pre-labelled `ham`/`spam`. Provides additional labelled spam
examples and neutral ham messages to improve generalisation. Sender set to `UNKNOWN` for this
source.

After cross-dataset deduplication on `body`, the combined corpus is **22,615 messages**.

| Class | Count | % |
|-------|-------|---|
| LEGIT | ~21,418 | ~94.7% |
| SPAM  | ~1,080  | ~4.8%  |
| FRAUD | ~117    | ~0.5%  |

---

### 10.3 Automatic Labeling

Manual annotation at scale is impractical. A rule-based auto-labeler (`label_message()` in
`train_real_model.py`) assigns labels using a priority-ordered decision tree:

**Rule 1 — Phone-number sender heuristic**
If `sender` matches `^\+\d{10,}$` (a real mobile number, not a DLT header), apply regex patterns
for known scam content: work-from-home job scams with `wa.me/` links, prize/lottery claims, and
unsolicited app-download requests. Matched → `fraud`; unmatched URL → `spam`; clean → `legit`.

**Rule 2 — Known LEGIT brand lookup**
The `LEGIT_SENDER_BRANDS` set contains ~100 known Indian DLT brand codes (extracted from the
sender field after stripping the 2-character prefix, e.g. `AX-AIRTEL` → `AIRTEL`). A sender
matching any entry is labelled `legit` unless two specific fraud-override patterns are matched
(KYC-expiry or account-suspension with a link).

**Rule 3 — Known SPAM brand lookup**
`SPAM_SENDER_BRANDS` contains ~20 codes known to be rummy/gambling promotional senders. Matched → `spam`.

**Rule 4 — DLT prefix heuristics**
Remaining DLT senders (those with a `-` separator or length ≤ 6) are checked for gambling/rummy
keywords + URL → `spam`; generic promo patterns + URL → `spam`; otherwise `legit`.

**Rule 5 — Content-only fraud patterns**
Five regex patterns capture high-confidence fraud signals regardless of sender:
account-suspension threats, KYC/Aadhaar expiry requests, legal-action threats,
unauthorized-transaction claims, and regulator impersonation.

This labeling strategy achieves high precision on LEGIT (the dominant class) and FRAUD while
accepting lower recall on SPAM edge cases — a deliberate trade-off given SPAM's lower risk.

---

### 10.4 Feature Engineering

Rather than raw text embeddings (TF-IDF, word2vec, BERT), 30 **behavioral features** are
extracted. This choice is driven by three constraints:

1. **On-device parity** — The exact same 30 features must be computed identically in both Python
   (training) and Dart (inference). Embedding models require a vocabulary or tokenizer that would
   add megabytes to the APK and milliseconds of latency.
2. **Interpretability** — Each feature has a human-readable meaning, enabling the rule-based reason
   engine to operate on the same signals the model uses.
3. **Size** — A 30-float input fits in 120 bytes; the resulting Dense model is 58.8 KB vs. the
   smallest BERT variant at ~25 MB.

Features are grouped into 5 behavioral dimensions plus structural/sender signals:

| Group | Features | Rationale |
|-------|----------|-----------|
| Urgency | `urgency_immediate`, `urgency_time_pressure` | Fraud creates artificial time pressure |
| Fear | `fear_account_threats`, `fear_loss_threats` | Threats of loss drive compliance |
| Reward | `reward_money`, `reward_prizes` | Prize/cash lures characterise spam & fraud |
| Authority | `authority_financial`, `authority_government` | Impersonation of trusted institutions |
| Action | `action_data_harvesting`, `action_immediate` | Demands for sensitive data or urgent action |
| Composite | `total_urgency/fear/reward/authority/action` | Summed signal strength per dimension |
| Structural | `length`, `word_count`, `uppercase_ratio`, `digit_ratio`, `special_char_ratio`, `exclamation_count`, `caps_words` | Style fingerprints differ across classes |
| Link/phone | `has_url`, `has_phone` | URL presence is a strong spam/fraud indicator |
| Sender | `sender_is_phone`, `sender_is_service`, `sender_length` | Sender type encodes risk prior |
| Composite risk | `fraud_score`, `spam_score`, `legit_score` | Pre-computed weighted scores summarising multiple signals |

All 30 features are normalized using `sklearn.preprocessing.StandardScaler` fitted on the training
split. The `mean_` and `scale_` arrays (30 values each) are exported to `behavioral_model_config.json`
and loaded by the Dart app at startup.

---

### 10.5 Model Architecture and Training

**Architecture choice — Dense MLP over sequence models**
LSTM and transformer architectures require sequential token inputs, which would need a vocabulary
embedded in the APK. A fully-connected MLP over the 30 pre-computed features is sufficient because
the features already encode all relevant semantics; no positional or sequential context is lost.

**Final architecture**:
```
InputLayer(30)  →  Dense(128, relu, L2=1e-4)  →  Dropout(0.3)
               →  Dense(64,  relu, L2=1e-4)  →  Dropout(0.2)
               →  Dense(32,  relu)
               →  Dense(3,   softmax)
```

No `BatchNormalization` layers — see Section 10.6 for the TFLite compatibility reason.

**Class imbalance handling**
With ~94.7% LEGIT, naïve training collapses to predicting LEGIT for everything. `compute_class_weight('balanced')`
from scikit-learn produces per-class weights inversely proportional to class frequency:

| Class | Weight |
|-------|--------|
| LEGIT | 0.352 |
| SPAM  | 6.980 |
| FRAUD | 64.156 |

These are passed to `model.fit(class_weight=...)`, causing the loss for each FRAUD sample to be
weighted ~182× more than a LEGIT sample.

**Optimizer and training**
- Optimizer: Adam (lr=0.001 default)
- Loss: `sparse_categorical_crossentropy`
- Epochs: up to 50, `EarlyStopping(patience=5, restore_best_weights=True)`
- Validation split: 20% stratified (4,523 messages)
- Best epoch: 14

---

### 10.6 TFLite Export and Compatibility

**The FC op v12 problem**
TensorFlow 2.16+ switched to an MLIR-based converter. When a model contains `BatchNormalization`,
the converter folds the BN statistics into the preceding `Dense` layer's weights — a valid
mathematical optimisation that produces `FULLY_CONNECTED` op version 12. The `tflite_flutter 0.11.0`
Flutter package bundles the TFLite 2.14 C++ runtime, which only supports `FULLY_CONNECTED` up to
version 11. The symptom at runtime is:

```
E/tflite: Didn't find op for builtin opcode 'FULLY_CONNECTED' version '12'
```

**Solution**
Remove all `BatchNormalization` layers from the model. Without BN, the MLIR converter emits
`FULLY_CONNECTED` op version 9, which is compatible with all TFLite runtimes since 2.0.

Additionally, `Optimize.DEFAULT` (post-training INT8 quantization) was removed from the converter
call. Quantization can also trigger higher op versions and introduces calibration complexity. The
float32 model at 58.8 KB is small enough that the size benefit of quantization (~50% reduction) is
not necessary.

**Export pipeline**:
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# No converter.optimizations — float32 keeps FC op v9
tflite_bytes = converter.convert()
open(TFLITE_OUT, 'wb').write(tflite_bytes)
```

---

### 10.7 Hybrid Inference: Neural Network + Rule Engine

The app uses a two-stage hybrid approach:

**Stage 1 — TFLite neural network** produces a probability distribution
`[P_LEGIT, P_SPAM, P_FRAUD]`. The argmax determines the classification result.

**Stage 2 — Rule-based reason engine** (`_detectReason()` in `advanced_fraud_detector.dart`)
applies 12 priority-ordered deterministic rules to produce a human-readable reason tag.

The two stages are **independent**: Stage 2 always runs regardless of whether Stage 1 succeeds.
This means:
- If TFLite fails to initialise (e.g. corrupt asset, unsupported device), all messages still
  receive a rule-based reason tag and a default LEGIT classification.
- The Threat Breakdown panel always shows non-zero counts as long as SMS can be read.

The rule engine runs on **all 500 scanned messages** — including those classified as LEGIT — so
the breakdown panel reflects patterns across the full inbox, not just flagged messages. The inline
reason badge in the message thread is only displayed for SPAM and FRAUD messages.

---

### 10.8 Evaluation

#### Model v3.0 (historical)

```
              precision    recall  f1-score   support

       LEGIT       0.99      0.94      0.96      4284
        SPAM       0.43      0.77      0.55       216
       FRAUD       0.30      0.96      0.46        23

    accuracy                           0.93      4523
```

#### Model v4.0 (baseline — 26,471 training messages, 3,899 injected fraud rows)

```
              precision    recall  f1-score   support

       LEGIT       0.99      0.95      0.97      4288
        SPAM       0.46      0.85      0.59       227
       FRAUD       0.98      1.00      0.99       780

    accuracy                           0.95      5295
   macro avg       0.81      0.93      0.85      5295
weighted avg       0.97      0.95      0.96      5295
```

**Key observations (v4.0)**:
- **LEGIT**: Precision 0.99 maintained. Recall improved to 0.95. Only 8 LEGIT messages misclassified
  as FRAUD (down from higher false-positive rate in v3.0).
- **FRAUD**: Precision jumped from 0.30 → **0.98** — the single biggest improvement. Only 2 FRAUD
  messages missed across 780 in the val set. Recall is effectively **100%**. This is the direct
  result of the 3,899-row `fraud_master.csv` injection reducing the class weight from 177× to 2.26×.
- **SPAM**: Recall improved from 0.77 → 0.85. Still the weakest class due to the broad diversity
  of promotional SMS styles not captured by the 25 synthetic fraud categories.

**Human-labelled validation (117 manually reviewed messages)**:
```
              precision    recall  f1-score   support

       LEGIT       1.00      0.40      0.58        42
        SPAM       0.54      0.72      0.62        43
       FRAUD       0.70      0.94      0.80        32

    accuracy                           0.67       117
```
Auto-label val accuracy (0.949) vs human-label accuracy (0.667) — gap of 0.282 indicates the model
has partially learned the auto-labeler's heuristics in addition to genuine fraud patterns. This is
expected: the 3,899 injected rows are synthetic (designed to match known fraud templates) and do not
fully represent the diversity of real-world borderline cases. Closing this gap requires more
manually-reviewed real-world SMS.

#### Model v4.1 (superseded — commit `d06ecb1`)

v4.1 updates only behavioral scoring logic (no architecture change, no dataset-size change):

- `_legit_score`/`_legitScore`: trusted URL `+0.5`, missed-call pattern `+0.3`
- `_spam_risk`/`_spamRisk`: gambling promo weight `+0.4`
- Trusted URL patterns expanded 14 → 52

Observed outcome from retrain:
- Auto-label validation accuracy: **0.945**
- Human-label accuracy (117 reviewed messages): **0.744**
- Auto-vs-human gap reduced: **0.282 → 0.202**

This confirms better alignment with manually corrected labels, especially in previously problematic
borderline categories (Airtel missed-call notifications and gambling/rummy promotional SMS).

#### Model v4.2 (current — 29,471 training messages, +3,000 injected SPAM rows)

```
              precision    recall  f1-score   support

       LEGIT       0.99      0.98      0.99      4288
        SPAM       0.93      0.91      0.92       827
       FRAUD       0.98      0.99      0.99       780

    accuracy                           0.98      5895
   macro avg       0.97      0.96      0.96      5895
weighted avg       0.98      0.98      0.98      5895
```

**Key observations (v4.2)**:
- **SPAM**: Recall 0.85 → **0.91** (+6 pp). Precision jumped from 0.46 → **0.93** because the
  3,000 injected synthetic rows teach the model clean promotional patterns, reducing confusion
  between SPAM and LEGIT service messages.
- **LEGIT**: Recall improved from 0.95 → **0.98**. The extra SPAM data reduces false-SPAM
  classification of borderline service messages.
- **FRAUD**: Recall maintained at **0.99** (772/780). No regression from SPAM injection.
- **Auto-label val accuracy**: 0.976 (up from 0.945 in v4.1) — largest single jump across all
  versions, driven primarily by SPAM class improvement.

**Human-labelled validation (117 manually reviewed messages)**:
```
              precision    recall  f1-score   support

       LEGIT       0.99      0.98      0.99      4288  (auto-label val)
        SPAM       0.93      0.91      0.92       827  (auto-label val)
       FRAUD       0.98      0.99      0.99       780  (auto-label val)

Human-label accuracy (117 reviewed): 0.761
```

**Qualitative (device testing — unchanged from v3.0 session)**
Running on 1,625 real inbox messages from SM E135F (Android 14):
- 477 LEGIT, 8 SPAM, 15 FRAUD detected
- Reason distribution: `phishing_link: 271`, `data_steal: 15`, `credential_harvest: 13`,
  `legal_threat: 12`, `kyc_fraud: 8`, `impersonation: 2`, `prize_fraud: 2`
- No false FRAUD positives on known-good banking OTP threads
- `phishing_link` dominant because most service SMS (Airtel billing, gas delivery, insurance)
  include URLs — confirms the known limitation in Section 9

---

## 11. System Evolution & Transformation

### **From Keyword-Only to Behavioral Intelligence**

The SMS fraud detection system has undergone a **complete paradigm shift** from simple keyword matching to advanced **behavioral pattern analysis and psychological manipulation detection**.

#### **BEFORE (Keyword-Only System)**
- Simple keyword matching (`urgent`, `verify`, `click here`)
- High false positive rate (11.2%)
- Missed obfuscated text and creative variations
- No context or intent understanding
- Limited fraud detection capability

#### **AFTER (Behavioral Analysis System)**
- 🧠 **Psychological pattern detection** — identifies manipulation tactics
- 😨 **Emotional intelligence analysis** — detects fear and urgency exploitation
- 👔 **Authority impersonation recognition** — identifies fake government/bank messages
- 🎯 **Intent and sentiment analysis** — understands message purpose beyond keywords
- 📊 **Multi-factor behavioral scoring** — combines multiple signals
- 🔍 **Intelligent reasoning system** — provides explanations

#### **Performance Improvement**
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Overall Accuracy | 89.1% | 93.8% | **+4.7%** |
| Fraud Detection | 85.4% | 100% | **+14.6%** |
| False Positive Rate | 11.2% | 6.25% | **-44.2%** |
| Processing Time | 42ms | 45ms | +3ms (acceptable) |

---

## 12. Enhanced Behavioral Analysis Engine

### **Multi-Factor Psychological Analysis**

The enhanced system analyzes messages across five behavioral dimensions:

#### **1. Psychological Manipulation Detection**
```
Urgency Tactics (0.00-0.30)
├── Time pressure keywords: "within 24 hours", "expires today"
├── Immediate action demands: "right now", "immediately", "urgent"
└── Deadline creation: "before", "by", "deadline"

Fear & Intimidation (0.00-0.25)
├── Account threats: "suspended", "blocked", "deactivated"
├── Loss threats: "penalty", "fine", "arrested", "legal action"
└── Consequence threats: "lose access", "account will be deleted"

Authority Impersonation (0.00-0.25)
├── Bank mimicking: "HDFC", "SBI", "ICICI", "Axis"
├── Government mimicking: "Income Tax", "Police", "RBI", "SEBI"
└── Service mimicking: Legitimate DLT sender spoofing

Reward/False Promises (0.00-0.35)
├── Money promises: "win", "earn", "cash", "prize"
├── Quick returns: "guaranteed", "fast", "easy money"
└── Exclusive offers: "limited", "special", "cashback"

Data Harvesting (0.00-0.15)
├── OTP requests: "share OTP", "provide code", "confirm verification"
├── Credential theft: "send password", "provide card details"
└── Personal info: "share Aadhar", "confirm PAN"
```

#### **2. Emotional Intelligence Analysis**
- Emotional intensity measurement
- Sentiment polarity analysis
- Psychological pressure assessment

#### **3. Structural & Composition Analysis**
- Writing style anomalies detection
- Capitalization abuse detection
- Punctuation manipulation recognition
- Language complexity assessment

#### **4. Sender Verification**
- Legitimate sender pattern validation
- Impersonation risk assessment
- Authority claim verification

---

## 13. Fraud Pattern Categories (Enhanced Format)

### **🚨 HIGH-RISK FRAUD PATTERNS**

#### **Account Suspension/Threat Scams**
- Pattern: Account threat + urgency + verification request
- Example: "URGENT: Your account SUSPENDED! Verify NOW!"
- Detection: fear_score > 0.05 AND urgency_score > 0.05 AND authority_score > 0.05

#### **Government Impersonation**
- Pattern: Authority mimicking + legal threats + deadline
- Example: "Income Tax: PAN disabled. Update within 24 hours or face legal action."
- Detection: authority_government > 0.05 AND fear_score > 0.05 AND urgency_score > 0.10

#### **Data Harvesting Attempts**
- Pattern: Information requests + impersonation + pressure
- Example: "Bank Security: Provide OTP and PIN to secure account."
- Detection: data_harvesting > 0.05 AND authority_score > 0.05

#### **Phishing & Malicious Links**
- Pattern: URL presence + credential/verification request
- Example: "Verify account: click [link] to confirm identity"
- Detection: has_url AND (data_harvesting OR phishing_pattern)

### **🟡 SPAM PROMOTION PATTERNS**

#### **Prize/Lottery Scams**
- Pattern: Reward promises + congratulations + urgency + contact request
- Example: "Congratulations! Won ₹50,000! Claim NOW before offer expires!"
- Detection: reward_score > 0.05 AND urgency_score > 0.03

#### **Investment/Income Fraud**
- Pattern: Money promises + work opportunities + guaranteed returns
- Example: "Earn ₹5000 daily from home! No investment! Guaranteed!"
- Detection: reward_money > 0.05 AND guarantee_promises > 0.03

#### **Marketing Manipulation**
- Pattern: Product promotion + time pressure + false scarcity
- Example: "SALE! 70% off! Limited time only! Buy NOW!"
- Detection: promotional_keywords AND urgency_tactics

### **🟢 LEGITIMATE SAFE PATTERNS**

#### **Bank Transaction Alerts**
- Pattern: Official bank code + transaction details + no request
- Example: "Rs.500 spent at Amazon. Balance: Rs.15,000 -HDFCBK"
- Protection: legitimate_bank_code AND transaction_pattern

#### **Service Notifications**
- Pattern: Status update from verified service
- Example: "Your order is out for delivery. ETA: 30 minutes."
- Protection: service_notification_pattern AND legitimate_sender

---

## 14. Setup & Installation Guide

### **Quick Start (5 minutes)**

```bash
# 1. Clone repository
git clone https://github.com/iDhanushp/-Smart-Detection-of-Malicious-SMS.git
cd "Smart Detection of Malicious SMS"

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Train the model
cd datasetgenerateor
python train_real_model.py

# 4. Export to Flutter
cd ../ML_Model
python export_tflite.py
```

### **Enhanced Behavioral Setup (15 minutes)**

```bash
# 1. Install enhanced dependencies
pip install -r requirements_enhanced.txt
# Includes: sentence-transformers, textstat, xgboost, lightgbm

# 2. Test enhanced behavioral labeler
cd datasetgenerateor
python enhanced_behavioral_labeler.py
python comprehensive_analysis_demo.py

# 3. Train with enhanced features
cd ../ML_Model
python train_enhanced.py --data "../datasetgenerateor/new csv/final_labeled_sms.csv" --use-semantic --use-behavioral
```

### **Flutter App Setup**

```bash
# 1. Navigate to app directory
cd sms_fraud_detectore_app

# 2. Get dependencies
flutter pub get

# 3. Copy model assets
copy ../ML_Model/advanced_fraud_detector.tflite assets/
copy ../ML_Model/behavioral_model_config.json assets/

# 4. Build APK
flutter build apk --debug

# 5. Install on device
adb install -r build/app/outputs/flutter-apk/app-debug.apk
```

---

## 15. Deployment Options

### **Option 1: Quick Behavioral Upgrade (Immediate)**
- Works with existing 29,471 messages
- 93.8% accuracy improvement
- <45ms processing per message
- No additional dependencies

### **Option 2: Full Semantic Intelligence**
- Advanced context understanding
- 384-dimensional embeddings
- Ensemble model optimization
- Real-time behavioral analysis

### **Option 3: Flutter App Integration**
- Mobile deployment ready
- Real-time classification
- Visual threat assessment
- User-friendly interface

---

## 16. Flutter App Features & UI

### **Enhanced Detection Dashboard**
- **Modern Material Design 3** interface with gradient backgrounds
- **Animated Status Cards** with real-time protection status
- **Statistics Grid** showing 4-metric dashboard
- **Professional Control Panel** with enhanced settings
- **Activity Feed** with recent detection history

### **Advanced Classification Display**
```
🟢 GREEN (Legitimate)
├── Bank transaction alerts
├── OTP codes and verification
├── Delivery updates
├── Personal messages
└── Confidence: 60-95%

🟡 YELLOW (Spam/Promotional)
├── Prize/lottery scams
├── Investment schemes
├── Marketing pressure
├── Unsolicited promotions
└── Confidence: 25-60%

🔴 RED (High-Risk Fraud)
├── Account suspension threats
├── Government impersonation
├── Data harvesting attempts
├── Credential theft schemes
└── Confidence: 60-95%
```

### **Key Features**
- ✅ Full SMS device sync (up to 500 messages)
- ✅ Real-time detection for new SMS
- ✅ Background processing
- ✅ Runtime permission management
- ✅ Offline operation (no internet required)
- ✅ User-friendly reasoning explanations
- ✅ Threat breakdown analysis
- ✅ Recent activity feed with timestamps

---

## 17. Configuration & Advanced Settings

### **Model Configuration (behavioral_model_config.json)**
```json
{
  "scaler_mean": [/* 30 float values */],
  "scaler_scale": [/* 30 float values */],
  "version": "4.2.0",
  "features": 30,
  "input_shape": [1, 30],
  "output_classes": 3
}
```

### **Feature Normalization**
All 30 features are normalized using StandardScaler fitted on training split:
- Each feature centered to mean 0
- Scaled to unit variance
- Ensures consistent inference across devices

### **Android App Configuration**
- API Level: 23+ (Android 6.0+)
- Permissions: SMS read, contacts (optional)
- Storage: <5 MB for model + config
- Memory: ~50 MB runtime

---

## 18. Troubleshooting & Common Issues

### **Issue: TFLite Model Fails to Initialize**
**Symptom**: App shows no detections
**Solution**: Fallback to rule-based classification (always runs)
**Details**: Rule engine operates independently of ML model

### **Issue: High false positive rate**
**Solution**: Adjust confidence thresholds in `_detectReason()`
**Details**: Tune rule boundaries for your dataset

### **Issue: Slow SMS sync**
**Symptom**: App freezes during sync
**Solution**: 500 message cap prevents ANR; uses batch processing
**Details**: Messages processed in chunks with 10-message yields

### **Issue: Model file corrupted**
**Symptom**: E/tflite error on app start
**Solution**: Re-export model from training script
**Details**: See Section 14 for export pipeline

---

## 19. Performance Testing & Results

### **Comprehensive Test Results**
```
Test Dataset: 16 carefully crafted messages
Overall Accuracy: 93.8% (15/16 correct)
Fraud Detection: 100% (4/4 identified)
Spam Detection: 100% (4/4 identified)
Legitimate Recognition: 87.5% (7/8 verified)
False Positive Rate: 6.25% (1/16 edge case)
Processing Time: <45ms per message
```

### **Real Dataset Analysis**
```
Sample: 100 messages from 10,946 device SMS
Fraud Detected: 21 messages (sophisticated patterns)
Legitimate Verified: 79 messages
Processing Speed: <50ms average
False Positives: Minimal (bank alerts protected)
```

### **Production Readiness Assessment**
- ✅ Conservative fraud detection (low false positive risk)
- ✅ Good overall accuracy (88-93% range)
- ✅ Handles large-scale data efficiently
- ✅ Clear reasoning for classifications
- ✅ Comprehensive logging

---

## 20. Complete Changelog

### **v4.2.0 (Current)**
- ✨ Synthetic SPAM injection (3,000 rows, 25 categories)
- 📈 SPAM recall improved: 85% → 91.3%
- 📊 Val accuracy: 97.6%
- 👥 Human-label accuracy: 76.1%
- Training messages: 26,471 → 29,471

### **v4.1.0**
- 🎯 Behavioral scoring logic improved
- 🔗 Trusted URL allowlist expanded: 14 → 52 entries
- 📉 Auto-vs-human accuracy gap: 28.2% → 20.2%
- 👥 Human-label accuracy: 66.7% → 74.4%

### **v4.0.0 (Baseline)**
- ✨ Fraud injection pipeline (3,899 synthetic rows)
- 📊 FRAUD precision: 0.30 → 0.98
- 🏆 FRAUD recall: 1.00 (100%)
- 🎯 Class weight reduction: 177× → 2.26×

### **v3.0.0 (Legacy)**
- 🚀 Initial TFLite + Flutter integration
- 📊 Val accuracy: 93.9%
- ⚠️ Low FRAUD precision (0.30)
- 🔴 Severe class imbalance

---

## 21. References & Related Documentation

This documentation integrates information from:
- `COMPLETE_DOCUMENTATION_UPDATE.md` — Behavioral analysis specifications
- `ENHANCED_SETUP_GUIDE.md` — Installation and configuration
- `ENHANCED_DEPLOYMENT_GUIDE.md` — Deployment strategies
- `PROJECT_SUMMARY.md` — System architecture overview
- `FINAL_ANALYSIS_REPORT.md` — Performance metrics and results
- `CHANGELOG.md` — Complete version history

---

**Version**: 4.2.0 · March 2026 · Flutter 3.32.5 · tflite_flutter 0.11.0 · TF 2.17 · Python 3.10  
**Data**: fraud_master.csv 3,899 rows · spam_master.csv 3,000 rows (25 promotional categories)  
**Model**: advanced_fraud_detector.tflite 58.8 KB · val acc 97.6% · SPAM recall 91.3% · human-label acc 76.1%  
**Repository**: [Smart Detection of Malicious SMS](https://github.com/iDhanushp/-Smart-Detection-of-Malicious-SMS)
