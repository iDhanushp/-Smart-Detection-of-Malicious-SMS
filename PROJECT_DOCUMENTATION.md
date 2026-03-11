# Project Documentation — Smart Detection of Malicious SMS

**Version**: 3.0.0  
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

## 2. Model v3.0

### Training Data

| Source | Messages |
|--------|----------|
| Phone CSV (real Indian SMS from device) | 17,447 (after within-source dedup) |
| sms_spam.csv (Kaggle SMS spam dataset) | 5,572 |
| Cross-dataset duplicates removed | −404 |
| **Total** | **22,615** |

Class distribution in training set: ~94.6% LEGIT, ~4.8% SPAM, ~0.5% FRAUD.
Class weights applied: `{LEGIT: 0.352, SPAM: 6.980, FRAUD: 64.156}`.

### Architecture

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

Best epoch: 14, val_accuracy: 0.9317

```
              precision    recall  f1-score   support

       LEGIT       0.99      0.94      0.96      4284
        SPAM       0.43      0.77      0.55       216
       FRAUD       0.30      0.96      0.46        23

    accuracy                           0.93      4523
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
| FRAUD precision 0.30 | Only 117 fraud training samples; many URLs are flagged as fraud even from legit senders |
| SPAM recall 77% | Significant SPAM class imbalance |
| phishing_link over-fires | Rule matches ANY URL — many legit service SMS contain URLs |
| 500 message cap | Older messages beyond 500 are never seen |

---

**Version**: 3.0.0 · March 2026 · Flutter 3.32.5 · tflite_flutter 0.11.0 · TF 2.17 · Python 3.10
