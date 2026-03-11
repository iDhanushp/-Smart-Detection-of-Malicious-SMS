# Project Documentation (Clean) — Smart Detection of Malicious SMS

**Version**: 3.0.0 · March 2026

---

## Overview

On-device Android SMS fraud detector. Reads up to 500 recent SMS, classifies each as LEGIT / SPAM /
FRAUD using a TFLite Dense neural network (30 features, 3 classes), then applies a deterministic
12-rule engine to produce a human-readable reason tag. No internet required.

---

## Model v3.0

**Dataset**: 22,615 messages (17,447 phone CSV + 5,572 sms_spam.csv − 404 cross-dataset duplicates)

**Architecture**:
```
Dense(128, relu, L2) -> Dropout(0.3) -> Dense(64, relu, L2) -> Dropout(0.2) -> Dense(32, relu) -> Dense(3, softmax)
```
No BatchNormalization (TF 2.17 MLIR + BN produces FC op v12, incompatible with tflite_flutter 0.11.0).
No Optimize.DEFAULT (float32 export keeps FC op v9).

**Validation accuracy**: 93.2% | **FRAUD recall**: 96% | **SPAM recall**: 77%

**Model file**: `assets/advanced_fraud_detector.tflite` — 58.8 KB, float32, FC op v9, input [1,30], output [1,3]

---

## 30 Behavioral Features

```
 0  urgency_immediate          16  word_count_normalized
 1  urgency_time_pressure      17  uppercase_ratio
 2  fear_account_threats       18  digit_ratio
 3  fear_loss_threats          19  special_char_ratio
 4  reward_money               20  exclamation_count_normalized
 5  reward_prizes              21  caps_words_normalized
 6  authority_financial        22  has_url
 7  authority_government       23  has_phone
 8  action_data_harvesting     24  sender_is_phone
 9  action_immediate           25  sender_is_service
10  total_urgency              26  sender_length_normalized
11  total_fear                 27  fraud_score
12  total_reward               28  spam_score
13  total_authority            29  legit_score
14  total_action
15  length_normalized
```

---

## 12 Reason Rules (priority order)

| # | Tag | Key Condition |
|---|-----|---------------|
| 1 | `account_threat` | suspended/blocked + account/card/upi |
| 2 | `kyc_fraud` | kyc/aadhaar/pan + update/verify |
| 3 | `legal_threat` | court/arrest/police/fir |
| 4 | `fraud_alert` | unauthorized + transaction/activity |
| 5 | `impersonation` | income tax/irdai/sebi + verify/action |
| 6 | `credential_harvest` | verify/confirm + otp/pin/cvv |
| 7 | `data_steal` | share/provide + otp/card/password (not "do not share") |
| 8 | `prize_fraud` | won/winner/lucky + prize/cash |
| 9 | `job_scam` | job/earn + wa.me/telegram or phone sender |
| 10 | `phishing_link` | any URL present |
| 11 | `suspicious` | FRAUD, no rule matched |
| 12 | `promotional` | SPAM, no rule matched |

Rules run on ALL messages. Badge shown only on SPAM/FRAUD. Threat Breakdown counts all tagged messages.

---

## Assets

| File | Size | Purpose |
|------|------|---------|
| `assets/advanced_fraud_detector.tflite` | 58.8 KB | Neural network (float32, FC v9) |
| `assets/behavioral_model_config.json` | ~2 KB | scaler_mean + scaler_scale (30 values each) |

---

## Key Implementation Notes

**classify() in advanced_fraud_detector.dart**:
_detectReason() is called OUTSIDE the TFLite try/catch — it runs even when TFLite fails.

**syncDeviceSms() in sms_log_state.dart**:
TFLite init failure is a warning, not an abort. Sync continues with rule-only classification.

**sms_permission_helper.dart**:
requestAll() returns sms.isGranted only. Contacts is optional.

---

## Build

```bash
# App
cd sms_fraud_detectore_app && flutter build apk --debug

# Retrain (Windows)
D:\venvs\sms_fraud_detector_env\Scripts\activate
cd datasetgenerateor && python train_real_model.py
```

---

## Limitations

| Issue | Notes |
|-------|-------|
| FRAUD precision 0.30 | Few fraud training samples; over-flags URLs from legit senders |
| SPAM recall 77% | Class imbalance |
| phishing_link over-fires | Matches any URL including legit service SMS |

---

**Flutter**: 3.32.5 · **tflite_flutter**: 0.11.0 · **TF**: 2.17 · **minSdk**: 23
