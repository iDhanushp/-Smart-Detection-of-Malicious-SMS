# Smart Detection of Malicious SMS 🛡️

An **AI-powered mobile fraud detection system** that classifies incoming SMS as **LEGIT**, **SPAM**, or **FRAUD** — entirely on-device, with no cloud dependency.

Trained on **22,615 real Indian SMS messages** using a **30-feature behavioral neural network** plus a **priority-ordered rule engine** that labels *why* each message was flagged.

---

## How It Works

```
SMS received
  ├── Extract 30 behavioral features (urgency, fear, authority, data-request, URL, sender type)
  ├── StandardScaler normalization  (fitted on 22,615 real messages)
  ├── TFLite Dense network  -->  [P_LEGIT, P_SPAM, P_FRAUD]
  └── Rule engine  -->  reason tag  (phishing_link / data_steal / job_scam / …)
```

Both steps run **independently** — if TFLite fails to load, the rule engine still classifies every message and populates the Threat Breakdown panel.

---

## 3 Output Classes

| Class | Badge | Examples |
|-------|-------|----------|
| LEGIT | Green  | OTPs, bank alerts, delivery notifications, personal messages |
| SPAM  | Amber  | Promotional offers, prize claims, gambling apps, marketing  |
| FRAUD | Red    | Account threats, phishing, impersonation, job scams         |

---

## 12 Reason Tags

Every flagged message (SPAM or FRAUD) gets an emoji tag explaining the specific threat pattern:

| Tag | Triggered By |
|-----|--------------|
| `account_threat`     | "account suspended / blocked / deactivated" |
| `kyc_fraud`          | KYC / Aadhaar / PAN update requests |
| `legal_threat`       | Court / police / arrest / FIR mentions |
| `fraud_alert`        | Unauthorized transaction / suspicious activity |
| `impersonation`      | Income Tax / IRDAI / SEBI / TRAI impersonation |
| `credential_harvest` | Verify / confirm your OTP / PIN / CVV |
| `data_steal`         | Share your card / OTP / password (excludes bank "do not share" warnings) |
| `prize_fraud`        | Won / lucky draw / prize |
| `job_scam`           | Work-from-home + wa.me / daily earnings scam |
| `phishing_link`      | Any URL in message body |
| `suspicious`         | FRAUD class but no specific rule matched |
| `promotional`        | SPAM class but no specific rule matched |

> Reason rules run on **all 500 scanned messages** (including LEGIT) so the Threat Breakdown panel
> shows full pattern counts. The tag badge in the message bubble is only shown on SPAM / FRAUD messages.

---

## Model Performance

**Model v3.0** — trained March 2026 on 22,615 real Indian SMS

```
Validation set: 4,523 messages

              precision    recall  f1-score   support

       LEGIT       0.99      0.94      0.96      4284
        SPAM       0.43      0.77      0.55       216
       FRAUD       0.30      0.96      0.46        23

    accuracy                           0.93      4523
```

| Metric | Value |
|--------|-------|
| Overall accuracy  | 93.2% |
| FRAUD recall      | 96% (catches 96 / 100 real fraud messages) |
| SPAM recall       | 77% |
| Inference time    | < 50 ms on-device |
| Model size        | 58.8 KB (float32 TFLite, FC op v9) |

---

## Architecture

### Project Structure

```
smart-sms-detection/
├── sms_fraud_detectore_app/              # Flutter Android app
│   ├── lib/
│   │   ├── main.dart
│   │   ├── sms_log_state.dart            # State: sync, classify, reasonCounts
│   │   ├── sms_log_model.dart            # SmsLogEntry, ClassificationOutput
│   │   ├── advanced_fraud_detector.dart  # TFLite + 12-rule reason engine
│   │   ├── thread_list_page.dart         # Thread list + Threat Breakdown panel
│   │   └── thread_page.dart              # Message bubbles + reason tags
│   └── assets/
│       ├── advanced_fraud_detector.tflite     # 58.8 KB float32 model (v3.0)
│       └── behavioral_model_config.json       # scaler_mean / scaler_scale (30 values)
├── datasetgenerateor/
│   └── train_real_model.py              # Full retraining pipeline (22,615 messages)
└── sms_extractor/                        # SMS export Flutter utility
```

### TFLite Model

| Property | Value |
|----------|-------|
| Input    | [1, 30] float32 |
| Output   | [1, 3] float32 → [P_LEGIT, P_SPAM, P_FRAUD] |
| Architecture | Dense(128,relu) → Dropout(0.3) → Dense(64,relu) → Dropout(0.2) → Dense(32,relu) → Dense(3,softmax) |
| Size | 58.8 KB (float32, no quantization) |
| FC op version | v9 (compatible with tflite_flutter 0.11.0 / TFLite 2.14) |
| BatchNormalization | REMOVED — BN causes TF 2.17 MLIR to emit FC op v12 which tflite_flutter 0.11.0 cannot run |

### 30 Behavioral Features

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

## App UI

### Thread List Screen
- **Stats Row**: Safe / Spam / Fraud counts update live during sync
- **Threat Breakdown Panel**: Collapsible; shows reason tag counts across all 500 scanned messages
  (e.g. `phishing link: 312`, `data steal: 47`)
- **Thread tiles**: Color-coded by worst classification in thread; last-message preview

### Message Thread Screen
- Message bubbles with inline reason badge for SPAM / FRAUD only
  (e.g. `credential harvest`)

### Classification Examples

```
FRAUD   credential harvest
"Dear Customer, your SBI account will be blocked. Verify OTP: bit.ly/xyz"

SPAM    promotional
"Play Rummy & Win Rs.10 Lakhs! Download now: gmg.im/rummy"

LEGIT
"Your OTP is 847291. Do NOT share this with anyone. -SBIINB"
```

---

## Setup & Build

### Prerequisites
- Flutter 3.x (`flutter doctor`)
- Android device / emulator — API 23+ (minSdk 23)
- Python 3.10+ with TF 2.17 (only needed for retraining)

### Build & Install

```bash
cd sms_fraud_detectore_app
flutter pub get
flutter build apk --debug
flutter install
```

### Retrain the Model

```bash
cd datasetgenerateor
# Windows: activate D:\venvs\sms_fraud_detector_env\Scripts\activate
python train_real_model.py
# Exports directly to ../sms_fraud_detectore_app/assets/
```

---

## Privacy

- 100% on-device — no data ever leaves the phone
- No network calls — works fully offline
- SMS read-only — no send / delete permissions used
- Contacts optional — contacts permission is requested but NOT required to read SMS

---

## Known Issues / Limitations

| Issue | Notes |
|-------|-------|
| FRAUD precision 0.30 — some false positives | Only ~115 fraud training samples; more fraud data would help |
| SPAM recall 77% — some promotional SMS missed | Class imbalance: 1,080 spam vs 21,418 legit in training set |
| `phishing_link` rule fires broadly | Matches any URL — many legit service SMS (Airtel, Indane, banks) contain URLs |

---

## License

MIT License

---

**Version**: 3.0.0 · **Last updated**: March 2026  
**Platform**: Android 7.0+ (API 23+) · **Flutter**: 3.32.5 · **tflite_flutter**: 0.11.0
