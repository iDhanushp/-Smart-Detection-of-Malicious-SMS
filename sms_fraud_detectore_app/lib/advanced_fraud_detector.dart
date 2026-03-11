import 'dart:convert';
import 'dart:math' as math;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'sms_log_model.dart';

/// SMS Fraud Detector — wraps the 30-feature TFLite behavioral model.
///
/// SINGLE public API:
///   [classify(sender, body)] → [DetectionResult]   (no Map, no strings)
///
/// Model output indices:  0 = LEGITIMATE, 1 = SPAM, 2 = FRAUD
class AdvancedFraudDetector {
  late Interpreter _interpreter;
  late List<double> _scalerMean;
  late List<double> _scalerScale;
  bool _ready = false;

  bool get isReady => _ready;

  // ── keyword banks ─────────────────────────────────────────────────────────
  static const _urgencyKw = [
    'urgent', 'immediately', 'asap', 'expire', 'deadline',
    'limited time', 'act now', 'hurry', 'last chance',
    'expire today', 'expires soon', 'time running out'
  ];
  static const _fearKw = [
    'suspended', 'blocked', 'terminated', 'legal action',
    'penalty', 'fine', 'arrest', 'court', 'lawsuit',
    'closed', 'cancelled', 'frozen', 'unauthorized'
  ];
  static const _rewardKw = [
    'congratulations', 'winner', 'won', 'prize', 'cash',
    'reward', 'lottery', 'jackpot', 'free', 'gift',
    'bonus', 'cashback', 'refund', 'lakh', 'crore'
  ];
  static const _actionKw = [
    'click', 'call', 'reply', 'text', 'visit', 'download',
    'verify', 'confirm', 'update', 'provide', 'share',
    'enter', 'submit', 'activate', 'redeem'
  ];

  // ── initialise ────────────────────────────────────────────────────────────
  Future<void> initialize() async {
    _interpreter = await Interpreter.fromAsset(
        'assets/advanced_fraud_detector.tflite');

    final raw = await rootBundle.loadString('assets/behavioral_model_config.json');
    final cfg = json.decode(raw) as Map<String, dynamic>;

    _scalerMean  = (cfg['scaler_mean']  as List).map((v) => (v as num).toDouble()).toList();
    _scalerScale = (cfg['scaler_scale'] as List).map((v) => (v as num).toDouble()).toList();

    _ready = true;

    if (kDebugMode) {
      debugPrint('✅ AdvancedFraudDetector ready  '
          'in=${_interpreter.getInputTensor(0).shape}  '
          'out=${_interpreter.getOutputTensor(0).shape}');
    }
  }

  // ── PUBLIC API ────────────────────────────────────────────────────────────

  /// Classifies a single SMS.  Returns a [ClassificationOutput] with result and optional reason — never throws.
  ClassificationOutput classify(String sender, String body) {
    final t = body.toLowerCase();
    final s = sender.toLowerCase();

    // ── Step 1: TFLite model (may fail gracefully) ────────────────────────
    DetectionResult result = DetectionResult.legitimate;

    if (!_ready) {
      if (kDebugMode) debugPrint('⚠️  classify called before initialize()');
      // fall through — rule engine still runs below
    } else {
      try {
        final raw  = _extractFeatures(body, sender);
        final norm = _normalize(raw);

        // TFLite: input [1,30] → output [1,3]
        final out = [List.filled(3, 0.0)];
        _interpreter.run([norm], out);

        final probs = out[0]; // [legit, spam, fraud]
        int best = 0;
        if (probs[1] > probs[best]) best = 1;
        if (probs[2] > probs[best]) best = 2;

        result = const [
          DetectionResult.legitimate,
          DetectionResult.spam,
          DetectionResult.fraudulent,
        ][best];

        if (kDebugMode) {
          debugPrint('DETECT "$sender" '
              'L=${probs[0].toStringAsFixed(3)} '
              'S=${probs[1].toStringAsFixed(3)} '
              'F=${probs[2].toStringAsFixed(3)} '
              '→ ${result.name.toUpperCase()}');
        }
      } catch (e, st) {
        if (kDebugMode) debugPrint('❌ TFLite error: $e\n$st');
        // result stays legitimate — rule engine still runs
      }
    }

    // ── Step 2: Rule-based reason detection ALWAYS runs ──────────────────
    final reason = _detectReason(t, s, result);
    if (kDebugMode && reason != null) {
      debugPrint('  reason → $reason');
    }
    return ClassificationOutput(result, reason);
  }

  // ── rule-based reason detection ───────────────────────────────────────────

  /// Returns a short rule key explaining WHY the message was flagged,
  /// or null for legitimate messages.
  String? _detectReason(String t, String s, DetectionResult result) {
    // NOTE: runs on ALL messages (LEGIT included) so the threat-breakdown
    // panel shows full pattern counts.  Bubble display is guarded in UI.

    // Highest-priority: account/card suspension threats
    if (RegExp(r'(suspended|blocked|deactivated|terminated)').hasMatch(t) &&
        RegExp(r'(account|card|upi|wallet)').hasMatch(t)) return 'account_threat';

    // KYC fraud
    if (RegExp(r'(kyc|aadhaar|aadhar|pan)').hasMatch(t) &&
        RegExp(r'(update|expire|verify|link|complete)').hasMatch(t)) return 'kyc_fraud';

    // Legal threats
    if (RegExp(r'(legal action|court|arrest|police|penalty|fine|lawsuit|fir)').hasMatch(t))
      return 'legal_threat';

    // Unauthorized transaction / fraud alert
    if (RegExp(r'(unauthorized|suspicious|unusual)').hasMatch(t) &&
        RegExp(r'(transaction|activity|login|access)').hasMatch(t)) return 'fraud_alert';

    // Government / authority impersonation
    if (RegExp(r'(income.?tax|irdai|sebi|rbi.?official|trai|government.?of.?india)').hasMatch(t) &&
        RegExp(r'(verify|update|link|action|notice)').hasMatch(t)) return 'impersonation';

    // Credential / OTP harvesting (verify/confirm OTP)
    if (RegExp(r'(verify|confirm|update)').hasMatch(t) &&
        RegExp(r'(otp|pin|cvv|password|card number|account number)').hasMatch(t))
      return 'credential_harvest';

    // Asking to share sensitive data — skip if it's a bank warning ("Do not share your OTP")
    if (RegExp(r'(share|provide|send|enter)').hasMatch(t) &&
        RegExp(r'(otp|pin|password|cvv|card|account|aadhaar|pan)').hasMatch(t) &&
        !RegExp(r'(do not|dont|never|don.t).{0,15}(share|give|enter|provide|send)').hasMatch(t))
      return 'data_steal';

    // Prize / lottery fraud
    if (RegExp(r'(won|winner|lucky|congratulations)').hasMatch(t) &&
        RegExp(r'(rs\.?|₹|\d{4,}|lakh|crore|cash|prize)').hasMatch(t))
      return 'prize_fraud';

    // Fake job / work-from-home — fires for phone senders OR DLT + wa.me/job redirect
    if (RegExp(r'(job|work.?from.?home|h0me.{0,10}j0b|salary|earn|per.?day|\d{3,}.?day)').hasMatch(t) &&
        (RegExp(r'(wa\.me|t\.me/|telegram|whatsapp|apply.?now)').hasMatch(t) || _isPhone(s)))
      return 'job_scam';

    // URL present but from a known-legit domain → not a phishing signal.
    // URL from an unknown domain → tag as phishing_link.
    if (_hasUrl(t) && !_hasTrustedUrl(t)) return 'phishing_link';

    // No specific pattern found — null for clean LEGIT, fallback label for flagged
    if (result == DetectionResult.legitimate) return null;
    return result == DetectionResult.fraudulent ? 'suspicious' : 'promotional';
  }

  // ── feature extraction (30 features, same order as training) ─────────────

  List<double> _extractFeatures(String text, String sender) {
    final t = text.toLowerCase();
    final s = sender.toLowerCase();

    final urgImm    = _score(t, _urgencyKw);
    final urgTime   = _timePressure(t)      ? 1.0 : 0.0;
    final fearAcc   = _score(t, _fearKw);
    final fearLoss  = _lossThreats(t)       ? 1.0 : 0.0;
    final rewMoney  = _moneyRewards(t)      ? 1.0 : 0.0;
    final rewPrize  = _score(t, _rewardKw);
    final authFin   = _impersonatesBank(t, s) ? 1.0 : 0.0;
    final authGov   = _impersonatesGov(t, s)  ? 1.0 : 0.0;
    final actData   = _requestsData(t)      ? 1.0 : 0.0;
    final actImm    = _score(t, _actionKw);

    return [
      urgImm, urgTime,
      fearAcc, fearLoss,
      rewMoney, rewPrize,
      authFin, authGov,
      actData, actImm,
      urgImm + urgTime,                           // totalUrgency
      fearAcc + fearLoss,                         // totalFear
      rewMoney + rewPrize,                        // totalReward
      authFin + authGov,                          // totalAuthority
      actData + actImm,                           // totalAction
      math.min(text.length / 500.0, 1.0),         // lengthNorm  (training: /500)
      math.min(text.split(' ').length / 100.0, 1.0), // wordCountNorm (training: /100)
      _upperRatio(text),
      _digitRatio(text),
      _specialRatio(text),
      math.min((text.split('!').length - 1) / 5.0, 1.0), // exclamCount (training: /5)
      math.min(_capsWords(text) / 10.0, 1.0),     // capsWords (training: /10)
      _hasUrl(text)   ? 1.0 : 0.0,
      _hasPhone(text) ? 1.0 : 0.0,
      _isPhone(sender)   ? 1.0 : 0.0,
      _isService(sender) ? 1.0 : 0.0,
      math.min(sender.length / 20.0, 1.0),        // senderLen (training: /20)
      _fraudRisk(t, s),
      _spamRisk(t),
      _legitScore(t, s),
    ];
  }

  List<double> _normalize(List<double> f) {
    final out = <double>[];
    for (int i = 0; i < f.length; i++) {
      final sc = (_scalerScale[i] == 0.0) ? 1.0 : _scalerScale[i];
      out.add((f[i] - _scalerMean[i]) / sc);
    }
    return out;
  }

  // ── helpers ───────────────────────────────────────────────────────────────

  double _score(String t, List<String> kws) {
    int h = 0;
    for (final k in kws) { if (t.contains(k)) h++; }
    return math.min(h / kws.length, 1.0);
  }

  bool _timePressure(String t) =>
      const ['expire','deadline','limited time','hurry','asap'].any(t.contains);

  bool _lossThreats(String t) =>
      const ['lose','loss','miss out','forfeit','penalty'].any(t.contains);

  bool _moneyRewards(String t) =>
      const ['₹','lakh','crore','cash','money','amount'].any(t.contains);

  bool _impersonatesBank(String t, String s) =>
      const ['bank','sbi','hdfc','icici','axis','rbi'].any(t.contains) &&
      _isPhone(s);

  bool _impersonatesGov(String t, String s) =>
      const ['government','ministry','department','income tax','aadhaar']
          .any(t.contains) && _isPhone(s);

  bool _requestsData(String t) =>
      const ['otp','pin','password','cvv','card number','account number']
          .any(t.contains);

  double _upperRatio(String t) {
    if (t.isEmpty) return 0;
    return t.codeUnits.where((c) => c >= 65 && c <= 90).length / t.length;
  }

  double _digitRatio(String t) {
    if (t.isEmpty) return 0;
    return t.codeUnits.where((c) => c >= 48 && c <= 57).length / t.length;
  }

  double _specialRatio(String t) {
    if (t.isEmpty) return 0;
    const sp = r'!@#$%^&*()_+-=[]{}|;:,.<>?';
    return t.split('').where(sp.contains).length / t.length;
  }

  int _capsWords(String t) =>
      t.split(' ').where((w) => w.length > 2 && w == w.toUpperCase()).length;
  // ── trusted-domain allowlist ──────────────────────────────────────────────
  // URLs from these domains are NOT tagged as phishing_link.
  // The raw _hasUrl() flag is still used for feature extraction (model input)
  // because the model was trained with it; only the reason-tag is suppressed.
  static const _trustedDomains = [
    // Telecom
    'airtel.in', 'airtel.com', 'jio.com', 'myairtel.app',
    'bsnl.in', 'vodafone.in', 'vi.in',
    // Banking & payments
    'hdfcbank.com', 'sbi.co.in', 'onlinesbi.sbi', 'icicibank.com',
    'axisbank.com', 'kotak.com', 'yesbank.in', 'rbl.in',
    'paytm.com', 'phonepe.com', 'gpay.app', 'upi.npci.org.in',
    // Insurance
    'icicilombard.com', 'hdfclife.com', 'licindia.in', 'starhealth.in',
    'bajajfinserv.in', 'reliancegeneral.co.in',
    // E-commerce & delivery
    'amazon.in', 'flipkart.com', 'myntra.com', 'meesho.com',
    'swiggy.in', 'zomato.com', 'blinkit.com',
    'bluedart.com', 'delhivery.com', 'ekart.in', 'dtdc.com',
    // Utilities & govt
    'irctc.co.in', 'indianrail.gov.in', 'india.gov.in',
    'incometax.gov.in', 'uidai.gov.in', 'epfindia.gov.in',
    'bescom.org', 'mahadiscom.in', 'tneb.in',
    // OTT & entertainment
    'jiocinema.com', 'hotstar.com', 'netflix.com', 'primevideo.com',
  ];

  /// Returns true if [t] contains a URL whose host matches a trusted domain.
  bool _hasTrustedUrl(String t) {
    final lower = t.toLowerCase();
    return _trustedDomains.any(lower.contains);
  }
  bool _hasUrl(String t) =>
      RegExp(r'https?://|www\.|\.com|\.in|\.org').hasMatch(t);

  bool _hasPhone(String t) => RegExp(r'\b\d{10,}\b').hasMatch(t);

  // Training: sender.startswith('+') — only international + numbers
  bool _isPhone(String s) => s.startsWith('+') && RegExp(r'^\+\d{10,}$').hasMatch(s);

  // Training: len(sender) <= 6 or '-' in sender
  bool _isService(String s) => !_isPhone(s) && (s.contains('-') || s.length <= 6);

  double _fraudRisk(String t, String s) => math.min(
      (_moneyRewards(t) ? 0.3 : 0) +
          (_requestsData(t) ? 0.4 : 0) +
          (_impersonatesBank(t, s) ? 0.4 : 0) +
          (t.contains('wa.me') ? 0.5 : 0) +
          (RegExp(r'(h0me|home).{0,20}(j0b|job)').hasMatch(t) ? 0.4 : 0) +
          (RegExp(r'earn.{0,15}(\d{3,}|rs\.?\s*\d{3,}).{0,10}(day|daily)').hasMatch(t) ? 0.3 : 0),
      1.0);

  double _spamRisk(String t) => math.min(
      (_hasUrl(t) ? 0.2 : 0) +
      (_score(t, _rewardKw) > 0.3 ? 0.3 : 0) +
      (RegExp(r'(rummy|poker|casino|bet|fantasy|ipl|cricket.?match|gaming)').hasMatch(t) ? 0.3 : 0),
      1.0);

  double _legitScore(String t, String s) {
    double score = 0;
    // OTP messages are almost always legitimate
    if (t.contains('otp') || RegExp(r'\b\d{4,6}\b').hasMatch(t)) score += 0.3;
    // Short message with no URL = service notification
    if (t.length < 160 && !_hasUrl(t)) score += 0.2;
    // Indian DLT bank/service sender code: starts with 2-letter prefix + dash
    // e.g. AX-AIRTEL-S, AD-SBIINB, VM-HDFC — training gave these +0.5 legit score
    if (RegExp(r'^[A-Z]{2}-').hasMatch(s)) score += 0.5;
    // Short numeric/alpha sender codes (e.g. IRCTC, BESCOM)
    if (!_isPhone(s) && s.length <= 6) score += 0.3;
    return math.min(score, 1.0);
  }

  void dispose() {
    if (_ready) _interpreter.close();
  }
}
