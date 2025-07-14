import 'dart:typed_data';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/foundation.dart';
import 'dart:math' as math;

/// 2-Class SMS detector: Handles existing 3-class model but implements 2-class logic.
/// Fraud detection: spam prediction + phone number sender pattern.
///
/// Logic:
///   • Model outputs: [legit, spam, fraud] but we combine spam+fraud = spam
///   • Final classes: 0=Legitimate, 1=Spam
///   • Fraud = (isSpam AND sender matches +countryCode pattern)
class FraudDetector {
  late Interpreter _intrp;

  // Thresholds for spam detection (increased for better precision)
  static const double _spamCutoffTrusted = 0.32; // alphanumeric senders
  static const double _spamCutoffUnverified = 0.32; // phone numbers

  // Keywords for fallback detection when model is uncertain
  static const List<String> _spamKeywords = [
    // Original keywords
    'win', 'prize', 'offer', 'discount', 'free', 'claim', 'cashback', 'loan',
    'lakh', '₹', 'congratulations', 'lottery', 'deal', 'urgent', 'limited time',
    'click here',
    // Enhanced spam keywords
    'gift', 'reward', 'bonus', 'cash', 'money', 'earn', 'income', 'profit',
    'investment', 'scheme', 'opportunity', 'business', 'work from home',
    'part time', 'easy money', 'guaranteed', 'instant', 'hurry', 'expires',
    'limited offer', 'special offer', 'exclusive', 'selected', 'winner',
    'lucky', 'surprise', 'scratch', 'coupon', 'voucher', 'code', 'redeem',
    'activate', 'confirm', 'verify', 'update', 'suspended', 'blocked',
    'crore', 'thousand', 'millions', 'rs.', 'rupees', 'paisa', 'amount',
    'transfer', 'deposit', 'withdraw', 'account', 'balance', 'credit',
    'commission', 'percentage', '%', 'daily', 'weekly', 'monthly'
  ];

  // Fraud-specific keywords (more aggressive patterns)
  static const List<String> _fraudKeywords = [
    'otp',
    'pin',
    'password',
    'cvv',
    'card number',
    'debit card',
    'credit card',
    'bank details',
    'account number',
    'ifsc',
    'mobile banking',
    'net banking',
    'phishing',
    'fraud',
    'scam',
    'fake',
    'suspicious',
    'unauthorized',
    'immediate action',
    'act now',
    'expire today',
    'last chance',
    'your account will be',
    'suspended',
    'closed',
    'terminated',
    'refund',
    'tax refund',
    'government',
    'income tax',
    'gst',
    'kyc',
    'documents',
    'verification required',
    'update kyc'
  ];

  // Legitimate Indian banking/telecom service patterns
  static const List<String> _legitimateBankCodes = [
    'AX-',
    'AD-',
    'JM-',
    'CP-',
    'VM-',
    'VK-',
    'BZ-',
    'TX-',
    'JD-',
    'BK-',
    'BP-',
    'JX-',
    'TM-',
    'QP-',
    'BV-',
    'JK-',
    'BH-',
    'TG-',
    'JG-',
    'VD-',
    'AIRTEL',
    'SBIINB',
    'SBIUPI',
    'AXISBK',
    'IOBCHN',
    'IOBBNK',
    'KOTAKB',
    'PHONPE',
    'PAYTM',
    'ADHAAR',
    'VAAHAN',
    'ESICIP',
    'EPFOHO',
    'BESCOM',
    'CBSSBI',
    'NBHOME',
    'NBCLUB',
    'GOKSSO',
    'TRAIND',
    'AIRXTM',
    'AIRMCA',
    'NSESMS',
    'CDSLEV',
    'CDSLTX',
    'BFDLTS',
    'BFDLPS',
    'BSELTD'
  ];

  // Promotional/E-commerce service patterns (typically spam)
  static const List<String> _promotionalCodes = [
    'MGLAMM', // Magicbricks (real estate promotions)
    'APLOTF', // Apollo (pharmacy promotions)
    'EVOKHN', // Evoking (promotions)
    'MYNTRA', // Myntra (fashion e-commerce)
    'FLPKRT', // Flipkart (e-commerce)
    'ZEPTON', // Zepto (grocery delivery)
    'DOMINO', // Domino's Pizza
    'ZOMATO', // Zomato (food delivery)
    'SWIGGY', // Swiggy (food delivery)
    'MEESHO', // Meesho (e-commerce)
    'BLUDRT', // Bluedart (courier promotions)
    'NOBRKR', // NoBroker (real estate)
    'GROWWZ', // Groww (investment promotions)
    'PAISAD', // Paisa (loan/finance promotions)
    'PRUCSH', // Promotions
    'HEDKAR', // Promotions
    'BOTNIC', // Mamaearth/Botanic promotions
    'EKARTL', // E-Kart logistics
    'RECHRG', // Recharge promotions
    'SMYTTN', // Smartton promotions
    // Additional promotional services
    'AMAZON', 'AMZNIN', 'AJIOCOM', 'NYKAA', 'LENSKRT', 'PAYTMM',
    'OLACABS', 'UBER', 'RAPIDO', 'DUNZO', 'BIGBSKT', 'GROFER',
    'JOCKEY', 'ADIDAS', 'NIKE', 'PUMA', 'LEVIS', 'UCB',
    'MCDONALD', 'KFC', 'SUBWAY', 'PIZZHUT', 'BURGKNG',
    'HOTSTAR', 'NETFLIX', 'PRIME', 'YOUTUB', 'SPOTIFY',
    'MAKEMT', 'POLICY', 'LENDIX', 'MONEYV', 'CRED',
    'FREECHARGE', 'MOBIKW', 'OXIGEN', 'CITRUS',
    'INDIGO', 'SPICEJET', 'AIRVST', 'GOIBIBO', 'MMTRIP',
    'BYJU', 'VEDANT', 'TOPPR', 'UNACAD'
  ];

  static const List<String> _legitimateKeywords = [
    'bank',
    'account',
    'balance',
    'credited',
    'debited',
    'transaction',
    'upi',
    'payment',
    'transfer',
    'received',
    'sent',
    'otp',
    'verification',
    'code',
    'airtel',
    'jio',
    'bsnl',
    'vodafone',
    'recharge',
    'validity',
    'statement',
    'emi',
    'loan approved',
    'insurance',
    'policy',
    'premium'
  ];

  Future<void> loadModel(String assetPath) async {
    _intrp = await Interpreter.fromAsset(assetPath);
  }

  /// Returns a map with keys: prediction (0/1/2) and reason (String).
  /// Final classification:
  /// - 0 = Legitimate
  /// - 1 = Spam
  /// - 2 = Fraud (spam + phone number sender)
  Map<String, dynamic> predictWithReasoning(
      String sender, String body, List<double> input) {
    // Get model predictions (3-class model: [legit, spam, fraud])
    final probs = _infer(input);
    final legitProb = probs[0];
    final spamProb = probs[1];
    final fraudProb = probs.length > 2 ? probs[2] : 0.0;

    // Combine spam and fraud probabilities for 2-class logic
    final combinedSpamProb = spamProb + fraudProb;

    // Determine sender type
    final bool isPhoneNumber = RegExp(r'^\+[0-9]{6,}').hasMatch(sender);
    final double cutoff =
        isPhoneNumber ? _spamCutoffUnverified : _spamCutoffTrusted;

    // Check for Indian service patterns
    final bool isLegitimateService = _isLegitimateServiceSender(sender);
    final bool isPromotionalService = _isPromotionalServiceSender(sender);
    final bool hasLegitKeywords = _hasLegitimateKeywords(body);

    // Calculate vector strength
    double magnitude = 0;
    int nonZero = 0;
    for (var v in input) {
      if (v != 0) {
        nonZero++;
        magnitude += v * v;
      }
    }
    magnitude = magnitude > 0 ? math.sqrt(magnitude) : 0;
    final bool isWeakVector = magnitude < 0.1 || nonZero < 3;

    // Enhanced classification logic
    bool isSpam;
    bool isSuspiciousFraud = false;

    // Check for fraud keywords first (highest priority)
    final lower = body.toLowerCase();
    final hasFraudKeywords = _hasFraudKeywords(body);

    if (hasFraudKeywords) {
      isSpam = true;
      isSuspiciousFraud = true;
    } else if (isWeakVector) {
      // For weak vectors (vocabulary mismatch), use pattern-based classification
      if (isPromotionalService) {
        // E-commerce/promotional services are always spam
        isSpam = true;
      } else if (isLegitimateService && !isPromotionalService) {
        // Only legitimate non-promotional services (banks/telecom)
        isSpam = false;
      } else if (hasLegitKeywords) {
        // Has legitimate keywords but not a known service
        isSpam = false;
      } else {
        // Check for spam keywords
        isSpam = _spamKeywords.any((kw) => lower.contains(kw));
      }
    } else {
      // For strong vectors, use model prediction with adjusted thresholds
      if (isPromotionalService && combinedSpamProb > 0.25) {
        // Promotional services with very low spam probability are likely spam
        isSpam = true;
      } else if (isLegitimateService &&
          !isPromotionalService &&
          combinedSpamProb < 0.65) {
        // Reduced benefit of doubt threshold for legitimate services
        isSpam = false;
      } else {
        isSpam = (combinedSpamProb >= cutoff) && (combinedSpamProb > legitProb);
      }
    }

    // Enhanced fraud detection
    bool isFraud = false;
    if (isSpam) {
      // Original fraud: spam + phone number
      if (isPhoneNumber) {
        isFraud = true;
      }
      // New fraud patterns: suspicious content regardless of sender
      else if (isSuspiciousFraud) {
        // Messages with fraud keywords are fraud even from alphanumeric senders
        isFraud = true;
      }
      // Additional fraud indicators
      else if (combinedSpamProb > 0.8 &&
          (_fraudKeywords.any((kw) => lower.contains(kw)) ||
              lower.contains('otp') ||
              lower.contains('pin') ||
              lower.contains('cvv') ||
              lower.contains('password'))) {
        isFraud = true;
      }
    }

    // Final prediction
    final int prediction = isFraud ? 2 : (isSpam ? 1 : 0);

    // Debug logging
    print('DETECT sender="$sender" '
        'legit=${legitProb.toStringAsFixed(3)} '
        'spam=${combinedSpamProb.toStringAsFixed(3)} '
        'cutoff=${cutoff.toStringAsFixed(2)} '
        'vecNZ=$nonZero vecNorm=${magnitude.toStringAsFixed(4)} '
        'phone=$isPhoneNumber '
        'legit_svc=$isLegitimateService '
        'promo_svc=$isPromotionalService '
        'weak=$isWeakVector '
        '-> pred=$prediction');

    // Generate reason
    String reason;
    if (isFraud) {
      reason =
          'Fraud: spam from phone number (+${sender.length > 1 ? sender.substring(1, math.min(4, sender.length)) : sender})';
    } else if (isSpam) {
      if (isPromotionalService) {
        reason = 'Spam: Promotional/E-commerce message';
      } else {
        reason =
            'Spam: ${(combinedSpamProb * 100).toStringAsFixed(1)}% confidence';
      }
    } else {
      if (isLegitimateService) {
        reason = 'Legitimate: Bank/Telecom service';
      } else if (hasLegitKeywords) {
        reason = 'Legitimate: Contains banking/service keywords';
      } else {
        reason =
            'Legitimate: ${(legitProb * 100).toStringAsFixed(1)}% confidence';
      }
    }

    return {
      'prediction': prediction,
      'reason': reason,
      'isSpam': isSpam,
      'isFraud': isFraud,
      'isPhoneNumber': isPhoneNumber,
      'spamProbability': combinedSpamProb,
      'legitProbability': legitProb,
    };
  }

  bool _isIndianServiceSender(String sender) {
    final upperSender = sender.toUpperCase();
    return _legitimateBankCodes.any((code) => upperSender.contains(code)) ||
        _promotionalCodes.any((code) => upperSender.contains(code));
  }

  bool _isLegitimateServiceSender(String sender) {
    final upperSender = sender.toUpperCase();
    return _legitimateBankCodes.any((code) => upperSender.contains(code));
  }

  bool _isPromotionalServiceSender(String sender) {
    final upperSender = sender.toUpperCase();
    return _promotionalCodes.any((code) => upperSender.contains(code));
  }

  bool _hasLegitimateKeywords(String body) {
    final lowerBody = body.toLowerCase();
    return _legitimateKeywords.any((keyword) => lowerBody.contains(keyword));
  }

  bool _hasFraudKeywords(String body) {
    final lowerBody = body.toLowerCase();
    return _fraudKeywords.any((keyword) => lowerBody.contains(keyword));
  }

  // Lightweight helper for simple classification
  int predict(List<double> input) {
    final probs = _infer(input);
    final legitProb = probs[0];
    final spamProb = probs[1];
    final fraudProb = probs.length > 2 ? probs[2] : 0.0;
    final combinedSpamProb = spamProb + fraudProb;

    return combinedSpamProb > legitProb ? 1 : 0;
  }

  // Internal inference method - handles both 2-class and 3-class models
  List<double> _infer(List<double> input) {
    try {
      // Wrap feature vector in batch dimension
      final inputTensor = [Float32List.fromList(input)]; // shape [1, N]

      // Try 3-class model first (existing model)
      var outputTensor = [List<double>.filled(3, 0.0)];

      _intrp.run(inputTensor, outputTensor);

      // Return 3-class output if successful
      return List<double>.from(outputTensor[0]);
    } catch (e) {
      // If 3-class fails, try 2-class model
      try {
        final inputTensor = [Float32List.fromList(input)];
        var outputTensor = [List<double>.filled(2, 0.0)];

        _intrp.run(inputTensor, outputTensor);

        // Convert 2-class to 3-class format for consistency
        final result = List<double>.from(outputTensor[0]);
        result.add(0.0); // Add dummy fraud probability
        return result;
      } catch (e2) {
        print('TFLite inference error: $e2');
        // Return balanced probabilities on error
        return [0.5, 0.3, 0.2]; // legit, spam, fraud
      }
    }
  }

  /// Helper method for Flutter app integration
  /// Returns classification with fraud logic
  Map<String, dynamic> classifyWithFraud(
      String sender, String body, List<double> input) {
    final result = predictWithReasoning(sender, body, input);

    final bool isSpam = result['isSpam'] as bool;
    final bool isFraud = result['isFraud'] as bool;
    final bool isPhoneNumber = result['isPhoneNumber'] as bool;

    return {
      'primary': isSpam ? 'spam' : 'legitimate',
      'isFraud': isFraud,
      'isSpam': isSpam,
      'isPhoneNumber': isPhoneNumber,
      'spamProbability': result['spamProbability'],
      'legitProbability': result['legitProbability'],
      'reason': result['reason'],
      'prediction': result['prediction'],
    };
  }
}
