import 'dart:typed_data';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/foundation.dart';
import 'dart:math' as math;

/// Minimal, single-purpose detector: returns 0 (Legit), 1 (Spam), 2 (Fraud).
/// Rules:
///   • Spam  = spamProb > legitProb.
///   • Fraud = Spam + sender starts with +countryCode AND spamProb ≥ _fraudThreshold.
class FraudDetector {
  late Interpreter _intrp;

  static const double _spamCutoffTrusted = 0.33; // alphanumeric senders
  static const double _spamCutoffUnverified = 0.28; // phone numbers
  static const double _fraudThreshold = 0.65;

  static const List<String> _spamKeywords = [
    'win',
    'prize',
    'offer',
    'discount',
    'free',
    'claim',
    'cashback',
    'loan',
    'lakh',
    '₹',
    'congratulations',
    'lottery',
    'deal'
  ];

  Future<void> loadModel(String assetPath) async {
    _intrp = await Interpreter.fromAsset(assetPath);
  }

  /// Returns a map with keys: prediction (0/1/2) and reason (String).
  Map<String, dynamic> predictWithReasoning(
      String sender, String body, List<double> input) {
    final probs = _infer(input);
    final spamProb = probs[1];
    final legitProb = probs[0];

    final bool phoneSender = RegExp(r'^\+[0-9]{6,}').hasMatch(sender);
    final double cutoff =
        phoneSender ? _spamCutoffUnverified : _spamCutoffTrusted;

    // Spam only when the model is actually more confident in Spam than Legit.
    bool isSpam = (spamProb >= cutoff) && (spamProb > legitProb);

    // Fallback keyword heuristic when the model is unsure (probabilities very
    // close: difference < 1e-6).
    if (!isSpam && (spamProb - legitProb).abs() < 1e-6) {
      final lower = body.toLowerCase();
      isSpam = _spamKeywords.any((kw) => lower.contains(kw));
    }
    final bool isFraud = isSpam && phoneSender && spamProb >= _fraudThreshold;

    final prediction = isFraud ? 2 : (isSpam ? 1 : 0);

    // Console log for every detection (sender, probs, final prediction)
    // Additional diagnostics: vector magnitude & non-zero feature count.
    double magnitude = 0;
    int nonZero = 0;
    for (var v in input) {
      if (v != 0) {
        nonZero++;
        magnitude += v * v;
      }
    }
    magnitude = magnitude > 0 ? math.sqrt(magnitude) : 0;

    print('DETECT sender="$sender" '
        'legit=${legitProb.toStringAsFixed(3)} '
        'spam=${spamProb.toStringAsFixed(3)} '
        'cutoff=${phoneSender ? _spamCutoffUnverified : _spamCutoffTrusted} '
        'vecNZ=$nonZero vecNorm=${magnitude.toStringAsFixed(4)} '
        '-> pred=$prediction');

    return {
      'prediction': prediction,
      'reason': 'spamProb ${(spamProb * 100).toStringAsFixed(2)}%',
    };
  }

  // Lightweight helpers --------------------------------------------------

  int predict(List<double> input) {
    final probs = _infer(input);
    return probs[1] > probs[0] ? 1 : 0;
  }

  List<double> _infer(List<double> input) {
    try {
      // Wrap feature vector in batch dimension
      final inputTensor = [Float32List.fromList(input)]; // shape [1, N]

      // Output tensor as List<double> inside batch list (shape [1,3])
      final outputTensor = [List<double>.filled(3, 0.0)];

      _intrp.run(inputTensor, outputTensor);

      return List<double>.from(outputTensor[0]);
    } catch (e) {
      print('TFLite inference error: $e');
      return [0.5, 0.5, 0.0];
    }
  }
}
