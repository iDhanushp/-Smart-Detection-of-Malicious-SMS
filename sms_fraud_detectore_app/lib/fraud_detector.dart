import 'package:tflite_flutter/tflite_flutter.dart';

class FraudDetector {
  late Interpreter _interpreter;

  Future<void> loadModel(String assetPath) async {
    // Use more compatible interpreter options
    final options = InterpreterOptions();

    // Try to load with basic options first
    try {
      _interpreter = await Interpreter.fromAsset(assetPath, options: options);
    } catch (e) {
      print('Failed to load model with options: $e');
      // Fallback to default loading
      _interpreter = await Interpreter.fromAsset(assetPath);
    }
  }

  /// Enhanced prediction with sender validation and reasoning for three classes
  /// Returns a map with 'prediction' (0, 1, or 2) and 'reason' (String)
  Map<String, dynamic> predictWithReasoning(String sender, List<double> input) {
    // First, check sender validation
    final senderValidation = _analyzeSender(sender);

    if (senderValidation['isTrusted']) {
      // Trusted sender (bank/app) - likely legitimate
      return {
        'prediction': 0,
        'reason': 'Trusted sender: ${senderValidation['reason']}'
      };
    }

    // If sender is suspicious, apply ML detection
    final mlPrediction = predict(input);
    final mlReason = _getPredictionReason(mlPrediction);

    return {
      'prediction': mlPrediction,
      'reason': 'Suspicious sender + $mlReason'
    };
  }

  /// Get human-readable reason for ML prediction
  String _getPredictionReason(int prediction) {
    switch (prediction) {
      case 0:
        return 'ML model classified as legitimate';
      case 1:
        return 'ML model flagged as spam (promotional content)';
      case 2:
        return 'ML model flagged as fraudulent (security threat)';
      default:
        return 'ML model classification unclear';
    }
  }

  /// Enhanced prediction with sender validation for three classes
  /// Returns 0 (legitimate), 1 (spam), or 2 (fraudulent)
  int predictWithSenderValidation(String sender, List<double> input) {
    // First, check sender validation
    if (_isTrustedSender(sender)) {
      // Trusted sender (bank/app) - likely legitimate
      return 0;
    }

    // If sender is suspicious (phone number), apply ML detection
    return predict(input);
  }

  /// Expects input as a List<double> (TF-IDF vector)
  /// Returns 0 (legitimate), 1 (spam), or 2 (fraudulent)
  int predict(List<double> input) {
    try {
      // Model expects shape [1, 3000] - add batch dimension
      final inputTensor = [input];

      // Create output tensor with proper shape [1, 3] for three classes
      var output = List.filled(1, List.filled(3, 0.0));

      // Run inference
      _interpreter.run(inputTensor, output);

      // Get probabilities from softmax output
      final probabilities = output[0] as List<double>;

      // Find the class with highest probability
      int maxIndex = 0;
      double maxProb = probabilities[0];
      for (int i = 1; i < probabilities.length; i++) {
        if (probabilities[i] > maxProb) {
          maxProb = probabilities[i];
          maxIndex = i;
        }
      }

      // Return the predicted class (0=legitimate, 1=spam, 2=fraudulent)
      return maxIndex;
    } catch (e) {
      print('Prediction error: $e');
      // Return safe default (legitimate)
      return 0;
    }
  }

  /// Analyze sender and return detailed information
  Map<String, dynamic> _analyzeSender(String sender) {
    if (sender.isEmpty) {
      return {'isTrusted': false, 'reason': 'Empty sender'};
    }

    // Remove any whitespace
    sender = sender.trim();

    // Check if sender starts with country code (suspicious)
    // Pattern: +91, +1, +44, etc.
    if (RegExp(r'^\+[0-9]{1,4}').hasMatch(sender)) {
      return {'isTrusted': false, 'reason': 'Phone number with country code'};
    }

    // Check if sender is alphanumeric (trusted)
    // Pattern: HDFCBK, VM-AIRTEL, BX-ICICIB, etc.
    if (RegExp(r'^[A-Z0-9\-]{3,15}$').hasMatch(sender.toUpperCase())) {
      return {'isTrusted': true, 'reason': 'Bank/App sender ID (alphanumeric)'};
    }

    // Check if it's a short numeric code (like 5-digit codes)
    if (RegExp(r'^[0-9]{4,6}$').hasMatch(sender)) {
      return {
        'isTrusted': true,
        'reason': 'Short numeric code (likely trusted service)'
      };
    }

    // Default to suspicious for unknown patterns
    return {'isTrusted': false, 'reason': 'Unknown sender pattern'};
  }

  /// Check if sender is trusted (alphanumeric) vs suspicious (phone number)
  bool _isTrustedSender(String sender) {
    if (sender.isEmpty) return false;

    // Remove any whitespace
    sender = sender.trim();

    // Check if sender starts with country code (suspicious)
    // Pattern: +91, +1, +44, etc.
    if (RegExp(r'^\+[0-9]{1,4}').hasMatch(sender)) {
      return false; // Suspicious - phone number
    }

    // Check if sender is alphanumeric (trusted)
    // Pattern: HDFCBK, VM-AIRTEL, BX-ICICIB, etc.
    if (RegExp(r'^[A-Z0-9\-]{3,15}$').hasMatch(sender.toUpperCase())) {
      return true; // Trusted - bank/app sender ID
    }

    // Check if it's a short numeric code (like 5-digit codes)
    if (RegExp(r'^[0-9]{4,6}$').hasMatch(sender)) {
      return true; // Trusted - short numeric codes
    }

    // Default to suspicious for unknown patterns
    return false;
  }
}
