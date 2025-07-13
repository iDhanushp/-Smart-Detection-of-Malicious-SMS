import 'package:flutter/material.dart';

enum DetectionResult { legitimate, spam, fraudulent }

// Advanced data models
class SenderInfo {
  final String? name;
  final bool isVerified;
  final String? type; // bank, app, unknown
  final double? reputation;

  SenderInfo({
    this.name,
    this.isVerified = false,
    this.type,
    this.reputation,
  });
}

class OTPResult {
  final bool isOTP;
  final String? otpCode;
  final String riskLevel; // low, medium, high
  final String? recommendations;

  OTPResult({
    this.isOTP = false,
    this.otpCode,
    this.riskLevel = 'low',
    this.recommendations,
  });
}

class SmsLogEntry {
  final String sender;
  final String body;
  final DetectionResult result;
  final DateTime timestamp;
  bool isMistake;
  final String? reason; // Why was it classified this way?

  // Advanced features
  final double? trustScore;
  final bool? isOTP;
  final String? otpRisk;
  final SenderInfo? senderInfo;

  SmsLogEntry({
    required this.sender,
    required this.body,
    required this.result,
    required this.timestamp,
    this.isMistake = false,
    this.reason,
    this.trustScore,
    this.isOTP,
    this.otpRisk,
    this.senderInfo,
  });

  /// Convert integer prediction to DetectionResult
  static DetectionResult fromPrediction(int prediction) {
    DetectionResult result;
    switch (prediction) {
      case 0:
        result = DetectionResult.legitimate;
        break;
      case 1:
        result = DetectionResult.spam;
        break;
      case 2:
        result = DetectionResult.fraudulent;
        break;
      default:
        result = DetectionResult.legitimate; // Safe default
    }
    // Logging removed to avoid console spam. Uncomment the lines below if
    // you ever need to debug class mapping again.
    // debugPrint('[fromPrediction] prediction: $prediction → $result');
    return result;
  }

  /// Get color for UI display
  Color get displayColor {
    switch (result) {
      case DetectionResult.legitimate:
        return Colors.green;
      case DetectionResult.spam:
        return Colors.orange;
      case DetectionResult.fraudulent:
        return Colors.red;
    }
  }

  /// Get icon for UI display
  IconData get displayIcon {
    switch (result) {
      case DetectionResult.legitimate:
        return Icons.check_circle;
      case DetectionResult.spam:
        return Icons.mark_email_read;
      case DetectionResult.fraudulent:
        return Icons.warning;
    }
  }

  /// Get human-readable result text
  String get resultText {
    switch (result) {
      case DetectionResult.legitimate:
        return 'Legitimate';
      case DetectionResult.spam:
        return 'Spam';
      case DetectionResult.fraudulent:
        return 'Fraudulent';
    }
  }

  /// Get advanced display information
  String get advancedDisplayText {
    final baseText = resultText;
    final parts = <String>[baseText];

    if (trustScore != null) {
      parts.add('Trust: ${(trustScore! * 100).toStringAsFixed(0)}%');
    }

    if (isOTP == true) {
      parts.add('OTP');
    }

    if (senderInfo?.isVerified == true) {
      parts.add('Verified');
    }

    return parts.join(' • ');
  }
}
