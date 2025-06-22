import 'package:flutter/material.dart';

enum DetectionResult { legitimate, spam, fraudulent }

class SmsLogEntry {
  final String sender;
  final String body;
  final DetectionResult result;
  final DateTime timestamp;
  bool isMistake;
  final String? reason; // Why was it classified this way?

  SmsLogEntry({
    required this.sender,
    required this.body,
    required this.result,
    required this.timestamp,
    this.isMistake = false,
    this.reason,
  });

  /// Convert integer prediction to DetectionResult
  static DetectionResult fromPrediction(int prediction) {
    switch (prediction) {
      case 0:
        return DetectionResult.legitimate;
      case 1:
        return DetectionResult.spam;
      case 2:
        return DetectionResult.fraudulent;
      default:
        return DetectionResult.legitimate; // Safe default
    }
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
}
