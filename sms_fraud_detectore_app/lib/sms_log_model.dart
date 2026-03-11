import 'package:flutter/material.dart';

// ── Detection result ──────────────────────────────────────────────────────

enum DetectionResult { legitimate, spam, fraudulent }

// ── Classification output (result + reason) ───────────────────────────────

class ClassificationOutput {
  final DetectionResult result;
  final String? reason; // rule key, e.g. 'phishing_link', 'prize_fraud'
  const ClassificationOutput(this.result, [this.reason]);
}

extension DetectionResultX on DetectionResult {
  String get label {
    switch (this) {
      case DetectionResult.legitimate: return 'Legitimate';
      case DetectionResult.spam:       return 'Spam';
      case DetectionResult.fraudulent: return 'Fraudulent';
    }
  }

  Color get color {
    switch (this) {
      case DetectionResult.legitimate: return Colors.green;
      case DetectionResult.spam:       return Colors.orange;
      case DetectionResult.fraudulent: return Colors.red;
    }
  }

  IconData get icon {
    switch (this) {
      case DetectionResult.legitimate: return Icons.check_circle;
      case DetectionResult.spam:       return Icons.mark_email_read;
      case DetectionResult.fraudulent: return Icons.warning_amber_rounded;
    }
  }
}

// ── SMS log entry ─────────────────────────────────────────────────────────

class SmsLogEntry {
  final String sender;
  final String body;
  final DetectionResult result;
  final String? reason;   // rule that triggered (e.g. 'phishing_link', 'prize_fraud')
  final DateTime timestamp;
  bool isMistake;

  SmsLogEntry({
    required this.sender,
    required this.body,
    required this.result,
    this.reason,
    required this.timestamp,
    this.isMistake = false,
  });
}

// ── Thread (grouped by sender) ────────────────────────────────────────────

class ThreadEntry {
  final String address;
  final List<SmsLogEntry> messages;

  ThreadEntry({required this.address, required this.messages});

  /// Most recent message
  SmsLogEntry get lastMessage => messages.first;

  /// Worst detection result in the thread (fraudulent > spam > legitimate)
  DetectionResult get worstResult {
    if (messages.any((m) => m.result == DetectionResult.fraudulent)) {
      return DetectionResult.fraudulent;
    }
    if (messages.any((m) => m.result == DetectionResult.spam)) {
      return DetectionResult.spam;
    }
    return DetectionResult.legitimate;
  }
}
