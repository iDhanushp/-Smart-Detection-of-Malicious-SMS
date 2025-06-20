enum DetectionResult { legitimate, fraudulent }

class SmsLogEntry {
  final String sender;
  final String body;
  final DetectionResult result;
  bool isMistake;
  final DateTime timestamp;

  SmsLogEntry({
    required this.sender,
    required this.body,
    required this.result,
    this.isMistake = false,
    required this.timestamp,
  });
} 