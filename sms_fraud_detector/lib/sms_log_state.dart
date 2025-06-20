import 'package:flutter/material.dart';
import 'sms_log_model.dart';
import 'tfidf_preprocessor.dart';
import 'fraud_detector.dart';
import 'sms_receiver.dart';

class SmsLogState extends ChangeNotifier {
  final List<SmsLogEntry> _log = [];
  bool detectionEnabled = true;

  late TfidfPreprocessor preprocessor;
  late FraudDetector detector;
  late SmsReceiver receiver;
  bool _initialized = false;

  List<SmsLogEntry> get log => List.unmodifiable(_log);

  Future<void> initialize() async {
    if (_initialized) return;
    preprocessor = TfidfPreprocessor();
    await preprocessor.loadVocab('assets/tfidf_vocab.json');
    detector = FraudDetector();
    await detector.loadModel('assets/fraud_detector.tflite');
    receiver = SmsReceiver();
    receiver.startListening(_onSmsReceived);
    _initialized = true;
  }

  void _onSmsReceived(String sender, String body) {
    if (!detectionEnabled) return;
    final vector = preprocessor.transform(body);
    final pred = detector.predict(vector);
    final result = pred == 1 ? DetectionResult.fraudulent : DetectionResult.legitimate;
    final entry = SmsLogEntry(
      sender: sender,
      body: body,
      result: result,
      timestamp: DateTime.now(),
    );
    _log.insert(0, entry);
    notifyListeners();
  }

  void toggleDetection(bool enabled) {
    detectionEnabled = enabled;
    notifyListeners();
  }

  void markAsMistake(int index) {
    if (index >= 0 && index < _log.length) {
      _log[index].isMistake = true;
      notifyListeners();
    }
  }
} 