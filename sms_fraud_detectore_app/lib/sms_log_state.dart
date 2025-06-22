import 'package:flutter/material.dart';
import 'package:telephony/telephony.dart';
import 'sms_log_model.dart';
import 'tfidf_preprocessor.dart';
import 'fraud_detector.dart';
import 'thread_models.dart';

class SmsLogState extends ChangeNotifier {
  List<SmsLogEntry> _log = [];
  bool _isSyncing = false;
  bool detectionEnabled = true;

  late TfidfPreprocessor preprocessor;
  late FraudDetector detector;
  bool _initialized = false;

  List<SmsLogEntry> get log => _log;
  bool get isSyncing => _isSyncing;

  /// Returns messages grouped by sender, newest first inside each thread
  List<ThreadEntry> get threads {
    final Map<String, List<SmsLogEntry>> grouped = {};
    for (final msg in _log) {
      grouped.putIfAbsent(msg.sender, () => []).add(msg);
    }
    final list = grouped.entries
        .map((e) => ThreadEntry(address: e.key, messages: e.value))
        .toList();
    list.sort(
        (a, b) => b.lastMessage.timestamp.compareTo(a.lastMessage.timestamp));
    return list;
  }

  Future<void> initialize() async {
    if (_initialized) return;
    preprocessor = TfidfPreprocessor();
    await preprocessor.loadVocab('assets/tfidf_vocab.json');
    detector = FraudDetector();
    await detector.loadModel('assets/fraud_detector.tflite');
    _initialized = true;
  }

  void _onSmsReceived(String sender, String body) {
    if (!detectionEnabled) return;
    final vector = preprocessor.transform(body);
    final result = detector.predictWithReasoning(sender, vector);
    final pred = result['prediction'] as int;
    final reason = result['reason'] as String;
    final detectionResult = SmsLogEntry.fromPrediction(pred);
    final entry = SmsLogEntry(
      sender: sender,
      body: body,
      result: detectionResult,
      timestamp: DateTime.now(),
      reason: reason,
    );
    _log.insert(0, entry);
    notifyListeners();
  }

  void _onSmsReceivedWithTs(String sender, String body, DateTime ts) {
    if (!detectionEnabled) return;
    final vector = preprocessor.transform(body);
    final result = detector.predictWithReasoning(sender, vector);
    final pred = result['prediction'] as int;
    final reason = result['reason'] as String;
    final detectionResult = SmsLogEntry.fromPrediction(pred);
    final entry = SmsLogEntry(
      sender: sender,
      body: body,
      result: detectionResult,
      timestamp: ts,
      reason: reason,
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

  void addLogEntry(SmsLogEntry entry) {
    _log.insert(0, entry);
    notifyListeners();
  }

  // Call this on app startup or manual sync
  Future<void> syncDeviceSms() async {
    // Ensure detector and preprocessor are initialized
    if (!_initialized) {
      await initialize();
    }

    _isSyncing = true;
    notifyListeners();

    Telephony telephony = Telephony.instance;
    List<SmsMessage> messages = await telephony.getInboxSms();

    // Convert Object dates to DateTime and sort
    messages.sort((a, b) {
      final dateA = a.date is DateTime ? a.date as DateTime : DateTime.now();
      final dateB = b.date is DateTime ? b.date as DateTime : DateTime.now();
      return dateA.compareTo(dateB);
    });

    final List<SmsLogEntry> newLog = [];
    for (final sms in messages) {
      final features = preprocessor.transform(sms.body ?? '');
      final result =
          detector.predictWithReasoning(sms.address ?? 'Unknown', features);
      final prediction = result['prediction'] as int;
      final reason = result['reason'] as String;
      final detectionResult = SmsLogEntry.fromPrediction(prediction);
      newLog.add(SmsLogEntry(
        sender: sms.address ?? 'Unknown',
        body: sms.body ?? '',
        result: detectionResult,
        timestamp: sms.date is DateTime ? sms.date as DateTime : DateTime.now(),
        reason: reason,
      ));
    }
    _log = newLog;
    notifyListeners();
    _isSyncing = false;
  }
}
