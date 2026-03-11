import 'package:flutter/material.dart';
import 'package:telephony/telephony.dart';
import 'sms_log_model.dart';
import 'advanced_fraud_detector.dart';

class SmsLogState extends ChangeNotifier {
  final List<SmsLogEntry> _log = [];
  final _detector = AdvancedFraudDetector();

  bool _syncing    = false;
  bool _detectorOk = false;
  int  _progress   = 0;   // 0-100
  String _statusMsg = 'Not synced';

  List<SmsLogEntry> get log       => _log;
  bool get isSyncing              => _syncing;
  int  get progress               => _progress;
  String get statusMsg            => _statusMsg;

  // ── grouped by sender, newest first ──────────────────────────────────────
  List<ThreadEntry> get threads {
    final Map<String, List<SmsLogEntry>> map = {};
    for (final e in _log) {
      map.putIfAbsent(e.sender, () => []).add(e);
    }
    final list = map.entries
        .map((e) => ThreadEntry(address: e.key, messages: e.value))
        .toList()
      ..sort((a, b) =>
          b.lastMessage.timestamp.compareTo(a.lastMessage.timestamp));
    return list;
  }

  // ── stats ─────────────────────────────────────────────────────────────────
  int get countLegitimate =>
      _log.where((e) => e.result == DetectionResult.legitimate).length;
  int get countSpam =>
      _log.where((e) => e.result == DetectionResult.spam).length;
  int get countFraud =>
      _log.where((e) => e.result == DetectionResult.fraudulent).length;

  /// Reason counts for all flagged (spam + fraud) messages, sorted descending.
  List<MapEntry<String, int>> get reasonCounts {
    final map = <String, int>{};
    for (final e in _log) {
      if (e.reason != null) {
        map[e.reason!] = (map[e.reason!] ?? 0) + 1;
      }
    }
    final sorted = map.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    return sorted;
  }

  // ── init ──────────────────────────────────────────────────────────────────
  Future<void> initialize() async {
    try {
      await _detector.initialize();
      _detectorOk = true;
      debugPrint('✅ SmsLogState: detector ready');
    } catch (e) {
      _detectorOk = false;
      debugPrint('❌ SmsLogState: detector init failed: $e');
    }
  }

  // ── sync ──────────────────────────────────────────────────────────────────
  Future<void> syncDeviceSms() async {
    if (_syncing) return;

    // Try to init the detector, but do NOT abort if it fails.
    // classify() handles _ready=false gracefully via rule-based fallback.
    if (!_detectorOk) {
      await initialize();
      if (!_detectorOk) {
        debugPrint('⚠️  Detector unavailable — proceeding with rule-based classification only');
      }
    }

    _syncing   = true;
    _progress  = 0;
    _statusMsg = 'Loading messages…';
    notifyListeners();

    try {
      final telephony = Telephony.instance;
      final msgs = await telephony.getInboxSms(columns: [
        SmsColumn.ADDRESS,
        SmsColumn.BODY,
        SmsColumn.DATE,
        SmsColumn.ID,
      ]);

      debugPrint('📱 Found ${msgs.length} SMS messages (will process up to 500 most recent)');

      // Sort newest first
      msgs.sort((a, b) {
        final ta = _toDateTime(a.date);
        final tb = _toDateTime(b.date);
        return tb.compareTo(ta);
      });

      // Cap at 500 most-recent to prevent ANR (1625 × ~3ms TFLite = 5s+ on main thread)
      final batch = msgs.length > 500 ? msgs.sublist(0, 500) : msgs;

      _log.clear();

      final total = batch.length;
      int done = 0;

      for (final sms in batch) {
        final sender = sms.address ?? 'Unknown';
        final body   = sms.body   ?? '';

        // ── THE ONLY CLASSIFICATION CALL ──────────────────────────────────
        final classified = _detector.classify(sender, body);
        // ─────────────────────────────────────────────────────────────────

        _log.add(SmsLogEntry(
          sender:    sender,
          body:      body,
          result:    classified.result,
          reason:    classified.reason,
          timestamp: _toDateTime(sms.date),
        ));

        done++;
        _progress = ((done / total) * 100).round();

        // Yield every 10 messages with a real delay to keep Android ANR watchdog happy
        // 500 msgs ÷ 10 = 50 yields × 4ms = 200ms overhead, well under 5s ANR limit
        if (done % 10 == 0 || done == total) {
          _statusMsg = 'Processing… $done / $total';
          notifyListeners();
          await Future.delayed(const Duration(milliseconds: 4));
        }
      }

      debugPrint('📊 Done: ${countLegitimate} legit, '
          '${countSpam} spam, ${countFraud} fraud');

      // Debug: show reason breakdown so we can verify it's populated
      final rc = reasonCounts;
      debugPrint('📋 Reason counts (${rc.length} categories): '
          '${rc.map((e) => "${e.key}:${e.value}").join(", ")}');

      _statusMsg = 'Synced ${_log.length} messages';
    } catch (e) {
      debugPrint('❌ syncDeviceSms error: $e');
      _statusMsg = 'Sync failed';
    } finally {
      _syncing = false;
      _progress = 100;
      notifyListeners();
    }
  }

  // ── helpers ───────────────────────────────────────────────────────────────
  DateTime _toDateTime(dynamic val) {
    if (val == null) return DateTime.now();
    if (val is DateTime) return val;
    if (val is int) return DateTime.fromMillisecondsSinceEpoch(val);
    return DateTime.now();
  }

  void markMistake(int index) {
    if (index >= 0 && index < _log.length) {
      _log[index].isMistake = true;
      notifyListeners();
    }
  }

  @override
  void dispose() {
    _detector.dispose();
    super.dispose();
  }
}
