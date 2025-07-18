import 'package:flutter/material.dart';
import 'package:telephony/telephony.dart';
import 'sms_log_model.dart';
import 'package:flutter/foundation.dart';
// import 'tfidf_preprocessor.dart';  // Commented out - old model
// import 'fraud_detector.dart';      // Commented out - old model
import 'advanced_fraud_detector.dart'; // Use advanced model for everything
import 'thread_models.dart';
import 'package:provider/provider.dart';
import 'services/realtime_detection_service.dart';

class SmsLogState extends ChangeNotifier {
  List<SmsLogEntry> _log = [];
  bool _isSyncing = false;
  bool detectionEnabled = true;
  int _syncProgress = 0; // Track sync progress (0-100)
  int _totalMessages = 0; // Total messages to process
  int _processedMessages = 0; // Messages processed so far

  // Use AdvancedFraudDetector for everything now
  late AdvancedFraudDetector detector;
  bool _initialized = false;

  List<SmsLogEntry> get log => _log;
  bool get isSyncing => _isSyncing;
  int get syncProgress => _syncProgress;
  int get totalMessages => _totalMessages;
  int get processedMessages => _processedMessages;

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
    // Initialize AdvancedFraudDetector for bulk processing
    detector = AdvancedFraudDetector();
    await detector.initialize();
    _initialized = true;
  }

  // Note: SMS processing now handled by RealtimeDetectionService  
  // using AdvancedFraudDetector for both real-time and bulk processing

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
    print(
        '[addLogEntry] sender: ${entry.sender}, result: ${entry.result}, reason: ${entry.reason}');
    _log.insert(0, entry);
    notifyListeners();
  }

  // Call this on app startup or manual sync
  Future<void> syncDeviceSms(BuildContext context) async {
    try {
      // Ensure detector and preprocessor are initialized
      if (!_initialized) {
        await initialize();
      }

      // Temporarily silence the Telephony plugin's noisy "Column is ..." prints
      final DebugPrintCallback originalDebugPrint = debugPrint;
      debugPrint = (String? message, {int? wrapWidth}) {};

      _isSyncing = true;
      notifyListeners();

      Telephony telephony = Telephony.instance;
      // Fetch only the columns we actually need; reduces internal logging too
      List<SmsMessage> messages = await telephony.getInboxSms(
        columns: [
          SmsColumn.ADDRESS,
          SmsColumn.BODY,
          SmsColumn.DATE,
          SmsColumn.ID,
        ],
      );

      print('📱 Found ${messages.length} SMS messages to process (processing ALL messages in batches)');
      
      // Initialize progress tracking
      _totalMessages = messages.length;
      _processedMessages = 0;
      _syncProgress = 0;
      notifyListeners();

      // Convert Object dates to DateTime and sort with null safety
      messages.sort((a, b) {
        DateTime dateA;
        DateTime dateB;
        
        // Handle a.date safely
        if (a.date != null) {
          if (a.date is DateTime) {
            dateA = a.date as DateTime;
          } else if (a.date is int) {
            dateA = DateTime.fromMillisecondsSinceEpoch(a.date as int);
          } else {
            dateA = DateTime.now();
          }
        } else {
          dateA = DateTime.now();
        }
        
        // Handle b.date safely
        if (b.date != null) {
          if (b.date is DateTime) {
            dateB = b.date as DateTime;
          } else if (b.date is int) {
            dateB = DateTime.fromMillisecondsSinceEpoch(b.date as int);
          } else {
            dateB = DateTime.now();
          }
        } else {
          dateB = DateTime.now();
        }
        
        return dateB.compareTo(dateA); // Most recent first
      });

      // Process ALL messages but in batches to avoid blocking the UI

      final List<SmsLogEntry> newLog = [];
      int processed = 0;
      const int batchSize = 100; // Process 100 messages at a time
      final int totalMessages = messages.length;
      
      print('📱 Starting batch processing: $totalMessages messages in batches of $batchSize');
      
      for (int i = 0; i < totalMessages; i += batchSize) {
        final int endIndex = (i + batchSize < totalMessages) ? i + batchSize : totalMessages;
        final List<SmsMessage> batch = messages.sublist(i, endIndex);
        
        print('📱 Processing batch ${(i ~/ batchSize) + 1}/${(totalMessages / batchSize).ceil()}: messages ${i + 1}-$endIndex');
        
        // Process current batch
        for (final sms in batch) {
          try {
            // Use AdvancedFraudDetector for behavioral analysis
            final result = detector.classifyWithBehavioralAnalysis(
                sms.address ?? 'Unknown', sms.body ?? '');
            
            // Convert string classification to integer prediction
            final classification = result['classification'] as String? ?? 'LEGITIMATE';
            int prediction;
            switch (classification) {
              case 'FRAUD':
                prediction = 2;
                break;
              case 'SPAM':
                prediction = 1;
                break;
              case 'LEGITIMATE':
              default:
                prediction = 0;
                break;
            }
            
            final reasoning = result['reasoning'] as List<String>? ?? ['No classification'];
            final reason = reasoning.join('; ');
            final detectionResult = SmsLogEntry.fromPrediction(prediction);
            
            // Handle timestamp with extra null safety
            DateTime timestamp;
            if (sms.date != null) {
              if (sms.date is DateTime) {
                timestamp = sms.date as DateTime;
              } else if (sms.date is int) {
                // Handle Unix timestamp (milliseconds)
                timestamp = DateTime.fromMillisecondsSinceEpoch(sms.date as int);
              } else {
                // Fallback for any other type
                timestamp = DateTime.now();
              }
            } else {
              timestamp = DateTime.now();
            }
            
            newLog.add(SmsLogEntry(
              sender: sms.address ?? 'Unknown',
              body: sms.body ?? '',
              result: detectionResult,
              timestamp: timestamp,
              reason: reason,
            ));
            
            processed++;
            _processedMessages = processed;
            _syncProgress = ((processed / totalMessages) * 100).round();
          } catch (e) {
            print('❌ Error processing SMS message: $e');
            // Skip this message and continue with the next one
            continue;
          }
        }
        
        // Update UI with progress after each batch
        print('📱 Batch completed: $processed/$totalMessages messages processed (${_syncProgress}%)');
        notifyListeners(); // Update UI with current progress
        
        // Allow UI to update and prevent blocking
        if (i + batchSize < totalMessages) {
          // Brief pause between batches to prevent UI blocking
          await Future.delayed(const Duration(milliseconds: 10));
          
          // Update UI with partial results every few batches
          if (((i ~/ batchSize) + 1) % 5 == 0) {
            // Update log with current progress every 5 batches (500 messages)
            _log = List.from(newLog);
            notifyListeners();
            print('📱 UI updated with ${newLog.length} messages so far...');
          }
        }
      }
      _log = newLog;
      _isSyncing = false;
      _syncProgress = 100; // Complete
      notifyListeners();

      print('✅ SMS batch processing completed: ${newLog.length} total messages processed');
      
      // Calculate and display statistics
      final legitimate = newLog.where((msg) => msg.result == DetectionResult.legitimate).length;
      final spam = newLog.where((msg) => msg.result == DetectionResult.spam).length;
      final fraud = newLog.where((msg) => msg.result == DetectionResult.fraudulent).length;
      
      print('📊 Classification results: $legitimate legitimate, $spam spam, $fraud fraudulent');

      // Update statistics in RealtimeDetectionService
      try {
        final detectionService =
            Provider.of<RealtimeDetectionService>(context, listen: false);
        detectionService.recalculateStatistics(_log);
      } catch (e) {
        debugPrint('Failed to update statistics: $e');
      }

      // Restore normal debugPrint behaviour
      debugPrint = originalDebugPrint;
    } catch (e) {
      print('❌ Error during SMS sync: $e');
      _isSyncing = false;
      _syncProgress = 0;
      _processedMessages = 0;
      _totalMessages = 0;
      notifyListeners();
      // Don't rethrow - let the app continue even if SMS sync fails
    }
  }
}
