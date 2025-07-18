import 'package:flutter/foundation.dart';
import 'lib/full_dataset_fraud_detector.dart';
import 'lib/advanced_fraud_detector.dart';
import 'lib/services/realtime_detection_service.dart';

/// Test script to verify Full Dataset Fraud Detector integration
/// 
/// This script tests:
/// 1. Full Dataset Detector initialization
/// 2. Advanced Detector fallback
/// 3. Dual detection service functionality
/// 4. Sample SMS classification with both models
void main() async {
  print('🚀 Testing Full Dataset Fraud Detector Integration...\n');

  // Test 1: Full Dataset Detector Standalone
  print('📊 Test 1: Full Dataset Detector Initialization');
  final fullDatasetDetector = FullDatasetFraudDetector();
  
  try {
    await fullDatasetDetector.initialize();
    print('✅ Full Dataset Detector initialized successfully');
    print('📈 Model info: ${fullDatasetDetector.modelInfo}');
    
    // Test sample classifications
    final testMessages = [
      {'sender': 'MGLAMM', 'body': 'Flash sale! 70% off on your favorite brands. Shop now!'},
      {'sender': 'AX-HDFCBK', 'body': 'Your account has been credited with Rs.1000. Thank you.'},
      {'sender': '+917890123456', 'body': 'Call 08712402972 immediately for urgent prize collection'},
      {'sender': 'KOTAKB', 'body': 'OTP for transaction: 123456. Do not share with anyone.'},
      {'sender': 'MYNTRA', 'body': 'Congratulations! You won Rs.50000 cash prize. Click here immediately'},
    ];
    
    print('\n🔍 Sample Classifications:');
    for (var msg in testMessages) {
      final result = await fullDatasetDetector.classifyMessage(msg['sender']!, msg['body']!);
      print('📱 "${msg['sender']}" → ${result['classification']} (${(result['confidence'] * 100).toStringAsFixed(1)}%)');
      print('   💭 ${result['reasoning'].take(2).join('; ')}');
      print('   ⏱️ Processing: ${result['processingTime']}ms\n');
    }
    
  } catch (e) {
    print('❌ Full Dataset Detector failed: $e');
  }

  // Test 2: Advanced Detector Fallback
  print('\n📊 Test 2: Advanced Detector Fallback');
  final advancedDetector = AdvancedFraudDetector();
  
  try {
    await advancedDetector.initialize();
    print('✅ Advanced Detector initialized successfully');
    
    // Quick test
    final result = advancedDetector.classifyWithBehavioralAnalysis('MGLAMM', 'Flash sale! 70% off');
    print('📱 Advanced result: ${result['classification']} (${(result['confidence'] * 100).toStringAsFixed(1)}%)');
    
  } catch (e) {
    print('❌ Advanced Detector failed: $e');
  }

  // Test 3: Dual Detection Service
  print('\n📊 Test 3: Dual Detection Service');
  final realtimeService = RealtimeDetectionService();
  
  try {
    final initialized = await realtimeService.initialize();
    if (initialized) {
      print('✅ Realtime Service initialized successfully');
      print('🎯 Primary detector: ${realtimeService.primaryDetector}');
      print('📊 Using full dataset: ${realtimeService.useFullDatasetDetector}');
    } else {
      print('❌ Realtime Service initialization failed');
    }
  } catch (e) {
    print('❌ Realtime Service error: $e');
  }

  print('\n🎉 Integration test completed!');
  print('📋 Next steps:');
  print('   1. Copy full_dataset_3class_fraud_detector.tflite to Flutter assets');
  print('   2. Run flutter pub get to update dependencies');
  print('   3. Test on device with real SMS messages');
  print('   4. Monitor performance and accuracy improvements');
}
