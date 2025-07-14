import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/services.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  print('🔍 Testing Advanced Fraud Detector Asset Loading...');
  
  // Test 1: Check if asset exists using rootBundle
  try {
    print('📁 Testing asset accessibility...');
    final data = await rootBundle.load('assets/advanced_fraud_detector.tflite');
    print('✅ Asset loaded successfully, size: ${data.lengthInBytes} bytes');
  } catch (e) {
    print('❌ Asset loading failed: $e');
    return;
  }
  
  // Test 2: Try different asset path variations
  final assetPaths = [
    'advanced_fraud_detector.tflite',
    'assets/advanced_fraud_detector.tflite',
  ];
  
  for (final path in assetPaths) {
    try {
      print('🔄 Testing path: $path');
      final interpreter = await Interpreter.fromAsset(path);
      print('✅ Successfully loaded TFLite model from: $path');
      print('📊 Input shape: ${interpreter.getInputTensor(0).shape}');
      print('📊 Output shape: ${interpreter.getOutputTensor(0).shape}');
      interpreter.close();
      break;
    } catch (e) {
      print('❌ Failed to load from $path: $e');
    }
  }
  
  // Test 3: Compare with working fraud_detector.tflite
  try {
    print('🔄 Testing working fraud_detector.tflite...');
    final workingInterpreter = await Interpreter.fromAsset('fraud_detector.tflite');
    print('✅ Working model loaded successfully');
    print('📊 Working model input shape: ${workingInterpreter.getInputTensor(0).shape}');
    print('📊 Working model output shape: ${workingInterpreter.getOutputTensor(0).shape}');
    workingInterpreter.close();
  } catch (e) {
    print('❌ Working model also failed: $e');
  }
  
  print('🏁 Test completed');
}
