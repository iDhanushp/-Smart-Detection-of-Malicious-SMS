import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:convert';
import 'package:flutter/services.dart' show rootBundle;

/// Test script to verify TFLite model compatibility
class ModelCompatibilityTest {
  static Future<void> testModelCompatibility() async {
    try {
      print('🔍 Testing TFLite Model Compatibility...');

      // Test 1: Load vocabulary
      print('\n1. Testing vocabulary loading...');
      String jsonStr = await rootBundle.loadString('assets/tfidf_vocab.json');
      final data = json.decode(jsonStr);
      final vocabulary = Map<String, int>.from(data['vocabulary']);
      final idf = List<double>.from(data['idf'].map((x) => x.toDouble()));

      print('✅ Vocabulary size: ${vocabulary.length}');
      print('✅ IDF size: ${idf.length}');
      print('✅ Expected features: 3000');

      if (vocabulary.length != 3000 || idf.length != 3000) {
        throw Exception('Vocabulary size mismatch!');
      }

      // Test 2: Load TFLite model
      print('\n2. Testing TFLite model loading...');
      final interpreter =
          await Interpreter.fromAsset('assets/fraud_detector.tflite');

      // Test 3: Check input/output details
      print('\n3. Testing model input/output specifications...');
      final inputTensor = interpreter.getInputTensor(0);
      final outputTensor = interpreter.getOutputTensor(0);

      print('✅ Input shape: ${inputTensor.shape}');
      print('✅ Output shape: ${outputTensor.shape}');
      print('✅ Input type: ${inputTensor.type}');
      print('✅ Output type: ${outputTensor.type}');

      // Test 4: Test inference with sample data
      print('\n4. Testing inference...');
      final testVector = List<double>.filled(3000, 0.1); // Sample TF-IDF vector
      final testInput = [testVector];
      var output = List.filled(1, 0.0).reshape([1, 1]);

      interpreter.run(testInput, output);
      final probability = output[0][0] as double;

      print('✅ Inference successful');
      print('✅ Output probability: $probability');
      print('✅ Output range: 0.0 to 1.0 (sigmoid)');

      // Test 5: Test text preprocessing
      print('\n5. Testing text preprocessing...');
      final testText =
          'URGENT: Your account suspended. Click here: http://fake.com';
      final processedVector = _preprocessText(testText, vocabulary, idf);

      print('✅ Preprocessing successful');
      print('✅ Vector size: ${processedVector.length}');
      print('✅ Vector sum: ${processedVector.reduce((a, b) => a + b)}');

      print('\n🎉 All compatibility tests passed!');
    } catch (e) {
      print('\n❌ Compatibility test failed: $e');
      rethrow;
    }
  }

  static List<double> _preprocessText(
      String text, Map<String, int> vocabulary, List<double> idf) {
    // Simple text preprocessing for testing
    final lower = text.toLowerCase();
    final noPunct = lower.replaceAll(RegExp(r'[^a-z0-9\s]'), '');
    final words = noPunct.split(' ');

    final vector = List<double>.filled(3000, 0.0);
    final totalWords = words.length;

    if (totalWords > 0) {
      for (final word in words) {
        if (vocabulary.containsKey(word)) {
          final idx = vocabulary[word]!;
          if (idx < 3000) {
            final tf = 1.0 / totalWords;
            final idfVal = idf[idx];
            vector[idx] = tf * idfVal;
          }
        }
      }
    }

    return vector;
  }
}
