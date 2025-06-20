import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:typed_data';

class FraudDetector {
  late Interpreter _interpreter;

  Future<void> loadModel(String assetPath) async {
    _interpreter = await Interpreter.fromAsset(assetPath);
  }

  /// Expects input as a List<double> (TF-IDF vector)
  /// Returns 0 (legitimate) or 1 (fraudulent)
  int predict(List<double> input) {
    // Model expects shape [1, N]
    final inputTensor = [input];
    var output = List.filled(1, 0).reshape([1, 1]);
    _interpreter.run(inputTensor, output);
    return output[0][0];
  }
} 