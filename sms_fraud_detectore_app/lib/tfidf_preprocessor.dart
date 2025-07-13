import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/services.dart' show rootBundle;

class TfidfPreprocessor {
  late Map<String, int> vocabulary;
  late List<double> idf;
  late Set<String> stopWords;
  static const int expectedFeatures = 3000;

  TfidfPreprocessor();

  Future<void> loadVocab(String assetPath) async {
    String jsonStr = await rootBundle.loadString(assetPath);
    final data = json.decode(jsonStr);
    vocabulary = Map<String, int>.from(data['vocabulary']);
    idf = List<double>.from(data['idf'].map((x) => x.toDouble()));
    stopWords = Set<String>.from(data['stop_words'] ?? []);

    // Validate that we have the expected number of features
    if (vocabulary.length != expectedFeatures ||
        idf.length != expectedFeatures) {
      throw Exception(
          'Vocabulary size mismatch. Expected $expectedFeatures, got ${vocabulary.length}');
    }
  }

  List<double> transform(String text) {
    // Clean text: lowercase, remove punctuation, remove stopwords
    final cleaned = _cleanText(text);
    final words = cleaned.split(' ');
    final wordCounts = <String, int>{};
    for (var w in words) {
      if (w.isEmpty || stopWords.contains(w)) continue;
      wordCounts[w] = (wordCounts[w] ?? 0) + 1;
    }

    // Create vector with exact size expected by model
    final vector = List<double>.filled(expectedFeatures, 0.0);
    final totalWords = words.length;
    if (totalWords == 0) return vector;

    // Build raw TF-IDF vector
    vocabulary.forEach((word, idx) {
      if (idx < expectedFeatures) {
        // Safety check
        final tf = (wordCounts[word] ?? 0) / totalWords;
        final idfVal = idf[idx];
        vector[idx] = tf * idfVal;
      }
    });

    // IMPORTANT: scikit-learn applies L2 normalisation to each row by default (norm='l2').
    // Without this step the magnitudes are inconsistent with the weights learnt by the
    // logistic-regression model, causing the classifier to bias towards the majority class.
    double l2 = 0.0;
    for (final v in vector) {
      l2 += v * v;
    }
    l2 = math.sqrt(l2);
    if (l2 > 0) {
      for (int i = 0; i < vector.length; i++) {
        vector[i] /= l2;
      }
    }

    return vector;
  }

  String _cleanText(String text) {
    final lower = text.toLowerCase();
    final noPunct = lower.replaceAll(RegExp(r'[^a-z0-9\s]'), '');
    return noPunct;
  }
}
