import 'package:flutter/material.dart';

/// Controls the app ThemeMode (system / light / dark) and notifies listeners.
class ThemeController extends ChangeNotifier {
  ThemeMode _mode = ThemeMode.system;

  ThemeMode get mode => _mode;

  /// Cycles through System → Light → Dark → System …
  void toggle() {
    switch (_mode) {
      case ThemeMode.system:
        _mode = ThemeMode.light;
        break;
      case ThemeMode.light:
        _mode = ThemeMode.dark;
        break;
      case ThemeMode.dark:
        _mode = ThemeMode.system;
        break;
    }
    notifyListeners();
  }
}
