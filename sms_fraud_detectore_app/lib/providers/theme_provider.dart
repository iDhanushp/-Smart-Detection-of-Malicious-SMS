import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../theme/app_theme.dart';

/// Provider for managing app theme state (dark/light mode)
class ThemeProvider extends ChangeNotifier {
  static const String _themeKey = 'theme_mode';

  bool _isDarkMode = false;
  bool _isSystemTheme = true;

  /// Current dark mode state
  bool get isDarkMode => _isDarkMode;

  /// Whether to follow system theme
  bool get isSystemTheme => _isSystemTheme;

  /// Current theme data
  ThemeData get theme => _isDarkMode ? AppTheme.darkTheme : AppTheme.lightTheme;

  /// Initialize theme provider
  ThemeProvider() {
    _loadThemePreference();
  }

  /// Load saved theme preference
  Future<void> _loadThemePreference() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      _isSystemTheme = prefs.getBool('${_themeKey}_system') ?? true;

      if (_isSystemTheme) {
        // Use system theme
        _isDarkMode =
            WidgetsBinding.instance.platformDispatcher.platformBrightness ==
                Brightness.dark;
      } else {
        // Use saved preference
        _isDarkMode = prefs.getBool(_themeKey) ?? false;
      }

      notifyListeners();
    } catch (e) {
      debugPrint('Error loading theme preference: $e');
    }
  }

  /// Toggle between dark and light mode
  Future<void> toggleTheme() async {
    _isDarkMode = !_isDarkMode;
    _isSystemTheme = false;

    await _saveThemePreference();
    notifyListeners();
  }

  /// Set specific theme mode
  Future<void> setThemeMode(bool isDark) async {
    _isDarkMode = isDark;
    _isSystemTheme = false;

    await _saveThemePreference();
    notifyListeners();
  }

  /// Enable system theme following
  Future<void> enableSystemTheme() async {
    _isSystemTheme = true;
    _isDarkMode =
        WidgetsBinding.instance.platformDispatcher.platformBrightness ==
            Brightness.dark;

    await _saveThemePreference();
    notifyListeners();
  }

  /// Disable system theme following
  Future<void> disableSystemTheme() async {
    _isSystemTheme = false;
    await _saveThemePreference();
  }

  /// Save theme preference to storage
  Future<void> _saveThemePreference() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setBool(_themeKey, _isDarkMode);
      await prefs.setBool('${_themeKey}_system', _isSystemTheme);
    } catch (e) {
      debugPrint('Error saving theme preference: $e');
    }
  }

  /// Update theme based on system changes
  void updateSystemTheme() {
    if (_isSystemTheme) {
      final isSystemDark =
          WidgetsBinding.instance.platformDispatcher.platformBrightness ==
              Brightness.dark;
      if (_isDarkMode != isSystemDark) {
        _isDarkMode = isSystemDark;
        notifyListeners();
      }
    }
  }

  /// Get theme mode description
  String get themeModeDescription {
    if (_isSystemTheme) {
      return 'System';
    }
    return _isDarkMode ? 'Dark' : 'Light';
  }

  /// Get theme icon
  IconData get themeIcon {
    if (_isSystemTheme) {
      return Icons.brightness_auto;
    }
    return _isDarkMode ? Icons.dark_mode : Icons.light_mode;
  }
}
