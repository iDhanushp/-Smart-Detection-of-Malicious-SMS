import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:provider/provider.dart';
import 'sms_log_state.dart';
import 'sms_log_model.dart';
import 'fraud_detector.dart';
import 'tfidf_preprocessor.dart';
import 'sms_log_page.dart';
import 'theme_controller.dart';
import 'thread_list_page.dart';
import 'sms_permission_helper.dart';
// Advanced features imports
import 'theme/app_theme.dart';
import 'providers/theme_provider.dart';
import 'services/realtime_detection_service.dart';
import 'widgets/realtime_detection_dashboard.dart';
import 'dart:async';

void main() {
  // Run the whole app in a custom Zone so we can filter *print* statements
  // from noisy plugins (e.g. Telephony).
  runZonedGuarded(() {
    runApp(const MyApp());
  }, (error, stack) {
    debugPrint('Unhandled error: $error');
  }, zoneSpecification: ZoneSpecification(print: (self, parent, zone, line) {
    if (line.startsWith('Column is')) return; // suppress noisy cursor logs
    parent.print(zone, line);
  }));
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => SmsLogState()),
        ChangeNotifierProvider(create: (_) => ThemeController()),
        // Advanced providers
        ChangeNotifierProvider(create: (_) => ThemeProvider()),
        ChangeNotifierProvider(create: (_) => RealtimeDetectionService()),
      ],
      child: Consumer2<ThemeController, ThemeProvider>(
        builder: (_, themeCtrl, themeProvider, __) => MaterialApp(
          title: 'Advanced SMS Fraud Detector',
          themeMode:
              themeProvider.isDarkMode ? ThemeMode.dark : ThemeMode.light,
          theme: AppTheme.lightTheme,
          darkTheme: AppTheme.darkTheme,
          home: const HomePage(),
        ),
      ),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({Key? key}) : super(key: key);
  @override
  HomePageState createState() => HomePageState();
}

class HomePageState extends State<HomePage> {
  late FraudDetector _fraudDetector;
  late TfidfPreprocessor _preprocessor;
  bool _isModelLoaded = false;
  int _currentIndex = 0;

  final List<Widget> _pages = [];

  @override
  void initState() {
    super.initState();
    _initializeApp();
  }

  Future<void> _initializeApp() async {
    await SmsPermissionHelper.requestAll();

    // Initialize basic services
    _fraudDetector = FraudDetector();
    await _fraudDetector.loadModel('assets/fraud_detector.tflite');
    _preprocessor = TfidfPreprocessor();
    await _preprocessor.loadVocab('assets/tfidf_vocab.json');

    // Initialize real-time detection service
    final detectionService =
        Provider.of<RealtimeDetectionService>(context, listen: false);
    await detectionService.initialize();

    // Set up detection callbacks
    detectionService.onDetectionResult = (entry) => _onDetectionResult(entry);
    detectionService.onError = (error) => _onDetectionError(error);
    detectionService.onMonitoringStarted = () => _onMonitoringStarted();
    detectionService.onMonitoringStopped = () => _onMonitoringStopped();

    // Start real-time monitoring
    await detectionService.startMonitoring();

    // Sync device SMS
    await Provider.of<SmsLogState>(context, listen: false).syncDeviceSms();

    if (!mounted) return;
    setState(() {
      _isModelLoaded = true;
      _pages.addAll([
        ThreadListPage(),
        const RealtimeDetectionDashboard(),
        SmsLogPage(),
      ]);
    });

    if (!mounted) return;
    final messenger = ScaffoldMessenger.of(context);
    messenger.hideCurrentSnackBar();
    messenger.showSnackBar(
      SnackBar(
        content: Row(
          children: [
            Icon(Icons.security, color: Colors.white),
            SizedBox(width: 8),
            Expanded(
              child: Text(
                'Real-time SMS Fraud Detection is now active!',
                style:
                    TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
              ),
            ),
          ],
        ),
        backgroundColor: Colors.green,
        duration: Duration(seconds: 3),
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      ),
    );
  }

  void _onDetectionResult(SmsLogEntry entry) {
    if (!mounted) return;

    // Add to log state
    Provider.of<SmsLogState>(context, listen: false).addLogEntry(entry);

    // Show notification
    _showDetectionNotification(entry);
  }

  void _onDetectionError(String error) {
    if (!mounted) return;

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('Detection Error: $error'),
        backgroundColor: Colors.red,
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  void _onMonitoringStarted() {
    if (!mounted) return;

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('Real-time monitoring started'),
        backgroundColor: Colors.green,
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  void _onMonitoringStopped() {
    if (!mounted) return;

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('Real-time monitoring stopped'),
        backgroundColor: Colors.orange,
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  void _showDetectionNotification(SmsLogEntry entry) {
    final color = entry.displayColor;
    final icon = entry.displayIcon;
    final message = _getDetectionMessage(entry);

    final messenger = ScaffoldMessenger.of(context);
    messenger.hideCurrentSnackBar();
    messenger.showSnackBar(
      SnackBar(
        content: Row(
          children: [
            Icon(icon, color: Colors.white),
            SizedBox(width: 8),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    message,
                    style: TextStyle(
                        color: Colors.white, fontWeight: FontWeight.bold),
                  ),
                  if (entry.trustScore != null)
                    Text(
                      'Trust: ${(entry.trustScore! * 100).toStringAsFixed(0)}%',
                      style: TextStyle(color: Colors.white70, fontSize: 12),
                    ),
                ],
              ),
            ),
          ],
        ),
        backgroundColor: color,
        duration: Duration(seconds: 4),
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        action: SnackBarAction(
          label: 'Details',
          textColor: Colors.white,
          onPressed: () {
            setState(() {
              _currentIndex = 2; // Switch to logs tab
            });
          },
        ),
      ),
    );
  }

  String _getDetectionMessage(SmsLogEntry entry) {
    if (entry.result == DetectionResult.fraudulent) {
      return '🚨 Fraudulent SMS detected!';
    } else if (entry.result == DetectionResult.spam) {
      return '⚠️ Spam SMS detected!';
    } else if (entry.isOTP == true) {
      return '🔐 OTP detected - Stay alert!';
    } else {
      return '✅ Legitimate SMS';
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!_isModelLoaded || _pages.isEmpty) {
      return Scaffold(
        appBar: AppBar(
          title: Text('Advanced SMS Fraud Detector'),
          backgroundColor: Theme.of(context).colorScheme.primary,
          foregroundColor: Theme.of(context).colorScheme.onPrimary,
        ),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircularProgressIndicator(),
              SizedBox(height: 16),
              Text('Loading advanced AI model...'),
              SizedBox(height: 8),
              Text(
                'Initializing real-time protection...',
                style: TextStyle(fontSize: 12, color: Colors.grey),
              ),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: _pages,
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _currentIndex,
        onDestinationSelected: (index) {
          setState(() {
            _currentIndex = index;
          });
        },
        destinations: [
          NavigationDestination(
            icon: Icon(Icons.chat_outlined),
            selectedIcon: Icon(Icons.chat),
            label: 'Messages',
          ),
          NavigationDestination(
            icon: Icon(Icons.security_outlined),
            selectedIcon: Icon(Icons.security),
            label: 'Protection',
          ),
          NavigationDestination(
            icon: Icon(Icons.history_outlined),
            selectedIcon: Icon(Icons.history),
            label: 'Logs',
          ),
        ],
      ),
      floatingActionButton: _currentIndex == 0
          ? FloatingActionButton.extended(
              onPressed: () {
                // TODO: Implement new message composition
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                    content: Text('New message feature coming soon!'),
                    behavior: SnackBarBehavior.floating,
                  ),
                );
              },
              icon: Icon(Icons.edit),
              label: Text('New Message'),
            )
          : null,
      appBar: AppBar(
        title: Text('Advanced SMS Fraud Detector'),
        backgroundColor: Theme.of(context).colorScheme.primary,
        foregroundColor: Theme.of(context).colorScheme.onPrimary,
        actions: [
          // Theme toggle
          Consumer<ThemeProvider>(
            builder: (context, themeProvider, child) {
              return IconButton(
                icon: Icon(
                  themeProvider.isDarkMode ? Icons.light_mode : Icons.dark_mode,
                ),
                onPressed: () => themeProvider.toggleTheme(),
                tooltip: 'Toggle theme',
              );
            },
          ),
          // Sync button
          Consumer<SmsLogState>(
            builder: (context, state, child) {
              return IconButton(
                icon: state.isSyncing
                    ? SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          valueColor: AlwaysStoppedAnimation<Color>(
                            Theme.of(context).colorScheme.onPrimary,
                          ),
                        ),
                      )
                    : Icon(Icons.sync),
                onPressed: state.isSyncing ? null : () => state.syncDeviceSms(),
                tooltip: 'Sync SMS',
              );
            },
          ),
        ],
      ),
    );
  }
}

class DetectionDashboardPage extends StatefulWidget {
  const DetectionDashboardPage({Key? key}) : super(key: key);

  @override
  DetectionDashboardPageState createState() => DetectionDashboardPageState();
}

class DetectionDashboardPageState extends State<DetectionDashboardPage>
    with TickerProviderStateMixin {
  bool _isDetectionEnabled = true;
  late AnimationController _pulseController;
  late AnimationController _slideController;
  late Animation<double> _pulseAnimation;
  late Animation<Offset> _slideAnimation;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      duration: const Duration(seconds: 2),
      vsync: this,
    );
    _slideController = AnimationController(
      duration: const Duration(milliseconds: 800),
      vsync: this,
    );

    _pulseAnimation = Tween<double>(
      begin: 1.0,
      end: 1.1,
    ).animate(CurvedAnimation(
      parent: _pulseController,
      curve: Curves.easeInOut,
    ));

    _slideAnimation = Tween<Offset>(
      begin: const Offset(0, 0.3),
      end: Offset.zero,
    ).animate(CurvedAnimation(
      parent: _slideController,
      curve: Curves.easeOutCubic,
    ));

    _pulseController.repeat(reverse: true);
    _slideController.forward();
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _slideController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              Colors.blue.shade50,
              Colors.white,
            ],
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              // Header Section
              _buildHeader(),

              // Main Content
              Expanded(
                child: SlideTransition(
                  position: _slideAnimation,
                  child: SingleChildScrollView(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      children: [
                        // Status Card
                        _buildStatusCard(),
                        const SizedBox(height: 20),

                        // Statistics Cards
                        _buildStatisticsSection(),
                        const SizedBox(height: 20),

                        // Control Card
                        _buildControlCard(),
                        const SizedBox(height: 20),

                        // Recent Activity
                        _buildRecentActivitySection(),
                        const SizedBox(height: 30),

                        // Scan Button
                        _buildScanButton(),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.all(20),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [Colors.blue.shade400, Colors.blue.shade600],
              ),
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Icon(
              Icons.security,
              color: Colors.white,
              size: 24,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Fraud Detection',
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    color: Colors.black87,
                  ),
                ),
                Text(
                  'AI-powered SMS protection',
                  style: TextStyle(
                    fontSize: 14,
                    color: Colors.grey[600],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatusCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: _isDetectionEnabled
              ? [Colors.green.shade400, Colors.green.shade600]
              : [Colors.orange.shade400, Colors.orange.shade600],
        ),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: (_isDetectionEnabled ? Colors.green : Colors.orange)
                .withOpacity(0.3),
            blurRadius: 15,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Row(
        children: [
          AnimatedBuilder(
            animation: _pulseAnimation,
            builder: (context, child) {
              return Transform.scale(
                scale: _isDetectionEnabled ? _pulseAnimation.value : 1.0,
                child: Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.2),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Icon(
                    _isDetectionEnabled ? Icons.shield : Icons.shield_outlined,
                    color: Colors.white,
                    size: 32,
                  ),
                ),
              );
            },
          ),
          const SizedBox(width: 20),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _isDetectionEnabled
                      ? 'Protection Active'
                      : 'Protection Paused',
                  style: const TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  _isDetectionEnabled
                      ? 'Your device is protected from fraudulent SMS'
                      : 'Detection is currently disabled',
                  style: TextStyle(
                    fontSize: 14,
                    color: Colors.white.withOpacity(0.9),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatisticsSection() {
    return Consumer<SmsLogState>(
      builder: (context, smsLogState, child) {
        final logs = smsLogState.log;
        final totalSms = logs.length;
        final legitimateSms = logs
            .where((log) => log.result == DetectionResult.legitimate)
            .length;
        final spamSms =
            logs.where((log) => log.result == DetectionResult.spam).length;
        final fraudulentSms = logs
            .where((log) => log.result == DetectionResult.fraudulent)
            .length;
        final fraudPercentage =
            totalSms > 0 ? (fraudulentSms / totalSms * 100) : 0.0;
        final spamPercentage = totalSms > 0 ? (spamSms / totalSms * 100) : 0.0;

        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Detection Statistics',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: Colors.black87,
              ),
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: _buildStatCard(
                    'Total Messages',
                    totalSms.toString(),
                    Icons.message_rounded,
                    Colors.blue,
                    'All scanned SMS',
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: _buildStatCard(
                    'Safe',
                    legitimateSms.toString(),
                    Icons.check_circle_rounded,
                    Colors.green,
                    'Legitimate messages',
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: _buildStatCard(
                    'Spam',
                    spamSms.toString(),
                    Icons.mark_email_read_rounded,
                    Colors.orange,
                    'Promotional content',
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: _buildStatCard(
                    'Threats',
                    fraudulentSms.toString(),
                    Icons.warning_rounded,
                    Colors.red,
                    'Fraudulent detected',
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: _buildStatCard(
                    'Spam Level',
                    '${spamPercentage.toStringAsFixed(1)}%',
                    Icons.email_rounded,
                    spamPercentage > 20 ? Colors.orange : Colors.grey,
                    'Spam percentage',
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: _buildStatCard(
                    'Risk Level',
                    '${fraudPercentage.toStringAsFixed(1)}%',
                    Icons.analytics_rounded,
                    fraudPercentage > 10
                        ? Colors.red
                        : fraudPercentage > 5
                            ? Colors.orange
                            : Colors.green,
                    'Fraud percentage',
                  ),
                ),
              ],
            ),
          ],
        );
      },
    );
  }

  Widget _buildStatCard(
      String title, String value, IconData icon, Color color, String subtitle) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.1),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: color.withOpacity(0.1),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(
              icon,
              color: color,
              size: 24,
            ),
          ),
          const SizedBox(height: 12),
          AnimatedSwitcher(
            duration: const Duration(milliseconds: 300),
            transitionBuilder: (child, anim) =>
                ScaleTransition(scale: anim, child: child),
            child: Text(
              value,
              key: ValueKey(value),
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
          ),
          const SizedBox(height: 4),
          Text(
            title,
            style: const TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: Colors.black87,
            ),
          ),
          Text(
            subtitle,
            style: TextStyle(
              fontSize: 10,
              color: Colors.grey[600],
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  Widget _buildControlCard() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.1),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: Colors.blue.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Icon(
                  Icons.settings,
                  color: Colors.blue,
                  size: 20,
                ),
              ),
              const SizedBox(width: 12),
              const Text(
                'Detection Controls',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: Colors.black87,
                ),
              ),
            ],
          ),
          const SizedBox(height: 20),
          Row(
            children: [
              const Icon(
                Icons.shield,
                color: Colors.blue,
                size: 20,
              ),
              const SizedBox(width: 12),
              const Expanded(
                child: Text(
                  'Real-time Protection',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),
              Switch(
                value: _isDetectionEnabled,
                onChanged: (value) {
                  setState(() {
                    _isDetectionEnabled = value;
                  });
                  final messenger = ScaffoldMessenger.of(context);
                  messenger.hideCurrentSnackBar();
                  messenger.showSnackBar(
                    SnackBar(
                      content: Row(
                        children: [
                          Icon(
                            value ? Icons.check_circle : Icons.pause,
                            color: Colors.white,
                          ),
                          const SizedBox(width: 8),
                          Text(value
                              ? 'Protection enabled'
                              : 'Protection paused'),
                        ],
                      ),
                      backgroundColor: value ? Colors.green : Colors.orange,
                      behavior: SnackBarBehavior.floating,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10),
                      ),
                    ),
                  );
                },
                activeColor: Colors.green,
                activeTrackColor: Colors.green.withOpacity(0.3),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildRecentActivitySection() {
    return Consumer<SmsLogState>(
      builder: (context, smsLogState, child) {
        final recentLogs = smsLogState.log.take(3).toList();

        return Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(16),
            boxShadow: [
              BoxShadow(
                color: Colors.grey.withOpacity(0.1),
                blurRadius: 10,
                offset: const Offset(0, 4),
              ),
            ],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: Colors.purple.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: const Icon(
                      Icons.history,
                      color: Colors.purple,
                      size: 20,
                    ),
                  ),
                  const SizedBox(width: 12),
                  const Text(
                    'Recent Activity',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Colors.black87,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              if (recentLogs.isEmpty)
                Container(
                  padding: const EdgeInsets.all(20),
                  child: Column(
                    children: [
                      Icon(
                        Icons.inbox_outlined,
                        size: 48,
                        color: Colors.grey[400],
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'No messages scanned yet',
                        style: TextStyle(
                          fontSize: 16,
                          color: Colors.grey[600],
                        ),
                      ),
                    ],
                  ),
                )
              else
                ...recentLogs.map((log) => _buildActivityItem(log)),
            ],
          ),
        );
      },
    );
  }

  Widget _buildActivityItem(SmsLogEntry log) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: log.result == DetectionResult.fraudulent
            ? Colors.red.withOpacity(0.1)
            : log.result == DetectionResult.spam
                ? Colors.orange.withOpacity(0.1)
                : Colors.green.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: log.result == DetectionResult.fraudulent
              ? Colors.red.withOpacity(0.3)
              : log.result == DetectionResult.spam
                  ? Colors.orange.withOpacity(0.3)
                  : Colors.green.withOpacity(0.3),
        ),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: log.result == DetectionResult.fraudulent
                  ? Colors.red.withOpacity(0.2)
                  : log.result == DetectionResult.spam
                      ? Colors.orange.withOpacity(0.2)
                      : Colors.green.withOpacity(0.2),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Icon(
              log.displayIcon,
              color: log.displayColor,
              size: 16,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  log.sender,
                  style: const TextStyle(
                    fontWeight: FontWeight.w600,
                    fontSize: 14,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  log.body.length > 30
                      ? '${log.body.substring(0, 30)}...'
                      : log.body,
                  style: TextStyle(
                    fontSize: 12,
                    color: Colors.grey[600],
                  ),
                ),
                if (log.reason != null) ...[
                  const SizedBox(height: 4),
                  Text(
                    log.reason!,
                    style: TextStyle(
                      fontSize: 10,
                      color: Colors.grey[500],
                      fontStyle: FontStyle.italic,
                    ),
                  ),
                ],
              ],
            ),
          ),
          Text(
            _formatTime(log.timestamp),
            style: TextStyle(
              fontSize: 10,
              color: Colors.grey[500],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildScanButton() {
    return Container(
      width: double.infinity,
      height: 56,
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [Colors.blue.shade400, Colors.blue.shade600],
        ),
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.blue.withOpacity(0.3),
            blurRadius: 15,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: ElevatedButton.icon(
        onPressed: _scanDetection,
        icon: const Icon(Icons.scanner, color: Colors.white),
        label: const Text(
          'Scan All Messages',
          style: TextStyle(
            color: Colors.white,
            fontSize: 16,
            fontWeight: FontWeight.w600,
          ),
        ),
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.transparent,
          shadowColor: Colors.transparent,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
        ),
      ),
    );
  }

  String _formatTime(DateTime dt) {
    final now = DateTime.now();
    final difference = now.difference(dt);

    if (difference.inMinutes < 1) {
      return 'Now';
    } else if (difference.inMinutes < 60) {
      return '${difference.inMinutes}m ago';
    } else if (difference.inHours < 24) {
      return '${difference.inHours}h ago';
    } else {
      return '${difference.inDays}d ago';
    }
  }

  Future<void> _scanDetection() async {
    try {
      // Show scanning indicator
      final messenger = ScaffoldMessenger.of(context);
      messenger.hideCurrentSnackBar();
      messenger.showSnackBar(
        const SnackBar(
          content: Row(
            children: [
              SizedBox(
                width: 20,
                height: 20,
                child: CircularProgressIndicator(strokeWidth: 2),
              ),
              SizedBox(width: 16),
              Text('Scanning all SMS messages...'),
            ],
          ),
          duration: Duration(seconds: 2),
        ),
      );

      // Get the state and scan all SMS using the state's sync method
      final state = Provider.of<SmsLogState>(context, listen: false);
      await state.syncDeviceSms();

      // Show completion message
      if (mounted) {
        final logs = state.log;
        final totalSms = logs.length;
        final legitimateSms = logs
            .where((log) => log.result == DetectionResult.legitimate)
            .length;
        final spamSms =
            logs.where((log) => log.result == DetectionResult.spam).length;
        final fraudulentSms = logs
            .where((log) => log.result == DetectionResult.fraudulent)
            .length;

        messenger.hideCurrentSnackBar();
        messenger.showSnackBar(
          SnackBar(
            content: Text(
              'Scan complete! Found $totalSms messages ($legitimateSms legitimate, $spamSms spam, $fraudulentSms fraudulent)',
            ),
            backgroundColor: fraudulentSms > 0
                ? Colors.red
                : spamSms > 0
                    ? Colors.orange
                    : Colors.green,
            duration: const Duration(seconds: 3),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        final messenger = ScaffoldMessenger.of(context);
        messenger.hideCurrentSnackBar();
        messenger.showSnackBar(
          SnackBar(
            content: Text('Scan failed: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }
}
