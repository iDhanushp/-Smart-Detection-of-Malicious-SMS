import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'sms_log_state.dart';
import 'sms_log_model.dart';
import 'fraud_detector.dart';
import 'tfidf_preprocessor.dart';
import 'sms_log_page.dart';
import 'theme_controller.dart';
import 'thread_list_page.dart';
import 'sms_permission_helper.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => SmsLogState()),
        ChangeNotifierProvider(create: (_) => ThemeController()),
      ],
      child: Consumer<ThemeController>(
        builder: (_, themeCtrl, __) => MaterialApp(
          title: 'SMS Fraud Detector',
          themeMode: themeCtrl.mode,
          theme: ThemeData.light(useMaterial3: true).copyWith(
            appBarTheme: const AppBarTheme(
              backgroundColor: Colors.blue,
              foregroundColor: Colors.white,
              elevation: 0,
            ),
            floatingActionButtonTheme: const FloatingActionButtonThemeData(
              backgroundColor: Colors.blue,
              foregroundColor: Colors.white,
            ),
          ),
          darkTheme: ThemeData.dark(useMaterial3: true).copyWith(
            appBarTheme: const AppBarTheme(
              backgroundColor: Colors.blue,
              foregroundColor: Colors.white,
              elevation: 0,
            ),
            floatingActionButtonTheme: const FloatingActionButtonThemeData(
              backgroundColor: Colors.blue,
              foregroundColor: Colors.white,
            ),
            snackBarTheme:
                const SnackBarThemeData(backgroundColor: Colors.black87),
          ),
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
  bool _isDetectionEnabled = true;
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
    _fraudDetector = FraudDetector();
    await _fraudDetector.loadModel('assets/fraud_detector.tflite');
    _preprocessor = TfidfPreprocessor();
    await _preprocessor.loadVocab('assets/tfidf_vocab.json');
    // Sync device SMS
    await Provider.of<SmsLogState>(context, listen: false).syncDeviceSms();

    if (!mounted) return;
    setState(() {
      _isModelLoaded = true;
      _pages.addAll([
        ThreadListPage(),
        const DetectionDashboardPage(),
        SmsLogPage(),
      ]);
    });

    if (!mounted) return;
    final messenger = ScaffoldMessenger.of(context);
    messenger.hideCurrentSnackBar();
    messenger.showSnackBar(
      const SnackBar(
        content: Text('SMS Fraud Detection is now active!'),
        backgroundColor: Colors.green,
        duration: Duration(seconds: 2),
      ),
    );
  }

  void _onSmsReceived(String sender, String body) {
    if (!_isDetectionEnabled || !_isModelLoaded) return;
    try {
      final features = _preprocessor.transform(body);
      final result = _fraudDetector.predictWithReasoning(sender, features);
      final prediction = result['prediction'] as int;
      final reason = result['reason'] as String;
      final detectionResult = SmsLogEntry.fromPrediction(prediction);
      final smsLog = SmsLogEntry(
        sender: sender,
        body: body,
        result: detectionResult,
        timestamp: DateTime.now(),
        reason: reason,
      );
      Provider.of<SmsLogState>(context, listen: false).addLogEntry(smsLog);
      _showDetectionResult(smsLog);
    } catch (e) {
      // ignore: avoid_print
      print('Error processing SMS: $e');
    }
  }

  void _showDetectionResult(SmsLogEntry smsLog) {
    final color = smsLog.displayColor;
    final icon = smsLog.displayIcon;
    final message = smsLog.result == DetectionResult.fraudulent
        ? 'Fraudulent SMS detected!'
        : smsLog.result == DetectionResult.spam
            ? 'Spam SMS detected!'
            : 'Legitimate SMS';
    final messenger = ScaffoldMessenger.of(context);
    messenger.hideCurrentSnackBar();
    messenger.showSnackBar(
      SnackBar(
        content: Row(
          children: [
            Icon(icon, color: Colors.white),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                message,
                style: const TextStyle(
                    color: Colors.white, fontWeight: FontWeight.bold),
              ),
            ),
          ],
        ),
        backgroundColor: color,
        duration: const Duration(seconds: 3),
        action: SnackBarAction(
          label: 'View',
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

  @override
  Widget build(BuildContext context) {
    if (!_isModelLoaded || _pages.isEmpty) {
      return Scaffold(
        appBar: AppBar(
          title: const Text('SMS Fraud Detector'),
          backgroundColor: Colors.blue,
          foregroundColor: Colors.white,
        ),
        body: const Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircularProgressIndicator(),
              SizedBox(height: 16),
              Text('Loading AI model...'),
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
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: (index) {
          setState(() {
            _currentIndex = index;
          });
        },
        type: BottomNavigationBarType.fixed,
        selectedItemColor: Colors.blue,
        unselectedItemColor: Colors.grey,
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.chat),
            label: 'Messages',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.security),
            label: 'Detection',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.history),
            label: 'Logs',
          ),
        ],
      ),
      floatingActionButton: _currentIndex == 0
          ? FloatingActionButton(
              onPressed: () {
                // TODO: Implement new message composition
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                    content: Text('New message feature coming soon!'),
                  ),
                );
              },
              child: const Icon(Icons.edit),
            )
          : null,
      appBar: AppBar(
        title: const Text('SMS Fraud Detector'),
        backgroundColor: Colors.blue,
        foregroundColor: Colors.white,
        actions: [
          Consumer<SmsLogState>(
            builder: (context, state, child) {
              return IconButton(
                icon: state.isSyncing
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          valueColor:
                              AlwaysStoppedAnimation<Color>(Colors.white),
                        ),
                      )
                    : const Icon(Icons.sync),
                tooltip: 'Sync SMS',
                onPressed: state.isSyncing
                    ? null
                    : () async {
                        try {
                          await state.syncDeviceSms();
                          if (mounted) {
                            ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(
                                content:
                                    Text('Device SMS synced successfully!'),
                                backgroundColor: Colors.green,
                              ),
                            );
                          }
                        } catch (e) {
                          if (mounted) {
                            ScaffoldMessenger.of(context).showSnackBar(
                              SnackBar(
                                content: Text('Sync failed: $e'),
                                backgroundColor: Colors.red,
                              ),
                            );
                          }
                        }
                      },
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
