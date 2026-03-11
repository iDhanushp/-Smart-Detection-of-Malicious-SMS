import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'sms_log_state.dart';
import 'thread_list_page.dart';
import 'sms_permission_helper.dart';
import 'dart:async';

void main() {
  runZonedGuarded(
    () => runApp(
      ChangeNotifierProvider(
        create: (_) => SmsLogState(),
        child: const SmsFraudApp(),
      ),
    ),
    (e, st) => debugPrint('Unhandled: $e'),
    zoneSpecification: ZoneSpecification(
      print: (self, parent, zone, line) {
        if (line.startsWith('Column is')) return; // suppress telephony noise
        parent.print(zone, line);
      },
    ),
  );
}

class SmsFraudApp extends StatelessWidget {
  const SmsFraudApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SMS Fraud Detector',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: Colors.indigo,
        useMaterial3: true,
      ),
      darkTheme: ThemeData(
        colorSchemeSeed: Colors.indigo,
        brightness: Brightness.dark,
        useMaterial3: true,
      ),
      home: const AppRoot(),
    );
  }
}

class AppRoot extends StatefulWidget {
  const AppRoot({Key? key}) : super(key: key);
  @override
  State<AppRoot> createState() => _AppRootState();
}

class _AppRootState extends State<AppRoot> {
  bool _permGranted = false;
  bool _checking    = true;

  @override
  void initState() {
    super.initState();
    _boot();
  }

  Future<void> _boot() async {
    final granted = await SmsPermissionHelper.requestAll();
    if (!mounted) return;
    setState(() {
      _permGranted = granted;
      _checking    = false;
    });
    if (granted) {
      final state = context.read<SmsLogState>();
      await state.initialize();
      if (mounted) await state.syncDeviceSms();
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_checking) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }
    if (!_permGranted) {
      return Scaffold(
        body: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.sms_failed, size: 64, color: Colors.red),
              const SizedBox(height: 16),
              const Text('SMS permission required',
                  style: TextStyle(fontSize: 18)),
              const SizedBox(height: 12),
              ElevatedButton(
                onPressed: _boot,
                child: const Text('Grant Permission'),
              ),
            ],
          ),
        ),
      );
    }
    return const ThreadListPage();
  }
}
