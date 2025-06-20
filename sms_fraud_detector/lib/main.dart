import 'package:flutter/material.dart';
import 'sms_log_page.dart';

void main() {
  runApp(SmsFraudDetectorApp());
}

class SmsFraudDetectorApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SMS Fraud Detector',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: SmsLogPage(),
    );
  }
} 