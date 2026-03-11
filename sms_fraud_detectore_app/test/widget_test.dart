// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:sms_fraud_detectore_app/main.dart';

void main() {
  testWidgets('App smoke test', (WidgetTester tester) async {
    // Just verify the app starts without crashing
    await tester.pumpWidget(const SmsFraudApp());

    expect(find.byType(MaterialApp), findsOneWidget);
  });
}
