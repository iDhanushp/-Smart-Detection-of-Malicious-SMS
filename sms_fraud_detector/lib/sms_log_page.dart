import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'sms_log_state.dart';
import 'widgets/sms_log_item.dart';

class SmsLogPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => SmsLogState()..initialize(),
      child: Consumer<SmsLogState>(
        builder: (context, state, _) {
          return Scaffold(
            appBar: AppBar(
              title: Text('SMS Fraud Detector'),
              actions: [
                Row(
                  children: [
                    Text('Detection'),
                    Switch(
                      value: state.detectionEnabled,
                      onChanged: (val) => state.toggleDetection(val),
                    ),
                  ],
                ),
                SizedBox(width: 8),
              ],
            ),
            body: state.log.isEmpty
                ? Center(child: Text('No SMS detected yet.'))
                : ListView.builder(
                    itemCount: state.log.length,
                    itemBuilder: (context, idx) {
                      final entry = state.log[idx];
                      return SmsLogItem(
                        entry: entry,
                        onMistake: entry.isMistake
                            ? null
                            : () => state.markAsMistake(idx),
                      );
                    },
                  ),
          );
        },
      ),
    );
  }
} 