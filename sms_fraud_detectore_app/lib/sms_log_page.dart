import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'sms_log_state.dart';
import 'widgets/sms_log_item.dart';
import 'sms_log_model.dart';

class SmsLogPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final logs = context.watch<SmsLogState>().log;
    return Scaffold(
      appBar: AppBar(title: const Text('Detection Log')),
      body: logs.isEmpty
          ? Center(child: Text('No SMS detected yet.'))
          : ListView.builder(
              itemCount: logs.length,
              itemBuilder: (context, idx) {
                final entry = logs[idx];
                return InkWell(
                  onTap: () => showDialog(
                    context: context,
                    builder: (_) => SmsLogDetailsDialog(entry: entry),
                  ),
                  child: SmsLogItem(entry: entry),
                );
              },
            ),
    );
  }
}

class SmsLogDetailsDialog extends StatelessWidget {
  final SmsLogEntry entry;
  const SmsLogDetailsDialog({required this.entry});

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: Row(
        children: [
          Icon(entry.displayIcon, color: entry.displayColor),
          const SizedBox(width: 8),
          Text(entry.resultText),
        ],
      ),
      content: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Sender: ${entry.sender}',
                style: TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            Text('Received: ${_formatTime(entry.timestamp)}'),
            const SizedBox(height: 16),
            Text(entry.body, style: TextStyle(fontSize: 16)),
            const SizedBox(height: 16),
            Row(
              children: [
                if (entry.result == DetectionResult.spam ||
                    entry.result == DetectionResult.fraudulent)
                  Container(
                    margin: const EdgeInsets.only(right: 6),
                    padding:
                        const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                    decoration: BoxDecoration(
                      color: Colors.orange.withOpacity(0.15),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Row(
                      children: [
                        Icon(Icons.report, color: Colors.orange, size: 14),
                        const SizedBox(width: 4),
                        Text('Spam',
                            style: TextStyle(
                                color: Colors.orange,
                                fontWeight: FontWeight.bold)),
                      ],
                    ),
                  ),
                if ((entry.result == DetectionResult.fraudulent) ||
                    (entry.result == DetectionResult.spam &&
                        entry.sender.startsWith('+')))
                  Container(
                    margin: const EdgeInsets.only(right: 6),
                    padding:
                        const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                    decoration: BoxDecoration(
                      color: Colors.red.withOpacity(0.15),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Row(
                      children: [
                        Icon(Icons.warning, color: Colors.red, size: 14),
                        const SizedBox(width: 4),
                        Text('Fraud',
                            style: TextStyle(
                                color: Colors.red,
                                fontWeight: FontWeight.bold)),
                      ],
                    ),
                  ),
              ],
            ),
            const SizedBox(height: 16),
            if (entry.reason != null)
              Text('Reason: ${entry.reason}',
                  style: TextStyle(
                      fontSize: 13,
                      color: Colors.grey[700],
                      fontStyle: FontStyle.italic)),
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Close'),
        ),
      ],
    );
  }

  String _formatTime(DateTime timestamp) {
    final now = DateTime.now();
    final difference = now.difference(timestamp);
    if (difference.inMinutes < 1) {
      return 'Just now';
    } else if (difference.inMinutes < 60) {
      return '${difference.inMinutes}m ago';
    } else if (difference.inHours < 24) {
      return '${difference.inHours}h ago';
    } else {
      return '${difference.inDays}d ago';
    }
  }
}
