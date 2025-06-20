import 'package:flutter/material.dart';
import '../sms_log_model.dart';

class SmsLogItem extends StatelessWidget {
  final SmsLogEntry entry;
  final VoidCallback? onMistake;

  const SmsLogItem({Key? key, required this.entry, this.onMistake}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: EdgeInsets.symmetric(vertical: 4, horizontal: 8),
      child: ListTile(
        leading: Icon(
          entry.result == DetectionResult.fraudulent ? Icons.warning : Icons.check_circle,
          color: entry.result == DetectionResult.fraudulent ? Colors.red : Colors.green,
        ),
        title: Text(entry.sender, style: TextStyle(fontWeight: FontWeight.bold)),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(entry.body),
            SizedBox(height: 4),
            Row(
              children: [
                Text(
                  entry.result == DetectionResult.fraudulent ? 'Fraudulent' : 'Legitimate',
                  style: TextStyle(
                    color: entry.result == DetectionResult.fraudulent ? Colors.red : Colors.green,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                SizedBox(width: 16),
                if (!entry.isMistake && onMistake != null)
                  TextButton(
                    onPressed: onMistake,
                    child: Text('Mark as mistake'),
                  ),
                if (entry.isMistake)
                  Text('Marked as mistake', style: TextStyle(color: Colors.orange)),
              ],
            ),
          ],
        ),
        trailing: Text(
          _formatTime(entry.timestamp),
          style: TextStyle(fontSize: 12, color: Colors.grey),
        ),
      ),
    );
  }

  String _formatTime(DateTime dt) {
    return "${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}";
  }
} 