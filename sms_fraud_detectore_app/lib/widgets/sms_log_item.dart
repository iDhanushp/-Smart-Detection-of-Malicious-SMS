import 'package:flutter/material.dart';
import '../sms_log_model.dart';

class SmsLogItem extends StatelessWidget {
  final SmsLogEntry entry;
  final VoidCallback? onMistake;

  const SmsLogItem({Key? key, required this.entry, this.onMistake})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: EdgeInsets.symmetric(vertical: 4, horizontal: 8),
      child: ListTile(
        leading: Icon(
          entry.displayIcon,
          color: entry.displayColor,
        ),
        title:
            Text(entry.sender, style: TextStyle(fontWeight: FontWeight.bold)),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(entry.body),
            SizedBox(height: 4),
            if (entry.reason != null)
              Container(
                padding: EdgeInsets.all(4),
                decoration: BoxDecoration(
                  color: Colors.grey[100],
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Text(
                  entry.reason!,
                  style: TextStyle(
                    fontSize: 12,
                    color: Colors.grey[700],
                    fontStyle: FontStyle.italic,
                  ),
                ),
              ),
            SizedBox(height: 4),
            Row(
              children: [
                Text(
                  entry.resultText,
                  style: TextStyle(
                    color: entry.displayColor,
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
                  Text('Marked as mistake',
                      style: TextStyle(color: Colors.orange)),
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
