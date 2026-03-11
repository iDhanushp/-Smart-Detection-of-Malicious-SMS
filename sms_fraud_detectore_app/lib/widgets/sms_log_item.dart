import 'package:flutter/material.dart';
import '../sms_log_model.dart';
import 'feedback_dialog.dart';

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
          entry.result.icon,
          color: entry.result.color,
        ),
        title: Row(
          children: [
            Icon(entry.result.icon, color: entry.result.color),
            const SizedBox(width: 8),
            Text(entry.result.label),
          ],
        ),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(entry.body),
            SizedBox(height: 4),
            Row(
              children: [
                // Show FRAUD badge for fraudulent messages
                if (entry.result == DetectionResult.fraudulent)
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
                        Text('FRAUD',
                            style: TextStyle(
                                color: Colors.red,
                                fontWeight: FontWeight.bold)),
                      ],
                    ),
                  ),
                // Show SPAM badge for spam messages
                if (entry.result == DetectionResult.spam)
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
                        Text('SPAM',
                            style: TextStyle(
                                color: Colors.orange,
                                fontWeight: FontWeight.bold)),
                      ],
                    ),
                  ),
                // Show SAFE badge for legitimate messages
                if (entry.result == DetectionResult.legitimate)
                  Container(
                    margin: const EdgeInsets.only(right: 6),
                    padding:
                        const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                    decoration: BoxDecoration(
                      color: Colors.green.withOpacity(0.15),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Row(
                      children: [
                        Icon(Icons.check_circle, color: Colors.green, size: 14),
                        const SizedBox(width: 4),
                        Text('SAFE',
                            style: TextStyle(
                                color: Colors.green,
                                fontWeight: FontWeight.bold)),
                      ],
                    ),
                  ),
                Expanded(
                  child: Text(
                    entry.result.label,
                    style: TextStyle(
                      color: entry.result.color,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
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
        trailing: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Flexible(
              child: Text(
                _formatTime(entry.timestamp),
                style: TextStyle(fontSize: 12, color: Colors.grey),
                overflow: TextOverflow.ellipsis,
              ),
            ),
            IconButton(
              icon: Icon(Icons.feedback_outlined, color: Colors.blue),
              tooltip: 'Report classification error',
              onPressed: () {
                showDialog(
                  context: context,
                  builder: (context) => FeedbackDialog(
                    message: entry,
                    currentClassification: entry.result.label,
                  ),
                );
              },
            ),
          ],
        ),
      ),
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
