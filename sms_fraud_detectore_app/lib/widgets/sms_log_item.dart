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
          entry.displayIcon,
          color: entry.displayColor,
        ),
        title: Row(
          children: [
            Icon(entry.displayIcon, color: entry.displayColor),
            const SizedBox(width: 8),
            Text(entry.resultText),
          ],
        ),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(entry.body),
            SizedBox(height: 4),
            Row(
              children: [
                // Treat both spam (1) and fraudulent (2) as spam for badge purposes.
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
                // Fraud badge when classified as fraudulent OR spam from phone number.
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
                    currentClassification: entry.resultText,
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
