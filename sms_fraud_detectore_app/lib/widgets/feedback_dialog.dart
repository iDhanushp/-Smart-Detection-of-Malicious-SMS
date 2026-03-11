import 'package:flutter/material.dart';
import '../sms_log_model.dart';

/// Dialog for users to report classification errors and provide feedback
class FeedbackDialog extends StatefulWidget {
  final SmsLogEntry message;
  final String currentClassification;

  const FeedbackDialog({
    Key? key,
    required this.message,
    required this.currentClassification,
  }) : super(key: key);

  @override
  State<FeedbackDialog> createState() => _FeedbackDialogState();
}

class _FeedbackDialogState extends State<FeedbackDialog> {
  String? _correctClassification;
  final TextEditingController _notesController = TextEditingController();
  bool _isSubmitting = false;

  final List<String> _classificationOptions = [
    'Legitimate',
    'Spam',
    'Fraudulent'
  ];

  @override
  void dispose() {
    _notesController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return AlertDialog(
      title: Row(
        children: [
          Icon(
            Icons.feedback_outlined,
            color: theme.colorScheme.primary,
          ),
          const SizedBox(width: 8),
          const Text('Report Classification Error'),
        ],
      ),
      content: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Current classification
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: _getClassificationColor(widget.currentClassification)
                    .withOpacity(0.1),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(
                  color: _getClassificationColor(widget.currentClassification)
                      .withOpacity(0.3),
                ),
              ),
              child: Row(
                children: [
                  Icon(
                    _getClassificationIcon(widget.currentClassification),
                    color:
                        _getClassificationColor(widget.currentClassification),
                    size: 20,
                  ),
                  const SizedBox(width: 8),
                  Text(
                    'Current: ${widget.currentClassification}',
                    style: theme.textTheme.titleMedium?.copyWith(
                      color:
                          _getClassificationColor(widget.currentClassification),
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 16),

            // Message preview
            Text(
              'Message Preview:',
              style: theme.textTheme.titleSmall?.copyWith(
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: theme.colorScheme.surface,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(
                    color: theme.colorScheme.outline.withOpacity(0.3)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'From: ${widget.message.sender}',
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: theme.colorScheme.onSurface.withOpacity(0.7),
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    widget.message.body,
                    style: theme.textTheme.bodyMedium,
                    maxLines: 3,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
              ),
            ),

            const SizedBox(height: 16),

            // Correct classification dropdown
            Text(
              'Correct Classification:',
              style: theme.textTheme.titleSmall?.copyWith(
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 8),
            DropdownButtonFormField<String>(
              value: _correctClassification,
              decoration: InputDecoration(
                hintText: 'Select correct classification',
                prefixIcon: const Icon(Icons.category_outlined),
              ),
              items: _classificationOptions.map((option) {
                return DropdownMenuItem<String>(
                  value: option,
                  child: Row(
                    children: [
                      Icon(
                        _getClassificationIcon(option),
                        color: _getClassificationColor(option),
                        size: 18,
                      ),
                      const SizedBox(width: 8),
                      Text(option),
                    ],
                  ),
                );
              }).toList(),
              onChanged: (value) {
                setState(() {
                  _correctClassification = value;
                });
              },
              validator: (value) {
                if (value == null) {
                  return 'Please select the correct classification';
                }
                return null;
              },
            ),

            const SizedBox(height: 16),

            // Additional notes
            Text(
              'Additional Notes (Optional):',
              style: theme.textTheme.titleSmall?.copyWith(
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: _notesController,
              maxLines: 3,
              decoration: InputDecoration(
                hintText: 'Explain why this classification is incorrect...',
                prefixIcon: const Icon(Icons.note_outlined),
              ),
            ),

            const SizedBox(height: 16),

            // Help text
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: theme.colorScheme.primaryContainer.withOpacity(0.3),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Row(
                children: [
                  Icon(
                    Icons.info_outline,
                    color: theme.colorScheme.primary,
                    size: 20,
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      'Your feedback helps improve the detection accuracy for everyone.',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: theme.colorScheme.onSurface.withOpacity(0.8),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: _isSubmitting ? null : () => Navigator.of(context).pop(),
          child: const Text('Cancel'),
        ),
        ElevatedButton(
          onPressed: _isSubmitting ? null : _submitFeedback,
          child: _isSubmitting
              ? const SizedBox(
                  width: 16,
                  height: 16,
                  child: CircularProgressIndicator(strokeWidth: 2),
                )
              : const Text('Submit Feedback'),
        ),
      ],
    );
  }

  /// Get color for classification type
  Color _getClassificationColor(String classification) {
    switch (classification.toLowerCase()) {
      case 'legitimate':
        return Colors.green;
      case 'spam':
        return Colors.orange;
      case 'fraudulent':
        return Colors.red;
      default:
        return Colors.grey;
    }
  }

  /// Get icon for classification type
  IconData _getClassificationIcon(String classification) {
    switch (classification.toLowerCase()) {
      case 'legitimate':
        return Icons.check_circle_outline;
      case 'spam':
        return Icons.warning_amber_outlined;
      case 'fraudulent':
        return Icons.dangerous_outlined;
      default:
        return Icons.help_outline;
    }
  }

  /// Submit feedback to the backend
  Future<void> _submitFeedback() async {
    if (_correctClassification == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please select the correct classification'),
          backgroundColor: Colors.red,
        ),
      );
      return;
    }

    setState(() {
      _isSubmitting = true;
    });

    try {
      final feedback = FeedbackData(
        messageId: widget.message.timestamp.toIso8601String(),
        originalClassification: widget.currentClassification,
        correctClassification: _correctClassification!,
        userNotes: _notesController.text.trim(),
        timestamp: DateTime.now(),
        messagePreview: widget.message.body.substring(
            0,
            widget.message.body.length > 100
                ? 100
                : widget.message.body.length),
        sender: widget.message.sender,
      );

      await FeedbackService().submitFeedback(feedback);

      if (mounted) {
        Navigator.of(context).pop(true); // Return true to indicate success

        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: const Text('Thank you for your feedback!'),
            backgroundColor: Theme.of(context).colorScheme.primary,
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error submitting feedback: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          _isSubmitting = false;
        });
      }
    }
  }
}

/// Data model for feedback
class FeedbackData {
  final String messageId;
  final String originalClassification;
  final String correctClassification;
  final String userNotes;
  final DateTime timestamp;
  final String messagePreview;
  final String sender;

  FeedbackData({
    required this.messageId,
    required this.originalClassification,
    required this.correctClassification,
    required this.userNotes,
    required this.timestamp,
    required this.messagePreview,
    required this.sender,
  });

  Map<String, dynamic> toJson() {
    return {
      'messageId': messageId,
      'originalClassification': originalClassification,
      'correctClassification': correctClassification,
      'userNotes': userNotes,
      'timestamp': timestamp.toIso8601String(),
      'messagePreview': messagePreview,
      'sender': sender,
    };
  }
}

/// Service for handling feedback submission
class FeedbackService {
  static final FeedbackService _instance = FeedbackService._internal();
  factory FeedbackService() => _instance;
  FeedbackService._internal();

  /// Submit feedback to backend
  Future<void> submitFeedback(FeedbackData feedback) async {
    // TODO: Implement actual API call to backend
    // For now, just simulate network delay
    await Future.delayed(const Duration(seconds: 1));

    // In a real implementation, you would:
    // 1. Send feedback to your backend API
    // 2. Store feedback locally for offline support
    // 3. Sync when network is available

    print('Feedback submitted: ${feedback.toJson()}');
  }

  /// Get feedback statistics
  Future<Map<String, dynamic>> getFeedbackStats() async {
    // TODO: Implement API call to get feedback statistics
    return {
      'totalFeedback': 0,
      'accuracyImprovement': 0.0,
      'lastUpdated': DateTime.now().toIso8601String(),
    };
  }
}
