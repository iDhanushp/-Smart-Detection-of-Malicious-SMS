import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'sms_log_state.dart';
import 'thread_models.dart';
import 'sms_log_model.dart';
import 'thread_page.dart';

class ThreadListPage extends StatelessWidget {
  const ThreadListPage({super.key});

  @override
  Widget build(BuildContext context) {
    final threads = context.watch<SmsLogState>().threads;
    final isSyncing = context.watch<SmsLogState>().isSyncing;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Messages'),
        backgroundColor: Colors.blue,
        foregroundColor: Colors.white,
        elevation: 0,
        actions: [
          if (isSyncing)
            const Padding(
              padding: EdgeInsets.all(16.0),
              child: SizedBox(
                width: 20,
                height: 20,
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                ),
              ),
            ),
          IconButton(
            icon: const Icon(Icons.search),
            onPressed: () {
              // TODO: Implement search functionality
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Search coming soon!')),
              );
            },
          ),
          PopupMenuButton<String>(
            onSelected: (value) {
              switch (value) {
                case 'settings':
                  // TODO: Navigate to settings
                  break;
                case 'help':
                  // TODO: Show help
                  break;
              }
            },
            itemBuilder: (context) => [
              const PopupMenuItem(
                value: 'settings',
                child: Row(
                  children: [
                    Icon(Icons.settings),
                    SizedBox(width: 8),
                    Text('Settings'),
                  ],
                ),
              ),
              const PopupMenuItem(
                value: 'help',
                child: Row(
                  children: [
                    Icon(Icons.help),
                    SizedBox(width: 8),
                    Text('Help'),
                  ],
                ),
              ),
            ],
          ),
        ],
      ),
      body: threads.isEmpty
          ? _buildEmptyState()
          : ListView.separated(
              itemCount: threads.length,
              separatorBuilder: (_, __) => const Divider(height: 1),
              itemBuilder: (context, index) {
                final thread = threads[index];
                final last = thread.lastMessage;
                return _buildThreadTile(context, thread, last);
              },
            ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.chat_bubble_outline,
            size: 64,
            color: Colors.grey[400],
          ),
          const SizedBox(height: 16),
          Text(
            'No messages yet',
            style: TextStyle(
              fontSize: 18,
              color: Colors.grey[600],
              fontWeight: FontWeight.w500,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'SMS messages will appear here\nwith fraud detection results',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 14,
              color: Colors.grey[500],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildThreadTile(
      BuildContext context, ThreadEntry thread, SmsLogEntry last) {
    final isFraudulent = last.result == DetectionResult.fraudulent;

    return ListTile(
      contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      leading: CircleAvatar(
        radius: 24,
        backgroundColor:
            isFraudulent ? Colors.red.shade100 : Colors.blue.shade100,
        child: Icon(
          isFraudulent ? Icons.warning : Icons.person,
          color: isFraudulent ? Colors.red : Colors.blue,
          size: 24,
        ),
      ),
      title: Row(
        children: [
          Expanded(
            child: Text(
              thread.address,
              style: const TextStyle(
                fontWeight: FontWeight.w600,
                fontSize: 16,
              ),
            ),
          ),
          if (isFraudulent)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
              decoration: BoxDecoration(
                color: Colors.red.shade100,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(
                'FRAUD',
                style: TextStyle(
                  color: Colors.red.shade700,
                  fontSize: 10,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
        ],
      ),
      subtitle: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const SizedBox(height: 4),
          Text(
            last.body,
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
            style: TextStyle(
              color: Colors.grey[600],
              fontSize: 14,
            ),
          ),
          const SizedBox(height: 4),
          Row(
            children: [
              Icon(
                isFraudulent ? Icons.warning : Icons.check_circle,
                size: 14,
                color: isFraudulent ? Colors.red : Colors.green,
              ),
              const SizedBox(width: 4),
              Text(
                isFraudulent ? 'Fraudulent' : 'Legitimate',
                style: TextStyle(
                  fontSize: 12,
                  color: isFraudulent ? Colors.red : Colors.green,
                  fontWeight: FontWeight.w500,
                ),
              ),
              const Spacer(),
              Text(
                _friendlyTime(last.timestamp),
                style: TextStyle(
                  fontSize: 12,
                  color: Colors.grey[500],
                ),
              ),
            ],
          ),
        ],
      ),
      onTap: () => Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => ThreadPage(thread: thread),
        ),
      ),
    );
  }

  String _friendlyTime(DateTime ts) {
    final now = DateTime.now();
    final difference = now.difference(ts);

    if (difference.inDays > 0) {
      if (difference.inDays == 1) {
        return 'Yesterday';
      } else if (difference.inDays < 7) {
        return '${difference.inDays}d ago';
      } else {
        return '${ts.month}/${ts.day}/${ts.year}';
      }
    } else if (difference.inHours > 0) {
      return '${difference.inHours}h ago';
    } else if (difference.inMinutes > 0) {
      return '${difference.inMinutes}m ago';
    } else {
      return 'Just now';
    }
  }
}
