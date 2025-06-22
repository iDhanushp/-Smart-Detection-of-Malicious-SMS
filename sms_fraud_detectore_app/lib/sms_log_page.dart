import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'sms_log_state.dart';
import 'sms_log_model.dart';
import 'widgets/sms_log_item.dart';

class SmsLogPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Consumer<SmsLogState>(
      builder: (context, state, _) {
        return Scaffold(
          appBar: AppBar(
            title: const Text('Detection Log'),
            backgroundColor: Colors.blue,
            foregroundColor: Colors.white,
            elevation: 0,
            actions: [
              IconButton(
                icon: const Icon(Icons.filter_list),
                onPressed: () {
                  _showFilterOptions(context, state);
                },
              ),
              IconButton(
                icon: const Icon(Icons.clear_all),
                onPressed: () {
                  _showClearConfirmation(context, state);
                },
              ),
            ],
          ),
          body: state.log.isEmpty
              ? _buildEmptyState()
              : Column(
                  children: [
                    _buildStatsBar(state),
                    Expanded(
                      child: ListView.builder(
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
                    ),
                  ],
                ),
        );
      },
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.history,
            size: 64,
            color: Colors.grey[400],
          ),
          const SizedBox(height: 16),
          Text(
            'No detection logs yet',
            style: TextStyle(
              fontSize: 18,
              color: Colors.grey[600],
              fontWeight: FontWeight.w500,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'SMS detection results will appear here',
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

  Widget _buildStatsBar(SmsLogState state) {
    final logs = state.log;
    final totalSms = logs.length;
    final fraudulentSms =
        logs.where((log) => log.result == DetectionResult.fraudulent).length;
    final legitimateSms = totalSms - fraudulentSms;
    final mistakes = logs.where((log) => log.isMistake).length;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.grey.shade50,
        border: Border(
          bottom: BorderSide(color: Colors.grey.shade300),
        ),
      ),
      child: Row(
        children: [
          Expanded(
            child: _buildStatItem(
              'Total',
              totalSms.toString(),
              Icons.message,
              Colors.blue,
            ),
          ),
          Expanded(
            child: _buildStatItem(
              'Legitimate',
              legitimateSms.toString(),
              Icons.check_circle,
              Colors.green,
            ),
          ),
          Expanded(
            child: _buildStatItem(
              'Fraudulent',
              fraudulentSms.toString(),
              Icons.warning,
              Colors.red,
            ),
          ),
          Expanded(
            child: _buildStatItem(
              'Mistakes',
              mistakes.toString(),
              Icons.error,
              Colors.orange,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatItem(
      String label, String value, IconData icon, Color color) {
    return Column(
      children: [
        Icon(icon, color: color, size: 20),
        const SizedBox(height: 4),
        Text(
          value,
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: color,
          ),
        ),
        Text(
          label,
          style: TextStyle(
            fontSize: 10,
            color: Colors.grey[600],
          ),
        ),
      ],
    );
  }

  void _showFilterOptions(BuildContext context, SmsLogState state) {
    showModalBottomSheet(
      context: context,
      builder: (context) => Container(
        padding: const EdgeInsets.all(16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text(
              'Filter Logs',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            ListTile(
              leading: const Icon(Icons.all_inclusive),
              title: const Text('Show All'),
              onTap: () {
                Navigator.pop(context);
                // TODO: Implement filter functionality
              },
            ),
            ListTile(
              leading: const Icon(Icons.check_circle, color: Colors.green),
              title: const Text('Legitimate Only'),
              onTap: () {
                Navigator.pop(context);
                // TODO: Implement filter functionality
              },
            ),
            ListTile(
              leading: const Icon(Icons.warning, color: Colors.red),
              title: const Text('Fraudulent Only'),
              onTap: () {
                Navigator.pop(context);
                // TODO: Implement filter functionality
              },
            ),
            ListTile(
              leading: const Icon(Icons.error, color: Colors.orange),
              title: const Text('Mistakes Only'),
              onTap: () {
                Navigator.pop(context);
                // TODO: Implement filter functionality
              },
            ),
          ],
        ),
      ),
    );
  }

  void _showClearConfirmation(BuildContext context, SmsLogState state) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Clear All Logs'),
        content: const Text(
            'Are you sure you want to clear all detection logs? This action cannot be undone.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              // TODO: Implement clear functionality
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                    content: Text('Clear functionality coming soon!')),
              );
            },
            child: const Text('Clear', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
  }
}
