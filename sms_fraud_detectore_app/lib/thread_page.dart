import 'package:flutter/material.dart';
import 'sms_log_model.dart';

class ThreadPage extends StatelessWidget {
  final ThreadEntry thread;
  const ThreadPage({Key? key, required this.thread}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final worst = thread.worstResult;

    return Scaffold(
      appBar: AppBar(
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(thread.address,
                style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
            Text('${thread.messages.length} messages',
                style: const TextStyle(fontSize: 12)),
          ],
        ),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 12),
            child: Chip(
              label: Text(worst.label.toUpperCase(),
                  style: TextStyle(
                      color: worst.color,
                      fontWeight: FontWeight.bold,
                      fontSize: 12)),
              backgroundColor: worst.color.withOpacity(0.12),
              side: BorderSide(color: worst.color.withOpacity(0.4)),
            ),
          ),
        ],
      ),
      body: Column(
        children: [
          // ── alert banner for fraud / spam ───────────────────────────────
          if (worst != DetectionResult.legitimate)
            Container(
              width: double.infinity,
              color: worst.color.withOpacity(0.12),
              padding:
                  const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              child: Row(
                children: [
                  Icon(worst.icon, color: worst.color, size: 20),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      worst == DetectionResult.fraudulent
                          ? '⚠️ This thread contains FRAUDULENT messages. Do not share personal or financial information.'
                          : '📢 This thread contains SPAM messages.',
                      style: TextStyle(
                          color: worst.color, fontWeight: FontWeight.w500),
                    ),
                  ),
                ],
              ),
            ),

          // ── message list ────────────────────────────────────────────────
          Expanded(
            child: ListView.builder(
              padding: const EdgeInsets.all(12),
              itemCount: thread.messages.length,
              itemBuilder: (_, i) => _MessageBubble(msg: thread.messages[i]),
            ),
          ),
        ],
      ),
    );
  }
}

// ── Single message bubble ─────────────────────────────────────────────────

class _MessageBubble extends StatelessWidget {
  final SmsLogEntry msg;
  const _MessageBubble({required this.msg});

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    Color bgColor;
    switch (msg.result) {
      case DetectionResult.fraudulent:
        bgColor = Colors.red.withOpacity(isDark ? 0.25 : 0.10);
        break;
      case DetectionResult.spam:
        bgColor = Colors.orange.withOpacity(isDark ? 0.25 : 0.10);
        break;
      case DetectionResult.legitimate:
        bgColor = isDark ? Colors.grey[800]! : Colors.grey[100]!;
        break;
    }

    final ts = _fmtFull(msg.timestamp);

    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(12),
        border: msg.result != DetectionResult.legitimate
            ? Border.all(color: msg.result.color.withOpacity(0.35))
            : null,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // badge + reason tag + timestamp
          Row(
            children: [
              _Badge(result: msg.result),
              if (msg.reason != null &&
                  msg.result != DetectionResult.legitimate) ...[  
                const SizedBox(width: 6),
                _ReasonTag(reason: msg.reason!),
              ],
              const Spacer(),
              Text(ts,
                  style: TextStyle(
                      fontSize: 11,
                      color: Theme.of(context).colorScheme.outline)),
            ],
          ),
          const SizedBox(height: 6),
          // body
          Text(msg.body, style: const TextStyle(fontSize: 14)),
        ],
      ),
    );
  }

  String _fmtFull(DateTime dt) {
    return '${dt.day}/${dt.month}/${dt.year}  '
        '${dt.hour.toString().padLeft(2, '0')}:'
        '${dt.minute.toString().padLeft(2, '0')}';
  }
}

class _Badge extends StatelessWidget {
  final DetectionResult result;
  const _Badge({required this.result});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 3),
      decoration: BoxDecoration(
        color: result.color.withOpacity(0.15),
        borderRadius: BorderRadius.circular(4),
        border: Border.all(color: result.color.withOpacity(0.4)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(result.icon, size: 12, color: result.color),
          const SizedBox(width: 4),
          Text(
            result.label.toUpperCase(),
            style: TextStyle(
                fontSize: 10,
                fontWeight: FontWeight.bold,
                color: result.color),
          ),
        ],
      ),
    );
  }
}

class _ReasonTag extends StatelessWidget {
  final String reason;
  const _ReasonTag({required this.reason});

  // Map rule keys to human-readable icon + label
  static const _icons = <String, String>{
    'phishing_link':      '🔗',
    'data_steal':         '💳',
    'prize_fraud':        '🎰',
    'account_threat':     '🔒',
    'credential_harvest': '🔑',
    'legal_threat':       '⚖️',
    'kyc_fraud':          '📋',
    'fraud_alert':        '🚨',
    'impersonation':      '🎭',
    'job_scam':           '💼',
    'promotional':        '📢',
    'suspicious':         '⚠️',
  };

  @override
  Widget build(BuildContext context) {
    final icon  = _icons[reason] ?? '❓';
    final label = reason.replaceAll('_', ' ');
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 3),
      decoration: BoxDecoration(
        color: Colors.grey.withOpacity(0.12),
        borderRadius: BorderRadius.circular(4),
        border: Border.all(color: Colors.grey.withOpacity(0.35)),
      ),
      child: Text(
        '$icon $label',
        style: const TextStyle(
            fontSize: 10,
            fontWeight: FontWeight.w600,
            color: Colors.grey),
      ),
    );
  }
}
