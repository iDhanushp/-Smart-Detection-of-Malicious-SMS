import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'sms_log_state.dart';
import 'sms_log_model.dart';
import 'thread_page.dart';

class ThreadListPage extends StatelessWidget {
  const ThreadListPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('SMS Fraud Detector'),
        actions: [
          Consumer<SmsLogState>(
            builder: (_, state, __) => IconButton(
              icon: state.isSyncing
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.refresh),
              tooltip: 'Re-scan',
              onPressed: state.isSyncing ? null : () => state.syncDeviceSms(),
            ),
          ),
        ],
      ),
      body: Consumer<SmsLogState>(
        builder: (context, state, _) {
          // ── progress bar while syncing ─────────────────────────────────
          if (state.isSyncing && state.threads.isEmpty) {
            return Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const CircularProgressIndicator(),
                  const SizedBox(height: 16),
                  Text(state.statusMsg),
                  const SizedBox(height: 8),
                  Text('${state.progress}%',
                      style: Theme.of(context).textTheme.headlineMedium),
                ],
              ),
            );
          }

          // ── stats row ─────────────────────────────────────────────────
          final threads = state.threads;
          return Column(
            children: [
              if (state.isSyncing)
                LinearProgressIndicator(value: state.progress / 100),
              _StatsRow(state: state),
              _ThreatBreakdown(state: state),
              Expanded(
                child: threads.isEmpty
                    ? const Center(child: Text('No messages found'))
                    : ListView.separated(
                        itemCount: threads.length,
                        separatorBuilder: (_, __) =>
                            const Divider(height: 1, indent: 72),
                        itemBuilder: (ctx, i) =>
                            _ThreadTile(thread: threads[i]),
                      ),
              ),
            ],
          );
        },
      ),
    );
  }
}

// ── Stats strip ───────────────────────────────────────────────────────────

class _StatsRow extends StatelessWidget {
  final SmsLogState state;
  const _StatsRow({required this.state});

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Theme.of(context).colorScheme.surfaceVariant,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _Stat(label: 'Safe',   value: state.countLegitimate, color: Colors.green),
          _Stat(label: 'Spam',   value: state.countSpam,       color: Colors.orange),
          _Stat(label: 'Fraud',  value: state.countFraud,      color: Colors.red),
        ],
      ),
    );
  }
}

class _Stat extends StatelessWidget {
  final String label;
  final int value;
  final Color color;
  const _Stat({required this.label, required this.value, required this.color});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text('$value',
            style: TextStyle(
                fontWeight: FontWeight.bold, fontSize: 20, color: color)),
        Text(label, style: const TextStyle(fontSize: 12)),
      ],
    );
  }
}

// ── Threat breakdown panel ────────────────────────────────────────────

class _ThreatBreakdown extends StatelessWidget {
  final SmsLogState state;
  const _ThreatBreakdown({required this.state});

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
    final counts = state.reasonCounts;
    if (counts.isEmpty) return const SizedBox.shrink();

    final total = counts.fold<int>(0, (s, e) => s + e.value);

    return Theme(
      data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
      child: ExpansionTile(
        tilePadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 0),
        title: Row(
          children: [
            const Text('🛡️  Threat Breakdown',
                style: TextStyle(fontWeight: FontWeight.w600, fontSize: 14)),
            const SizedBox(width: 8),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
              decoration: BoxDecoration(
                color: Colors.red.withOpacity(0.12),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Text('$total flagged',
                  style: const TextStyle(
                      fontSize: 11, color: Colors.red, fontWeight: FontWeight.bold)),
            ),
          ],
        ),
        initiallyExpanded: true,
        childrenPadding:
            const EdgeInsets.only(left: 12, right: 12, bottom: 10),
        children: [
          Wrap(
            spacing: 8,
            runSpacing: 6,
            children: counts.map((entry) {
              final icon  = _icons[entry.key] ?? '❓';
              final label = entry.key.replaceAll('_', ' ');
              return _ReasonChip(
                icon: icon,
                label: label,
                count: entry.value,
              );
            }).toList(),
          ),
        ],
      ),
    );
  }
}

class _ReasonChip extends StatelessWidget {
  final String icon;
  final String label;
  final int count;
  const _ReasonChip({
    required this.icon,
    required this.label,
    required this.count,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 5),
      decoration: BoxDecoration(
        color: isDark
            ? Colors.grey[800]!
            : Colors.grey.withOpacity(0.10),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.grey.withOpacity(0.3)),
      ),
      child: RichText(
        text: TextSpan(
          style: DefaultTextStyle.of(context)
              .style
              .copyWith(fontSize: 12),
          children: [
            TextSpan(text: '$icon  '),
            TextSpan(
                text: label,
                style: const TextStyle(fontWeight: FontWeight.w500)),
            TextSpan(
                text: '  •  $count',
                style: const TextStyle(
                    fontWeight: FontWeight.bold, color: Colors.redAccent)),
          ],
        ),
      ),
    );
  }
}

// ── Thread tile ───────────────────────────────────────────────────────────

class _ThreadTile extends StatelessWidget {
  final ThreadEntry thread;
  const _ThreadTile({required this.thread});

  @override
  Widget build(BuildContext context) {
    final worst  = thread.worstResult;
    final last   = thread.lastMessage;
    final ts     = _fmtTime(last.timestamp);

    return ListTile(
      leading: CircleAvatar(
        backgroundColor: worst.color.withOpacity(0.15),
        child: Icon(worst.icon, color: worst.color, size: 20),
      ),
      title: Row(
        children: [
          Expanded(
            child: Text(thread.address,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(fontWeight: FontWeight.w600)),
          ),
          const SizedBox(width: 8),
          Text(ts,
              style: TextStyle(
                  fontSize: 11,
                  color: Theme.of(context).colorScheme.outline)),
        ],
      ),
      subtitle: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(last.body,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(fontSize: 13)),
          const SizedBox(height: 2),
          _ResultBadge(result: worst),
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

  String _fmtTime(DateTime dt) {
    final now = DateTime.now();
    if (dt.year == now.year && dt.month == now.month && dt.day == now.day) {
      return '${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}';
    }
    return '${dt.day}/${dt.month}';
  }
}

// ── Shared badge ──────────────────────────────────────────────────────────

class _ResultBadge extends StatelessWidget {
  final DetectionResult result;
  const _ResultBadge({required this.result});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: result.color.withOpacity(0.15),
        borderRadius: BorderRadius.circular(4),
        border: Border.all(color: result.color.withOpacity(0.4)),
      ),
      child: Text(
        result.label.toUpperCase(),
        style: TextStyle(
            fontSize: 10,
            fontWeight: FontWeight.bold,
            color: result.color),
      ),
    );
  }
}
