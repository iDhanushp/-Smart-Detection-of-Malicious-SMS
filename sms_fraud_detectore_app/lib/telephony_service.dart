import 'package:telephony/telephony.dart';

typedef SmsCallback = void Function(
    String address, String body, DateTime timestamp);

class TelephonyService {
  final Telephony _telephony = Telephony.instance;

  Future<bool> requestPermission() async {
    final granted = await _telephony.requestSmsPermissions ?? false;
    return granted;
  }

  Future<void> fetchInbox(SmsCallback onMessage) async {
    final messages = await _telephony.getInboxSms(columns: [
      SmsColumn.ADDRESS,
      SmsColumn.BODY,
      SmsColumn.DATE,
    ]);
    for (final msg in messages) {
      final address = msg.address ?? 'Unknown';
      final body = msg.body ?? '';
      final ts = DateTime.fromMillisecondsSinceEpoch(msg.date ?? 0);
      onMessage(address, body, ts);
    }
  }

  void listenIncoming(SmsCallback onMessage) {
    _telephony.listenIncomingSms(onNewMessage: (SmsMessage msg) {
      final address = msg.address ?? 'Unknown';
      final body = msg.body ?? '';
      final ts = DateTime.fromMillisecondsSinceEpoch(
          msg.date ?? DateTime.now().millisecondsSinceEpoch);
      onMessage(address, body, ts);
    });
  }
}
