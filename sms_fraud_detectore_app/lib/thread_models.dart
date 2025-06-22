import 'sms_log_model.dart';

class ThreadEntry {
  final String address;
  final List<SmsLogEntry> messages;

  ThreadEntry({required this.address, required this.messages});

  SmsLogEntry get lastMessage => messages.first;
}
