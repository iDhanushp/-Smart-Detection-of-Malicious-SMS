import 'package:telephony/telephony.dart';

typedef SmsCallback = void Function(String sender, String body);

class SmsReceiver {
  final Telephony telephony = Telephony.instance;

  void startListening(SmsCallback onSmsReceived) {
    telephony.listenIncomingSms(
      onNewMessage: (SmsMessage message) {
        if (message.body != null && message.address != null) {
          onSmsReceived(message.address!, message.body!);
        }
      },
      listenInBackground: false,
    );
  }
} 