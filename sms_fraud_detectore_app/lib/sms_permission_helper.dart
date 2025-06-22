import 'package:permission_handler/permission_handler.dart';

class SmsPermissionHelper {
  static Future<bool> requestAll() async {
    final sms = await Permission.sms.request();
    final contacts = await Permission.contacts.request();
    return sms.isGranted && contacts.isGranted;
  }
}
