import 'package:permission_handler/permission_handler.dart';

class SmsPermissionHelper {
  static Future<bool> requestAll() async {
    final sms = await Permission.sms.request();
    // Contacts is optional — resolves display names but is NOT required for inbox reading.
    // Do NOT block sync if contacts is denied.
    await Permission.contacts.request();
    return sms.isGranted;
  }
}
