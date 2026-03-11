import 'dart:convert';
import 'dart:io';
import 'package:csv/csv.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter_sms_inbox/flutter_sms_inbox.dart';

class SmsExporter {
  static Future<String?> exportInboxToCsv() async {
    // Request SMS and storage permissions
    final smsStatus = await Permission.sms.request();
    final storageStatus = await Permission.manageExternalStorage.request();

    if (!smsStatus.isGranted || !storageStatus.isGranted) {
      return null;
    }

    final query = SmsQuery();
    final messages = await query.querySms(
      kinds: [SmsQueryKind.inbox],
      // null for no limit
    );

    // Build CSV rows
    final List<List<dynamic>> rows = [
      ['id', 'address', 'body', 'date'],
    ];

    for (final msg in messages) {
      rows.add([
        msg.id,
        msg.address ?? '',
        msg.body?.replaceAll('\n', ' ') ?? '',
        msg.date?.toIso8601String() ?? '',
      ]);
    }

    final csvStr = const ListToCsvConverter().convert(rows);

    // Prepare file path in a custom folder directly in internal storage (e.g., /storage/emulated/0/SMSExports)
    final extDir = await getExternalStorageDirectory();
    if (extDir == null) return null;

    // Strip the /Android/... part to reach the root of internal storage
    final rootPath = extDir.path.split('/Android').first;
    final exportDir = Directory('$rootPath/SMSExports');
    if (!await exportDir.exists()) {
      await exportDir.create(recursive: true);
    }

    final timestamp = DateTime.now().toIso8601String().replaceAll(':', '-');
    final filePath = '${exportDir.path}/phone_sms_export_$timestamp.csv';

    final file = File(filePath);
    await file.writeAsString(csvStr);
    return filePath;
  }
}
