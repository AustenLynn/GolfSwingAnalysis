import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../providers/ble_provider.dart';

class BleStatusBar extends ConsumerWidget {
  const BleStatusBar({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(bleProvider);
    final notifier = ref.read(bleProvider.notifier);

    final (label, color, loading) = switch (state.status) {
      BleStatus.idle => ('Disconnected', Colors.grey, false),
      BleStatus.scanning => ('Scanning...', Colors.orange, true),
      BleStatus.connecting => ('Connecting...', Colors.orange, true),
      BleStatus.connected => ('ESP32_IMU_GOLF', Colors.green, false),
      BleStatus.disconnected => ('Disconnected', Colors.grey, false),
      BleStatus.error => (state.errorMessage ?? 'Error', Colors.red, false),
    };

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      color: const Color(0xFF1E1E1E),
      child: Row(
        children: [
          if (loading)
            const SizedBox(
              width: 12,
              height: 12,
              child: CircularProgressIndicator(strokeWidth: 2),
            )
          else
            Container(
              width: 10,
              height: 10,
              decoration: BoxDecoration(color: color, shape: BoxShape.circle),
            ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(label,
                style: TextStyle(color: color, fontWeight: FontWeight.w500)),
          ),
          if (state.status == BleStatus.connected)
            TextButton(
              onPressed: notifier.disconnect,
              child: const Text('Disconnect',
                  style: TextStyle(color: Colors.red, fontSize: 12)),
            )
          else if (!loading)
            TextButton(
              onPressed: notifier.connect,
              child: const Text('Connect',
                  style: TextStyle(color: Colors.blue, fontSize: 12)),
            ),
        ],
      ),
    );
  }
}
