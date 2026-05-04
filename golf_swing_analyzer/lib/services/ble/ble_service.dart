import 'dart:async';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import '../../core/constants/ble_constants.dart';
import '../../models/imu_sample.dart';
import 'imu_stream_parser.dart';

class BleService {
  BluetoothDevice? _device;
  BluetoothCharacteristic? _txChar;
  final _parser = ImuStreamParser();
  StreamSubscription? _notifySub;

  Stream<ImuSample> get imuStream => _parser.stream;

  bool get isConnected =>
      _device?.isConnected == true && _txChar != null;

  /// Scan for ESP32_IMU_GOLF and connect. Returns true on success.
  Future<bool> connectToDevice() async {
    final completer = Completer<BluetoothDevice?>();

    final sub = FlutterBluePlus.scanResults.listen((results) {
      for (final r in results) {
        if (r.device.platformName == BleConstants.deviceName &&
            !completer.isCompleted) {
          completer.complete(r.device);
        }
      }
    });

    await FlutterBluePlus.startScan(
      timeout: const Duration(seconds: 10),
      withNames: [BleConstants.deviceName],
    );

    final found = await completer.future
        .timeout(const Duration(seconds: 12), onTimeout: () => null);

    await sub.cancel();
    await FlutterBluePlus.stopScan();

    if (found == null) return false;
    _device = found;

    await _device!.connect(autoConnect: false);
    await _device!.requestMtu(512);
    final services = await _device!.discoverServices();

    for (final svc in services) {
      if (svc.uuid.toString().toLowerCase() ==
          BleConstants.serviceUuid.toLowerCase()) {
        for (final char in svc.characteristics) {
          if (char.uuid.toString().toLowerCase() ==
              BleConstants.txCharUuid.toLowerCase()) {
            _txChar = char;
            break;
          }
        }
      }
    }

    if (_txChar == null) {
      await _device!.disconnect();
      return false;
    }

    _parser.reset();
    await _txChar!.setNotifyValue(true);
    _notifySub = _txChar!.onValueReceived.listen(_parser.onData);
    return true;
  }

  Future<void> disconnect() async {
    await _notifySub?.cancel();
    _notifySub = null;
    await _txChar?.setNotifyValue(false);
    await _device?.disconnect();
    _device = null;
    _txChar = null;
  }

  void dispose() {
    _notifySub?.cancel();
    _parser.dispose();
  }
}
