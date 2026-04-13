import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../services/ble/ble_service.dart';
import '../models/imu_sample.dart';

enum BleStatus { idle, scanning, connecting, connected, disconnected, error }

class BleState {
  final BleStatus status;
  final String? errorMessage;

  const BleState({this.status = BleStatus.idle, this.errorMessage});

  BleState copyWith({BleStatus? status, String? errorMessage}) => BleState(
        status: status ?? this.status,
        errorMessage: errorMessage,
      );
}

class BleNotifier extends StateNotifier<BleState> {
  final BleService _service = BleService();

  BleNotifier() : super(const BleState());

  BleService get service => _service;

  Stream<ImuSample> get imuStream => _service.imuStream;

  Future<void> connect() async {
    state = state.copyWith(status: BleStatus.scanning);
    try {
      final ok = await _service.connectToDevice();
      state = state.copyWith(
        status: ok ? BleStatus.connected : BleStatus.error,
        errorMessage: ok ? null : 'Device not found',
      );
    } catch (e) {
      state = state.copyWith(
          status: BleStatus.error, errorMessage: e.toString());
    }
  }

  Future<void> disconnect() async {
    await _service.disconnect();
    state = state.copyWith(status: BleStatus.disconnected);
  }

  @override
  void dispose() {
    _service.dispose();
    super.dispose();
  }
}

final bleProvider = StateNotifierProvider<BleNotifier, BleState>((ref) {
  return BleNotifier();
});

final imuStreamProvider = StreamProvider<ImuSample>((ref) {
  final notifier = ref.watch(bleProvider.notifier);
  return notifier.imuStream;
});
