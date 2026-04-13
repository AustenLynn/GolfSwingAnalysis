import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:uuid/uuid.dart';
import '../models/imu_sample.dart';
import '../models/swing_session.dart';
import '../models/analyze_response.dart';
import '../core/utils/swing_detector.dart';
import '../core/constants/ble_constants.dart';
import '../services/api/swing_api.dart';
import 'settings_provider.dart';

enum CaptureStatus { idle, capturing, analyzing, done, error }

class CaptureState {
  final CaptureStatus status;
  final List<ImuSample> samples;
  final bool inSwing;
  final int swingCount;
  final AnalyzeResponse? result;
  final String? errorMessage;

  const CaptureState({
    this.status = CaptureStatus.idle,
    this.samples = const [],
    this.inSwing = false,
    this.swingCount = 0,
    this.result,
    this.errorMessage,
  });

  CaptureState copyWith({
    CaptureStatus? status,
    List<ImuSample>? samples,
    bool? inSwing,
    int? swingCount,
    AnalyzeResponse? result,
    String? errorMessage,
  }) =>
      CaptureState(
        status: status ?? this.status,
        samples: samples ?? this.samples,
        inSwing: inSwing ?? this.inSwing,
        swingCount: swingCount ?? this.swingCount,
        result: result ?? this.result,
        errorMessage: errorMessage,
      );
}

class CaptureNotifier extends StateNotifier<CaptureState> {
  final Ref _ref;
  SwingDetector? _detector;

  CaptureNotifier(this._ref) : super(const CaptureState());

  void addSample(ImuSample sample) {
    if (state.status != CaptureStatus.capturing) return;
    _detector ??= _buildDetector();
    _detector!.update(sample);

    state = state.copyWith(
      samples: [...state.samples, sample],
      inSwing: _detector!.inSwing,
      swingCount: _detector!.completedSwings,
    );
  }

  SwingDetector _buildDetector() {
    final settings = _ref.read(settingsProvider);
    return SwingDetector(
      startThreshold: settings.omegaStart,
      endThreshold: settings.omegaEnd,
      minSamples: BleConstants.minSwingSamples,
    );
  }

  void startCapture() {
    _detector = _buildDetector();
    state = CaptureState(status: CaptureStatus.capturing);
  }

  Future<AnalyzeResponse?> stopAndAnalyze() async {
    if (state.samples.isEmpty) {
      state = state.copyWith(
          status: CaptureStatus.error, errorMessage: 'No samples captured');
      return null;
    }

    state = state.copyWith(status: CaptureStatus.analyzing);
    final session = SwingSession(
      sessionId: const Uuid().v4(),
      capturedAt: DateTime.now().toUtc(),
      samples: state.samples,
    );

    try {
      final result = await SwingApi.postAnalyze(session);
      state = state.copyWith(status: CaptureStatus.done, result: result);
      return result;
    } catch (e) {
      state = state.copyWith(
          status: CaptureStatus.error, errorMessage: e.toString());
      return null;
    }
  }

  void reset() {
    _detector = null;
    state = const CaptureState();
  }
}

final captureProvider =
    StateNotifierProvider<CaptureNotifier, CaptureState>((ref) {
  return CaptureNotifier(ref);
});
