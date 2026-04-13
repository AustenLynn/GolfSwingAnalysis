import '../../models/imu_sample.dart';

enum SwingPhase { idle, inSwing }

/// Real-time omega threshold detector — mirrors find_swings() in analyze_swing.py.
/// Emits phase transitions so the UI can highlight active swing detection.
class SwingDetector {
  final double startThreshold;
  final double endThreshold;
  final int minSamples;

  SwingPhase _phase = SwingPhase.idle;
  int _swingSampleCount = 0;
  int _completedSwings = 0;

  SwingDetector({
    required this.startThreshold,
    required this.endThreshold,
    required this.minSamples,
  });

  SwingPhase get phase => _phase;
  int get completedSwings => _completedSwings;
  bool get inSwing => _phase == SwingPhase.inSwing;

  /// Feed a new sample. Returns true if the phase just changed.
  bool update(ImuSample sample) {
    final omega = sample.omegaMag;
    final before = _phase;

    if (_phase == SwingPhase.idle && omega > startThreshold) {
      _phase = SwingPhase.inSwing;
      _swingSampleCount = 1;
    } else if (_phase == SwingPhase.inSwing) {
      _swingSampleCount++;
      if (omega < endThreshold) {
        if (_swingSampleCount >= minSamples) {
          _completedSwings++;
        }
        _phase = SwingPhase.idle;
        _swingSampleCount = 0;
      }
    }

    return _phase != before;
  }

  void reset() {
    _phase = SwingPhase.idle;
    _swingSampleCount = 0;
    _completedSwings = 0;
  }
}
