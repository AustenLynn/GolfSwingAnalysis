import 'imu_sample.dart';

class SwingSession {
  final String sessionId;
  final DateTime capturedAt;
  final List<ImuSample> samples;
  final int? label;

  const SwingSession({
    required this.sessionId,
    required this.capturedAt,
    required this.samples,
    this.label,
  });

  int get sampleCount => samples.length;

  Duration get duration => samples.isEmpty
      ? Duration.zero
      : Duration(
          milliseconds: (samples.last.tMs - samples.first.tMs).round(),
        );

  Map<String, dynamic> toJson() => {
        'session_id': sessionId,
        'captured_at': capturedAt.toIso8601String(),
        'label': label,
        'samples': samples.map((s) => s.toJson()).toList(),
      };
}
