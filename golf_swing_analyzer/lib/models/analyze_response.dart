import 'swing_result.dart';

class AnalyzeResponse {
  final String sessionId;
  final int swingsDetected;
  final List<SwingResult> swings;

  const AnalyzeResponse({
    required this.sessionId,
    required this.swingsDetected,
    required this.swings,
  });

  factory AnalyzeResponse.fromJson(Map<String, dynamic> j) => AnalyzeResponse(
        sessionId: j['session_id'] as String,
        swingsDetected: j['swings_detected'] as int,
        swings: (j['swings'] as List)
            .map((e) => SwingResult.fromJson(e as Map<String, dynamic>))
            .toList(),
      );
}
