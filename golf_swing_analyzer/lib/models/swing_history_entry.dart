class SwingHistoryEntry {
  final String id;
  final String sessionId;
  final int swingIndex;
  final DateTime capturedAt;
  final String verdict;
  final int label;
  final int? userLabel;
  final double confidence;
  final double? accMean;
  final double? omegaMean;
  final double? tempoRatio;

  const SwingHistoryEntry({
    required this.id,
    required this.sessionId,
    required this.swingIndex,
    required this.capturedAt,
    required this.verdict,
    required this.label,
    this.userLabel,
    required this.confidence,
    this.accMean,
    this.omegaMean,
    this.tempoRatio,
  });

  bool get isGood => (userLabel ?? label) == 1;

  factory SwingHistoryEntry.fromJson(Map<String, dynamic> j) =>
      SwingHistoryEntry(
        id: j['id'] as String,
        sessionId: j['session_id'] as String,
        swingIndex: j['swing_index'] as int,
        capturedAt: DateTime.parse(j['captured_at'] as String),
        verdict: j['verdict'] as String,
        label: j['label'] as int,
        userLabel: j['user_label'] as int?,
        confidence: (j['confidence'] as num).toDouble(),
        accMean: (j['acc_mean'] as num?)?.toDouble(),
        omegaMean: (j['omega_mean'] as num?)?.toDouble(),
        tempoRatio: (j['tempo_ratio'] as num?)?.toDouble(),
      );
}
