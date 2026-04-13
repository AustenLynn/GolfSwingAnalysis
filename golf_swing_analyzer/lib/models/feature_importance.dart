class FeatureImportance {
  final String feature;
  final double importance;
  final double value;
  final String verdict; // "good" | "bad" | "neutral"

  const FeatureImportance({
    required this.feature,
    required this.importance,
    required this.value,
    required this.verdict,
  });

  factory FeatureImportance.fromJson(Map<String, dynamic> j) =>
      FeatureImportance(
        feature: j['feature'] as String,
        importance: (j['importance'] as num).toDouble(),
        value: (j['value'] as num).toDouble(),
        verdict: j['verdict'] as String? ?? 'neutral',
      );

  bool get isGood => verdict == 'good';
}
