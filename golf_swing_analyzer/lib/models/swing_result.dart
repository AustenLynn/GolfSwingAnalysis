import 'feature_importance.dart';
import 'phase_timing.dart';

class SwingResult {
  final String? id;
  final int swingIndex;
  final String verdict;
  final int label;
  final double confidence;
  final Map<String, dynamic> features;
  final List<FeatureImportance> featureImportances;
  final PhaseTiming phaseTiming;

  const SwingResult({
    this.id,
    required this.swingIndex,
    required this.verdict,
    required this.label,
    required this.confidence,
    required this.features,
    required this.featureImportances,
    required this.phaseTiming,
  });

  bool get isGood => label == 1;

  factory SwingResult.fromJson(Map<String, dynamic> j) => SwingResult(
        id: j['id'] as String?,
        swingIndex: j['swing_index'] as int,
        verdict: j['verdict'] as String,
        label: j['label'] as int,
        confidence: (j['confidence'] as num).toDouble(),
        features: Map<String, dynamic>.from(j['features'] as Map),
        featureImportances: (j['feature_importances'] as List)
            .map((e) => FeatureImportance.fromJson(e as Map<String, dynamic>))
            .toList(),
        phaseTiming: PhaseTiming.fromJson(j['phase_timing'] as Map<String, dynamic>),
      );
}
