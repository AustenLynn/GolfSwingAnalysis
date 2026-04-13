class PhaseTiming {
  final double tStartMs;
  final double tTopMs;
  final double tImpactMs;
  final double tEndMs;
  final double tBackswingS;
  final double tDownswingS;
  final double tempoRatio;

  const PhaseTiming({
    required this.tStartMs,
    required this.tTopMs,
    required this.tImpactMs,
    required this.tEndMs,
    required this.tBackswingS,
    required this.tDownswingS,
    required this.tempoRatio,
  });

  factory PhaseTiming.fromJson(Map<String, dynamic> j) => PhaseTiming(
        tStartMs: (j['t_start_ms'] as num).toDouble(),
        tTopMs: (j['t_top_ms'] as num).toDouble(),
        tImpactMs: (j['t_impact_ms'] as num).toDouble(),
        tEndMs: (j['t_end_ms'] as num).toDouble(),
        tBackswingS: (j['T_backswing_s'] as num).toDouble(),
        tDownswingS: (j['T_downswing_s'] as num).toDouble(),
        tempoRatio: (j['tempo_ratio'] as num).toDouble(),
      );
}
