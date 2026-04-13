import 'package:flutter/material.dart';
import '../../../core/theme/app_theme.dart';
import '../../../models/phase_timing.dart';

class PhaseTimeline extends StatelessWidget {
  final PhaseTiming timing;

  const PhaseTimeline({super.key, required this.timing});

  @override
  Widget build(BuildContext context) {
    final total = timing.tBackswingS + timing.tDownswingS;
    final followThrough =
        ((timing.tEndMs - timing.tImpactMs) / 1000).clamp(0.0, 10.0);
    final fullTotal = total + followThrough;

    final backFrac = total > 0 ? timing.tBackswingS / fullTotal : 0.33;
    final downFrac = total > 0 ? timing.tDownswingS / fullTotal : 0.33;
    final followFrac = 1.0 - backFrac - downFrac;

    final tempoColor = (timing.tempoRatio - 3.0).abs() < 1.5
        ? AppTheme.good
        : AppTheme.bad;

    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text('Swing Phases',
                  style: TextStyle(
                      fontWeight: FontWeight.bold, color: Colors.white70)),
              Row(
                children: [
                  const Text('Tempo: ',
                      style: TextStyle(color: Colors.white54, fontSize: 13)),
                  Text(
                    '${timing.tempoRatio.toStringAsFixed(1)}:1',
                    style: TextStyle(
                        color: tempoColor,
                        fontWeight: FontWeight.bold,
                        fontSize: 13),
                  ),
                  Text(' (target 3:1)',
                      style:
                          const TextStyle(color: Colors.white38, fontSize: 12)),
                ],
              ),
            ],
          ),
          const SizedBox(height: 10),
          ClipRRect(
            borderRadius: BorderRadius.circular(6),
            child: SizedBox(
              height: 28,
              child: Row(
                children: [
                  _Segment(
                    flex: (backFrac * 100).round(),
                    color: AppTheme.accent,
                    label:
                        'Backswing ${timing.tBackswingS.toStringAsFixed(2)}s',
                  ),
                  _Segment(
                    flex: (downFrac * 100).round(),
                    color: AppTheme.good,
                    label: 'Down ${timing.tDownswingS.toStringAsFixed(2)}s',
                  ),
                  _Segment(
                    flex: (followFrac * 100).round(),
                    color: Colors.purple.shade300,
                    label: 'Follow ${followThrough.toStringAsFixed(2)}s',
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _Segment extends StatelessWidget {
  final int flex;
  final Color color;
  final String label;

  const _Segment(
      {required this.flex, required this.color, required this.label});

  @override
  Widget build(BuildContext context) => Expanded(
        flex: flex.clamp(1, 100),
        child: Container(
          color: color.withOpacity(0.7),
          alignment: Alignment.center,
          child: Text(
            label,
            style: const TextStyle(
                fontSize: 10, color: Colors.white, fontWeight: FontWeight.w600),
            overflow: TextOverflow.ellipsis,
          ),
        ),
      );
}
