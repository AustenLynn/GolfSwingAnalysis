import 'package:flutter/material.dart';
import '../core/theme/app_theme.dart';

/// Shows a named feature with its value positioned on a [goodMin, goodMax] range bar.
class FeatureBar extends StatelessWidget {
  final String label;
  final double value;
  final double goodMin;
  final double goodMax;
  final double rangeMin;
  final double rangeMax;
  final String? unit;
  final bool isGood;

  const FeatureBar({
    super.key,
    required this.label,
    required this.value,
    required this.goodMin,
    required this.goodMax,
    required this.rangeMin,
    required this.rangeMax,
    this.unit,
    required this.isGood,
  });

  @override
  Widget build(BuildContext context) {
    final span = rangeMax - rangeMin;
    final goodStartFrac = ((goodMin - rangeMin) / span).clamp(0.0, 1.0);
    final goodWidthFrac = ((goodMax - goodMin) / span).clamp(0.0, 1.0);
    final valueFrac = ((value - rangeMin) / span).clamp(0.0, 1.0);
    final dotColor = isGood ? AppTheme.good : AppTheme.bad;

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(label,
                  style: const TextStyle(fontSize: 13, color: Colors.white70)),
              Text(
                '${value.toStringAsFixed(2)}${unit != null ? ' $unit' : ''}',
                style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.bold,
                    color: dotColor),
              ),
            ],
          ),
          const SizedBox(height: 4),
          LayoutBuilder(builder: (_, constraints) {
            final w = constraints.maxWidth;
            return SizedBox(
              height: 8,
              child: Stack(
                children: [
                  // Track
                  Container(
                    decoration: BoxDecoration(
                      color: Colors.white12,
                      borderRadius: BorderRadius.circular(4),
                    ),
                  ),
                  // Good zone
                  Positioned(
                    left: goodStartFrac * w,
                    width: goodWidthFrac * w,
                    top: 0,
                    bottom: 0,
                    child: Container(
                      decoration: BoxDecoration(
                        color: AppTheme.good.withOpacity(0.25),
                        borderRadius: BorderRadius.circular(4),
                      ),
                    ),
                  ),
                  // Value marker
                  Positioned(
                    left: (valueFrac * w - 5).clamp(0.0, w - 10),
                    top: -2,
                    child: Container(
                      width: 10,
                      height: 12,
                      decoration: BoxDecoration(
                        color: dotColor,
                        borderRadius: BorderRadius.circular(3),
                      ),
                    ),
                  ),
                ],
              ),
            );
          }),
        ],
      ),
    );
  }
}
