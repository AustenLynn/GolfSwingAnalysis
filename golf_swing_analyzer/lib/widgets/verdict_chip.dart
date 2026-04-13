import 'package:flutter/material.dart';
import '../core/theme/app_theme.dart';

class VerdictChip extends StatelessWidget {
  final bool isGood;
  final double confidence;
  final double fontSize;

  const VerdictChip({
    super.key,
    required this.isGood,
    required this.confidence,
    this.fontSize = 12,
  });

  @override
  Widget build(BuildContext context) {
    final color = isGood ? AppTheme.good : AppTheme.bad;
    final label = isGood ? 'GOOD' : 'BAD';
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.15),
        border: Border.all(color: color, width: 1.5),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Text(
        '$label  ${(confidence * 100).round()}%',
        style: TextStyle(
          color: color,
          fontWeight: FontWeight.bold,
          fontSize: fontSize,
        ),
      ),
    );
  }
}
