import 'package:flutter/material.dart';
import '../../../core/theme/app_theme.dart';
import '../../../models/swing_result.dart';

class VerdictBanner extends StatelessWidget {
  final SwingResult result;

  const VerdictBanner({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    final color = result.isGood ? AppTheme.good : AppTheme.bad;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(vertical: 28),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [color.withOpacity(0.4), color.withOpacity(0.1)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        border: Border(bottom: BorderSide(color: color.withOpacity(0.3))),
      ),
      child: Column(
        children: [
          Text(
            result.verdict,
            style: TextStyle(
              fontSize: 40,
              fontWeight: FontWeight.w900,
              color: color,
              letterSpacing: 4,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            '${(result.confidence * 100).round()}% confidence',
            style: const TextStyle(fontSize: 16, color: Colors.white60),
          ),
        ],
      ),
    );
  }
}
