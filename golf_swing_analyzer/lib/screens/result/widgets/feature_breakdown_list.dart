import 'package:flutter/material.dart';
import '../../../core/theme/app_theme.dart';
import '../../../models/feature_importance.dart';
import '../../../widgets/feature_bar.dart';

// Reference ranges seeded from explore_features.py good/bad means
const _ranges = <String, Map<String, double>>{
  'acc_mean':    {'good_min': 0.94, 'good_max': 1.60, 'min': 0.90, 'max': 2.20},
  'omega_mean':  {'good_min': 48.0, 'good_max': 260.0, 'min': 0.0, 'max': 500.0},
  'yaw_at_top':  {'good_min': 250.0, 'good_max': 310.0, 'min': 200.0, 'max': 370.0},
  'acc_std':     {'good_min': 0.02, 'good_max': 0.55, 'min': 0.0, 'max': 1.0},
  'omega_at_top':{'good_min': 87.0, 'good_max': 600.0, 'min': 0.0, 'max': 1300.0},
  'T_backswing_s':{'good_min': 0.3, 'good_max': 1.0, 'min': 0.0, 'max': 1.5},
};

const _units = <String, String>{
  'acc_mean': 'g', 'acc_std': 'g', 'acc_peak': 'g',
  'omega_mean': '°/s', 'omega_peak': '°/s', 'omega_at_top': '°/s',
  'yaw_at_top': '°', 'yaw_range': '°',
  'T_backswing_s': 's', 'T_downswing_s': 's',
};

class FeatureBreakdownList extends StatelessWidget {
  final List<FeatureImportance> importances;
  final bool showAll;

  const FeatureBreakdownList({
    super.key,
    required this.importances,
    this.showAll = false,
  });

  @override
  Widget build(BuildContext context) {
    final displayed = showAll ? importances : importances.take(6).toList();

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Feature Breakdown',
              style: TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 15,
                  color: Colors.white70)),
          const SizedBox(height: 8),
          ...displayed.map((imp) {
            final r = _ranges[imp.feature];
            if (r != null) {
              return FeatureBar(
                label: _friendlyName(imp.feature),
                value: imp.value,
                goodMin: r['good_min']!,
                goodMax: r['good_max']!,
                rangeMin: r['min']!,
                rangeMax: r['max']!,
                unit: _units[imp.feature],
                isGood: imp.isGood,
              );
            }
            return _SimpleRow(imp: imp);
          }),
        ],
      ),
    );
  }

  String _friendlyName(String key) => switch (key) {
        'acc_mean' => 'Avg Acceleration',
        'omega_mean' => 'Avg Angular Velocity',
        'yaw_at_top' => 'Yaw at Top',
        'acc_std' => 'Acceleration Variability',
        'omega_at_top' => 'Angular Velocity at Top',
        'T_backswing_s' => 'Backswing Duration',
        _ => key,
      };
}

class _SimpleRow extends StatelessWidget {
  final FeatureImportance imp;
  const _SimpleRow({required this.imp});

  @override
  Widget build(BuildContext context) {
    final color = imp.isGood ? AppTheme.good : AppTheme.bad;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(imp.feature,
              style: const TextStyle(color: Colors.white70, fontSize: 13)),
          Text(imp.value.toStringAsFixed(3),
              style: TextStyle(
                  color: color, fontWeight: FontWeight.bold, fontSize: 13)),
        ],
      ),
    );
  }
}
