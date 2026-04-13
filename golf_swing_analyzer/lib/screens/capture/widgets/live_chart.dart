import 'dart:collection';
import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../core/constants/ble_constants.dart';
import '../../../core/theme/app_theme.dart';
import '../../../providers/ble_provider.dart';

class LiveChart extends ConsumerStatefulWidget {
  const LiveChart({super.key});

  @override
  ConsumerState<LiveChart> createState() => _LiveChartState();
}

class _LiveChartState extends ConsumerState<LiveChart> {
  final _omegaPoints = Queue<FlSpot>();
  final _accPoints = Queue<FlSpot>();
  int _index = 0;
  double _maxY = 120;

  @override
  void initState() {
    super.initState();
    ref.listenManual(imuStreamProvider, (_, next) {
      next.whenData((sample) {
        setState(() {
          final x = _index.toDouble();
          _omegaPoints.add(FlSpot(x, sample.omegaMag));
          _accPoints.add(FlSpot(x, sample.accMag * 50)); // scale g → deg/s-like
          _index++;

          while (_omegaPoints.length > BleConstants.chartWindowSize) {
            _omegaPoints.removeFirst();
            _accPoints.removeFirst();
          }

          final peak = _omegaPoints.fold(
              0.0, (m, s) => s.y > m ? s.y : m);
          _maxY = (peak * 1.15).clamp(120, double.infinity);
        });
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    final omega = _omegaPoints.toList();
    final acc = _accPoints.toList();

    return RepaintBoundary(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(8, 16, 16, 8),
        child: LineChart(
          duration: Duration.zero,
          LineChartData(
            minY: 0,
            maxY: _maxY,
            gridData: FlGridData(
              show: true,
              horizontalInterval: 60,
              getDrawingHorizontalLine: (_) =>
                  const FlLine(color: Colors.white10, strokeWidth: 1),
              drawVerticalLine: false,
            ),
            borderData: FlBorderData(show: false),
            titlesData: FlTitlesData(
              leftTitles: AxisTitles(
                sideTitles: SideTitles(
                  showTitles: true,
                  interval: 60,
                  reservedSize: 36,
                  getTitlesWidget: (v, _) => Text(
                    v.toInt().toString(),
                    style:
                        const TextStyle(color: Colors.white38, fontSize: 10),
                  ),
                ),
              ),
              rightTitles:
                  const AxisTitles(sideTitles: SideTitles(showTitles: false)),
              topTitles:
                  const AxisTitles(sideTitles: SideTitles(showTitles: false)),
              bottomTitles:
                  const AxisTitles(sideTitles: SideTitles(showTitles: false)),
            ),
            extraLinesData: ExtraLinesData(
              horizontalLines: [
                HorizontalLine(
                  y: BleConstants.omegaStartThreshold,
                  color: Colors.red.withOpacity(0.5),
                  strokeWidth: 1.5,
                  dashArray: [6, 4],
                  label: HorizontalLineLabel(
                    show: true,
                    labelResolver: (_) => 'Swing threshold',
                    style: const TextStyle(color: Colors.red, fontSize: 10),
                  ),
                ),
              ],
            ),
            lineBarsData: [
              LineChartBarData(
                spots: omega,
                isCurved: false,
                color: AppTheme.accent,
                barWidth: 2,
                dotData: const FlDotData(show: false),
                belowBarData: BarAreaData(
                  show: true,
                  color: AppTheme.accent.withOpacity(0.05),
                ),
              ),
              LineChartBarData(
                spots: acc,
                isCurved: false,
                color: Colors.orange,
                barWidth: 1.5,
                dotData: const FlDotData(show: false),
                dashArray: [4, 4],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
