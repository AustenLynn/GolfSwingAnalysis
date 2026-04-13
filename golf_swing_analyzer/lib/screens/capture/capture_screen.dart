import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../providers/ble_provider.dart';
import '../../providers/capture_provider.dart';
import '../../core/theme/app_theme.dart';
import 'widgets/ble_status_bar.dart';
import 'widgets/live_chart.dart';

class CaptureScreen extends ConsumerWidget {
  const CaptureScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final captureState = ref.watch(captureProvider);
    final captureNotifier = ref.read(captureProvider.notifier);
    final bleState = ref.watch(bleProvider);

    // Feed BLE samples into capture provider
    ref.listen(imuStreamProvider, (_, next) {
      next.whenData(captureNotifier.addSample);
    });

    final isCapturing = captureState.status == CaptureStatus.capturing;
    final isAnalyzing = captureState.status == CaptureStatus.analyzing;
    final isConnected = bleState.status == BleStatus.connected;

    return Scaffold(
      appBar: AppBar(title: const Text('Capture Swing')),
      body: Column(
        children: [
          const BleStatusBar(),

          // Live chart
          Expanded(
            child: Stack(
              children: [
                AnimatedContainer(
                  duration: const Duration(milliseconds: 300),
                  color: captureState.inSwing
                      ? AppTheme.good.withOpacity(0.05)
                      : Colors.transparent,
                  child: const LiveChart(),
                ),
                if (captureState.inSwing)
                  Positioned(
                    top: 12,
                    right: 16,
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                        color: AppTheme.good,
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: const Text('SWING DETECTED',
                          style: TextStyle(
                              color: Colors.white,
                              fontSize: 11,
                              fontWeight: FontWeight.bold)),
                    ),
                  ),
              ],
            ),
          ),

          // Legend
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: Row(
              children: [
                _Legend(color: AppTheme.accent, label: 'ω mag (deg/s)'),
                const SizedBox(width: 16),
                _Legend(color: Colors.orange, label: 'acc mag ×50'),
              ],
            ),
          ),

          // Controls
          Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    _Stat(
                        label: 'Samples',
                        value: captureState.samples.length.toString()),
                    _Stat(
                        label: 'Swings',
                        value: captureState.swingCount.toString()),
                    _Stat(
                        label: 'Duration',
                        value: captureState.samples.isEmpty
                            ? '0s'
                            : '${((captureState.samples.last.tMs - captureState.samples.first.tMs) / 1000).toStringAsFixed(1)}s'),
                  ],
                ),
                const SizedBox(height: 16),
                if (isAnalyzing)
                  const CircularProgressIndicator()
                else
                  SizedBox(
                    width: double.infinity,
                    height: 52,
                    child: ElevatedButton(
                      style: ElevatedButton.styleFrom(
                        backgroundColor:
                            isCapturing ? AppTheme.bad : AppTheme.good,
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12)),
                      ),
                      onPressed: isConnected
                          ? () async {
                              if (!isCapturing) {
                                captureNotifier.startCapture();
                              } else {
                                final result =
                                    await captureNotifier.stopAndAnalyze();
                                if (result != null && context.mounted) {
                                  context.push('/result',
                                      extra: result);
                                }
                              }
                            }
                          : null,
                      child: Text(
                        isCapturing ? 'Stop & Analyze' : 'Start Recording',
                        style: const TextStyle(
                            fontSize: 16, fontWeight: FontWeight.bold),
                      ),
                    ),
                  ),
                if (captureState.status == CaptureStatus.error)
                  Padding(
                    padding: const EdgeInsets.only(top: 8),
                    child: Text(captureState.errorMessage ?? 'Error',
                        style: const TextStyle(color: AppTheme.bad)),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _Legend extends StatelessWidget {
  final Color color;
  final String label;
  const _Legend({required this.color, required this.label});

  @override
  Widget build(BuildContext context) => Row(
        children: [
          Container(width: 16, height: 3, color: color),
          const SizedBox(width: 4),
          Text(label, style: const TextStyle(color: Colors.white54, fontSize: 11)),
        ],
      );
}

class _Stat extends StatelessWidget {
  final String label;
  final String value;
  const _Stat({required this.label, required this.value});

  @override
  Widget build(BuildContext context) => Column(
        children: [
          Text(value,
              style: const TextStyle(
                  fontSize: 20, fontWeight: FontWeight.bold, color: Colors.white)),
          Text(label,
              style: const TextStyle(fontSize: 11, color: Colors.white54)),
        ],
      );
}
