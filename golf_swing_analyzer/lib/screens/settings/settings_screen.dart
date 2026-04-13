import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../providers/settings_provider.dart';
import '../../services/api/swing_api.dart';
import '../../core/theme/app_theme.dart';

class SettingsScreen extends ConsumerStatefulWidget {
  const SettingsScreen({super.key});

  @override
  ConsumerState<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends ConsumerState<SettingsScreen> {
  late TextEditingController _urlController;
  bool? _connectionOk;
  bool _testing = false;
  bool _retraining = false;
  Map<String, dynamic>? _modelInfo;

  @override
  void initState() {
    super.initState();
    final settings = ref.read(settingsProvider);
    _urlController = TextEditingController(text: settings.apiBaseUrl);
    _loadModelInfo();
  }

  Future<void> _loadModelInfo() async {
    try {
      final info = await SwingApi.getModelInfo();
      setState(() => _modelInfo = info);
    } catch (_) {}
  }

  Future<void> _testConnection() async {
    setState(() {
      _testing = true;
      _connectionOk = null;
    });
    await ref.read(settingsProvider.notifier).setApiUrl(_urlController.text.trim());
    final ok = await SwingApi.checkHealth();
    setState(() {
      _testing = false;
      _connectionOk = ok;
    });
    if (ok) _loadModelInfo();
  }

  @override
  Widget build(BuildContext context) {
    final settings = ref.watch(settingsProvider);

    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // API URL
          const Text('Backend API URL',
              style: TextStyle(
                  color: Colors.white70, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _urlController,
                  style: const TextStyle(color: Colors.white),
                  decoration: InputDecoration(
                    hintText: 'http://192.168.x.x:8000',
                    hintStyle: const TextStyle(color: Colors.white38),
                    filled: true,
                    fillColor: const Color(0xFF2A2A2A),
                    border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(8),
                        borderSide: BorderSide.none),
                    suffixIcon: _connectionOk == null
                        ? null
                        : Icon(
                            _connectionOk!
                                ? Icons.check_circle
                                : Icons.error,
                            color: _connectionOk!
                                ? AppTheme.good
                                : AppTheme.bad),
                  ),
                ),
              ),
              const SizedBox(width: 10),
              _testing
                  ? const SizedBox(
                      width: 36,
                      height: 36,
                      child: CircularProgressIndicator(strokeWidth: 2))
                  : ElevatedButton(
                      onPressed: _testConnection,
                      child: const Text('Test')),
            ],
          ),

          const SizedBox(height: 28),

          // Omega thresholds
          const Text('Swing Detection Thresholds',
              style: TextStyle(
                  color: Colors.white70, fontWeight: FontWeight.bold)),
          const SizedBox(height: 4),
          _ThresholdSlider(
            label: 'Start threshold',
            value: settings.omegaStart,
            min: 20,
            max: 150,
            unit: 'deg/s',
            onChanged: (v) =>
                ref.read(settingsProvider.notifier).setOmegaStart(v),
          ),
          _ThresholdSlider(
            label: 'End threshold',
            value: settings.omegaEnd,
            min: 10,
            max: 80,
            unit: 'deg/s',
            onChanged: (v) =>
                ref.read(settingsProvider.notifier).setOmegaEnd(v),
          ),

          const SizedBox(height: 28),

          // Model info
          const Text('ML Model',
              style: TextStyle(
                  color: Colors.white70, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          if (_modelInfo != null) ...[
            _InfoRow('Type', _modelInfo!['model_class'] ?? '—'),
            _InfoRow('Training samples',
                '${_modelInfo!['n_training_samples']} (${_modelInfo!['n_good']}G / ${_modelInfo!['n_bad']}B)'),
            _InfoRow('Features',
                (_modelInfo!['features'] as List).join(', ')),
          ],
          const SizedBox(height: 12),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF333333)),
              onPressed: _retraining
                  ? null
                  : () async {
                      setState(() => _retraining = true);
                      try {
                        await SwingApi.retrain();
                        await _loadModelInfo();
                        if (context.mounted) {
                          ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(
                                  content: Text('Model retrained!')));
                        }
                      } catch (e) {
                        if (context.mounted) {
                          ScaffoldMessenger.of(context).showSnackBar(
                              SnackBar(content: Text('Error: $e')));
                        }
                      } finally {
                        setState(() => _retraining = false);
                      }
                    },
              child: _retraining
                  ? const SizedBox(
                      height: 18,
                      width: 18,
                      child: CircularProgressIndicator(strokeWidth: 2))
                  : const Text('Retrain Model'),
            ),
          ),
        ],
      ),
    );
  }
}

class _ThresholdSlider extends StatelessWidget {
  final String label;
  final double value;
  final double min, max;
  final String unit;
  final ValueChanged<double> onChanged;

  const _ThresholdSlider({
    required this.label,
    required this.value,
    required this.min,
    required this.max,
    required this.unit,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) => Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(label,
                  style: const TextStyle(color: Colors.white54, fontSize: 13)),
              Text('${value.toInt()} $unit',
                  style: const TextStyle(
                      color: Colors.white, fontWeight: FontWeight.bold)),
            ],
          ),
          Slider(
            value: value,
            min: min,
            max: max,
            divisions: ((max - min) / 5).round(),
            onChanged: onChanged,
          ),
        ],
      );
}

class _InfoRow extends StatelessWidget {
  final String label, value;
  const _InfoRow(this.label, this.value);

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 3),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(
              width: 120,
              child: Text(label,
                  style: const TextStyle(
                      color: Colors.white54, fontSize: 12)),
            ),
            Expanded(
              child: Text(value,
                  style: const TextStyle(color: Colors.white, fontSize: 12)),
            ),
          ],
        ),
      );
}
