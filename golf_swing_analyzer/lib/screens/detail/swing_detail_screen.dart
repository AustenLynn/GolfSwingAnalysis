import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../models/swing_result.dart';
import '../../services/api/swing_api.dart';
import '../result/widgets/verdict_banner.dart';
import '../result/widgets/phase_timeline.dart';
import '../result/widgets/feature_breakdown_list.dart';

class SwingDetailScreen extends ConsumerStatefulWidget {
  final String swingId;
  const SwingDetailScreen({super.key, required this.swingId});

  @override
  ConsumerState<SwingDetailScreen> createState() => _SwingDetailScreenState();
}

class _SwingDetailScreenState extends ConsumerState<SwingDetailScreen> {
  SwingResult? _result;
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final r = await SwingApi.getSwingDetail(widget.swingId);
      setState(() {
        _result = r;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Swing Detail')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? Center(
                  child: Text(_error!,
                      style: const TextStyle(color: Colors.red)))
              : SingleChildScrollView(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      VerdictBanner(result: _result!),
                      const SizedBox(height: 8),
                      PhaseTimeline(timing: _result!.phaseTiming),
                      const Divider(color: Colors.white12),
                      FeatureBreakdownList(
                        importances: _result!.featureImportances,
                        showAll: true,
                      ),
                      const SizedBox(height: 24),
                    ],
                  ),
                ),
    );
  }
}
