import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../models/analyze_response.dart';
import '../../models/swing_result.dart';
import '../../services/api/swing_api.dart';
import 'widgets/verdict_banner.dart';
import 'widgets/phase_timeline.dart';
import 'widgets/feature_breakdown_list.dart';

class ResultScreen extends ConsumerStatefulWidget {
  final AnalyzeResponse response;
  const ResultScreen({super.key, required this.response});

  @override
  ConsumerState<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends ConsumerState<ResultScreen> {
  int _page = 0;
  bool _showAll = false;

  @override
  Widget build(BuildContext context) {
    if (widget.response.swingsDetected == 0) {
      return Scaffold(
        appBar: AppBar(title: const Text('Results')),
        body: const Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.sports_golf, size: 64, color: Colors.white24),
              SizedBox(height: 16),
              Text('No swings detected in this session.',
                  style: TextStyle(color: Colors.white54)),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: Text(
          widget.response.swingsDetected > 1
              ? 'Swing ${_page + 1} of ${widget.response.swingsDetected}'
              : 'Swing Result',
        ),
      ),
      body: PageView.builder(
        onPageChanged: (i) => setState(() => _page = i),
        itemCount: widget.response.swingsDetected,
        itemBuilder: (_, i) {
          final result = widget.response.swings[i];
          return _SwingPage(
            result: result,
            showAll: _showAll,
            onToggleShowAll: () => setState(() => _showAll = !_showAll),
          );
        },
      ),
    );
  }
}

class _SwingPage extends StatelessWidget {
  final SwingResult result;
  final bool showAll;
  final VoidCallback onToggleShowAll;

  const _SwingPage({
    required this.result,
    required this.showAll,
    required this.onToggleShowAll,
  });

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          VerdictBanner(result: result),
          const SizedBox(height: 8),
          PhaseTimeline(timing: result.phaseTiming),
          const Divider(color: Colors.white12),
          FeatureBreakdownList(
            importances: result.featureImportances,
            showAll: showAll,
          ),
          if (result.featureImportances.length > 6)
            TextButton(
              onPressed: onToggleShowAll,
              child: Text(showAll ? 'Show less' : 'Show all features'),
            ),
          const SizedBox(height: 12),
          _LabelControl(result: result),
          const SizedBox(height: 24),
        ],
      ),
    );
  }
}

class _LabelControl extends StatefulWidget {
  final SwingResult result;
  const _LabelControl({required this.result});

  @override
  State<_LabelControl> createState() => _LabelControlState();
}

class _LabelControlState extends State<_LabelControl> {
  int? _selected;
  bool _saving = false;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Correct the label for training:',
              style: TextStyle(color: Colors.white54, fontSize: 13)),
          const SizedBox(height: 8),
          Row(
            children: [
              _LabelBtn(
                label: 'Good',
                selected: _selected == 1,
                color: Colors.green,
                onTap: () => setState(() => _selected = 1),
              ),
              const SizedBox(width: 12),
              _LabelBtn(
                label: 'Bad',
                selected: _selected == 0,
                color: Colors.red,
                onTap: () => setState(() => _selected = 0),
              ),
              const SizedBox(width: 12),
              if (_selected != null)
                _saving
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(strokeWidth: 2))
                    : ElevatedButton(
                        onPressed: () async {
                          if (widget.result.id == null) return;
                          setState(() => _saving = true);
                          await SwingApi.labelSwing(
                              widget.result.id!, _selected!);
                          setState(() => _saving = false);
                          if (context.mounted) {
                            ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(content: Text('Label saved')));
                          }
                        },
                        child: const Text('Save'),
                      ),
            ],
          ),
        ],
      ),
    );
  }
}

class _LabelBtn extends StatelessWidget {
  final String label;
  final bool selected;
  final Color color;
  final VoidCallback onTap;

  const _LabelBtn(
      {required this.label,
      required this.selected,
      required this.color,
      required this.onTap});

  @override
  Widget build(BuildContext context) => GestureDetector(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          decoration: BoxDecoration(
            color: selected ? color.withOpacity(0.3) : Colors.transparent,
            border: Border.all(color: selected ? color : Colors.white24),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Text(label,
              style: TextStyle(
                  color: selected ? color : Colors.white54,
                  fontWeight:
                      selected ? FontWeight.bold : FontWeight.normal)),
        ),
      );
}
