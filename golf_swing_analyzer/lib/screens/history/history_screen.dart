import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:intl/intl.dart';
import '../../providers/history_provider.dart';
import '../../models/swing_history_entry.dart';
import '../../core/theme/app_theme.dart';
import '../../widgets/verdict_chip.dart';

class HistoryScreen extends ConsumerStatefulWidget {
  const HistoryScreen({super.key});

  @override
  ConsumerState<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends ConsumerState<HistoryScreen> {
  int? _filterLabel; // null = all

  @override
  Widget build(BuildContext context) {
    final historyAsync = ref.watch(historyProvider);

    return Scaffold(
      appBar: AppBar(title: const Text('Swing History')),
      body: Column(
        children: [
          // Filter chips
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: Row(
              children: [
                _FilterChip(
                    label: 'All',
                    selected: _filterLabel == null,
                    onTap: () {
                      setState(() => _filterLabel = null);
                      ref.read(historyProvider.notifier).load();
                    }),
                const SizedBox(width: 8),
                _FilterChip(
                    label: 'Good',
                    selected: _filterLabel == 1,
                    color: AppTheme.good,
                    onTap: () {
                      setState(() => _filterLabel = 1);
                      ref.read(historyProvider.notifier).load(label: 1);
                    }),
                const SizedBox(width: 8),
                _FilterChip(
                    label: 'Bad',
                    selected: _filterLabel == 0,
                    color: AppTheme.bad,
                    onTap: () {
                      setState(() => _filterLabel = 0);
                      ref.read(historyProvider.notifier).load(label: 0);
                    }),
              ],
            ),
          ),
          Expanded(
            child: historyAsync.when(
              loading: () =>
                  const Center(child: CircularProgressIndicator()),
              error: (e, _) => Center(
                  child: Text('Error: $e',
                      style: const TextStyle(color: AppTheme.bad))),
              data: (swings) => swings.isEmpty
                  ? const Center(
                      child: Text('No swings yet.',
                          style: TextStyle(color: Colors.white38)))
                  : RefreshIndicator(
                      onRefresh: () =>
                          ref.read(historyProvider.notifier).refresh(),
                      child: ListView.separated(
                        padding: const EdgeInsets.all(12),
                        itemCount: swings.length,
                        separatorBuilder: (_, __) =>
                            const SizedBox(height: 6),
                        itemBuilder: (_, i) =>
                            _SwingTile(entry: swings[i]),
                      ),
                    ),
            ),
          ),
        ],
      ),
    );
  }
}

class _SwingTile extends StatelessWidget {
  final SwingHistoryEntry entry;
  const _SwingTile({required this.entry});

  @override
  Widget build(BuildContext context) {
    final fmt = DateFormat('MMM d, HH:mm');
    return Card(
      child: ListTile(
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        leading: VerdictChip(
            isGood: entry.isGood, confidence: entry.confidence),
        title: Text(fmt.format(entry.capturedAt),
            style: const TextStyle(color: Colors.white)),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 4),
            Row(
              children: [
                if (entry.tempoRatio != null)
                  _Kv(k: 'Tempo', v: '${entry.tempoRatio!.toStringAsFixed(1)}:1'),
                if (entry.accMean != null) ...[
                  const SizedBox(width: 12),
                  _Kv(k: 'acc', v: '${entry.accMean!.toStringAsFixed(2)}g'),
                ],
                if (entry.omegaMean != null) ...[
                  const SizedBox(width: 12),
                  _Kv(k: 'ω', v: '${entry.omegaMean!.toStringAsFixed(0)}°/s'),
                ],
              ],
            ),
          ],
        ),
        trailing: const Icon(Icons.chevron_right, color: Colors.white38),
        onTap: () => context.push('/history/${entry.id}'),
      ),
    );
  }
}

class _Kv extends StatelessWidget {
  final String k, v;
  const _Kv({required this.k, required this.v});

  @override
  Widget build(BuildContext context) => Text(
        '$k: $v',
        style: const TextStyle(color: Colors.white54, fontSize: 12),
      );
}

class _FilterChip extends StatelessWidget {
  final String label;
  final bool selected;
  final Color color;
  final VoidCallback onTap;

  const _FilterChip({
    required this.label,
    required this.selected,
    this.color = Colors.white,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) => GestureDetector(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
          decoration: BoxDecoration(
            color: selected ? color.withOpacity(0.15) : Colors.transparent,
            border: Border.all(color: selected ? color : Colors.white24),
            borderRadius: BorderRadius.circular(20),
          ),
          child: Text(label,
              style: TextStyle(
                  color: selected ? color : Colors.white54,
                  fontSize: 13,
                  fontWeight:
                      selected ? FontWeight.bold : FontWeight.normal)),
        ),
      );
}
