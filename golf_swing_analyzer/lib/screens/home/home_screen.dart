import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../providers/ble_provider.dart';
import '../../providers/history_provider.dart';
import '../../core/theme/app_theme.dart';
import '../../widgets/verdict_chip.dart';

class HomeScreen extends ConsumerWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final historyAsync = ref.watch(historyProvider);
    final bleState = ref.watch(bleProvider);
    final isConnected = bleState.status == BleStatus.connected;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Golf Swing Analyzer'),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 16),
            child: Row(
              children: [
                Container(
                  width: 8,
                  height: 8,
                  decoration: BoxDecoration(
                    color: isConnected ? AppTheme.good : Colors.grey,
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 6),
                Text(
                  isConnected ? 'Connected' : 'Disconnected',
                  style: TextStyle(
                      fontSize: 12,
                      color: isConnected ? AppTheme.good : Colors.grey),
                ),
              ],
            ),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Session summary card
            historyAsync.when(
              loading: () => const _SummaryCardLoading(),
              error: (_, __) => const _SummaryCardError(),
              data: (swings) => _SummaryCard(swings: swings),
            ),

            const SizedBox(height: 24),

            // Start session button
            SizedBox(
              width: double.infinity,
              height: 60,
              child: ElevatedButton.icon(
                icon: const Icon(Icons.sports_golf, size: 24),
                label: const Text('Start New Session',
                    style:
                        TextStyle(fontSize: 17, fontWeight: FontWeight.bold)),
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppTheme.accent,
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(14)),
                ),
                onPressed: () => context.push('/capture'),
              ),
            ),

            const SizedBox(height: 28),

            // Recent swings
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('Recent Swings',
                    style: TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.bold,
                        color: Colors.white70)),
                TextButton(
                  onPressed: () => context.push('/history'),
                  child: const Text('See all'),
                ),
              ],
            ),
            const SizedBox(height: 8),
            historyAsync.when(
              loading: () =>
                  const Center(child: CircularProgressIndicator()),
              error: (_, __) => const SizedBox.shrink(),
              data: (swings) => swings.isEmpty
                  ? const Text('No swings yet — start a session above.',
                      style: TextStyle(color: Colors.white38))
                  : SingleChildScrollView(
                      scrollDirection: Axis.horizontal,
                      child: Row(
                        children: swings.take(5).map((s) {
                          return Padding(
                            padding: const EdgeInsets.only(right: 10),
                            child: GestureDetector(
                              onTap: () =>
                                  context.push('/history/${s.id}'),
                              child: VerdictChip(
                                  isGood: s.isGood,
                                  confidence: s.confidence,
                                  fontSize: 13),
                            ),
                          );
                        }).toList(),
                      ),
                    ),
            ),
          ],
        ),
      ),
    );
  }
}

class _SummaryCard extends StatelessWidget {
  final List swings;
  const _SummaryCard({required this.swings});

  @override
  Widget build(BuildContext context) {
    final good = swings.where((s) => s.isGood).length;
    final bad = swings.length - good;
    final goodFrac = swings.isEmpty ? 0.5 : good / swings.length;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('${swings.length} swings total',
                style: const TextStyle(
                    fontSize: 13, color: Colors.white54)),
            const SizedBox(height: 12),
            Row(
              children: [
                Text('$good',
                    style: const TextStyle(
                        fontSize: 28,
                        fontWeight: FontWeight.bold,
                        color: AppTheme.good)),
                const Text(' good  ',
                    style: TextStyle(color: Colors.white54)),
                Text('$bad',
                    style: const TextStyle(
                        fontSize: 28,
                        fontWeight: FontWeight.bold,
                        color: AppTheme.bad)),
                const Text(' bad',
                    style: TextStyle(color: Colors.white54)),
              ],
            ),
            const SizedBox(height: 12),
            ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: SizedBox(
                height: 8,
                child: Row(
                  children: [
                    Expanded(
                      flex: (goodFrac * 100).round(),
                      child: Container(color: AppTheme.good),
                    ),
                    Expanded(
                      flex: ((1 - goodFrac) * 100).round(),
                      child: Container(color: AppTheme.bad),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _SummaryCardLoading extends StatelessWidget {
  const _SummaryCardLoading();
  @override
  Widget build(BuildContext context) =>
      const Card(child: Padding(
        padding: EdgeInsets.all(20),
        child: Center(child: CircularProgressIndicator()),
      ));
}

class _SummaryCardError extends StatelessWidget {
  const _SummaryCardError();
  @override
  Widget build(BuildContext context) =>
      const Card(child: Padding(
        padding: EdgeInsets.all(20),
        child: Text('Could not load history — check API connection in Settings.',
            style: TextStyle(color: Colors.white38)),
      ));
}
