import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/swing_history_entry.dart';
import '../services/api/swing_api.dart';

class HistoryNotifier
    extends StateNotifier<AsyncValue<List<SwingHistoryEntry>>> {
  HistoryNotifier() : super(const AsyncValue.loading()) {
    load();
  }

  Future<void> load({int? label}) async {
    state = const AsyncValue.loading();
    try {
      final swings = await SwingApi.getSwings(label: label);
      state = AsyncValue.data(swings);
    } catch (e, st) {
      state = AsyncValue.error(e, st);
    }
  }

  Future<void> refresh() => load();
}

final historyProvider = StateNotifierProvider<HistoryNotifier,
    AsyncValue<List<SwingHistoryEntry>>>((ref) {
  return HistoryNotifier();
});
