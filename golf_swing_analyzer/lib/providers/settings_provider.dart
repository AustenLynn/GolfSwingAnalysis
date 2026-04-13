import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../services/storage/local_storage.dart';
import '../services/api/api_client.dart';
import '../core/constants/ble_constants.dart';

class AppSettings {
  final String apiBaseUrl;
  final double omegaStart;
  final double omegaEnd;

  const AppSettings({
    required this.apiBaseUrl,
    required this.omegaStart,
    required this.omegaEnd,
  });
}

class SettingsNotifier extends StateNotifier<AppSettings> {
  SettingsNotifier()
      : super(AppSettings(
          apiBaseUrl: LocalStorage.apiBaseUrl,
          omegaStart: LocalStorage.omegaStart,
          omegaEnd: LocalStorage.omegaEnd,
        ));

  Future<void> setApiUrl(String url) async {
    await LocalStorage.setApiBaseUrl(url);
    ApiClient.reset();
    state = AppSettings(
        apiBaseUrl: url, omegaStart: state.omegaStart, omegaEnd: state.omegaEnd);
  }

  Future<void> setOmegaStart(double v) async {
    await LocalStorage.setOmegaStart(v);
    state = AppSettings(
        apiBaseUrl: state.apiBaseUrl, omegaStart: v, omegaEnd: state.omegaEnd);
  }

  Future<void> setOmegaEnd(double v) async {
    await LocalStorage.setOmegaEnd(v);
    state = AppSettings(
        apiBaseUrl: state.apiBaseUrl, omegaStart: state.omegaStart, omegaEnd: v);
  }
}

final settingsProvider =
    StateNotifierProvider<SettingsNotifier, AppSettings>((ref) {
  return SettingsNotifier();
});
