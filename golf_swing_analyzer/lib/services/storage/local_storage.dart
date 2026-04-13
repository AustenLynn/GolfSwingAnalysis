import 'package:shared_preferences/shared_preferences.dart';
import '../../core/constants/app_constants.dart';
import '../../core/constants/ble_constants.dart';

class LocalStorage {
  static SharedPreferences? _prefs;

  static Future<void> init() async {
    _prefs = await SharedPreferences.getInstance();
  }

  static String get apiBaseUrl =>
      _prefs?.getString(AppConstants.apiUrlKey) ?? AppConstants.defaultApiUrl;

  static Future<void> setApiBaseUrl(String url) async =>
      _prefs?.setString(AppConstants.apiUrlKey, url);

  static double get omegaStart =>
      _prefs?.getDouble(AppConstants.omegaStartKey) ??
      BleConstants.omegaStartThreshold;

  static Future<void> setOmegaStart(double v) async =>
      _prefs?.setDouble(AppConstants.omegaStartKey, v);

  static double get omegaEnd =>
      _prefs?.getDouble(AppConstants.omegaEndKey) ??
      BleConstants.omegaEndThreshold;

  static Future<void> setOmegaEnd(double v) async =>
      _prefs?.setDouble(AppConstants.omegaEndKey, v);
}
