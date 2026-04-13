import 'package:dio/dio.dart';
import '../../services/storage/local_storage.dart';

class ApiClient {
  static Dio? _dio;

  static Dio get instance {
    _dio ??= _build();
    return _dio!;
  }

  static Dio _build() => Dio(
        BaseOptions(
          baseUrl: LocalStorage.apiBaseUrl,
          connectTimeout: const Duration(seconds: 10),
          receiveTimeout: const Duration(seconds: 30),
          headers: {'Content-Type': 'application/json'},
        ),
      );

  /// Call this when the user saves a new API URL in Settings.
  static void reset() => _dio = null;
}
