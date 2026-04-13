import '../../models/analyze_response.dart';
import '../../models/swing_history_entry.dart';
import '../../models/swing_result.dart';
import '../../models/swing_session.dart';
import 'api_client.dart';

class SwingApi {
  static Future<bool> checkHealth() async {
    try {
      final resp = await ApiClient.instance.get('/health');
      return resp.data['status'] == 'ok';
    } catch (_) {
      return false;
    }
  }

  static Future<AnalyzeResponse> postAnalyze(SwingSession session) async {
    final resp = await ApiClient.instance.post(
      '/analyze',
      data: session.toJson(),
    );
    return AnalyzeResponse.fromJson(resp.data as Map<String, dynamic>);
  }

  static Future<List<SwingHistoryEntry>> getSwings({
    int page = 1,
    int pageSize = 20,
    int? label,
  }) async {
    final resp = await ApiClient.instance.get(
      '/swings',
      queryParameters: {
        'page': page,
        'page_size': pageSize,
        if (label != null) 'label': label,
      },
    );
    final list = resp.data['swings'] as List;
    return list
        .map((e) => SwingHistoryEntry.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  static Future<SwingResult> getSwingDetail(String id) async {
    final resp = await ApiClient.instance.get('/swings/$id');
    return SwingResult.fromJson(resp.data as Map<String, dynamic>);
  }

  static Future<void> labelSwing(String id, int label) async {
    await ApiClient.instance.post('/swings/$id/label', data: {'label': label});
  }

  static Future<Map<String, dynamic>> getModelInfo() async {
    final resp = await ApiClient.instance.get('/model/info');
    return resp.data as Map<String, dynamic>;
  }

  static Future<Map<String, dynamic>> retrain() async {
    final resp = await ApiClient.instance.post('/model/retrain');
    return resp.data as Map<String, dynamic>;
  }
}
