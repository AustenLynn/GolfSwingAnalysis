import 'dart:async';
import '../../models/imu_sample.dart';

/// Reassembles 20-byte BLE notification chunks into complete CSV lines,
/// then parses each line into an ImuSample.
///
/// Mirrors the partial-line buffer logic in capture_imu_ble.py:
///   while "\n" in partial:
///       line, partial = partial.split("\n", 1)
///       ... parse line ...
class ImuStreamParser {
  final _controller = StreamController<ImuSample>.broadcast();
  String _partial = '';
  bool _headerFound = false;

  Stream<ImuSample> get stream => _controller.stream;

  void onData(List<int> bytes) {
    final chunk = String.fromCharCodes(bytes);
    _partial += chunk;

    while (_partial.contains('\n')) {
      final idx = _partial.indexOf('\n');
      final line = _partial.substring(0, idx).trim();
      _partial = _partial.substring(idx + 1);

      if (line.isEmpty || line.startsWith('#')) continue;

      // Skip the CSV header row
      if (!_headerFound && line.startsWith('t_ms')) {
        _headerFound = true;
        continue;
      }

      final sample = ImuSample.fromCsvLine(line);
      if (sample != null) {
        _headerFound = true; // implicit header detection
        _controller.add(sample);
      }
    }
  }

  void reset() {
    _partial = '';
    _headerFound = false;
  }

  void dispose() {
    _controller.close();
  }
}
