class BleConstants {
  static const String deviceName = 'ESP32_IMU_GOLF';
  static const String serviceUuid = '6e400001-b5a3-f393-e0a9-e50e24dcca9e';
  static const String txCharUuid = '6e400003-b5a3-f393-e0a9-e50e24dcca9e';

  // Swing detection thresholds (mirrors analyze_swing.py)
  static const double omegaStartThreshold = 60.0; // deg/s
  static const double omegaEndThreshold = 30.0;   // deg/s
  static const int minSwingSamples = 25;

  // Live chart rolling window (samples)
  static const int chartWindowSize = 200;
}
