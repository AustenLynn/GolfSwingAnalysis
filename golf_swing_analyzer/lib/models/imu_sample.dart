import 'dart:math' as math;

class ImuSample {
  final double tMs;
  final double ax, ay, az;
  final double gx, gy, gz;
  final double yaw, pitch, roll;

  const ImuSample({
    required this.tMs,
    required this.ax,
    required this.ay,
    required this.az,
    required this.gx,
    required this.gy,
    required this.gz,
    required this.yaw,
    required this.pitch,
    required this.roll,
  });

  double get omegaMag => math.sqrt(gx * gx + gy * gy + gz * gz);
  double get accMag => math.sqrt(ax * ax + ay * ay + az * az);

  /// Parse a comma-separated CSV line: t_ms,ax,ay,az,gx,gy,gz,yaw,pitch,roll
  static ImuSample? fromCsvLine(String line) {
    final parts = line.trim().split(',');
    if (parts.length != 10) return null;
    final nums = parts.map(double.tryParse).toList();
    if (nums.any((v) => v == null)) return null;
    return ImuSample(
      tMs: nums[0]!,
      ax: nums[1]!,
      ay: nums[2]!,
      az: nums[3]!,
      gx: nums[4]!,
      gy: nums[5]!,
      gz: nums[6]!,
      yaw: nums[7]!,
      pitch: nums[8]!,
      roll: nums[9]!,
    );
  }

  Map<String, dynamic> toJson() => {
        't_ms': tMs,
        'ax': ax,
        'ay': ay,
        'az': az,
        'gx': gx,
        'gy': gy,
        'gz': gz,
        'yaw': yaw,
        'pitch': pitch,
        'roll': roll,
      };
}
