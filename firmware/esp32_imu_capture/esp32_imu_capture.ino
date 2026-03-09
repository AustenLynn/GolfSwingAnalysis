#include <Wire.h>
#include <MadgwickAHRS.h>
#include <math.h>

#define SDA_PIN 6
#define SCL_PIN 5
#define ADXL345_ADDR   0x53
#define ITG3200_ADDR   0x68
#define HMC5883L_ADDR  0x1E

Madgwick filter;

unsigned long lastMicros = 0;
float sampleFreq = 100.0f;

float gx_offset = 0.0f;
float gy_offset = 0.0f;
float gz_offset = 0.0f;

// -------------------------
// I2C helpers
// -------------------------
void writeRegister8(uint8_t addr, uint8_t reg, uint8_t value) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

void readRegisters(uint8_t addr, uint8_t startReg, uint8_t count, uint8_t *dest) {
  Wire.beginTransmission(addr);
  Wire.write(startReg);
  Wire.endTransmission(false);
  Wire.requestFrom((int)addr, (int)count);

  uint8_t i = 0;
  while (Wire.available() && i < count) {
    dest[i++] = Wire.read();
  }
}

int16_t read16BE(uint8_t addr, uint8_t reg) {
  uint8_t buf[2];
  readRegisters(addr, reg, 2, buf);
  return (int16_t)((buf[0] << 8) | buf[1]);
}

int16_t read16LE(uint8_t addr, uint8_t reg) {
  uint8_t buf[2];
  readRegisters(addr, reg, 2, buf);
  return (int16_t)((buf[1] << 8) | buf[0]);
}

// -------------------------
// Sensor init
// -------------------------
void initADXL345() {
  writeRegister8(ADXL345_ADDR, 0x2D, 0x08); // measure mode
  writeRegister8(ADXL345_ADDR, 0x31, 0x08); // full resolution, +-2g
  writeRegister8(ADXL345_ADDR, 0x2C, 0x0A); // 100 Hz
}

void initITG3200() {
  writeRegister8(ITG3200_ADDR, 0x3E, 0x01); // PLL with X gyro
  writeRegister8(ITG3200_ADDR, 0x16, 0x1B); // +-2000 deg/s
  writeRegister8(ITG3200_ADDR, 0x15, 0x09); // ~100 Hz
}

void initHMC5883L() {
  writeRegister8(HMC5883L_ADDR, 0x00, 0x70);
  writeRegister8(HMC5883L_ADDR, 0x01, 0x20);
  writeRegister8(HMC5883L_ADDR, 0x02, 0x00);
}

// -------------------------
// Sensor read
// -------------------------
void readADXL345(float &ax, float &ay, float &az) {
  int16_t rawX = read16LE(ADXL345_ADDR, 0x32);
  int16_t rawY = read16LE(ADXL345_ADDR, 0x34);
  int16_t rawZ = read16LE(ADXL345_ADDR, 0x36);

  ax = rawX * 0.0039f;
  ay = rawY * 0.0039f;
  az = rawZ * 0.0039f;
}

void readITG3200(float &gx, float &gy, float &gz) {
  int16_t rawX = read16BE(ITG3200_ADDR, 0x1D);
  int16_t rawY = read16BE(ITG3200_ADDR, 0x1F);
  int16_t rawZ = read16BE(ITG3200_ADDR, 0x21);

  gx = ((float)rawX / 14.375f) - gx_offset;
  gy = ((float)rawY / 14.375f) - gy_offset;
  gz = ((float)rawZ / 14.375f) - gz_offset;
}

void readHMC5883L(float &mx, float &my, float &mz) {
  int16_t rawX = read16BE(HMC5883L_ADDR, 0x03);
  int16_t rawZ = read16BE(HMC5883L_ADDR, 0x05);
  int16_t rawY = read16BE(HMC5883L_ADDR, 0x07);

  mx = (float)rawX;
  my = (float)rawY;
  mz = (float)rawZ;
}

// -------------------------
// Gyro calibration
// -------------------------
void calibrateGyro(int samples = 500) {
  Serial.println("# Calibrando gyro. No muevas el sensor...");
  float sx = 0, sy = 0, sz = 0;

  for (int i = 0; i < samples; i++) {
    float gx, gy, gz;
    readITG3200(gx, gy, gz);
    sx += gx;
    sy += gy;
    sz += gz;
    delay(5);
  }

  gx_offset = sx / samples;
  gy_offset = sy / samples;
  gz_offset = sz / samples;

  Serial.print("# Offsets: ");
  Serial.print(gx_offset); Serial.print(", ");
  Serial.print(gy_offset); Serial.print(", ");
  Serial.println(gz_offset);
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);

  initADXL345();
  delay(50);
  initITG3200();
  delay(50);
  initHMC5883L();
  delay(50);

  filter.begin(sampleFreq);

  calibrateGyro();

  lastMicros = micros();

  // encabezado CSV
  Serial.println("t_ms,ax,ay,az,gx,gy,gz,yaw,pitch,roll");
}

void loop() {
  float ax, ay, az;
  float gx, gy, gz;
  float mx, my, mz;

  readADXL345(ax, ay, az);
  readITG3200(gx, gy, gz);
  readHMC5883L(mx, my, mz);

  unsigned long nowMicros = micros();
  float dt = (nowMicros - lastMicros) * 1e-6f;
  lastMicros = nowMicros;

  if (dt > 0.0001f && dt < 0.1f) {
    sampleFreq = 1.0f / dt;
  }

  filter.begin(sampleFreq);

  float gx_rad = gx * DEG_TO_RAD;
  float gy_rad = gy * DEG_TO_RAD;
  float gz_rad = gz * DEG_TO_RAD;

  filter.update(gx_rad, gy_rad, gz_rad, ax, ay, az, mx, my, mz);

  float roll  = filter.getRoll();
  float pitch = filter.getPitch();
  float yaw   = filter.getYaw();

  unsigned long t_ms = millis();

  Serial.print(t_ms);   Serial.print(",");
  Serial.print(ax, 4);  Serial.print(",");
  Serial.print(ay, 4);  Serial.print(",");
  Serial.print(az, 4);  Serial.print(",");
  Serial.print(gx, 4);  Serial.print(",");
  Serial.print(gy, 4);  Serial.print(",");
  Serial.print(gz, 4);  Serial.print(",");
  Serial.print(yaw, 2); Serial.print(",");
  Serial.print(pitch, 2); Serial.print(",");
  Serial.println(roll, 2);

  delay(10); // ~100 Hz
}