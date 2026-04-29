#include <Arduino_LSM6DSOX.h>

float Ax, Ay, Az;
float Gx, Gy, Gz;

unsigned long last = 0;
const int interval = 10;

void setup() {
  Serial.begin(115200);  // faster baud rate (important!)

  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  Serial.println("t,ax,ay,az,gx,gy,gz");
}

void loop() {

  unsigned long now = millis();

  if (now - last >= interval) {
    last = now;

    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      IMU.readAcceleration(Ax, Ay, Az);
      IMU.readGyroscope(Gx, Gy, Gz);


      Serial.print(now);
      Serial.print(",");
      Serial.print(Ax); Serial.print(",");
      Serial.print(Ay); Serial.print(",");
      Serial.print(Az); Serial.print(",");
      Serial.print(Gx); Serial.print(",");
      Serial.print(Gy); Serial.print(",");
      Serial.println(Gz);
    }
  }
}