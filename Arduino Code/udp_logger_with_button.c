// Add a button state to the message to control trial recording
#include <Arduino_LSM6DSOX.h>
#include <WiFiNINA.h>
#include <WiFiUdp.h>
#include "secrets.h"

char ssid[] = SECRET_SSID;
char pass[] = SECRET_PASS;

WiFiUDP udp;

const char* receiverIP = "192.168.50.1";  // your laptop IP
const int receiverPort = 5005;

float Ax, Ay, Az;
float Gx, Gy, Gz;

// Button Variables

#define button 9

bool buttonState = false;
bool lastReading = false;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 30; // ms

void setup() {
  Serial.begin(115200);
  udp.begin(5006);
  pinMode(button, INPUT_PULLUP);
  // while (!Serial);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  Serial.print("Connecting to WiFi");
  while (WiFi.begin(ssid, pass) != WL_CONNECTED) {
    Serial.print(".");
    delay(1000);
  }

  Serial.println();
  Serial.println("Connected!");
  Serial.print("Arduino IP: ");
  Serial.println(WiFi.localIP());

}

void loop() {

  bool reading = !digitalRead(button); // pressed = true

  if (reading != lastReading) {
    lastDebounceTime = millis();
  }
  if ((millis() - lastDebounceTime) > debounceDelay) {
    buttonState = reading;
  }
  lastReading = reading;


  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    IMU.readAcceleration(Ax, Ay, Az);
    IMU.readGyroscope(Gx, Gy, Gz);

    unsigned long t = millis();

    String packet = String(t) + "," +
                    String(Ax, 4) + "," +
                    String(Ay, 4) + "," +
                    String(Az, 4) + "," +
                    String(Gx, 4) + "," +
                    String(Gy, 4) + "," +
                    String(Gz, 4) + "," +
                    String(buttonState);

    udp.beginPacket(receiverIP, receiverPort);
    udp.print(packet);
    udp.endPacket();

    Serial.println(packet);
  }

  delay(10);
}