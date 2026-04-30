from flask import Flask, jsonify, render_template_string
import threading
import socket
from collections import deque
import numpy as np
import joblib
import time

from features import extract_features

app = Flask(__name__)

# Shared state
latest_prediction = "Waiting..."
latest_confidence = 0.0
packet_count = 0
motion_level = 0
MOTION_THRESHOLD = 0.4

WINDOW = 20
buffer = deque(maxlen=WINDOW)
pred_buffer = deque(maxlen=1)

model = joblib.load("motion_model.joblib")

UDP_IP = "0.0.0.0"
UDP_PORT = 5005


def udp_loop():
    global latest_prediction, latest_confidence, packet_count, motion_level

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.setblocking(False)

    while True:
        latest_data = None

        # Drain all waiting packets and keep only newest one
        while True:
            try:
                data, addr = sock.recvfrom(1024)
                latest_data = data
            except BlockingIOError:
                break

        if latest_data is None:    
            time.sleep(0.001)
            continue

        line = latest_data.decode(errors="ignore").strip()

        parts = line.split(",")
        if len(parts) != 8:
            continue

        try:
            values = [float(x) for x in parts]
        except ValueError:
            continue

        sample = values[1:7]
        buffer.append(sample)
        packet_count += 1

        if len(buffer) == WINDOW:
            window = np.array(buffer)

            features = extract_features(window)
            features = np.array(features).reshape(1, -1)

            accel_mag = np.linalg.norm(window[:, 0:3], axis=1)
            motion_level = accel_mag.std()

            if motion_level < MOTION_THRESHOLD:   # tune this
                prediction = "idle"
                confidence = 0.9
            else:
                prediction = model.predict(features)[0]

                if hasattr(model, "predict_proba"):
                    confidence = model.predict_proba(features).max()
                else:
                    confidence = 0.0

            pred_buffer.append(prediction)
            smoothed = max(set(pred_buffer), key=pred_buffer.count)

            latest_prediction = smoothed
            latest_confidence = confidence

@app.route("/")
def index():
    return render_template_string("""
    <html>
    <head>
        <title>IMU Gesture Classifier</title>
        <style>
            body {
                font-family: Arial;
                text-align: center;
                margin-top: 50px;
            }
            .gesture {
                font-size: 60px;
                font-weight: bold;
                color: blue;
            }
            .info {
                font-size: 24px;
                margin-top: 20px;
            }
            .motion_level {
                font-size: 24px;
                margin-top: 20px;
                color: #444;
            }
        </style>
    </head>
    <body>
        <h1>Wireless IMU Gesture</h1>
        <div class="gesture" id="gesture">Waiting...</div>
        <div class="info" id="confidence"></div>
        <div class="info" id="packets"></div>
        <div class="motion_level" id="motion_level"></div>

        <script>
            async function update() {
              try {
                  const res = await fetch('/data');
                  const data = await res.json();

                  document.getElementById('gesture').innerText = data.prediction;
                                  
                  document.getElementById('confidence').innerText =
                      "Confidence: " + data.confidence.toFixed(2);
                                  
                  document.getElementById('packets').innerText =
                      "Packets: " + data.packets;
                                  
                    document.getElementById('motion_level').innerText =
                        "Motion Level: " + data.motion_level.toFixed(4) +
                         " | Threshold: " + data.motion_threshold.toFixed(4);
                                  
              } catch (err) {
                  console.log(err);
              }

              setTimeout(update, 50);
          }

          update();
        </script>
    </body>
    </html>
    """)


@app.route("/data")
def data():
    return jsonify({
        "prediction": latest_prediction,
        "confidence": latest_confidence,
        "packets": packet_count,
        "motion_level": motion_level,
        "motion_threshold": MOTION_THRESHOLD
    })


if __name__ == "__main__":
    thread = threading.Thread(target=udp_loop, daemon=True)
    thread.start()

    app.run(host="0.0.0.0", port=5000, threaded=True)