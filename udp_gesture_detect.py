import socket
from collections import deque
import joblib
from features import extract_features
import numpy as np

model = joblib.load("motion_model.joblib")

WINDOW = 40

buffer = deque(maxlen=WINDOW)
pred_buffer = deque(maxlen=3)
PREDICT_EVERY = 3

UDP_IP = "0.0.0.0"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for UDP packets on port {UDP_PORT}...")

count = 0

while True:
    data, addr = sock.recvfrom(1024)
    line = data.decode(errors="ignore").strip()

    parts = line.split(",")

    if len(parts) != 7:
        continue

    try:
        values = [float(x) for x in parts]
    except ValueError:
        continue

    sample = values[1:7]  # skip timestamp
    buffer.append(sample)
    count += 1

    if len(buffer) == WINDOW and count % PREDICT_EVERY == 0:
        window = np.array(buffer)

        features = extract_features(window)
        features = np.array(features).reshape(1, -1)

        prediction = model.predict(features)[0]
        pred_buffer.append(prediction)

        final = max(set(pred_buffer), key=pred_buffer.count)

        print(f"Prediction: {prediction} | Smoothed: {final}")