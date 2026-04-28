import socket
from collections import deque
import joblib
from features import extract_features

model = joblib.load("motion_model.joblib")

WINDOW = 75
buffer = deque(maxlen=WINDOW)
pred_buffer = deque(maxlen=5)

UDP_IP = "0.0.0.0"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for UDP packets on port {UDP_PORT}...")

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

    if len(buffer) == WINDOW:
      window = np.array(buffer)

      features = extract_features(window)
      features = np.array(features).reshape(1, -1)

      prediction = model.predict(features)[0]

      print("Prediction:", prediction)

      pred_buffer.append(prediction)

      if len(pred_buffer) == 5:
          final = max(set(pred_buffer), key=pred_buffer.count)
          print("Smoothed:", final)