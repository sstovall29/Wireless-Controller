import tkinter as tk
import threading
import socket
from collections import deque
import numpy as np
import joblib

from features import extract_features

WINDOW = 40
UDP_IP = "0.0.0.0"
UDP_PORT = 5005

model = joblib.load("motion_model.joblib")

buffer = deque(maxlen=WINDOW)
pred_buffer = deque(maxlen=3)

latest_prediction = "Waiting..."
latest_confidence = 0.0
packet_count = 0


def udp_loop():
    global latest_prediction, latest_confidence, packet_count

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

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

        sample = values[1:7]
        buffer.append(sample)
        packet_count += 1

        if len(buffer) == WINDOW:
            window = np.array(buffer)

            features = extract_features(window)
            features = np.array(features).reshape(1, -1)

            prediction = model.predict(features)[0]

            if hasattr(model, "predict_proba"):
                confidence = model.predict_proba(features).max()
            else:
                confidence = 0.0

            pred_buffer.append(prediction)
            smoothed = max(set(pred_buffer), key=pred_buffer.count)

            latest_prediction = smoothed
            latest_confidence = confidence


def update_gui():
    prediction_label.config(text=f"Gesture: {latest_prediction}")
    confidence_label.config(text=f"Confidence: {latest_confidence:.2f}")
    packet_label.config(text=f"Packets: {packet_count}")

    root.after(100, update_gui)


root = tk.Tk()
root.title("IMU Gesture Classifier")
root.geometry("400x250")

title = tk.Label(root, text="Wireless IMU Gesture Classifier", font=("Arial", 18))
title.pack(pady=15)

prediction_label = tk.Label(root, text="Gesture: Waiting...", font=("Arial", 24))
prediction_label.pack(pady=10)

confidence_label = tk.Label(root, text="Confidence: 0.00", font=("Arial", 16))
confidence_label.pack(pady=5)

packet_label = tk.Label(root, text="Packets: 0", font=("Arial", 14))
packet_label.pack(pady=5)

thread = threading.Thread(target=udp_loop, daemon=True)
thread.start()

update_gui()
root.mainloop()