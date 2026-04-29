import serial
import joblib
import numpy as np
from collections import deque

PORT = "/dev/cu.usbmodem11101" 
BAUD = 115200

WINDOW_SIZE = 75

model = joblib.load("motion_model.joblib")

buffer = deque(maxlen=WINDOW_SIZE)

def extract_features(window):
    features = []

    for channel in range(window.shape[1]):
        values = window[:, channel]

        features.append(values.mean())
        features.append(values.std())
        features.append(values.min())
        features.append(values.max())

        # NEW features (important)
        diff = np.diff(values)

        features.append(diff.mean())
        features.append(diff.std())

        # energy
        features.append(np.sum(values**2))

    accel_mag = np.linalg.norm(window[:, 0:3], axis=1)
    gyro_mag = np.linalg.norm(window[:, 3:6], axis=1)

    features.append(accel_mag.mean())
    features.append(accel_mag.std())
    features.append(accel_mag.max())

    features.append(gyro_mag.mean())
    features.append(gyro_mag.std())
    features.append(gyro_mag.max())

    return np.array(features).reshape(1, -1)


ser = serial.Serial(PORT, BAUD, timeout=1)

print("Live classifier running...")
print("Move the IMU. Press Ctrl+C to stop.\n")

try:
    while True:
        line = ser.readline().decode(errors="ignore").strip()

        if not line:
            continue

        parts = line.split(",")

        if len(parts) != 7:
            print(f"Skipping bad line: {line}")
            continue

        try:
            values = [float(x) for x in parts]
        except ValueError:
            print(f"Skipping non-numeric line: {line}")
            continue

        # Ignore timestamp, keep ax, ay, az, gx, gy, gz
        sample = values[1:7]

        buffer.append(sample)

        if len(buffer) == WINDOW_SIZE:
            window = np.array(buffer)

            X_live = extract_features(window)
            prediction = model.predict(X_live)[0]

            if hasattr(model, "predict_proba"):
                confidence = model.predict_proba(X_live).max()
                print(f"Prediction: {prediction}  confidence: {confidence:.2f}")
            else:
                print(f"Prediction: {prediction}")

except KeyboardInterrupt:
    print("\nStopping.")

finally:
    ser.close()