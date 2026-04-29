import numpy as np

def extract_features(window):
    features = []

    for channel in range(window.shape[1]):
        axis = window[:, channel]

        features.append(axis.mean())
        features.append(axis.std())
        features.append(axis.min())
        features.append(axis.max())

        diff = np.diff(axis)
        features.append(diff.mean())
        features.append(diff.std())

        features.append(np.sum(axis**2))

    accel_mag = np.linalg.norm(window[:, 0:3], axis=1)
    gyro_mag = np.linalg.norm(window[:, 3:6], axis=1)

    features.append(accel_mag.mean())
    features.append(accel_mag.std())
    features.append(accel_mag.max())

    features.append(gyro_mag.mean())
    features.append(gyro_mag.std())
    features.append(gyro_mag.max())

    return features