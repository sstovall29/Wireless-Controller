import glob
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


DATA_DIR = "Data/v2_imu_logs"
CLASSES = ["idle", "shake", "jab", "uppercut", "hook"]
WINDOW_SIZE = 50
STEP_SIZE = 10

# train_files = []
# test_files = []

# for label in CLASSES:
#     pattern = os.path.join(DATA_DIR, label, "*.csv")
#     files = sorted(glob.glob(pattern))  # important: sorted

#     if len(files) < 4:
#         print(f"Not enough files in {label} to split properly")
#         continue

#     test = files[-3:]      # last 3 → test
#     train = files[:-3]     # rest → train

#     train_files.extend([(f, label) for f in train])
#     test_files.extend([(f, label) for f in test])

# print(f"Train files: {len(train_files)}")
# print(f"Test files: {len(test_files)}")

def load_trials():
    all_trials = []

    for label in CLASSES:
        pattern = os.path.join(DATA_DIR, label, "*.csv")
        files = glob.glob(pattern)

        for file_path in files:
            df = pd.read_csv(file_path)
            df["label"] = label
            df["trial"] = file_path
            all_trials.append(df)

    return all_trials


def load_from_file_list(file_list):
    trials = []

    for file_path, label in file_list:
        df = pd.read_csv(file_path)
        df["label"] = label
        df["trial"] = file_path
        trials.append(df)

    return trials

def make_windows(trials):
    windows = []
    labels = []

    for df in trials:
        sensor_data = df[["ax", "ay", "az", "gx", "gy", "gz"]].values
        label = df["label"].iloc[0]

        for start in range(0, len(sensor_data) - WINDOW_SIZE, STEP_SIZE):
            end = start + WINDOW_SIZE
            window = sensor_data[start:end]

            windows.append(window)
            labels.append(label)

    return np.array(windows), np.array(labels)


def extract_features(window):
    features = []

    for channel in range(window.shape[1]):
        axis = window[:, channel]

        # basic features
        features.append(axis.mean())
        features.append(axis.std())
        features.append(axis.min())
        features.append(axis.max())

        # NEW features (important)
        diff = np.diff(axis)

        features.append(diff.mean())
        features.append(diff.std())

        # energy
        features.append(np.sum(axis**2))

    # magnitude features (unchanged)
    accel_mag = np.linalg.norm(window[:, 0:3], axis=1)
    gyro_mag = np.linalg.norm(window[:, 3:6], axis=1)

    features.append(accel_mag.mean())
    features.append(accel_mag.std())
    features.append(accel_mag.max())

    features.append(gyro_mag.mean())
    features.append(gyro_mag.std())
    features.append(gyro_mag.max())

    return features

def main():
    trials = load_trials()
    print(f"Loaded {len(trials)} trial files")
    trial_labels = [df["label"].iloc[0] for df in trials]

    train_trials, test_trials = train_test_split(
        trials,
        test_size=0.3,
        stratify=trial_labels,
        random_state=42
    )

    # train_trials = load_from_file_list(train_files)
    # test_trials = load_from_file_list(test_files)

    print(f"Loaded {len(train_trials)} training trials")
    print(f"Loaded {len(test_trials)} testing trials")

    train_windows, train_labels = make_windows(train_trials)
    test_windows, test_labels = make_windows(test_trials)

    X_train = np.array([extract_features(w) for w in train_windows])
    y_train = train_labels

    X_test = np.array([extract_features(w) for w in test_windows])
    y_test = test_labels


    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "motion_model.joblib")
    print("Saved model to motion_model.joblib")

    predictions = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    main()