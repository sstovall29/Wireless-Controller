import glob
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from features import extract_features

DATA_DIR = "../Data/v3_imu_logs"
CLASSES = ["idle", "shake", "jab", "uppercut", "hook"]
WINDOW_SIZE = 50
STEP_SIZE = 5

def load_trials():
    all_trials = []

    for label in CLASSES:
        pattern = os.path.join(DATA_DIR, label, "*.csv")
        files = glob.glob(pattern)
        #print(label, len(files), files[:3])

        for file_path in files:
            df = pd.read_csv(file_path)
            df["label"] = label
            df["trial"] = file_path
            all_trials.append(df)

    return all_trials


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
