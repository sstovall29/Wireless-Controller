import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = "imu_logs/hook/hook_7_20260428_140353.csv"

df = pd.read_csv(file_path)

print(df.head())
print(df.columns)
print(df.shape)

plt.plot(df["ax"], label="ax")
plt.plot(df["ay"], label="ay")
plt.plot(df["az"], label="az")

plt.title("Raw Accelerometer Data - Hook Trial")
plt.xlabel("Sample")
plt.ylabel("Acceleration")
plt.legend()
plt.show()

plt.plot(df["gx"], label="gx")
plt.plot(df["gy"], label="gy")
plt.plot(df["gz"], label="gz")

plt.title("Raw Gyroscope Data - Hook Trial")
plt.xlabel("Sample")
plt.ylabel("Angular Velocity")
plt.legend()
plt.show()

hook = pd.read_csv("imu_logs/hook/hook_7_20260428_140353.csv")
uppercut = pd.read_csv("imu_logs/uppercut/uppercut_7_20260428_140234.csv")

plt.plot(hook["gx"], label="hook gx")
plt.plot(uppercut["gx"], label="uppercut gx")

plt.title("Hook vs Uppercut - gx")
plt.xlabel("Sample")
plt.ylabel("Angular Velocity")
plt.legend()
plt.show()


# fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)

channels = ["ax", "ay", "az", "gx", "gy", "gz"]

# for i, ch in enumerate(channels):
#     axes[i].plot(hook[ch], label=f"hook {ch}")
#     axes[i].plot(uppercut[ch], label=f"uppercut {ch}")
#     axes[i].set_ylabel(ch)
#     axes[i].legend()

# axes[-1].set_xlabel("Sample")

# plt.suptitle("Hook vs Uppercut - All IMU Channels")
# plt.tight_layout()
# plt.show()

hook_accel_mag = np.linalg.norm(hook[["ax","ay","az"]], axis=1)
upper_accel_mag = np.linalg.norm(uppercut[["ax","ay","az"]], axis=1)

plt.plot(hook_accel_mag, label="hook accel mag")
plt.plot(upper_accel_mag, label="uppercut accel mag")
plt.legend()
plt.title("Acceleration Magnitude")
plt.show()


def crop_around_peak(df, before=75, after=75):
    accel_mag = np.linalg.norm(df[["ax", "ay", "az"]].values, axis=1)
    peak = np.argmax(accel_mag)

    start = max(0, peak - before)
    end = min(len(df), peak + after)

    return df.iloc[start:end].reset_index(drop=True)

hook_crop = crop_around_peak(hook)
uppercut_crop = crop_around_peak(uppercut)

fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)

for i, ch in enumerate(channels):
    axes[i].plot(hook_crop[ch], label=f"hook {ch}")
    axes[i].plot(uppercut_crop[ch], label=f"uppercut {ch}")
    axes[i].set_ylabel(ch)
    axes[i].legend()

axes[-1].set_xlabel("Aligned sample")
plt.suptitle("Hook vs Uppercut - Cropped Around Peak")
plt.tight_layout()
plt.show()