import serial
import time
import os

PORT = "/dev/cu.usbmodem11101"   # change to actual port
BAUD = 115200
OUTPUT_DIR = "imu_logs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

ser = serial.Serial(PORT, BAUD, timeout=1)

print("Connected.")
print("Press Ctrl+C to stop recording.\n")

trial_num = 0

try:
    while True:
        input("Press ENTER to start a new trial...")

        trial_num += 1
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        label = input("Enter label (shake, idle, etc): ")
        OUTPUT_DIR = "imu_logs/" + label
        filename = os.path.join(OUTPUT_DIR, f"{label}_{trial_num}_{timestamp}.csv")

        with open(filename, "w") as f:
            f.write("host_time_ms,arduino_time_ms,ax,ay,az,gx,gy,gz\n")

            print(f"Recording → {filename}")

            start_time = time.time()

            while True:
                line = ser.readline().decode(errors="ignore").strip()

                if not line:
                    continue

                parts = line.split(",")
                if len(parts) != 7:
                    continue

                host_time = int(time.time() * 1000)
                f.write(f"{host_time},{line}\n")

                print(line)

                # Stop after fixed duration (example: 5 seconds)
                if time.time() - start_time > 3:
                    print("Trial complete.\n")
                    break

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    ser.close()