import socket
import time
import os

UDP_IP = "0.0.0.0"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for UDP packets on port {UDP_PORT}...")
print("Press Ctrl+C to stop recording.\n")

# Hardcoded label
label = input("What gesture would you like to log (idle, jab, etc): ")   # change this manually between sessions

OUTPUT_DIR = os.path.join("../Data/v2_imu_logs", label)
os.makedirs(OUTPUT_DIR, exist_ok=True)

trial_num = 0
last_button_state = 0  # track previous state

try:
    while True:
        data, addr = sock.recvfrom(1024)
        line = data.decode(errors="ignore").strip()

        parts = line.split(",")

        if len(parts) != 8:
            continue

        try:
            button_state = int(parts[7])  # last value
        except ValueError:
            continue

        # Detect button press (0 → 1)
        if last_button_state == 0 and button_state == 1:
            trial_num += 1
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            filename = os.path.join(
                OUTPUT_DIR,
                f"{label}_{trial_num}_{timestamp}.csv"
            )

            print(f"Recording → {filename}")

            with open(filename, "w") as f:
                f.write("host_time_ms,arduino_time_ms,ax,ay,az,gx,gy,gz\n")

                # Record until button released
                while True:
                    data, addr = sock.recvfrom(1024)
                    line = data.decode(errors="ignore").strip()
                    parts = line.split(",")

                    if len(parts) != 8:
                        continue

                    try:
                        button_state = int(parts[7])
                    except ValueError:
                        continue

                    host_time = int(time.time() * 1000)
                    f.write(f"{host_time},{line}\n")

                    # Detect release (1 → 0)
                    if button_state == 0:
                        print("Trial complete.\n")
                        break

        last_button_state = button_state

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    print("Closing")