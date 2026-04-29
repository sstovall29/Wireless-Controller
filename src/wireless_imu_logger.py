import socket
import time
import os

UDP_IP = "0.0.0.0"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for UDP packets on port {UDP_PORT}...")

OUTPUT_DIR = "imu_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

trial_num = 0
duration = 3 # duration of each trial

try:
    while True:
        input("Press ENTER to start a new trial...")
        try:
          duration = float(input("How long do you want this trial (sec)? "))
        except ValueError:
          duration = 3.0        
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
                data, addr = sock.recvfrom(1024)
                line = data.decode(errors="ignore").strip()

                print(f"{addr}: {line}")

                if not line:
                    continue

                parts = line.split(",")
                if len(parts) != 7:
                    continue

                host_time = int(time.time() * 1000)
                f.write(f"{host_time},{line}\n")

                print(line)

                # Stop after fixed duration (example: 5 seconds)
                if time.time() - start_time > duration:
                    print("Trial complete.\n")
                    break

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    print("Done")