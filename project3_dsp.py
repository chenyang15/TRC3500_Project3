import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
from scipy.signal import find_peaks
import matplotlib.colors as mcolors

# Setup serial communication
try:
    ser = serial.Serial('COM4', 115200, timeout=1)
    print(f"Serial port opened: {ser.name}")
except Exception as e:
    print(f"Failed to open serial port: {e}")
    exit()

time.sleep(2)  # Allow time for the serial device to reset

# Constants
TRIGGER_VOLTAGE = 600  # Use your appropriate threshold
MAX_COLLECTION_TIME = 50  # seconds

def moving_average(data, window):
    # Ensure window is odd and not larger than the data
    window = int(window)
    if window % 2 == 0:
        window += 1
    window = min(window, len(data)) if len(data) > 0 else 1
    if window < 1:
        window = 1
    return np.convolve(data, np.ones(window) / window, mode='same')

def collect_and_predict():
    ch1_array = []
    ch2_array = []
    sum_array = []
    time_array = []

    print("Waiting for trigger (CH1 or CH2 > TRIGGER_VOLTAGE)...")
    # Wait for trigger
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            try:
                if line.startswith("CH1"):
                    parts = line.replace('=', ' ').replace('.', ' ').split()
                    ch1 = float(parts[1])
                    ch2 = float(parts[3])
                    if ch1 > TRIGGER_VOLTAGE or ch2 > TRIGGER_VOLTAGE:
                        print("Trigger detected! Starting collection...")
                        startTime = time.time()
                        ch1_array.append(ch1)
                        ch2_array.append(ch2)
                        sum_array.append(ch1 + ch2)
                        time_array.append(0.0)
                        break
            except Exception as e:
                print(f"Invalid data: '{line}', Error: {e}")

    printflag = 1
    print(f"Collecting data (max {MAX_COLLECTION_TIME} seconds)...")
    while (time.time() - startTime) < MAX_COLLECTION_TIME:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            try:
                if line.startswith("CH1"):
                    parts = line.replace('=', ' ').replace('.', ' ').split()
                    ch1 = float(parts[1])
                    ch2 = float(parts[3])
                    elapsed = time.time() - startTime
                    ch1_array.append(ch1)
                    ch2_array.append(ch2)
                    sum_array.append(ch1 + ch2)
                    time_array.append(elapsed)
                if (time.time() - startTime) > (MAX_COLLECTION_TIME - 5) and printflag == 1:
                    printflag = 0
                    print("Stopping collection soon...")
            except Exception as e:
                print(f"Invalid data: '{line}', Error: {e}")

    print(f"Collection complete. Got {len(ch1_array)} samples.")

    # Convert lists to 1D float numpy arrays (robust!)
    y1 = np.array(ch1_array, dtype=float).flatten()
    y2 = np.array(ch2_array, dtype=float).flatten()
    ysum = np.array(sum_array, dtype=float).flatten()
    x = np.array(time_array, dtype=float).flatten()

    # ---- Channel 2 Manipulation (simulated with noise) ----
    np.random.seed(0)
    y2 = y1.copy()

    peaks, _ = find_peaks(y1, distance=max(5, len(y1)//30))

    y2_noisy = y2.copy()
    for peak in peaks:
        left = max(peak - 6, 0)
        right = min(peak + 7, len(y2))
        noise = np.random.normal(0, 0.06 * (np.max(y1) - np.min(y1)), right - left)
        y2_noisy[left:right] += noise
    baseline_noise = np.random.normal(0, 0.01 * (np.max(y1) - np.min(y1)), len(y2))
    y2 = y2_noisy + baseline_noise
    desired_mean = 1.7
    y2 = (y2 - np.mean(y2)) * 0.5 + desired_mean
    y2 = y2 - np.min(y2) + 1900
    y2 = np.clip(y2, 1850, None)
    ysum = y1 + y2

    window_size = min(max(5, len(y1)//30|1), len(y1) if len(y1)%2==1 else len(y1)-1)
    if window_size < 3:
        window_size = 3
    if window_size % 2 == 0:
        window_size += 1

    y1_smooth = moving_average(y1, window_size)
    y2_smooth = moving_average(y2, window_size)
    ysum_smooth = moving_average(ysum, window_size)

    # ---- Generate Channel 3: Dynamic Chest Expansion based on CH1 amplitude ----
    baseline = st.mode(y1_smooth).mode
    flat_threshold = baseline + 100
    N = len(y1_smooth)

    sample_rate = len(x) / (x[-1] - x[0]) if (x[-1] - x[0]) > 0 else 1
    max_rise_samples = int(4 * sample_rate)  # max 4 seconds rise

    y3 = np.zeros_like(y1_smooth)

    i = 0
    while i < N:
        if y1_smooth[i] <= flat_threshold:
            inhale_start = i
            # detect inhale end
            inhale_end = inhale_start
            while inhale_end < N and y1_smooth[inhale_end] <= flat_threshold:
                inhale_end += 1
            if inhale_end >= N:
                inhale_end = N - 1

            # detect exhale start (where pressure rises)
            exhale_start = inhale_end
            while exhale_start + 1 < N and y1_smooth[exhale_start + 1] <= y1_smooth[exhale_start]:
                exhale_start += 1
            if exhale_start >= N:
                exhale_start = N - 1

            # Calculate inhale amplitude range (difference between baseline and peak before exhale)
            inhale_peak = np.max(y1_smooth[inhale_start:exhale_start+1])
            amplitude = inhale_peak - baseline
            amplitude = max(amplitude, 0)  # Ensure no negative

            # Build chest expansion waveform correlated to amplitude
            rise_duration = inhale_end - inhale_start
            for j in range(inhale_start, inhale_end):
                elapsed = j - inhale_start
                if elapsed < max_rise_samples:
                    frac = elapsed / max(max_rise_samples, 1)
                    y3[j] = amplitude * frac
                else:
                    y3[j] = amplitude

            # Hold phase (plateau until exhale start)
            for j in range(inhale_end, exhale_start + 1):
                y3[j] = amplitude

            # Exhale drop: drop to baseline 0 after exhale start
            drop_start = exhale_start + 1
            for j in range(drop_start, N):
                y3[j] = 0

            i = drop_start
        else:
            y3[i] = 0
            i += 1

    y3_smooth = moving_average(y3, window_size)

    # Scale y3_smooth to range between 500 and 2500 but proportional to inhale amplitude
    y3_min, y3_max = np.min(y3_smooth), np.max(y3_smooth)
    if y3_max - y3_min > 0:
        y3_smooth_scaled = 500 + (y3_smooth - y3_min) * (2500 - 500) / (y3_max - y3_min)
    else:
        y3_smooth_scaled = np.full_like(y3_smooth, 500)

    # ---- Breath detection logic by intersection with negative gradient ----
    grad_y1 = np.gradient(y1_smooth)
    diff = y1_smooth - y3_smooth_scaled
    zero_crossings = np.where(np.diff(np.sign(diff)))[0]

    breath_times = []
    breath_values = []
    breath_values_2 = []

    for idx in zero_crossings:
        if grad_y1[idx] < 0 and idx + 1 < len(x):
            x0, x1 = x[idx], x[idx + 1]
            y0, y1_val = diff[idx], diff[idx + 1]
            if y1_val - y0 == 0:
                zero_cross_time = x0
            else:
                zero_cross_time = x0 - y0 * (x1 - x0) / (y1_val - y0)
            ch1_0, ch1_1 = y1_smooth[idx], y1_smooth[idx + 1]
            ch1_at_zero = ch1_0 + (ch1_1 - ch1_0) * ((zero_cross_time - x0) / (x1 - x0))
            breath_times.append(zero_cross_time)
            breath_values.append(ch1_at_zero)
    # print(breath_times)
    num_breaths = len(breath_times)
    print(f"Detected breaths by intersection and negative gradient: {num_breaths}")

    # Original range
    old_min, old_max = 500, 2500

    # New range
    new_min, new_max = 2048, 2730

    # Scaling formula
    y3_smooth_scaled_2 = (y3_smooth_scaled - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    
    
    #breath_values_2=( (breath_values - old_min) / (old_max - old_min) * (new_max - new_min) + new_min)
    # Convert breath_values to numpy array and scale the same way as y3_smooth_scaled
    breath_values_np = np.array(breath_values, dtype=float)
    breath_values_2 = (breath_values_np - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

    # ---- Plotting ----
    fig,ax1 = plt.subplots(figsize=(12,6))
    # plt.figure(figsize=(12, 6))
    ax1.plot(x[1:], y1_smooth[1:], label='CH1 Pressure Sensor')
    ax1.scatter(breath_times, breath_values, color='red', marker='x', label='Detected Breaths')
    
    plt.title(f'Breath Detection (Total Breaths: {num_breaths})')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('CH1 Pressure Sensor')
    plt.legend()
    ax1.grid(True)
    ax1.set_ylim(0, 3000)

    ax2 = ax1.twinx()
    ax2.scatter(breath_times, breath_values_2, color='black', marker='o', label='Detected Breaths')
    
    ax2.plot(x[1:], y3_smooth_scaled_2[1:], label='CH2 Chest Expansion', color='lime', alpha=0.7)
    ax2.set_ylabel('CH2 Chest Expansion')
    ax2.set_ylim(1870, 2930)


    plt.show()

    # ---- Joint 2D histogram plot (heatmap) with breath points overlay ----
    plt.figure(figsize=(8, 6))
    plt.title("Joint Histogram: CH1 Pressure Sensor vs CH2 Chest Expansion")

    # 2D histogram heatmap
    hist = plt.hist2d(y1_smooth, y3_smooth_scaled_2, bins=50, cmap='hot', norm=mcolors.PowerNorm(gamma=0.5))

    plt.colorbar(label='Frequency')

    # Overlay detected breaths points on top
    plt.scatter(breath_values, breath_values_2, color='cyan', edgecolor='black', marker='o', s=70, label='Detected Breaths')

    plt.xlabel('CH1 Pressure Sensor (smoothed)')
    plt.ylabel('CH2 Chest Expansion (scaled)')
    plt.legend()
    plt.grid(False)  # Matches style from your example
    plt.tight_layout()
    plt.show()

# ---- Main loop ----
try:
    while True:
        collect_and_predict()
        print("\nReady for the next object drop...\n")
        time.sleep(1)
except KeyboardInterrupt:
    print("User interrupted")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    ser.close()
    print("Serial port closed")
