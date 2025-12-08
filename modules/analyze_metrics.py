# analyze_metrics.py
import json
import glob
import statistics as stats
import matplotlib.pyplot as plt

files = glob.glob("metrics/session_*.json")
if not files:
    print("No metrics files found.")
    raise SystemExit

all_calib = []
all_lat = []

for path in files:
    with open(path) as f:
        data = json.load(f)
    all_calib.extend(data.get("calibration_errors_px", []))
    all_lat.extend(data.get("summary_latencies_ms", []))

print(f"Loaded {len(files)} sessions.")

if all_calib:
    print(f"Gaze error (px): mean={stats.mean(all_calib):.1f}, "
          f"stdev={stats.pstdev(all_calib):.1f}, "
          f"max={max(all_calib):.1f}")

    # Histogram of calibration error
    plt.figure()
    plt.hist(all_calib, bins=10)
    plt.title("Calibration Error (pixels)")
    plt.xlabel("Error (px)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

if all_lat:
    print(f"Summary latency (ms): mean={stats.mean(all_lat):.1f}, "
          f"stdev={stats.pstdev(all_lat):.1f}, "
          f"max={max(all_lat):.1f}")

    # Histogram of summary latency
    plt.figure()
    plt.hist(all_lat, bins=10)
    plt.title("Summary Latency (ms)")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
