# uap_tracker.py
# Smart multi-frame motion filtering + auto-increment output directory

import cv2
import numpy as np
import matplotlib.pyplot as plt
import string
import os
import re

# === USER CONFIG ===
VIDEO_FILE = "20250614_021240.mp4"
ZOOM = 1.5
MIN_BRIGHTNESS = 200
MAX_TIME = 20.0
SAVE_FRAMES = True
MOTION_THRESHOLD = 15
MEMORY_FRAMES = 3

# === AUTO-INCREMENT OUTPUT DIR ===
def get_next_output_dir(base_name="pass"):
    existing = [d for d in os.listdir() if os.path.isdir(d) and re.match(f"{base_name}_\\d+", d)]
    numbers = [int(re.findall(r"\\d+", d)[0]) for d in existing if re.findall(r"\\d+", d)]
    next_num = max(numbers, default=4) + 1
    return f"{base_name}_{next_num:02d}"

OUTPUT_DIR = get_next_output_dir()

cap = cv2.VideoCapture(VIDEO_FILE)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Loaded video: {VIDEO_FILE} ({frame_count} frames @ {fps:.2f} FPS, {width}x{height})")
print(f"Saving output to: {OUTPUT_DIR}")

if SAVE_FRAMES and not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

motion_paths = []
frame_idx = 0
track_memory = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or cap.get(cv2.CAP_PROP_POS_MSEC) > MAX_TIME * 1000:
        break

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ZOOM != 1.0:
        gray = cv2.resize(gray, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_LINEAR)

    _, thresh = cv2.threshold(gray, MIN_BRIGHTNESS, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    marked = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    confirmed_positions = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 5:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Match with previous frame tracks
        matched = False
        for key, history in track_memory.items():
            px, py = history[-1]
            dist = np.sqrt((cx - px)**2 + (cy - py)**2)
            if dist < MOTION_THRESHOLD:
                history.append((cx, cy))
                if len(history) >= MEMORY_FRAMES:
                    confirmed_positions.append((key, cx, cy))
                matched = True
                break

        if not matched:
            label = string.ascii_uppercase[len(track_memory) % 26]
            track_memory[label] = [(cx, cy)]

    # Draw only confirmed moving objects
    for label, cx, cy in confirmed_positions:
        cv2.circle(marked, (cx, cy), 10, (0, 255, 0), 2)
        cv2.putText(marked, label, (cx + 12, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        motion_paths.append((frame_idx / fps, cx, cy))

    if SAVE_FRAMES:
        filename = os.path.join(OUTPUT_DIR, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(filename, marked)

    # Prune memory
    for key in list(track_memory):
        if len(track_memory[key]) > MEMORY_FRAMES:
            track_memory[key].pop(0)

    frame_idx += 1

cap.release()

# === PLOT MOTION PATHS ===
if motion_paths:
    times, xs, ys = zip(*motion_paths)
    plt.figure(figsize=(10, 6))
    plt.scatter(xs, ys, c=times, cmap='viridis', s=15)
    plt.gca().invert_yaxis()
    plt.title("Filtered Object Motion Paths")
    plt.xlabel("X Position (pixels)")
    plt.ylabel("Y Position (pixels)")
    plt.grid(True)
    plt.colorbar(label='Time (s)')
    plt.show()
else:
    print("No moving objects detected.")
