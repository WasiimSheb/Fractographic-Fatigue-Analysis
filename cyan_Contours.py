import cv2
import numpy as np
import os
import csv
from scipy.spatial import cKDTree

# === Paths ===
base_path = "C:\\Users\\shifa\\final project\\Enternal_Contours"
input_folder = os.path.join(base_path, "SLM-P1-CrackZone-NEW")
output_folder = os.path.join(base_path, "Cyan_CrackFront-P1")
os.makedirs(output_folder, exist_ok=True)
csv_folder = os.path.join(output_folder, "contours_csv")
os.makedirs(csv_folder, exist_ok=True)

# === HSV Range for Cyan
cyan_lower = np.array([80, 30, 60])
cyan_upper = np.array([110, 255, 255])

# === Morphology Kernel
kernel = np.ones((5, 5), np.uint8)

# === Parameters
DISTANCE_TO_ELLIPSE = 150  # maximum distance allowed from ellipse to accept cyan region
MIN_AREA = 200  # Minimum contour area to accept

# === Process All Images ===
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".png"):
        continue

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # === Step 1: Create masks
    cyan_mask = cv2.inRange(hsv, cyan_lower, cyan_upper)
    cyan_mask = cv2.morphologyEx(cyan_mask, cv2.MORPH_OPEN, kernel)
    cyan_mask = cv2.morphologyEx(cyan_mask, cv2.MORPH_CLOSE, kernel)

    # Create ellipse mask
    ellipse_mask = np.zeros_like(gray)
    pink_mask = cv2.inRange(hsv, (140, 50, 50), (170, 255, 255))
    pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_CLOSE, kernel)
    contours_pink, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours_pink:
        largest_pink = max(contours_pink, key=cv2.contourArea)
        if len(largest_pink) >= 5:
            ellipse = cv2.fitEllipse(largest_pink)
            cv2.ellipse(ellipse_mask, ellipse, 255, -1)
        else:
            print(f"⚠ Not enough points for ellipse in {filename}")
            continue
    else:
        print(f"⚠ No pink ellipse found in {filename}")
        continue

    # === Step 2: Find all cyan points close to ellipse
    ellipse_points = np.column_stack(np.where(ellipse_mask > 0))  # (y,x)
    cyan_points = np.column_stack(np.where(cyan_mask > 0))        # (y,x)

    if cyan_points.size == 0:
        print(f"❌ No cyan region detected in {filename}")
        continue

    # Build KDTree once
    tree = cKDTree(ellipse_points)

    # Query all cyan points at once!
    distances, _ = tree.query(cyan_points)
    selected = cyan_points[distances <= DISTANCE_TO_ELLIPSE]

    if selected.size == 0:
        print(f"❌ No cyan points close enough to ellipse in {filename}")
        continue

    selected_points = np.array([[x, y] for y, x in selected])

    # === Step 3: Create one contour (full envelope)
    crack_mask = np.zeros_like(gray)
    for point in selected_points:
        cv2.circle(crack_mask, tuple(point), 1, 255, -1)

    crack_mask = cv2.dilate(crack_mask, np.ones((5,5), np.uint8), iterations=2)
    crack_mask = cv2.morphologyEx(crack_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"❌ No final contour detected in {filename}")
        continue

    # Take the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest_contour) < MIN_AREA:
        print(f"⚠ Detected contour too small in {filename}")
        continue

    # === Step 4: Save overlay
    overlay = img.copy()
    cv2.drawContours(overlay, [largest_contour], -1, (0, 0, 0), thickness=10)

    out_path = os.path.join(output_folder, f"{filename[:-4]}_envelope_overlay.png")
    cv2.imwrite(out_path, overlay)

    # === Step 5: Save contour points to CSV
    csv_path = os.path.join(csv_folder, f"{filename[:-4]}_envelope_contour.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x", "y"])
        for point in largest_contour.squeeze():
            writer.writerow(point)

    print(f"✅ Full crack front envelope saved for {filename}")

print("🎯 Crack front envelopes created successfully and FAST!")
