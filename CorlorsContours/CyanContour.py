import cv2
import numpy as np
import os
import csv
from scipy.interpolate import splprep, splev
from scipy.ndimage import binary_fill_holes

# === Paths ===
base_path = "C:\\Users\\shifa\\final project\\Enternal_Contours"
    input_folder = os.path.join(base_path, "EBM6-CrackZone-New")
output_folder = os.path.join(base_path, "cyan_Internal-EBM6")
os.makedirs(output_folder, exist_ok=True)
csv_folder = os.path.join(output_folder, "crackzone_contour_csv")
os.makedirs(csv_folder, exist_ok=True)

# === HSV Ranges for Red + Orange + Yellow + Cyan
combined_ranges = [
    ([0, 50, 50], [10, 255, 255]),     # Red low
    ([160, 50, 50], [180, 255, 255]),  # Red high
    ([11, 80, 80], [22, 255, 255]),    # Orange
    ([23, 90, 90], [38, 255, 255]),    # Yellow
    ([85, 50, 80], [105, 255, 255])    # Cyan
]

# === Morphological Kernels
dilate_kernel = np.ones((25, 25), np.uint8)    # Stronger dilation now
close_kernel = np.ones((35, 35), np.uint8)     # Stronger closing
open_kernel = np.ones((5, 5), np.uint8)

# === Other Parameters
DILATION_PIXELS = 100
SMOOTHNESS = 0.001   # Slightly smoother spline
NUM_POINTS = 600     # Resample points along the spline

# === Process All Images ===
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".png"):
        continue

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # === Step 1: Detect Pink Ellipse Mask
    ellipse_mask = np.zeros_like(gray)
    pink_mask = cv2.inRange(hsv, (140, 50, 50), (170, 255, 255))
    pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_CLOSE, open_kernel)
    contours_pink, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_pink:
        largest = max(contours_pink, key=cv2.contourArea)
        if len(largest) >= 5:
            ellipse = cv2.fitEllipse(largest)
            cv2.ellipse(ellipse_mask, ellipse, 255, -1)
        else:
            print(f"⚠ Not enough points for ellipse in {filename}")
            continue
    else:
        print(f"⚠ No pink ellipse found in {filename}")
        continue

    allowed_area = cv2.dilate(ellipse_mask, np.ones((DILATION_PIXELS, DILATION_PIXELS), np.uint8))

    # === Step 2: Combine Masks from All Color Ranges
    combined_mask = np.zeros_like(gray)
    for lower, upper in combined_ranges:
        lower_np = np.array(lower)
        upper_np = np.array(upper)
        combined_mask |= cv2.inRange(hsv, lower_np, upper_np)

    combined_mask = cv2.bitwise_and(combined_mask, allowed_area)

    # === Step 3: Improve the Mask
    combined_mask = cv2.dilate(combined_mask, dilate_kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, close_kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, open_kernel)

    # === Step 4: Fill Holes Inside
    combined_mask = binary_fill_holes(combined_mask > 0).astype(np.uint8) * 255

    # === Step 5: Find External Contour
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"❌ No crack zone contour found in {filename}")
        continue

    largest_contour = max(contours, key=cv2.contourArea)
    largest_contour = largest_contour.squeeze()

    if len(largest_contour.shape) != 2 or largest_contour.shape[0] < 10:
        print(f"⚠ Contour too small or broken in {filename}")
        continue

    # === Step 6: Fit a Spline Curve
    x, y = largest_contour[:, 0], largest_contour[:, 1]
    tck, u = splprep([x, y], s=SMOOTHNESS, per=True)
    u_fine = np.linspace(0, 1, NUM_POINTS)
    x_fine, y_fine = splev(u_fine, tck)

    smooth_contour = np.stack((x_fine, y_fine), axis=1).astype(np.int32)

    # === Step 7: Save Overlay Image
    overlay = img.copy()
    cv2.polylines(overlay, [smooth_contour], isClosed=True, color=(0, 0, 0), thickness=10)
    out_path = os.path.join(output_folder, f"{filename[:-4]}_crackzone_contour_overlay.png")
    cv2.imwrite(out_path, overlay)

    # === Step 8: Save Contour Points to CSV
    csv_path = os.path.join(csv_folder, f"{filename[:-4]}_crackzone_contour.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x", "y"])
        for pt in smooth_contour:
            writer.writerow(pt)

    print(f"✅ Saved final crack zone contour for {filename}")

print("🎯 All final crack zone envelopes extracted and smoothed successfully!")
