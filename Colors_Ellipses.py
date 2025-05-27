import cv2
import numpy as np
import os
from scipy.ndimage import binary_fill_holes

# === Parameters ====
PIXEL_SIZE_MICRONS = 1.34375
DILATION_PIXELS = 200
MIN_AREA = 5000

# === HSV Ranges for Each Color ===
COLOR_RANGES = {
    "dark_red": [([0, 200, 100], [10, 255, 180]), ([160, 200, 100], [180, 255, 180])],
    "red": [([0, 50, 50], [10, 255, 255]), ([160, 50, 50], [180, 255, 255])],
    "yellow": [([11, 80, 80], [22, 255, 255]), ([23, 90, 90], [38, 255, 255])],
    "cyan": [([85, 50, 80], [105, 255, 255])],
    "blue": [([105, 50, 50], [125, 255, 255])]
}

# === Paths ===
input_folder = r"C:\Users\shifa\final project\ellipses\SLM-P1-CrackZone-New"
output_folder = r"C:\Users\shifa\final project\ellipses\ellipses_SLM-P1"
os.makedirs(output_folder, exist_ok=True)

# === Kernels ===
open_kernel = np.ones((5, 5), np.uint8)
close_kernel = np.ones((30, 30), np.uint8)
expand_kernel = np.ones((100, 100), np.uint8)

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".png"):
        continue

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: Detect Pink Ellipse (crack zone)
    pink_mask = cv2.inRange(hsv, (140, 50, 50), (170, 255, 255))
    pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_CLOSE, open_kernel)
    contours_pink, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_pink or len(max(contours_pink, key=cv2.contourArea)) < 5:
        print(f"❌ No valid pink crack zone ellipse in {filename}")
        continue
    ellipse_mask = np.zeros_like(gray)
    pink_ellipse = cv2.fitEllipse(max(contours_pink, key=cv2.contourArea))
    cv2.ellipse(ellipse_mask, pink_ellipse, 255, -1)

    # Step 2: For each color – generate ellipse
    for color, hsv_ranges in COLOR_RANGES.items():
        color_mask = np.zeros_like(gray)
        for lower, upper in hsv_ranges:
            color_mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

        allowed_area = cv2.dilate(ellipse_mask, np.ones((DILATION_PIXELS, DILATION_PIXELS), np.uint8))
        color_mask = cv2.bitwise_and(color_mask, allowed_area)
        color_mask = cv2.dilate(color_mask, np.ones((25, 25), np.uint8))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, close_kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, open_kernel)
        color_mask = binary_fill_holes(color_mask > 0).astype(np.uint8) * 255
        color_mask = cv2.dilate(color_mask, expand_kernel)

        # Keep largest component
        num_labels, labels_im = cv2.connectedComponents(color_mask)
        max_area = 0
        largest_label = 0
        for label_idx in range(1, num_labels):
            area = np.count_nonzero(labels_im == label_idx)
            if area > max_area:
                max_area = area
                largest_label = label_idx

        if max_area < MIN_AREA:
            print(f"⚠ Not enough area for {color} in {filename}")
            continue

        final_mask = np.uint8(labels_im == largest_label) * 255
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours or len(max(contours, key=cv2.contourArea)) < 5:
            print(f"⚠ Not enough contour points for ellipse for {color} in {filename}")
            continue

        largest_contour = max(contours, key=cv2.contourArea)
        fitted_ellipse = cv2.fitEllipse(largest_contour)

        # Draw on the original heatmap
        overlay = img.copy()
        cv2.ellipse(overlay, fitted_ellipse, (0, 0, 0), thickness=10)
        out_path = os.path.join(output_folder, f"{filename[:-4]}_{color}_ellipse_overlay.png")
        cv2.imwrite(out_path, overlay)

        print(f"✅ {color} ellipse saved for {filename}")

print("🎯 All ellipses drawn and saved.")
