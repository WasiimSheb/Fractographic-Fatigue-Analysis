import cv2
import numpy as np
import os
from scipy.ndimage import binary_fill_holes

# === Parameters ===
PIXEL_SIZE_MICRONS = 1.34375
DILATION_PIXELS = 200
MIN_AREA = 5000

# === HSV Range for Dark Red ===
dark_red_ranges = [
    ([0, 200, 100], [10, 255, 180]),
    ([160, 200, 100], [180, 255, 180])
]

# === Paths ===
base_path = r"C:\Users\shifa\final project\Enternal_Contours"
input_folder = os.path.join(base_path, "SLM-P3-CrackZone-NEW")
output_folder = os.path.join(base_path, "DarkRed_Contours-SLM-P3--2")
os.makedirs(output_folder, exist_ok=True)
mask_folder = os.path.join(output_folder, "contour_masks")
os.makedirs(mask_folder, exist_ok=True)

# === Kernels ===
open_kernel = np.ones((5, 5), np.uint8)
close_kernel = np.ones((30, 30), np.uint8)
expand_kernel = np.ones((100, 100), np.uint8)

# === Process All Images ===
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".png"):
        continue

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # === Step 1: Detect Pink Ellipse (crack zone)
    pink_mask = cv2.inRange(hsv, (140, 50, 50), (170, 255, 255))
    pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_CLOSE, open_kernel)
    contours_pink, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_pink or len(max(contours_pink, key=cv2.contourArea)) < 5:
        print(f"❌ No valid pink crack zone ellipse in {filename}")
        continue
    ellipse_mask = np.zeros_like(gray)
    pink_ellipse = cv2.fitEllipse(max(contours_pink, key=cv2.contourArea))
    cv2.ellipse(ellipse_mask, pink_ellipse, 255, -1)

    # === Step 2: Detect Dark Red Inside Ellipse
    dark_red_mask = np.zeros_like(gray)
    for lower, upper in dark_red_ranges:
        dark_red_mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

    allowed_area = cv2.dilate(ellipse_mask, np.ones((DILATION_PIXELS, DILATION_PIXELS), np.uint8))
    dark_red_mask = cv2.bitwise_and(dark_red_mask, allowed_area)
    dark_red_mask = cv2.dilate(dark_red_mask, np.ones((25, 25), np.uint8))
    dark_red_mask = cv2.morphologyEx(dark_red_mask, cv2.MORPH_CLOSE, close_kernel)
    dark_red_mask = cv2.morphologyEx(dark_red_mask, cv2.MORPH_OPEN, open_kernel)
    dark_red_mask = binary_fill_holes(dark_red_mask > 0).astype(np.uint8) * 255
    dark_red_mask = cv2.dilate(dark_red_mask, expand_kernel)

    # === Step 3: Keep Largest Component
    num_labels, labels_im = cv2.connectedComponents(dark_red_mask)
    max_area = 0
    largest_label = 0
    for label_idx in range(1, num_labels):
        area = np.count_nonzero(labels_im == label_idx)
        if area > max_area:
            max_area = area
            largest_label = label_idx

    if max_area < MIN_AREA:
        print(f"⚠ Not enough dark red area in {filename}")
        continue

    final_mask = np.uint8(labels_im == largest_label) * 255

    # === Step 4: Contour and Ellipse Fit
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or len(max(contours, key=cv2.contourArea)) < 5:
        print(f"⚠ Not enough points to fit ellipse in {filename}")
        continue

    largest_contour = max(contours, key=cv2.contourArea)
    fitted_ellipse = cv2.fitEllipse(largest_contour)

    # === Step 5: Save Overlay with Black Ellipse
    overlay = img.copy()
    cv2.ellipse(overlay, fitted_ellipse, (0, 0, 0), thickness=15)
    overlay_path = os.path.join(output_folder, f"{filename[:-4]}_darkred_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    # === Step 6: Save Binary Mask (white inside ellipse)
    ellipse_mask_img = np.zeros_like(gray)
    cv2.ellipse(ellipse_mask_img, fitted_ellipse, 255, thickness=-1)
    mask_path = os.path.join(mask_folder, f"{filename[:-4]}_darkred_mask.png")
    cv2.imwrite(mask_path, ellipse_mask_img)

    print(f"✅ Saved dark red ellipse and mask for {filename}")

print("🎯 All dark red ellipses and binary masks generated successfully.")
