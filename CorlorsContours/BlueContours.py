import cv2
import numpy as np
import os
from scipy.ndimage import binary_fill_holes

# === Paths ===
base_path = "C:\\Users\\shifa\\final project\\Enternal_Contours"
input_folder = os.path.join(base_path, "New Samples-CrackZones")
output_folder = os.path.join(base_path, "blue_Contours-New Samples")
os.makedirs(output_folder, exist_ok=True)
mask_folder = os.path.join(output_folder, "ellipse_masks")
os.makedirs(mask_folder, exist_ok=True)

# === HSV Ranges for All Crack Zone Colors
combined_ranges = [
    ([0, 50, 50], [10, 255, 255]),      # Dark Red
    ([160, 50, 50], [180, 255, 255]),   # Red High
    ([11, 80, 80], [22, 255, 255]),     # Orange
    ([23, 90, 90], [38, 255, 255]),     # Yellow
    ([85, 50, 80], [105, 255, 255]),    # Cyan
    ([105, 50, 50], [125, 255, 255])    # Blue
]

# === Morphological Kernels
initial_dilate_kernel = np.ones((25, 25), np.uint8)
expand_dilate_kernel = np.ones((180, 180), np.uint8)# adjust as needed
close_kernel = np.ones((30, 30), np.uint8)
open_kernel = np.ones((5, 5), np.uint8)

# === Parameters
DILATION_PIXELS = 200
MIN_AREA_THRESHOLD = 5000

# === Process All Images ===
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".png"):
        continue

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # === Step 1: Pink Ellipse Mask
    ellipse_mask = np.zeros_like(gray)
    pink_mask = cv2.inRange(hsv, (140, 50, 50), (170, 255, 255))
    pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_CLOSE, open_kernel)
    contours_pink, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_pink:
        largest = max(contours_pink, key=cv2.contourArea)
        if len(largest) >= 5:
            pink_ellipse = cv2.fitEllipse(largest)
            cv2.ellipse(ellipse_mask, pink_ellipse, 255, -1)
        else:
            print(f"⚠ Not enough points for ellipse in {filename}")
            continue
    else:
        print(f"⚠ No pink ellipse found in {filename}")
        continue

    allowed_area = cv2.dilate(ellipse_mask, np.ones((DILATION_PIXELS, DILATION_PIXELS), np.uint8))

    # === Step 2: Crack Zone Color Mask
    combined_mask = np.zeros_like(gray)
    for lower, upper in combined_ranges:
        lower_np = np.array(lower)
        upper_np = np.array(upper)
        combined_mask |= cv2.inRange(hsv, lower_np, upper_np)

    combined_mask = cv2.bitwise_and(combined_mask, allowed_area)

    # === Step 3: Morphological Cleaning
    combined_mask = cv2.dilate(combined_mask, initial_dilate_kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, close_kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, open_kernel)
    combined_mask = binary_fill_holes(combined_mask > 0).astype(np.uint8) * 255

    # === Step 4: Big Dilation to Expand
    expanded_mask = cv2.dilate(combined_mask, expand_dilate_kernel)
    expanded_mask = cv2.dilate(expanded_mask, np.ones((60, 60), np.uint8))  # Optional second dilation

    # === Step 5: Keep Largest Component
    num_labels, labels_im = cv2.connectedComponents(expanded_mask)
    max_area = 0
    largest_label = 0
    for label_idx in range(1, num_labels):
        area = np.count_nonzero(labels_im == label_idx)
        if area > max_area:
            max_area = area
            largest_label = label_idx

    if max_area < MIN_AREA_THRESHOLD:
        print(f"⚠ No significant crack zone found in {filename}")
        continue

    final_mask = (labels_im == largest_label).astype(np.uint8) * 255

    # === Step 6: Contour + Convex Hull
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"❌ No contour found in {filename}")
        continue

    largest_contour = max(contours, key=cv2.contourArea)
    if len(largest_contour) < 5:
        print(f"⚠ Contour too small to fit ellipse in {filename}")
        continue

    hull = cv2.convexHull(largest_contour)
    fitted_ellipse = cv2.fitEllipse(hull)

    # === Step 7: Save Overlay Image
    overlay = img.copy()
    cv2.ellipse(overlay, fitted_ellipse, (0, 0, 0), thickness=20)  # black
    out_path = os.path.join(output_folder, f"{filename[:-4]}_ellipse_overlay.png")
    cv2.imwrite(out_path, overlay)

    # === Step 8: Generate and Save Binary Mask
    ellipse_mask_img = np.zeros_like(gray)
    cv2.ellipse(ellipse_mask_img, fitted_ellipse, 255, thickness=-1)  # fill
    mask_path = os.path.join(mask_folder, f"{filename[:-4]}_ellipse_mask.png")
    cv2.imwrite(mask_path, ellipse_mask_img)

    print(f"✅ Saved mask and overlay for {filename}")

print("🎯 All overlays and masks generated successfully!")
