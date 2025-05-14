import cv2
import numpy as np
import os
import csv
from scipy.ndimage import binary_fill_holes

# === Paths ===
base_path = "C:\\Users\\shifa\\final project\\Enternal_Contours"
input_folder = os.path.join(base_path, "SLM-P2-CrackZone-NEW")
output_folder = os.path.join(base_path, "blue_Contours-P2")
os.makedirs(output_folder, exist_ok=True)
csv_folder = os.path.join(output_folder, "ellipse_parameters_csv")
os.makedirs(csv_folder, exist_ok=True)

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
expand_dilate_kernel = np.ones((100, 100), np.uint8)  # BIG dilation to expand!
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

    # === Step 1: Detect Pink Ellipse Mask
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

    # === Step 2: Combine Masks from All Color Ranges
    combined_mask = np.zeros_like(gray)
    for lower, upper in combined_ranges:
        lower_np = np.array(lower)
        upper_np = np.array(upper)
        combined_mask |= cv2.inRange(hsv, lower_np, upper_np)

    # Mask only inside allowed pink region
    combined_mask = cv2.bitwise_and(combined_mask, allowed_area)

    # === Step 3: Morphological Cleaning
    combined_mask = cv2.dilate(combined_mask, initial_dilate_kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, close_kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, open_kernel)

    # === Step 4: Fill Holes
    combined_mask = binary_fill_holes(combined_mask > 0).astype(np.uint8) * 255

    # === Step 5: BIG Expand the Mask
    expanded_mask = cv2.dilate(combined_mask, expand_dilate_kernel)

    # === Step 6: Keep Largest Connected Component
    num_labels, labels_im = cv2.connectedComponents(expanded_mask)
    max_area = 0
    largest_label = 0

    for label_idx in range(1, num_labels):
        component = (labels_im == label_idx).astype(np.uint8)
        area = cv2.countNonZero(component)
        if area > max_area:
            max_area = area
            largest_label = label_idx

    if max_area < MIN_AREA_THRESHOLD:
        print(f"⚠ No significant crack zone found in {filename}")
        continue

    final_mask = (labels_im == largest_label).astype(np.uint8) * 255

    # === Step 7: Find External Contour
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"❌ No external crack zone contour found in {filename}")
        continue

    largest_contour = max(contours, key=cv2.contourArea)

    if len(largest_contour) < 5:
        print(f"⚠ Contour too small to fit ellipse in {filename}")
        continue

    # === Step 8: Fit an Ellipse
    fitted_ellipse = cv2.fitEllipse(largest_contour)

    # === Step 9: Save Overlay Image
    overlay = img.copy()
    cv2.ellipse(overlay, fitted_ellipse, (0, 0, 0), thickness=10)  # BLACK ellipse
    out_path = os.path.join(output_folder, f"{filename[:-4]}_full_crackzone_expanded_ellipse_overlay.png")
    cv2.imwrite(out_path, overlay)

    # === Step 10: Save Ellipse Parameters
    csv_path = os.path.join(csv_folder, f"{filename[:-4]}_full_crackzone_expanded_ellipse.csv")
    (center_x, center_y), (major_axis, minor_axis), angle = fitted_ellipse

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["center_x", "center_y", "major_axis_length", "minor_axis_length", "rotation_angle"])
        writer.writerow([center_x, center_y, major_axis, minor_axis, angle])

    print(f"✅ Saved expanded crack zone ellipse for {filename}")

print("🎯 All expanded crack zone ellipses extracted and saved successfully!")
