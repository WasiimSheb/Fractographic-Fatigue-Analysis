import cv2
import numpy as np
import os
import pandas as pd
from scipy.ndimage import binary_fill_holes
from scipy.interpolate import splprep, splev

"""
This script processes images of crack zones from specimens to:
1. Detect and segment internal contours corresponding to different colors (dark red, red, yellow, cyan, blue).
2. Calculate the internal area of each color region:
   - In pixels² (image units).
   - In micrometers² (physical units) based on a known pixel size.
3. Compute and store the scaling factor (micrometers²/pixels²).
4. Generate overlay images with the detected contours drawn.
5. Save a structured CSV file containing:
   - Pixel areas.
   - Micrometer² areas.
   - Scale factors for each region.
"""

# === Base Paths ===
base_path = "C:\\Users\\shifa\\final project\\Area-pixles\\"
internal_contours_folder = os.path.join(base_path, "Internal_Contours_EBM6")
overlay_base_folder = os.path.join(internal_contours_folder, "Overlays")
os.makedirs(overlay_base_folder, exist_ok=True)

# Create separate overlay folders for each color
overlay_folders = {}
for color in ["dark_red", "red", "yellow", "cyan", "blue"]:
    folder = os.path.join(overlay_base_folder, f"{color}_overlays")
    os.makedirs(folder, exist_ok=True)
    overlay_folders[color] = folder

# === Output Excel ===
output_csv = os.path.join(internal_contours_folder, "Internal_Contour_Areas_EBM6.csv")

# === Input folder with images ===
input_folder = os.path.join(base_path, "EBM6-CrackZone-New")

# === Parameters ===
DILATION_PIXELS = 100
SMOOTHNESS = 0.001
NUM_POINTS = 600

# === Micrometer Parameters ===
PIXEL_SIZE_MICRONS = 1.34375  # (יחס ההמרה מיקרון/פיקסל)
MICRON_AREA_FACTOR = PIXEL_SIZE_MICRONS ** 2  # מיקרון בריבוע לכל פיקסל

# === Morphological Kernels ===
kernel_small = np.ones((3, 3), np.uint8)
kernel_medium = np.ones((5, 5), np.uint8)
dilate_kernel = np.ones((25, 25), np.uint8)
close_kernel = np.ones((35, 35), np.uint8)
open_kernel = np.ones((5, 5), np.uint8)
expand_dilate_kernel = np.ones((100, 100), np.uint8)

# === Initialize results list ===
results = []

# === Process All Images ===
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".png"):
        continue

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sample_name = filename[:-4]

    # Step 1: Detect Pink Ellipse (Crack Zone)
    ellipse_mask = np.zeros_like(gray)
    pink_mask = cv2.inRange(hsv, (140, 50, 50), (170, 255, 255))
    pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_CLOSE, open_kernel)
    contours_pink, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_pink or len(max(contours_pink, key=cv2.contourArea)) < 5:
        print(f"⚠ No valid pink ellipse for {filename}")
        continue
    ellipse = cv2.fitEllipse(max(contours_pink, key=cv2.contourArea))
    cv2.ellipse(ellipse_mask, ellipse, 255, -1)

    row = {"sample": sample_name}

    # === DARK RED ===
    dark_red_mask = np.zeros_like(gray)
    for lower, upper in [([0, 200, 100], [10, 255, 180]), ([160, 200, 100], [180, 255, 180])]:
        dark_red_mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    dark_red_mask = cv2.bitwise_and(dark_red_mask, ellipse_mask)
    dark_red_mask = cv2.morphologyEx(dark_red_mask, cv2.MORPH_OPEN, kernel_small)
    dark_red_mask = cv2.morphologyEx(dark_red_mask, cv2.MORPH_CLOSE, kernel_small)

    ys, xs = np.where(dark_red_mask > 0)
    if len(xs) > 0:
        points = np.vstack((xs, ys)).T
        hull = cv2.convexHull(points)
        area = cv2.contourArea(hull)
        overlay = img.copy()
        cv2.drawContours(overlay, [hull], -1, (0, 0, 0), thickness=10)
        cv2.imwrite(os.path.join(overlay_folders["dark_red"], f"{sample_name}_dark_red_overlay.png"), overlay)
        row["dark_red_area_pixels"] = area
        row["dark_red_area_microns²"] = area * MICRON_AREA_FACTOR
    else:
        row["dark_red_area_pixels"] = 0
        row["dark_red_area_microns²"] = 0

    # === RED ===
    red_mask = np.zeros_like(gray)
    for lower, upper in [([0, 50, 50], [10, 255, 255]), ([160, 50, 50], [180, 255, 255])]:
        red_mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    red_mask = cv2.bitwise_and(red_mask, ellipse_mask)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_small)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_small)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        overlay = img.copy()
        cv2.drawContours(overlay, [largest], -1, (0, 0, 0), thickness=10)
        cv2.imwrite(os.path.join(overlay_folders["red"], f"{sample_name}_red_overlay.png"), overlay)
        row["red_area_pixels"] = area
        row["red_area_microns²"] = area * MICRON_AREA_FACTOR
    else:
        row["red_area_pixels"] = 0
        row["red_area_microns²"] = 0

    # === YELLOW ===
    yellow_mask = np.zeros_like(gray)
    for lower, upper in [([0, 50, 50], [10, 255, 255]),
                         ([160, 50, 50], [180, 255, 255]),
                         ([11, 80, 80], [22, 255, 255]),
                         ([23, 90, 90], [38, 255, 255])]:
        yellow_mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    yellow_mask = cv2.bitwise_and(yellow_mask, ellipse_mask)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel_small)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel_small)

    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        overlay = img.copy()
        cv2.drawContours(overlay, [largest], -1, (0, 0, 0), thickness=10)
        cv2.imwrite(os.path.join(overlay_folders["yellow"], f"{sample_name}_yellow_overlay.png"), overlay)
        row["yellow_area_pixels"] = area
        row["yellow_area_microns²"] = area * MICRON_AREA_FACTOR
    else:
        row["yellow_area_pixels"] = 0
        row["yellow_area_microns²"] = 0

    # === CYAN (Spline Based) ===
    allowed_area = cv2.dilate(ellipse_mask, np.ones((DILATION_PIXELS, DILATION_PIXELS), np.uint8))
    combined_mask = np.zeros_like(gray)
    for lower, upper in [([0, 50, 50], [10, 255, 255]),
                         ([160, 50, 50], [180, 255, 255]),
                         ([11, 80, 80], [22, 255, 255]),
                         ([23, 90, 90], [38, 255, 255]),
                         ([85, 50, 80], [105, 255, 255])]:
        combined_mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    combined_mask = cv2.bitwise_and(combined_mask, allowed_area)
    combined_mask = cv2.dilate(combined_mask, dilate_kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, close_kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, open_kernel)
    combined_mask = binary_fill_holes(combined_mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea).squeeze()
        if len(largest.shape) == 2 and largest.shape[0] >= 10:
            x, y = largest[:, 0], largest[:, 1]
            tck, u = splprep([x, y], s=SMOOTHNESS, per=True)
            u_fine = np.linspace(0, 1, NUM_POINTS)
            x_fine, y_fine = splev(u_fine, tck)
            smooth_contour = np.stack((x_fine, y_fine), axis=1).astype(np.int32)
            area = cv2.contourArea(smooth_contour)
            overlay = img.copy()
            cv2.polylines(overlay, [smooth_contour], isClosed=True, color=(0, 0, 0), thickness=10)
            cv2.imwrite(os.path.join(overlay_folders["cyan"], f"{sample_name}_cyan_overlay.png"), overlay)
            row["cyan_area_pixels"] = area
            row["cyan_area_microns²"] = area * MICRON_AREA_FACTOR
        else:
            row["cyan_area_pixels"] = 0
            row["cyan_area_microns²"] = 0
    else:
        row["cyan_area_pixels"] = 0
        row["cyan_area_microns²"] = 0

    # === BLUE (Big Expansion + Ellipse) ===
    allowed_area = cv2.dilate(ellipse_mask, np.ones((DILATION_PIXELS, DILATION_PIXELS), np.uint8))
    blue_mask = np.zeros_like(gray)
    for lower, upper in [([0, 50, 50], [10, 255, 255]),
                         ([160, 50, 50], [180, 255, 255]),
                         ([11, 80, 80], [22, 255, 255]),
                         ([23, 90, 90], [38, 255, 255]),
                         ([85, 50, 80], [105, 255, 255]),
                         ([105, 50, 50], [125, 255, 255])]:
        blue_mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    blue_mask = cv2.bitwise_and(blue_mask, allowed_area)
    blue_mask = cv2.dilate(blue_mask, dilate_kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, close_kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, open_kernel)
    blue_mask = binary_fill_holes(blue_mask > 0).astype(np.uint8) * 255
    expanded_mask = cv2.dilate(blue_mask, expand_dilate_kernel)

    num_labels, labels_im = cv2.connectedComponents(expanded_mask)
    max_area = 0
    largest_label = 0
    for label_idx in range(1, num_labels):
        component = (labels_im == label_idx).astype(np.uint8)
        area = cv2.countNonZero(component)
        if area > max_area:
            max_area = area
            largest_label = label_idx
    if max_area < 5000:
        print(f"⚠ No significant blue zone in {filename}")
        row["blue_area_pixels"] = 0
        row["blue_area_microns²"] = 0
        continue
    final_mask = (labels_im == largest_label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if len(largest) >= 5:
            ellipse_fit = cv2.fitEllipse(largest)
            (cx, cy), (major_axis, minor_axis), angle = ellipse_fit
            area = np.pi * (major_axis / 2) * (minor_axis / 2)
            overlay = img.copy()
            cv2.ellipse(overlay, ellipse_fit, (0, 0, 0), thickness=25)
            cv2.imwrite(os.path.join(overlay_folders["blue"], f"{sample_name}_blue_overlay.png"), overlay)
            row["blue_area_pixels"] = area
            row["blue_area_microns²"] = area * MICRON_AREA_FACTOR
        else:
            row["blue_area_pixels"] = 0
            row["blue_area_microns²"] = 0
    else:
        row["blue_area_pixels"] = 0
        row["blue_area_microns²"] = 0

    results.append(row)

# === Save All Results  as structured Excel
df = pd.DataFrame(results)

# === Create a NEW structured DataFrame ===
structured_df = pd.DataFrame()

structured_df["specimen"] = df["sample"]

# ==== PIXELS SECTION ====
structured_df["dark_red"] = df["dark_red_area_pixels"]
structured_df["red"] = df["red_area_pixels"]
structured_df["yellow"] = df["yellow_area_pixels"]
structured_df["cyan"] = df["cyan_area_pixels"]
structured_df["blue"] = df["blue_area_pixels"]

# ==== MICROMETER² SECTION ====
structured_df["dark_red"] = df["dark_red_area_microns²"]
structured_df["red"] = df["red_area_microns²"]
structured_df["yellow"] = df["yellow_area_microns²"]
structured_df["cyan"] = df["cyan_area_microns²"]
structured_df["blue"] = df["blue_area_microns²"]

# ==== SCALE FACTOR SECTION ====
scale_value = round((PIXEL_SIZE_MICRONS ** 2), 8)
structured_df["dark_red"] = scale_value
structured_df["red"] = scale_value
structured_df["yellow"] = scale_value
structured_df["cyan"] = scale_value
structured_df["blue"] = scale_value

# === Save the final structured Excel ===
structured_output_csv = os.path.join(internal_contours_folder, "Internal_Contour_Areas_EBM6_Structured.csv")
structured_df.to_csv(structured_output_csv, index=False)

print("\n🎯 Structured Internal Contour Areas for all colorssaved successfully!")
