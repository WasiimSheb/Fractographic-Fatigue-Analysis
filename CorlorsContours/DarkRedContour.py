import cv2
import numpy as np
import os

# === Paths ===
base_path = "C:\\Users\\shifa\\final Project\\Enternal_Contours"
input_folder = os.path.join(base_path, "New Samples-CrackZones")
output_folder = os.path.join(base_path, "DarkRed_Contours-NewSamples")
os.makedirs(output_folder, exist_ok=True)
mask_folder = os.path.join(output_folder, "contour_masks")
os.makedirs(mask_folder, exist_ok=True)

# === HSV Range for Dark Red
dark_red_ranges = [
    ([0, 200, 100], [10, 255, 180]),
    ([160, 200, 100], [180, 255, 180])
]

# === Morphological Kernel
kernel = np.ones((5, 5), np.uint8)

# === Process All Images ===
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".png"):
        continue

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # === Step 1: Mask Pink Ellipse Area
    ellipse_mask = np.zeros_like(gray)
    pink_mask = cv2.inRange(hsv, (140, 50, 50), (170, 255, 255))
    pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_CLOSE, kernel)
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

    # === Step 2: Create Mask for Dark Red Inside Ellipse
    dark_red_mask = np.zeros_like(gray)
    for lower, upper in dark_red_ranges:
        dark_red_mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

    dark_red_mask = cv2.bitwise_and(dark_red_mask, ellipse_mask)
    dark_red_mask = cv2.morphologyEx(dark_red_mask, cv2.MORPH_OPEN, kernel)
    dark_red_mask = cv2.morphologyEx(dark_red_mask, cv2.MORPH_CLOSE, kernel)

    # === Step 3: Extract Contours and Keep Only Inside Ellipse
    contours, _ = cv2.findContours(dark_red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"❌ No dark red contour in {filename}")
        continue

    inside_points = []
    for cnt in contours:
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        clipped = cv2.bitwise_and(mask, ellipse_mask)
        ys, xs = np.where(clipped > 0)
        if len(xs) > 0:
            inside_points.append(np.column_stack((xs, ys)))

    if not inside_points:
        print(f"⚠ No valid points inside ellipse for {filename}")
        continue

    all_inside = np.vstack(inside_points)
    hull = cv2.convexHull(all_inside)

    # === Step 4: Save Overlay Image with Black Contour
    overlay = img.copy()
    cv2.drawContours(overlay, [hull], -1, (0, 0, 0), thickness=10)
    overlay_path = os.path.join(output_folder, f"{filename[:-4]}_darkred_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    # === Step 5: Save Binary Mask (white inside)
    mask_img = np.zeros_like(gray)
    cv2.drawContours(mask_img, [hull], -1, 255, thickness=cv2.FILLED)
    mask_path = os.path.join(mask_folder, f"{filename[:-4]}_darkred_mask.png")
    cv2.imwrite(mask_path, mask_img)

    print(f"✅ Saved overlay and mask for {filename}")

print("🎯 All dark red overlays and binary masks generated successfully.")
