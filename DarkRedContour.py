import cv2
import numpy as np
import os
import csv

# === Base Paths ===
base_path = "C:\\Users\\shifa\\final project\\Enternal_Contours"
input_folder = os.path.join(base_path, "SLM-P3-CrackZone")
output_folder = os.path.join(base_path, "DarkRed_Contours-SLM-P3")
overlay_folder = os.path.join(output_folder, "overlays")
mask_folder = os.path.join(output_folder, "masks")              # ✅ NEW
csv_folder = os.path.join(output_folder, "csv_pixels")          # ✅ NEW
os.makedirs(overlay_folder, exist_ok=True)
os.makedirs(mask_folder, exist_ok=True)                         # ✅ NEW
os.makedirs(csv_folder, exist_ok=True)                          # ✅ NEW

# === Color Range for Dark Red in HSV ===
dark_red_ranges = [
    ([0, 200, 100], [10, 255, 180]),
    ([160, 200, 100], [180, 255, 180])
]

# === Kernel ===
kernel = np.ones((5, 5), np.uint8)

# === Process All Images ===
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".png"):
        continue

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # === Step 1: Detect Ellipse (White or Pink) === ✅ MODIFIED
    ellipse_mask = np.zeros_like(gray)
    success = False

    white_mask = cv2.inRange(img, (255, 255, 255), (255, 255, 255))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    contours_white, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_white:
        largest = max(contours_white, key=cv2.contourArea)
        if len(largest) >= 5:
            ellipse = cv2.fitEllipse(largest)
            cv2.ellipse(ellipse_mask, ellipse, 255, -1)
            success = True

    if not success:
        pink_mask = cv2.inRange(hsv, (140, 50, 50), (170, 255, 255))  # ✅ pink instead of yellow
        pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_CLOSE, kernel)
        contours_pink, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_pink:
            largest = max(contours_pink, key=cv2.contourArea)
            if len(largest) >= 5:
                ellipse = cv2.fitEllipse(largest)
                cv2.ellipse(ellipse_mask, ellipse, 255, -1)
                success = True

    if not success:
        print(f"⚠ No ellipse found in {filename}")
        continue

    # === Step 2: Create Dark Red Mask and Filter by Ellipse
    dark_red_mask = np.zeros_like(gray)
    for lower, upper in dark_red_ranges:
        dark_red_mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

    # Limit to crack zone
    dark_red_mask = cv2.bitwise_and(dark_red_mask, ellipse_mask)

    # Clean mask
    dark_red_mask = cv2.morphologyEx(dark_red_mask, cv2.MORPH_OPEN, kernel)
    dark_red_mask = cv2.morphologyEx(dark_red_mask, cv2.MORPH_CLOSE, kernel)

    # === ✅ Step 2.5: Save binary mask
    mask_path = os.path.join(mask_folder, f"{filename[:-4]}_darkred_mask.png")
    cv2.imwrite(mask_path, dark_red_mask)

    # === ✅ Step 2.6: Export pixel coordinates to CSV
    ys, xs = np.where(dark_red_mask > 0)
    csv_path = os.path.join(csv_folder, f"{filename[:-4]}_pixels.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x", "y"])
        for x, y in zip(xs, ys):
            writer.writerow([x, y])

    # === Step 3: Get Contours and Filter Inside Ellipse
    contours, _ = cv2.findContours(dark_red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"❌ No dark red regions in {filename}")
        continue

    # Merge all valid points (inside ellipse only)
    inside_points = []
    for cnt in contours:
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mask = cv2.bitwise_and(mask, ellipse_mask)
        ys, xs = np.where(mask > 0)
        points = np.array(list(zip(xs, ys)))
        if len(points) > 0:
            inside_points.append(points)

    if not inside_points:
        print(f"⚠ No valid dark red points inside ellipse for {filename}")
        continue

    all_inside = np.vstack(inside_points)
    hull = cv2.convexHull(all_inside)

    # === Step 4: Draw White Contour on Overlay Only
    overlay = img.copy()
    cv2.drawContours(overlay, [hull], -1, (255, 255, 255), thickness=6)

    overlay_path = os.path.join(overlay_folder, f"{filename[:-4]}_overlay_clipped.png")
    cv2.imwrite(overlay_path, overlay)
    print(f"✅ Overlay saved for {filename}")

print("🎯 Finished filtering, saving overlays, masks, and pixel CSVs.")
