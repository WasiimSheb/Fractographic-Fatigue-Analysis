import cv2
import numpy as np
import os
import csv

## in this code we find the red internal contour of the crack zone
# === Paths ===
base_path = "C:\\Users\\shifa\\final Project\\Enternal_Contours"
input_folder = os.path.join(base_path, "EBM9-CrackZone-NEW")
output_folder = os.path.join(base_path, "Red_Contours-EBM9")
os.makedirs(output_folder, exist_ok=True)
csv_folder = os.path.join(output_folder, "contours_csv")
os.makedirs(csv_folder, exist_ok=True)

# === HSV Range for Red Colors
red_ranges = [
    ([0, 50, 50], [10, 255, 255]),     # Low red
    ([160, 50, 50], [180, 255, 255])   # High red (wrap-around)
]

# === Morphological Kernel
kernel = np.ones((3, 3), np.uint8)
MIN_AREA = 150

# === Process All Images ===
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".png"):
        continue

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # === Step 1: Ellipse mask from pink 
    ellipse_mask = np.zeros_like(gray)
    pink_mask = cv2.inRange(hsv, (140, 50, 50), (170, 255, 255))  # HSV range for pink
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


    # === Step 2: Extract red area inside ellipse
    red_mask = np.zeros_like(gray)
    for lower, upper in red_ranges:
        red_mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

    red_mask = cv2.bitwise_and(red_mask, ellipse_mask)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    # === Step 3: Get largest valid contour
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"❌ No red contour found in {filename}")
        continue

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_AREA:
        print(f"⚠ Contour too small in {filename}")
        continue

    # === Step 4: Save thick black contour overlay (optional)
    overlay = img.copy()
    cv2.drawContours(overlay, [largest], -1, (0, 0, 0), thickness=10)
    overlay_path = os.path.join(output_folder, f"{filename[:-4]}_red_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    # === Step 5: Export contour as CSV
    csv_path = os.path.join(csv_folder, f"{filename[:-4]}_red_contour.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x", "y"])
        for point in largest.squeeze():
            writer.writerow(point)

    print(f"✅ Saved red contour and overlay for {filename}")

print("🎯 All red internal contours extracted.")
