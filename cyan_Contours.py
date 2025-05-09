import cv2
import numpy as np
import os
import csv

# in this code we find the cyan contours 
# but cyan region often lies on or very close to the fitted ellipse boundary of the crack zone in some images

# === Paths ===
base_path = "C:\\Users\\shifa\\final project\\Enternal_Contours"
input_folder = os.path.join(base_path, "SLM-P2-CrackZone-NEW")
output_folder = os.path.join(base_path, "Cyan_Contours-SLM-P2")
os.makedirs(output_folder, exist_ok=True)
csv_folder = os.path.join(output_folder, "contours_csv")
os.makedirs(csv_folder, exist_ok=True)

# === HSV Range for Cyan Only
cyan_range = ([85, 50, 80], [105, 255, 255])

# === Morphological Kernel and Area Threshold
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

    # === Step 2: Cyan mask filtered by ellipse
    lower, upper = cyan_range
    cyan_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    cyan_mask = cv2.bitwise_and(cyan_mask, ellipse_mask)
    cyan_mask = cv2.morphologyEx(cyan_mask, cv2.MORPH_OPEN, kernel)
    cyan_mask = cv2.morphologyEx(cyan_mask, cv2.MORPH_CLOSE, kernel)

    # === Step 3: Extract largest contour
    contours, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"❌ No cyan contours found in {filename}")
        continue

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_AREA:
        print(f"⚠ Contour too small in {filename}")
        continue

    # === Step 4: Save overlay image with black contour
    overlay = img.copy()
    cv2.drawContours(overlay, [largest], -1, (0, 0, 0), thickness=10)
    out_path = os.path.join(output_folder, f"{filename[:-4]}_cyan_overlay.png")
    cv2.imwrite(out_path, overlay)

    # === Step 5: Save contour points to CSV
    csv_path = os.path.join(csv_folder, f"{filename[:-4]}_contour.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x", "y"])
        for point in largest.squeeze():
            writer.writerow(point)

    print(f"✅ Saved cyan contour and overlay for {filename}")

print("🎯 All cyan contours extracted and saved.")
