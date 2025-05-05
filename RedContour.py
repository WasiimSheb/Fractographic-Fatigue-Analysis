import cv2
import numpy as np
import os

# === Paths ===
base_path = "C:\\Users\\shifa\\final project\\Enternal_Contours"
input_folder = os.path.join(base_path, "EBM6-CrackZone")
output_folder = os.path.join(base_path, "Red_Contours-EBM6")
overlay_folder = os.path.join(output_folder, "overlays")
os.makedirs(overlay_folder, exist_ok=True)

# === HSV Range for Red ===
red_ranges = [
    ([0, 180, 180], [10, 255, 255]),
    ([160, 180, 180], [180, 255, 255])
]

kernel = np.ones((5, 5), np.uint8)

# === Process Each Image ===
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".png"):
        continue

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # === Step 1: Detect crack zone ellipse ===
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
        yellow_mask = cv2.inRange(hsv, (25, 150, 150), (35, 255, 255))
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        contours_yellow, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_yellow:
            largest = max(contours_yellow, key=cv2.contourArea)
            if len(largest) >= 5:
                ellipse = cv2.fitEllipse(largest)
                cv2.ellipse(ellipse_mask, ellipse, 255, -1)
                success = True

    if not success:
        print(f"⚠ No ellipse found in {filename}")
        continue

    # === Step 2: Segment red area inside ellipse ===
    red_mask = np.zeros_like(gray)
    for lower, upper in red_ranges:
        red_mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

    red_mask = cv2.bitwise_and(red_mask, ellipse_mask)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    # === Step 3: Extract largest contour (no convex hull)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"❌ No red contours in {filename}")
        continue

    largest_contour = max(contours, key=cv2.contourArea)

    # === Step 4: Draw and Save Overlay ===
    overlay = img.copy()
    cv2.drawContours(overlay, [largest_contour], -1, (255, 255, 255), thickness=6)

    overlay_path = os.path.join(overlay_folder, f"{filename[:-4]}_red_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    print(f"✅ Saved red overlay for {filename}")

print("🎯 Done extracting realistic red internal contours (no convex hull).")
