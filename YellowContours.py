import cv2
import numpy as np
import os

# \This is the initial processing of the yellow color in the heatmaps...

# === Paths ===
base_path = "C:\\Users\\wasim\\Projects\\heatmaps-classifier\\crackedsamples"
input_folder = os.path.join(base_path, "EBM6_CrackZone")
output_folder = os.path.join(base_path, "Cyan_Contour_P1")
os.makedirs(output_folder, exist_ok=True)

# Combine both cyan and yellow HSV ranges
cyan_ranges = [
    ([15, 100, 100], [40, 255, 255]),   # Yellow-Orange
    ([80, 100, 100], [100, 255, 255])   # Cyan
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

    # === Step 1: Detect crack zone ellipse
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
        yellow_mask_for_ellipse = cv2.inRange(hsv, (25, 150, 150), (35, 255, 255))
        yellow_mask_for_ellipse = cv2.morphologyEx(yellow_mask_for_ellipse, cv2.MORPH_CLOSE, kernel)
        contours_yellow, _ = cv2.findContours(yellow_mask_for_ellipse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_yellow:
            largest = max(contours_yellow, key=cv2.contourArea)
            if len(largest) >= 5:
                ellipse = cv2.fitEllipse(largest)
                cv2.ellipse(ellipse_mask, ellipse, 255, -1)
                success = True

    if not success:
        print(f"⚠ No ellipse found in {filename}")
        continue

    # === Step 2: Segment cyan parts inside ellipse
    cyan_mask = np.zeros_like(gray)
    for lower, upper in cyan_ranges:
        cyan_mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

    cyan_inside = cv2.bitwise_and(cyan_mask, ellipse_mask)
    cyan_inside = cv2.morphologyEx(cyan_inside, cv2.MORPH_OPEN, kernel)
    cyan_inside = cv2.morphologyEx(cyan_inside, cv2.MORPH_CLOSE, kernel)

    # === Step 3: Extract contours and draw in white
    contours, _ = cv2.findContours(cyan_inside, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"❌ No cyan areas found in {filename}")
        continue

    overlay = img.copy()
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), thickness=6)

    output_path = os.path.join(output_folder, f"{filename[:-4]}_cyan_contour.png")
    cv2.imwrite(output_path, overlay)
    print(f"✅ Saved cyan contour overlay for {filename}")

print("🎯 Done highlighting cyan areas with white contours.")
