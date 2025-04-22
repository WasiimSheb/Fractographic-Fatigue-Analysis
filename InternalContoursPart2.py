import cv2
import numpy as np
import os
#find internal contours based on the results of convex hull to detect the crack area
# === Base Paths ===
base_path = "C:\\Users\\shifa\\final project\\Enternal_Contours"
input_folder = os.path.join(base_path, "Cracks_Part2")
output_folder = os.path.join(base_path, "internal contpurs part2")
os.makedirs(output_folder, exist_ok=True)

# === Define Color Departments ===
color_ranges = {
    "dark_red": [([0, 200, 100], [10, 255, 180]), ([160, 200, 100], [180, 255, 180])],
    "red": [([0, 180, 180], [10, 255, 255]), ([160, 180, 180], [180, 255, 255])],
    "orange": [([10, 100, 100], [25, 255, 255])],
    "yellow": [([25, 150, 150], [35, 255, 255])],
    "cyan": [([75, 100, 100], [95, 255, 255])],
    "blue": [([100, 100, 100], [130, 255, 255])]
}

# === Morphological Kernel ===
kernel = np.ones((5, 5), np.uint8)

# === Create subfolders ===
output_folders = {}
for color in color_ranges:
    folder = os.path.join(output_folder, f"{color}_contours")
    os.makedirs(folder, exist_ok=True)
    output_folders[color] = folder

# === Process Each Image ===
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".png"):
        continue

    image_path = os.path.join(input_folder, filename)
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # === STEP 1: Detect White Elliptical Contour ===
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white_mask = cv2.inRange(img, (255, 255, 255), (255, 255, 255))

    # Clean ellipse mask
    white_mask = cv2.dilate(white_mask, kernel, iterations=2)
    white_mask = cv2.erode(white_mask, kernel, iterations=2)

    # Find the white elliptical contour
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ellipse_mask = np.zeros_like(white_mask)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if len(largest) >= 5:
            ellipse = cv2.fitEllipse(largest)
            cv2.ellipse(ellipse_mask, ellipse, 255, -1)
        else:
            print(f"⚠ Not enough points to fit ellipse in {filename}")
            continue
    else:
        print(f"⚠ No white ellipse found in {filename}")
        continue

    # === STEP 2: For each color department ===
    for phase, ranges in color_ranges.items():
        color_mask = np.zeros_like(white_mask)

        for lower, upper in ranges:
            color_mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

        # Limit to the crack zone (white ellipse area)
        masked_inside_ellipse = cv2.bitwise_and(color_mask, ellipse_mask)

        contours, _ = cv2.findContours(masked_inside_ellipse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"❌ No {phase} contours found in {filename}")
            continue

        # Get largest internal contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Draw on image in BLACK
        overlay = img.copy()
        cv2.drawContours(overlay, [largest_contour], -1, (0, 0, 0), thickness=8)

        # Save result
        save_path = os.path.join(output_folders[phase], f"{filename[:-4]}_{phase}_overlay.png")
        cv2.imwrite(save_path, overlay)

        print(f"✅ {phase} contour saved for {filename}")

print("🎯 All black contour overlays saved in organized folders.")
