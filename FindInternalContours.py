import cv2
import numpy as np
import os

# === Base Paths ===
base_path = "C:\\Users\\shifa\\final project\\Enternal_Contours"
input_folder = os.path.join(base_path, "EBM6_CrackZone")
output_folder = os.path.join(base_path, "internal_contours_combined")
os.makedirs(output_folder, exist_ok=True)

# === Color Phase Ranges (HSV) ===
color_ranges = {
    "dark_red": [([0, 200, 100], [10, 255, 180]), ([160, 200, 100], [180, 255, 180])],
    "red": [([0, 180, 180], [10, 255, 255]), ([160, 180, 180], [180, 255, 255])],
    "orange": [([10, 100, 100], [25, 255, 255])],
    "yellow": [([25, 150, 150], [35, 255, 255])],
    "cyan": [([75, 100, 100], [95, 255, 255])],
    "blue": [([100, 100, 100], [130, 255, 255])]
}

# === Kernel for Morphology ===
kernel = np.ones((5, 5), np.uint8)

# === Create Output Folders ===
output_folders = {}
for phase in color_ranges:
    folder = os.path.join(output_folder, f"{phase}_contours")
    os.makedirs(folder, exist_ok=True)
    overlay_folder = os.path.join(folder, "overlays")
    os.makedirs(overlay_folder, exist_ok=True)
    output_folders[phase] = (folder, overlay_folder)

# === Process Each Image ===
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".png"):
        continue

    image_path = os.path.join(input_folder, filename)
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # === Try Method 1: White Ellipse ===
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white_mask = cv2.inRange(img, (255, 255, 255), (255, 255, 255))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    contours_white, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ellipse_mask = np.zeros_like(gray)
    success = False

    if contours_white:
        largest = max(contours_white, key=cv2.contourArea)
        if len(largest) >= 5:
            ellipse = cv2.fitEllipse(largest)
            cv2.ellipse(ellipse_mask, ellipse, 255, -1)
            success = True

    # === Fallback to Method 2: Yellow Ellipse ===
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
        print(f"⚠ No ellipse found in {filename}, skipping.")
        continue

    # === Extract Each Color Region Inside Ellipse ===
    for phase, ranges in color_ranges.items():
        phase_mask = np.zeros_like(gray)
        for lower, upper in ranges:
            phase_mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Clean phase mask
        phase_mask = cv2.morphologyEx(phase_mask, cv2.MORPH_OPEN, kernel)
        phase_mask = cv2.morphologyEx(phase_mask, cv2.MORPH_CLOSE, kernel)

        # Apply ellipse mask
        masked = cv2.bitwise_and(phase_mask, ellipse_mask)

        contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"❌ No {phase} contours in {filename}")
            continue

        largest_contour = max(contours, key=cv2.contourArea)

        # === Save Mask ===
        mask_img = np.zeros_like(img)
        cv2.drawContours(mask_img, [largest_contour], -1, (0, 255, 255), thickness=-1)

        folder, overlay_folder = output_folders[phase]
        mask_path = os.path.join(folder, f"{filename[:-4]}_{phase}.png")
        overlay_path = os.path.join(overlay_folder, f"{filename[:-4]}_{phase}_overlay.png")

        # === Save Overlay ===
        overlay_img = img.copy()
        cv2.drawContours(overlay_img, [largest_contour], -1, (255, 255, 255), thickness=6)
        cv2.imwrite(mask_path, mask_img)
        cv2.imwrite(overlay_path, overlay_img)

        print(f"✅ {phase} saved for {filename}")

print("🎯 Finished extracting internal contours using dual ellipse strategy.")
