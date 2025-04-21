import cv2
import numpy as np
import os

# === Base Path Setup ===
base_path = "C:\\Users\\shifa\\final project\\Enternal_Contours"
input_folder = os.path.join(base_path, "EBM6_CrackZone")
main_output_folder = os.path.join(base_path, "Internal crack contour")
os.makedirs(main_output_folder, exist_ok=True)

# === Define Color Departments and Ranges ===
color_ranges = {
    "dark_red": [([0, 200, 100], [10, 255, 180]), ([160, 200, 100], [180, 255, 180])],
    "red": [([0, 180, 180], [10, 255, 255]), ([160, 180, 180], [180, 255, 255])],
    "orange": [([10, 100, 100], [25, 255, 255])],
    "yellow": [([25, 150, 150], [35, 255, 255])],
    "cyan": [([75, 100, 100], [95, 255, 255])],
    "blue": [([100, 100, 100], [130, 255, 255])]
}

# === Create Output Folders per Color ===
output_folders = {}
for phase in color_ranges:
    phase_folder = os.path.join(main_output_folder, f"{phase}_contours")
    os.makedirs(phase_folder, exist_ok=True)
    output_folders[phase] = phase_folder

kernel = np.ones((5, 5), np.uint8)

# === Process Images ===
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".png"):
        continue

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # === STEP 1: Detect Yellow Ellipse Contour and Create Mask ===
    yellow_mask = cv2.inRange(hsv, (25, 150, 150), (35, 255, 255))
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ellipse_mask = np.zeros_like(yellow_mask)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if len(largest) >= 5:
            ellipse = cv2.fitEllipse(largest)
            cv2.ellipse(ellipse_mask, ellipse, 255, -1)
        else:
            print(f"⚠ Not enough points for ellipse in {filename}")
            continue
    else:
        print(f"⚠ No yellow ellipse found in {filename}")
        continue

        # === STEP 2: For Each Color Phase ===
    for phase, ranges in color_ranges.items():
        phase_mask = np.zeros_like(yellow_mask)

        for lower, upper in ranges:
            phase_mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Clean noise
        phase_mask = cv2.morphologyEx(phase_mask, cv2.MORPH_OPEN, kernel)
        phase_mask = cv2.morphologyEx(phase_mask, cv2.MORPH_CLOSE, kernel)

        # Apply ellipse mask to limit to crack zone
        limited_mask = cv2.bitwise_and(phase_mask, ellipse_mask)

        contours, _ = cv2.findContours(limited_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"❌ No {phase} contours found in {filename}")
            continue

        # Largest internal contour
        largest_contour = max(contours, key=cv2.contourArea)

        # === Save Mask ===
        mask_out = np.zeros_like(img)
        cv2.drawContours(mask_out, [largest_contour], -1, (0, 255, 255), thickness=-1)

        phase_folder = output_folders[phase]
        mask_save_path = os.path.join(phase_folder, f"{filename[:-4]}_{phase}.png")
        cv2.imwrite(mask_save_path, mask_out)

        # === Save Overlay ===
        overlay_folder = os.path.join(phase_folder, "overlays")
        os.makedirs(overlay_folder, exist_ok=True)

        overlay_img = img.copy()
        cv2.drawContours(overlay_img, [largest_contour], -1, (255, 255, 255), thickness=6)  # White & bold
        overlay_save_path = os.path.join(overlay_folder, f"{filename[:-4]}_{phase}_overlay.png")
        cv2.imwrite(overlay_save_path, overlay_img)

        print(f"✅ Saved {phase} mask and overlay for {filename}")
