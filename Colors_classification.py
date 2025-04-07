import cv2
import numpy as np
import os

# === Configuration ===
input_folder = "EBM6_heatmaps"
output_base_folder = os.path.join(input_folder, "heatmap_colors_output")

# === Define HSV color ranges for different heatmap zones ===
color_ranges = {
    "dark_red": [([0, 200, 100], [10, 255, 180]), ([160, 200, 100], [180, 255, 180])],
    "red": [([0, 180, 180], [10, 255, 255]), ([160, 180, 180], [180, 255, 255])],
    "orange": [([10, 100, 100], [25, 255, 255])],
    "yellow": [([26, 100, 100], [35, 255, 255])],
    "green": [([36, 80, 80], [85, 255, 255])],
    "cyan": [([86, 80, 80], [100, 255, 255])],
    "blue": [([101, 80, 80], [130, 255, 255])],
    "purple": [([131, 80, 80], [160, 255, 255])]
}

# === Morphology kernel ===
kernel = np.ones((5, 5), np.uint8)

# === Process Each Image ===
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    filepath = os.path.join(input_folder, filename)
    image = cv2.imread(filepath)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for color_name, ranges in color_ranges.items():
        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)

        # Build combined mask for color
        for lower, upper in ranges:
            lower_np = np.array(lower)
            upper_np = np.array(upper)
            mask = cv2.inRange(hsv, lower_np, upper_np)
            mask_total = cv2.bitwise_or(mask_total, mask)

        # Clean the mask
        mask_clean = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        # Apply mask to original image
        extracted = cv2.bitwise_and(image, image, mask=mask_clean)
        mask_vis = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)

        # Composite view
        combined = np.hstack((image, mask_vis, extracted))

        # Save to color-specific output folder
        color_folder = os.path.join(output_base_folder, color_name)
        os.makedirs(color_folder, exist_ok=True)

        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(color_folder, f"{base_name}_{color_name}_composite.png")
        cv2.imwrite(output_path, combined)
        print(f"✔ Saved for {color_name}: {filename}")

print("✅ All color extractions completed.")
