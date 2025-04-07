import cv2
import numpy as np
import os

# === Paths ===
base_path = "C:\\Users\\shifa\\final project\\extract_mask"
input_folder = os.path.join(base_path, "Al_Z27_01_heatmaps")
output_folder = os.path.join(base_path, "Al_Z27_01_Contours")
mask_output_folder = os.path.join(base_path, "Al_Z27_01d_masks_internal")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(mask_output_folder, exist_ok=True)

# === Process each .png heatmap ===
for image_name in os.listdir(input_folder):
    if not image_name.lower().endswith(".png"):
        continue

    input_path = os.path.join(input_folder, image_name)
    output_path = os.path.join(output_folder, image_name.replace("_heatmap", "_internal.png"))
    mask_path = os.path.join(mask_output_folder, image_name.replace("_heatmap", "_internal_mask.png"))

    # Load image and convert to HSV
    img = cv2.imread(input_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # === Red HSV ranges ===
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 50])
    upper_red2 = np.array([180, 255, 255])

    # === Yellow/Orange HSV range (to extend around red)
    lower_yellow = np.array([20, 80, 50])
    upper_yellow = np.array([35, 255, 255])

    # Create masks
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combine red + yellow masks
    heat_mask = cv2.bitwise_or(mask_red, mask_yellow)

    # === Morphological smoothing ===
    kernel = np.ones((9, 9), np.uint8)
    heat_mask = cv2.morphologyEx(heat_mask, cv2.MORPH_CLOSE, kernel)
    heat_mask = cv2.morphologyEx(heat_mask, cv2.MORPH_OPEN, kernel)

    # === Find and draw largest contour ===
    contours, _ = cv2.findContours(heat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)

        # === Draw thick green contour on original image ===
        cv2.drawContours(img, [largest], -1, (0, 255, 0), 14, cv2.LINE_AA)

        # === Add "Internal" label ===
        cv2.putText(
            img, "Internal", (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2, (0, 255, 0), 4, cv2.LINE_AA
        )

        # === Create color mask (green fill on black background) ===
        mask_output = np.zeros_like(img)  # 3-channel black background
        cv2.drawContours(mask_output, [largest], -1, (0, 255, 255), thickness=-1)

        # === Save color mask ===
        cv2.imwrite(mask_path, mask_output)

    # === Save annotated image ===
    cv2.imwrite(output_path, img)
    print(f"✔ Saved: {output_path}")
    print(f"✔ Mask:  {mask_path}")
