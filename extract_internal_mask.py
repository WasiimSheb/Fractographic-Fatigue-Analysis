import cv2
import numpy as np
import os
from scipy.spatial import ConvexHull

# === Paths ===
base_path = "C:\\Users\\wasim\\Projects\\extract_mask"
input_folder = os.path.join(base_path, "EBM6_heatmaps")
output_folder = os.path.join(base_path, "EBM6_internal")
os.makedirs(output_folder, exist_ok=True)

# === Process each .png heatmap ===
for image_name in os.listdir(input_folder):
    if not image_name.lower().endswith(".png"):
        continue

    input_path = os.path.join(input_folder, image_name)
    output_path = os.path.join(output_folder, image_name.replace("_heatmap", "_internal"))

    # Load image and convert to HSV
    img = cv2.imread(input_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # === Red HSV ranges ===
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 50])
    upper_red2 = np.array([180, 255, 255])

    # === Yellow/Orange HSV range (extends contour outward) ===
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
    contours, _ = cv2.findContours(heat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours:
        largest = max(contours, key=cv2.contourArea)

        # Extract contour points
        contour_points = largest.reshape(-1, 2)

        # Apply Convex Hull to find the boundary of the internal contour
        hull = ConvexHull(contour_points)
        hull_points = contour_points[hull.vertices]

        # Draw the hull as a red line
        cv2.polylines(img, [hull_points], isClosed=True, color=(50, 50, 255), thickness=25)

        # Draw the original internal contour in green
        cv2.drawContours(img, [largest], -1, (0, 255, 0), 2)

    # Save and display the result
    cv2.imwrite(output_path, img)
    print(f"✔ Saved: {output_path}")
