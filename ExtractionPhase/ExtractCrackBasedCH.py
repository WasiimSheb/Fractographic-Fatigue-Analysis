@@ -0,0 +1,66 @@
import cv2
import numpy as np
import os
from scipy.spatial import ConvexHull

# === Paths ===
base_path = "C:\\Users\\shifa\\final project\\Enternal_Contours"
input_folder = os.path.join(base_path, "SLM-Problamtic-HM")
output_folder = os.path.join(base_path, "SLM-output")
os.makedirs(output_folder, exist_ok=True)

# === Process each .png heatmap ===
for image_name in os.listdir(input_folder):
    if not image_name.lower().endswith(".png"):
        continue

    input_path = os.path.join(input_folder, image_name)
    output_path = os.path.join(output_folder, image_name.replace("_heatmap", "_heatmap_highlighted"))

    # Load image and convert to HSV
    img = cv2.imread(input_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # === Extended Warm HSV range (red to yellow) ===
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_orange = np.array([11, 70, 50])
    upper_orange = np.array([35, 255, 255])

    # Create warm zone masks
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    heat_mask = cv2.bitwise_or(mask_red, mask_orange)

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

        # Apply Convex Hull
        hull = ConvexHull(contour_points)
        hull_points = contour_points[hull.vertices]

        # Draw convex hull in WHITE
        cv2.polylines(img, [hull_points], isClosed=True, color=(255, 255, 255), thickness=25)

    else:
        print(f"⚠ No contours found in: {image_name}")

    # Save the final result
    cv2.imwrite(output_path, img)
    print(f"✔ Saved: {output_path}")
