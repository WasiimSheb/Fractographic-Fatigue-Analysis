import cv2
import numpy as np
import os

# === Directory Configuration ===
base_path = "C:\\Users\\shifa\\final project\\Enternal_Contours"
heatmap_folder = os.path.join(base_path, "Al_heatmaps")
highlighted_output_folder = os.path.join(base_path, "Al_CrackZone")
mask_output_folder = os.path.join(base_path, "Al_Z27_01_orange_in_red_masks")
os.makedirs(highlighted_output_folder, exist_ok=True)
os.makedirs(mask_output_folder, exist_ok=True)

# === HSV Color Ranges ===
color_ranges = {
    "dark_red": [([0, 200, 100], [10, 255, 180]), ([160, 200, 100], [180, 255, 180])],
    "red": [([0, 180, 180], [10, 255, 255]), ([160, 180, 180], [180, 255, 255])],
    "orange": [([10, 100, 100], [25, 255, 255])],
}

kernel = np.ones((5, 5), np.uint8)

# === Process Heatmaps ===
for filename in os.listdir(heatmap_folder):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    filepath = os.path.join(heatmap_folder, filename)
    image = cv2.imread(filepath)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    orange_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    # Build red mask
    for lower, upper in color_ranges["dark_red"] + color_ranges["red"]:
        red_mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    # Build orange mask
    for lower, upper in color_ranges["orange"]:
        orange_mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)

    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_orange, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    max_area = 0

    for o in contours_orange:
        if cv2.contourArea(o) < 50:
            continue

        M = cv2.moments(o)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        for r in contours_red:
            if cv2.pointPolygonTest(r, (cx, cy), False) >= 0:
                area = cv2.contourArea(o)
                if area > max_area:
                    max_area = area
                    largest_contour = o
                break

    # === If a valid largest orange-in-red was found:
    if largest_contour is not None:
        overlay = image.copy()
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        radius = int(radius)

        # Draw outer black ring + inner yellow
        cv2.circle(overlay, center, radius + 6, (0, 0, 0), 4)
        cv2.circle(overlay, center, radius, (0, 255, 255), 4)

        # Label
        text = "Internal Orange Zone"
        text_position = (center[0] - radius, center[1] - radius - 10)
        cv2.putText(overlay, text, text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Blend overlay with image
        image = cv2.addWeighted(overlay, 0.75, image, 0.25, 0)

        # Save highlighted image
        output_path = os.path.join(
            highlighted_output_folder,
            f"{os.path.splitext(filename)[0]}_highlighted.png"
        )
        cv2.imwrite(output_path, image)

        # Save yellow-filled mask
        mask_img = np.zeros_like(image)
        cv2.drawContours(mask_img, [largest_contour], -1, (0, 255, 255), thickness=-1)

        mask_save_path = os.path.join(
            mask_output_folder,
            f"{os.path.splitext(filename)[0]}_orange_in_red_mask.png"
        )
        cv2.imwrite(mask_save_path, mask_img)

        print(f"✔ Saved largest orange-in-red for {filename}")

print("✅ Done: Largest orange zone in red area saved for all heatmaps.")
