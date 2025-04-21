import cv2
import numpy as np
import os
import math

# === Configuration ===
input_folder = "EBM6_heatmaps"
output_base_folder = os.path.join(input_folder, "heatmap_colors_output")
highlighted_output_folder = os.path.join(output_base_folder, "highlighted_defects")
os.makedirs(highlighted_output_folder, exist_ok=True)

# === Static dark red range ===
color_ranges = {
    "dark_red": [([0, 200, 100], [10, 255, 180]), ([160, 200, 100], [180, 255, 180])]
}

# === Morphology kernel ===
kernel = np.ones((5, 5), np.uint8)

# === Helper: Check if circle is inside another ===
def is_inside(c1, c2):
    cx1, cy1 = c1["center"]
    cx2, cy2 = c2["center"]
    dist = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    return dist + c1["radius"] <= c2["radius"]

# === HSV presets ===
red_presets = [
    [([0, 180, 180], [10, 255, 255]), ([160, 180, 180], [180, 255, 255])],
    [([0, 150, 150], [10, 255, 255]), ([160, 150, 150], [180, 255, 255])],
    [([0, 130, 100], [10, 255, 255]), ([160, 130, 100], [180, 255, 255])]
]

orange_presets = [
    [([10, 100, 100], [25, 255, 255])],
    [([8, 80, 80], [24, 255, 255])],
    [([10, 70, 70], [26, 255, 255])]
]

yellow_range = [([26, 100, 100], [35, 255, 255])]
blue_range = [([100, 80, 80], [130, 255, 255])]

# === Process images ===
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    filepath = os.path.join(input_folder, filename)
    image = cv2.imread(filepath)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])

    best_combo = None
    max_orange_area = 0

    for red_range in red_presets:
        for orange_range in orange_presets:
            temp_red = np.zeros(hsv.shape[:2], dtype=np.uint8)
            temp_orange = np.zeros(hsv.shape[:2], dtype=np.uint8)

            for lower, upper in color_ranges["dark_red"] + red_range:
                temp_red |= cv2.inRange(hsv, np.array(lower), np.array(upper))
            for lower, upper in orange_range:
                temp_orange |= cv2.inRange(hsv, np.array(lower), np.array(upper))

            r_clean = cv2.morphologyEx(temp_red, cv2.MORPH_OPEN, kernel)
            r_clean = cv2.morphologyEx(r_clean, cv2.MORPH_CLOSE, kernel)
            o_clean = cv2.morphologyEx(temp_orange, cv2.MORPH_OPEN, kernel)
            o_clean = cv2.morphologyEx(o_clean, cv2.MORPH_CLOSE, kernel)

            combined = cv2.bitwise_or(r_clean, o_clean)
            contours_r, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_o, _ = cv2.findContours(o_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            total_area = 0
            for o in contours_o:
                if cv2.contourArea(o) < 100:
                    continue
                M = cv2.moments(o)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                for r in contours_r:
                    if cv2.pointPolygonTest(r, (cx, cy), False) >= 0:
                        total_area += cv2.contourArea(o)
                        break

            if total_area > max_orange_area:
                max_orange_area = total_area
                best_combo = (r_clean, o_clean)

    if not best_combo:
        print(f"⚠️ No match found in: {filename}")
        continue

    red_mask, orange_mask = best_combo
    combined_mask = cv2.bitwise_or(red_mask, orange_mask)

    contours_red, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_orange, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate_circles = []
    for o in contours_orange:
        if cv2.contourArea(o) < 100:
            continue
        M = cv2.moments(o)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        for r in contours_red:
            if cv2.pointPolygonTest(r, (cx, cy), False) >= 0:
                (x, y), radius = cv2.minEnclosingCircle(o)
                candidate_circles.append({
                    "center": (int(x), int(y)),
                    "radius": int(radius),
                    "area": cv2.contourArea(o)
                })
                break

    filtered_circles = []
    for i, c1 in enumerate(candidate_circles):
        inside_any = False
        for j, c2 in enumerate(candidate_circles):
            if i != j and is_inside(c1, c2):
                inside_any = True
                break
        if not inside_any:
            filtered_circles.append(c1)

    # === Create yellow, blue, and dark red masks ===
    yellow_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    blue_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    dark_red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in yellow_range:
        yellow_mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    for lower, upper in blue_range:
        blue_mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    for lower, upper in color_ranges["dark_red"]:
        dark_red_mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

    # === Final color-layer validation ===
    valid_circles = []
    for circle in filtered_circles:
        cx, cy = circle["center"]
        r = circle["radius"]

        circle_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.circle(circle_mask, (cx, cy), r, 255, -1)

        yellow_inside = cv2.countNonZero(cv2.bitwise_and(yellow_mask, yellow_mask, mask=circle_mask))
        blue_inside = cv2.countNonZero(cv2.bitwise_and(blue_mask, blue_mask, mask=circle_mask))

        outer_ring_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.circle(outer_ring_mask, (cx, cy), r + 8, 255, -1)
        cv2.circle(outer_ring_mask, (cx, cy), r + 4, 0, -1)
        dark_red_ring = cv2.countNonZero(cv2.bitwise_and(dark_red_mask, dark_red_mask, mask=outer_ring_mask))

        if (yellow_inside > 10 or blue_inside > 10) and dark_red_ring > 10:
            valid_circles.append(circle)

    # === Draw only valid layered-circle defects ===
    if valid_circles:
        overlay = image.copy()
        for circle in valid_circles:
            center = circle["center"]
            radius = circle["radius"]
            cv2.circle(overlay, center, radius + 6, (0, 0, 0), 4)
            cv2.circle(overlay, center, radius, (0, 255, 255), 4)
            text = "Defect (Layered)"
            text_position = (center[0] - radius, center[1] - radius - 10)
            cv2.putText(overlay, text, text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        final_img = cv2.addWeighted(overlay, 0.75, image, 0.25, 0)
        output_path = os.path.join(highlighted_output_folder, f"{filename}_highlighted.png")
        cv2.imwrite(output_path, final_img)
        print(f"✔ Saved layered defect: {filename}")
    else:
        print(f"⚠️ No valid layered defect in: {filename}")

print("✅ All images processed.")
