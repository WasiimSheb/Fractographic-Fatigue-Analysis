import cv2
import numpy as np
import os

# === Base Paths ===
base_path = r"C:\Users\shifa\final project"

original_folder = os.path.join(base_path, r"Final_Project_Fractographic_Failure_Analysis_with_CV_in_AM-main\P3")
inner_folder = os.path.join(base_path, r"Final_Project_Fractographic_Failure_Analysis_with_CV_in_AM-main\SLM-P3-segmented_inner_Shape")
heatmap_folder = os.path.join(base_path, r"Final_Project_Fractographic_Failure_Analysis_with_CV_in_AM-main\SLM-P3-CrackZone-NEW")

contour_folders = {
    "Dark Red Contour": os.path.join(base_path, r"Enternal_Contours\Dark Red\DarkRed_Contours-SLM-P3"),
    "Red Contour": os.path.join(base_path, r"Enternal_Contours\Red\Red_Contours-SLM-P3"),
    "Yellow Contour": os.path.join(base_path, r"Enternal_Contours\Yellow\yellow_Contours-SLM-P3"),
    "Cyan Contour": os.path.join(base_path, r"Enternal_Contours\cyan\cyan_Crack-SLM-P3"),
    "Blue Contour": os.path.join(base_path, r"Enternal_Contours\Blue\blue_Contours-SLM-P3")
}

output_folder = os.path.join(base_path, "internal contours results_SLM-P3")
os.makedirs(output_folder, exist_ok=True)

# === Text Parameters ===
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.0
thickness = 2
text_color = (0, 0, 0)  # Black

# === Process ===
for filename in os.listdir(original_folder):
    if not filename.lower().endswith('.png'):
        continue

    name_base = filename[:-4]  # Strip '.png'

    # Special check for dark red (dash vs underscore fallback)
    dark_red_path = os.path.join(contour_folders["Dark Red Contour"], f"{name_base}_heatmap_highlighted_darkred_overlay.png")
    if not os.path.exists(dark_red_path):
        alt_name_base = name_base.replace('-', '_')
        dark_red_path = os.path.join(contour_folders["Dark Red Contour"], f"{alt_name_base}_heatmap_highlighted_overlay_clipped.png")

    # Build file mappings in desired order
    file_mappings = {
        "Original Image": os.path.join(original_folder, f"{name_base}.png"),
        "Segmented Inner Shape": os.path.join(inner_folder, f"{name_base}_segmented_inner.png"),
        "Heatmap + Crack Zone": os.path.join(heatmap_folder, f"{name_base}_heatmap_highlighted.png"),
        "Dark Red Contour": dark_red_path,
        "Red Contour": os.path.join(contour_folders["Red Contour"], f"{name_base}_heatmap_highlighted_red_overlay.png"),
        "Yellow Contour": os.path.join(contour_folders["Yellow Contour"], f"{name_base}_heatmap_highlighted_envelope_overlay.png"),
        "Cyan Contour": os.path.join(contour_folders["Cyan Contour"], f"{name_base}_heatmap_highlighted_crackzone_contour_overlay.png"),
        "Blue Contour": os.path.join(contour_folders["Blue Contour"], f"{name_base}_heatmap_highlighted_ellipse_overlay.png"),
    }

    images = []
    titles = []

    for label, path in file_mappings.items():
        if os.path.exists(path):
            img = cv2.imread(path)
            img = cv2.resize(img, (512, 512))
            images.append(img)
            titles.append(label)
        else:
            print(f"⚠ {label} NOT found: {path}")

    # Combine images if any exist
    if images:
        combined_rows = []
        row = []

        for i, (img, title) in enumerate(zip(images, titles)):
            # Create title bar
            title_bar = np.ones((40, img.shape[1], 3), dtype=np.uint8) * 255
            cv2.putText(title_bar, title, (20, 30), font, font_scale, text_color, thickness, cv2.LINE_AA)
            full_img = np.vstack((title_bar, img))
            row.append(full_img)

            # Group 4 per row (2x4 layout)
            if (i + 1) % 4 == 0 or (i + 1) == len(images):
                combined_row = np.hstack(row)
                combined_rows.append(combined_row)
                row = []

        # === Pad rows to equal width ===
        max_width = max(row_img.shape[1] for row_img in combined_rows)
        padded_rows = []
        for row_img in combined_rows:
            height, width, channels = row_img.shape
            if width < max_width:
                pad_width = max_width - width
                padding = np.ones((height, pad_width, 3), dtype=np.uint8) * 255  # White padding
                padded_row = np.hstack((row_img, padding))
            else:
                padded_row = row_img
            padded_rows.append(padded_row)

        combined_all = np.vstack(padded_rows)

        # Add sample title bar
        total_width = combined_all.shape[1]
        sample_title_bar = np.ones((70, total_width, 3), dtype=np.uint8) * 255
        cv2.putText(sample_title_bar, f"Sample: {name_base}", (30, 50), font, 1.5, text_color, 3, cv2.LINE_AA)

        final_img = np.vstack((sample_title_bar, combined_all))

        # Save final image
        save_name = f"{name_base}_combined.png"
        save_path = os.path.join(output_folder, save_name)
        cv2.imwrite(save_path, final_img)
        print(f"✅ Saved combined image for {name_base}")

    else:
        print(f"⚠ No images found for {name_base}, skipped.")

print("🎉 Done generating all combined summary images!")
