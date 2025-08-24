

# Computer Vision Methods for Fractographic Defect Detection and analysis

Python pipeline for **automated, reproducible** analysis of fatigue fracture surfaces in additively manufactured (AM) **titanium (Ti-6Al-4V)** and **aluminum** alloys using computer vision.
It converts **SEM images → specimen masks → gradient heatmaps → phase-wise crack contours → calibrated areas** (pixels & µm²) **+ visual overlays and CSV reports**.&#x20;

## ✨ Key Features

* **Inner-shape extraction** of specimens (robust masks, noise removal).
* **Gradient heatmaps** (Sobel + sliding-window aggregation + histogram equalization).
* **Primary crack-zone detection** (HSV segmentation + centroid-in-red test; convex-hull fallback).
* **Multi-phase contour extraction** for 5 color bands: **dark-red → red → yellow → cyan → blue**.
* **Quantitative analysis** with pixel-to-micron calibration
  (defaults for **4096 px ↔ 5500 µm → 1.342773438 µm/px, 1.803040504 µm²/px**).
* **CSV outputs** per specimen & per category, plus **overlay images** for visual validation.

---

## 📁 Repository Structure (workflow order)

### 1) `Pipeline.ipynb`

**Main entry point**. Demonstrates the **end-to-end workflow** step by step:

1. Preprocess SEM images
2. Extract specimen inner shape
3. Generate heatmaps
4. Detect crack zones

> Use this first if you want to **see the entire pipeline interactively**.

---

### 2) `ExtractionPhase/`

Scripts for **extracting the crack area** from SEM images.

* **`ExtractCrackArea.py`**
  Primary crack-zone detection using **HSV segmentation + centroid-inclusion logic**:

  * Red mask = high-stress envelope
  * Orange mask = local critical zones
  * Keeps the **orange contour whose centroid lies inside the red region**
    **Output:** main crack-zone mask/contour

* **`ExtractCrackBasedCH.py`**
  **Alternative** method using **Convex Hull** when the crack is fragmented/irregular; ensures one robust closed boundary around **red+orange**.
  **Output:** convex polygon enclosing the fracture zone

> Together these scripts guarantee **reliable crack-area extraction** even for challenging specimens.

---

### 3) `CorlorsContours/` 

Implements **multi-phase contour extraction** inside the crack zone and the **geometric modeling** per phase:

* Dark-red (initiation)
* Red (early growth)
* Yellow (cumulative envelope)
* Cyan (advanced front)
* Blue (final failure)

---

### 4) `Area-Colors.py`

**Quantitative analysis** of the extracted crack phases.

* Pixel-to-micron calibration (default: **1.342773438 µm/px**, **1.803040504 µm²/px**)
* Compute **area per phase** (dark-red, red, yellow, cyan, blue)
* Export **CSV files** per specimen and per alloy category (SLM, EBM, Al)
* Generate **overlay images** for visual validation
  **Produces:** numerical results (CSV) + graphical validation

---

### 5) `data/`
```
data/
├─ SLM_Ti64/
├─ EBM_Ti64/
└─ Aluminum/
```
Raw files are typically **`.tif`**; the pipeline converts them to **`.png`** during preprocessing.

---


### 7) `requirements.txt`  

Dependencies: `opencv-python`, `numpy`, `scipy`, `pandas`, `matplotlib`, `Pillow`.

---
---


### 8 ) `Documentation & Reports`  

Fractographic Defect Detection Research.pdf – full research paper: Computer Vision Methods for Fractographic Defect Detection in Titanium and Aluminum Alloys.

final Presentation.pptx – final presentation slides summarizing the research and results.

---
## 🚀 Quick Start

```bash
# 1) Create & activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt
# or
pip install opencv-python numpy scipy pandas matplotlib Pillow

# 3) Put SEM images in data/ (see structure above)

# 4) Run the pipeline
#open Pipeline.ipynb and run cells sequentially
```

---

## ⚙️ Configuration Tips

* **Calibration:** If your SEM is not **4096↔5500 µm**, update
  `PIXEL_SIZE_MICRONS` and `MICRON_AREA_FACTOR` in the analysis step.
* **HSV thresholds:** Tune for your colormap/camera (red/orange bounds).
* **Morphology & smoothing:** Adjust kernel/window sizes for noisy datasets.
* **Folder names:** Consider renaming `CorlorsContours/` → `ColorsContours/` for clarity.

---


## 📚 Methodology Reference

This codebase implements the pipeline described in:
**“Computer Vision Methods for Fractographic Defect Detection in Titanium and Aluminum Alloys.”** (Ariel University);

---
