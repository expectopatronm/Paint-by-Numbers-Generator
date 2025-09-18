# Paint-by-Numbers Generator

<p align="center">
  <img src="docs/Screenshot%202025-09-18%20003029.png" width="45%" />
  <img src="docs/Screenshot%202025-09-18%20003107.png" width="45%" />
</p>

This project converts any input image into a structured **paint-by-numbers guide**, complete with:

- **Simplified color clusters** (via K-Means).
- **Mixing recipes** using a fixed set of real paint colors (Titanium White, Lemon Yellow, Vermillion Red, Carmine, Ultramarine, Pthalo Green, Yellow Ochre, Lamp Black).

<p align="center">
  <img src="docs/Screenshot%202025-09-18%20003932.png" width="75%" />
</p>

- **Component swatches** to visualize which paints contribute to each cluster.
- **Progress frames** that guide the painting process step-by-step (Darks → Midtones → Neutrals → Highlights → Completed).
- An **Initial Sketch with Grid** (from the original image edges) to help transfer proportions onto canvas.

<p align="center">
  <img src="docs/Screenshot%202025-09-18%20094714.png" width="50%" />
</p>
 
- A clean **A4 landscape PDF** output where each page contains both the working image for that stage and the matching color key.

<p align="center">
  <img src="docs/Screenshot%202025-09-18%20003813.png" width="90%" />
</p>

---

## 1. Color Clustering (Image Simplification)

1. The input image is **downsampled** (to speed up computation).
2. K-Means clustering groups all pixels into a fixed number of clusters (e.g., 15 colors).
3. Each cluster centroid becomes a “paint number” — a representative color in the paint-by-numbers map.
4. The clustered image is then **upsampled back to the original resolution** with nearest-neighbor interpolation, ensuring each pixel is assigned to a single cluster region.

---

## 2. Mapping to Real Paints

A cluster’s average color is rarely an exact match to a tube of paint.  
Instead, we solve for **mixing ratios** of your available paints:

- Paint palette:  
  `Titanium White, Lemon Yellow, Vermillion Red, Carmine, Ultramarine, Pthalo Green, Yellow Ochre, Lamp Black`

- For each cluster centroid (RGB), we run **non-negative least squares (NNLS)** to find the best combination of base paints (weights ≥ 0, sum = 1).

- We convert to **linear sRGB space** for accuracy, then back to display RGB.

- The weights are normalized and scaled to “parts”, so the output is human-friendly:  
  e.g. `2 parts Yellow Ochre + 1 part Vermillion Red`

- To avoid clutter, only the top 3–4 contributors per color are shown.

- Each recipe is paired with **small swatches of the contributing paints**, so you see exactly what goes in.

- Each legend entry also shows:  
  - **CIE L\*** (lightness / value) of the cluster  
  - If multiple clusters reduce to the same rounded recipe, **value tweaks** are suggested:  
    - `+ tiny White` → mix slightly lighter  
    - `+ tiny Black` → mix slightly darker  
    - `none (base)` → use the baseline recipe as-is  

This ensures visually similar clusters remain distinguishable when painting.

---

## 3. Progress Frames (Painting Order)

Painting is not just about colors — the **sequence matters**.  
The script generates logical **frames** that guide the painting process:

1. **Shadows / Darks**  
   - Lowest 25% luminance values (deep shadows, black mixes).  
   - Block in the structure and tonal foundation.

2. **Mid-tone Masses**  
   - Colors that are neither dark, neutral, nor highlights.  
   - Establish the main body of the subject.

3. **Neutrals / Background**  
   - Low-saturation colors (greys, ochres, muted tones).  
   - Often background or transitional areas.

4. **Highlights**  
   - Top 20% luminance values (lightest tones).  
   - Add volume and luminosity.

5. **Completed**  
   - All clusters together, for the finished guide.

⚠️ If a group has no clusters (e.g., no neutrals), that frame is skipped.

This sequencing reflects classical oil painting logic:
- Work **Dark → Light** (so lights remain clean).  
- Work **Thin → Thick** (so highlights sit on top).  
- Work **Background → Foreground** (so details overlap correctly).

---

## 4. PDF Output

- The script outputs an **A4 landscape PDF**.  
- **Page 1 (Overview):**  
  - Left column: Original (top) and Paint-by-Numbers map (bottom).  
  - Right column: Full color key for all clusters.  
- **Page 2 (Initial Sketch with Grid):**  
  - Edge-based sketch of the original image, with a light grid overlay (default every 80px).  
  - Helps transfer proportions accurately onto canvas.  
- **Pages 3+:**  
  - Left: One progress frame (darks, mids, neutrals, etc.).  
  - Right: A frame-specific key showing **only the clusters used in that frame**, with their recipes, swatches, L\*, and value tweaks.  

### Legend Improvements
- **Wrapped text** prevents long recipes from overlapping the figures.  
- **L\*** and **value tweak** hints are displayed for each cluster, ensuring clarity when clusters share the same mix.  
- Component swatches are shown in a consistent palette order.  

This makes the PDF usable both as an **overall map** and as a **step-by-step workbook**.

---

## 5. Customization

- **Number of clusters** (`--colors`) controls detail level.  
- **Resize resolution** (`--resize`) balances speed vs. accuracy for clustering.  
- **Max components per recipe** (`--components`) limits how many base paints appear in each mix.  
- **Grid step** (`--grid-step`) controls spacing of the sketch overlay grid (default: 80px).  
- **Edge percentile** (`--edge-percentile`) adjusts sensitivity of the sketch edges (default: 75).  
- **CSV export** (`--csv`) generates a table of numeric ratios for all clusters.  
- **Manual grouping** is possible by editing `progress_order` in the script if you want absolute control.

---

## 6. Why This Approach Works

- **K-Means** simplifies the image while preserving its overall tonal structure.  
- **NNLS mixing** ensures recipes are realistic (no negative paint).  
- **Linear RGB math** makes color approximation perceptually accurate.  
- **Progress frames** mimic a painter’s natural workflow, turning a flat “coloring book” into a logical painting roadmap.  
- **Value tweaks and L\*** values keep near-identical mixes visually distinct.  
- **Wrapped PDF legends** make the guide easy to read at the easel.  
- **Initial Sketch with Grid** helps you transfer proportions and placement like a traditional painter.

---

## 7. Example Workflow

1. Run the script on your chosen image.  
2. Print the generated PDF.  
3. Start with **Page 2 (Initial Sketch)** to lightly sketch your canvas.  
4. Move to **Page 3 (Frame 1 – Shadows)**, mix the paints shown, and block them in.  
5. Continue page by page until the painting is complete.  
6. Use **Page 1 (Overview)** as a reference throughout.

---

Happy painting 🎨