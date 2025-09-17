
# Paint-by-Numbers Generator

This project converts any input image into a structured **paint-by-numbers guide**, complete with:

- **Simplified color clusters** (via K-Means).
- **Mixing recipes** using a fixed set of real paint colors (Titanium White, Lemon Yellow, Vermillion Red, Carmine, Ultramarine, Pthalo Green, Yellow Ochre, Lamp Black).
- **Component swatches** to visualize which paints contribute to each cluster.
- **Progress frames** that guide the painting process step-by-step (Darks → Midtones → Neutrals → Highlights → Completed).
- A clean **A4 landscape PDF** output where each page contains both the working image for that stage and the matching color key.

This README emphasizes **how it works and why**, rather than just how to run it.

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
- **Pages 2+:**  
  - Left: One progress frame (darks, mids, neutrals, etc.).  
  - Right: A frame-specific key showing **only the clusters used in that frame**, with their recipes and swatches.

This makes the PDF usable both as an **overall map** and as a **step-by-step workbook**.

---

## 5. Customization

- **Number of clusters** (`--colors`) controls detail level.  
- **Resize resolution** (`--resize`) balances speed vs. accuracy for clustering.  
- **Max components per recipe** (`--components`) limits how many base paints appear in each mix.  
- **CSV export** (`--csv`) generates a table of numeric ratios for all clusters.  
- **Manual grouping** is possible by editing `progress_order` in the script if you want absolute control.

---

## 6. Why This Approach Works

- **K-Means** simplifies the image while preserving its overall tonal structure.  
- **NNLS mixing** ensures recipes are realistic (no negative paint).  
- **Linear RGB math** makes color approximation perceptually accurate.  
- **Progress frames** mimic a painter’s natural workflow, turning a flat “coloring book” into a logical painting roadmap.  
- **PDF layout** makes the guide easy to follow at the easel.

---

## 7. Example Workflow

1. Run the script on your chosen image.  
2. Print the generated PDF.  
3. Start with **Page 2 (Frame 1 – Shadows)**, mix the paints shown, and block them in.  
4. Continue page by page until the painting is complete.  
5. Use **Page 1 (Overview)** as a reference throughout.

---

Happy painting 🎨
