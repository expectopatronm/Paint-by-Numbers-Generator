
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

- **Advanced mixing models** (`--mix-model`) to better approximate real paint behavior.
- **Progress frames** that guide the painting process step-by-step (Darks → Midtones → Neutrals → Highlights → Completed).

<p align="center">
  <img src="docs/Screenshot%202025-09-18%20003813.png" width="85%" />
</p> 

- A clean **A4 landscape PDF** output where each page contains both the working image for that stage and the matching color key.
- An **edge sketch + grid page** to use for initial drawing transfer.

<p align="center">
  <img src="docs/Screenshot%202025-09-18%20094714.png" width="45%" />
</p>

---

## 1. Color Clustering (Image Simplification)

1. The input image is optionally **downsampled** (to speed up computation).
2. K-Means clustering groups all pixels into a fixed number of clusters (e.g., 15 or 20 colors).
3. Each cluster centroid becomes a “paint number” — a representative color in the paint-by-numbers map.
4. The clustered image is recolored with **the actual integer-mix approximation** (depends on `--max-parts`, `--components`, and `--mix-model`).
5. The result is **upsampled back to the original resolution** with nearest-neighbor interpolation.

This ensures that the paint-by-numbers image *really reflects your chosen mixing constraints*.

---

## 2. Mapping to Real Paints (Recipes)

A cluster’s average color is rarely an exact match to a tube of paint.  
Instead, the script solves for **integer mixing ratios** of your available paints:

- Paint palette (default):  
  `Titanium White, Lemon Yellow, Vermillion Red, Carmine, Ultramarine, Pthalo Green, Yellow Ochre, Lamp Black`

- For each cluster centroid, we test **all integer partitions** of `--max-parts` distributed over up to `--components` base paints.

- Mixing can be evaluated under different models (`--mix-model`):  
  - `linear`: mix in linear sRGB space (mathematically simple, not very physical).  
  - `lab`: mix in CIE Lab space (perceptually uniform).  
  - `subtractive`: multiplicative model, mimics simple pigment absorption.  
  - `km`: **Kubelka–Munk–like** model (default), averages absorbance, better for real paints.

- The recipe with the lowest **ΔE (Lab color difference)** to the cluster centroid is chosen.

- Recipes are expressed in **parts**, capped by `--max-parts`.  
  Example:  
  - With `--max-parts 11`: `8 parts White + 2 parts Yellow Ochre + 1 part Red`  
  - With `--max-parts 3`: `2 parts White + 1 part Ochre`  
  (shorter recipes mean fewer scoops from paint tubes).

- If a cluster is pure and matches a tube closely, the recipe is collapsed to `1 part <color>`.

- Each recipe is paired with **component swatches** of the contributing paints (optional with `--hide-components`).

- The key also displays:  
  - **L*** (perceptual lightness).  
  - **ΔE** (how close the mix is to the target).  
  - A suggested **value tweak** (`+ tiny White` / `+ tiny Black`) if multiple clusters share the same recipe.

---

## 3. Progress Frames (Painting Order)

Painting is not just about colors — the **sequence matters**.  
The script generates logical **frames** that guide the painting process:

1. **Shadows / Darks**  
   - Lowest ~25% luminance values.  
   - Block in structure and depth.

2. **Mid-tone Masses**  
   - Main body colors.  
   - Establish form and volume.

3. **Neutrals / Background**  
   - Low-saturation, muted tones.  
   - Often backdrop or transitions.

4. **Highlights**  
   - Top ~20% luminance values.  
   - Add sparkle and dimensionality.

5. **Completed**  
   - All clusters together, for the finished map.

### Value-sliced (5 levels of value)
1. Deep Shadows (lowest ~10%)  
2. Core Shadows (to ~25%)  
3. Midtones (to ~70%)  
4. Half-Lights (to ~85%)  
5. Highlights (top ~15%)  

### Combined (9-step sequence, default)
Interleaves the two systems into one logical painting order, avoiding duplicates:  

1. Deep Shadows  
2. Core Shadows  
3. Shadows / Dark Blocks (remaining)  
4. Value Midtones  
5. Mid-tone Masses (remaining)  
6. Neutrals / Background  
7. Half-Lights  
8. Highlights  
9. Highlight Accents (remaining highs)  
10. Completed (overview of all clusters)  

⚠️ **Note:** If a step has no remaining clusters (because they were already painted in earlier steps), that step is **skipped automatically**. This is why some numbered steps may not appear in the PDF.  

Future option: `--include-empty-steps` could force those pages to appear with a “No new clusters” note.  

This follows classical oil painting logic:
- Work **Dark → Light** (to keep lights clean).  
- Work **Thin → Thick** (so highlights sit on top).  
- Work **Background → Foreground** (for overlaps).

---

## 4. PDF Output

The script outputs an **A4 landscape PDF** with:

- **Page 1 (Overview):**  
  - Left: Original (top) and Paint-by-Numbers map (bottom, recolored with mixes).  
  - Right: Full color key (all clusters, recipes, L*, ΔE, component chips).  

- **Page 2 (Edge Sketch):**  
  - A high-contrast sketch of the original image, with grid lines for proportional transfer.  

- **Pages 3+:**  
  - Left: Progress frame (darks, mids, etc.).  
  - Right: Frame-specific key showing only the colors used in that stage.

This makes the PDF both a **big-picture reference** and a **step-by-step workbook**.

---

## 5. Customization

- **Number of clusters:**  
  `--colors N` (default: 15).  
  More clusters = more detail.

- **Resize resolution:**  
  `--resize W H` (default: 120×120).  
  Affects speed and clustering accuracy.

- **Max parts per recipe:**  
  `--max-parts N` (default: 5).  
  Constrains recipe complexity.  
  Low values → simpler recipes.  
  High values → more accurate but longer.

- **Max components per recipe:**  
  `--components N` (default: 3).  
  Caps how many different paints appear in one recipe.

- **Mixing model:**  
  `--mix-model {km,subtractive,lab,linear}` (default: km).  
  Switch to experiment with different pigment behaviors.

- **Hide component chips:**  
  `--hide-components` to suppress the right-hand base paint swatches.

- **Legend wrapping:**  
  `--wrap N` (default: 55) to control text wrapping in the PDF.

- **Edge sketch:**  
  Control grid and edge detection via `--grid-step` and `--edge-percentile`.

---

## 6. Why This Approach Works

- **K-Means** simplifies the image into stable color regions.  
- **Integer mix search** guarantees recipes you can actually scoop out with a palette knife.  
- **Advanced mixing models** mimic physical paint behavior better than plain averages.  
- **ΔE optimization** ensures accuracy against the cluster target.  
- **Progress frames** provide a painter’s roadmap rather than just a coloring book.  
- **PDF layout** makes it practical to print and use at the easel.

---

## 7. Example Workflow

1. Run the script on your chosen image:  
   ```bash
   python paint_by_numbers_generic_v16_robust_masks_mix_pbn.py portrait.jpg \
     --colors 20 \
     --max-parts 3 \
     --components 3 \
     --mix-model km \
     --pdf portrait_guide.pdf
   ```

2. Print the generated PDF.  
3. Start with **Page 3 (Frame 1 – Shadows)**, mix the paints shown, and block them in.  
4. Continue frame by frame until the painting is complete.  
5. Use **Page 1 (Overview)** as your overall reference.  
6. If needed, use **Page 2 (Sketch)** to transfer outlines and proportions.

---

## 8. Future Extensions

- Import custom palettes (`--palette-json` with hex values).  
- Allow manual progress group editing.  
- Add watercolor/acrylic mixing presets.  

---

Happy painting 🎨

