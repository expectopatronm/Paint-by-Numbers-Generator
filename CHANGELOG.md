# Changelog

## 2026-06-21

### Enabled Potts/MRF Label Smoothing
Added an optional Potts/MRF smoothing pass after clustering and before final region cleanup. It is enabled by default to reduce tiny speckled regions while still keeping the paint map based on the clustered image colors.

### Added Cluster Preview Script
Added `scripts/cluster_preview.py` to compare anti-speckling approaches without running the full PDF pipeline. It can render baseline K-Means, mean-shift prefiltering, spatial K-Means, Potts/MRF smoothing, and connected-component cleanup previews.

### Modularized the Generator
Split the former monolithic generator into the `pbn/` package so image operations, mixing, PDF rendering, SVG tracing, integrations, and orchestration live in separate files. The original `paint_by_numbers_generic_v8_pdf.py` now remains as a compatibility entry point.

### Simplified Paint Mixing
Removed the older KM-style mixing paths and kept only the learned Mixbox-based model. This makes recipe generation match the intended `model=learned` behavior consistently.

### External Sketch Workflow
Added support for a user-provided sketch file that is used for PDF outline pages, frame underlays, per-color underlays, and centerline SVG tracing. When provided, the generator does not need to create its own sketch for those outputs.

### Removed Closed-Region Sketch Generation
Removed the code path that tried to generate closed bounded regions from labels. It was unreliable for this artwork workflow, so the project now relies on image/external sketch outlines instead.
