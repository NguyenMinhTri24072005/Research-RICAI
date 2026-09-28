# Rice-surface detector review — 2026-09-28

## Change

The default detector now segments the local saturated rice region with Otsu,
fits its convex envelope, and checks diameter/center stability under nearby
thresholds. This allows the center to move away from the reflection-contaminated
seed. The crop is the surface envelope, not an expanded outer glass ellipse.
The old algorithm is available explicitly as `surface_method="radial"` for
comparison; it is not an automatic fallback after an uncertain adaptive fit.

CODE, DATASET_BUILDER and AI_SERVICES have byte-identical detector modules.
Their segmentation inputs now exclude pixels outside the surface. Full-frame
mode retains original dimensions and coordinates; cropped mode uses its crop
offset. Original images are unchanged. No new library/model is required.

## Real-image evidence

Open `surface_review/M0293_comparison.jpg`: left is radial v2.2, right is adaptive.
The other three samples also have comparison and crop images in that folder.
`review_surface.py` regenerates them and `surface_review/measurements.json`.
The extra threshold panels are exploratory diagnostics, not annotated truth.

| Image | Old inner diameter (px) | New inner diameter (px) | New px/mm |
|---|---:|---:|---:|
| M0261 | 1710 | 1636 | 71.739 |
| M0270 | 527 | 524 | 22.995 |
| M0293 | 945 | 935 | 40.993 |
| M0301 | 537 | 534 | 23.425 |

All rows use 22.8 mm solely to match the notebook comparison parameters.
This does not verify that 22.8 mm is the correct physical diameter for every
sample. The visible improvement on M0293 is the shifted center and exclusion
of the pale reflected grains below/around the rice, not simply a smaller scale.
M0261 has a less complete grain envelope near the rim; its scale remains a
proxy rather than an independently measured wall boundary.

M0293 normalized scale changed by 0.011% after a 90-degree rotation, 0.155%
after resizing to 75%, and 0.035% after reducing pixel brightness to 80%.
These are consistency tests, not absolute accuracy measurements.

## Verification

- Detector synthetic known-geometry checks, notebook SAHI input/coordinate
  checks, API runner and artifact tests: 12 passed on the final retry.
- Dataset static contract/parity: 8/8; mocked orchestration: 14/14.
- Python syntax, changed notebook cell syntax and three-way detector byte parity: pass.
- Four legacy `test_main_pipeline_crop.py` checks still fail: they require
  `prepare_square_container_crop`, already absent in the notebook before this
  change, and a crop-by-default contract inconsistent with the current notebook.
  Those tests were not edited to hide the failures.
- A combined test attempt encountered Windows 0x8007000e/OpenCV resource errors.
  Retry with OpenCV/BLAS limited to one thread exited 0 with 12 passed, although
  the environment still emitted a Windows diagnostic during NumPy/SciPy import.
- Full real-image YOLO/CNN/regression E2E: NOT VERIFIED. Notebook saved outputs
  were retained and are not evidence of running this new version.

## Limits and use

This estimates the filled cylindrical cross-section from the grain envelope.
Sparse/non-flat piles, pale rice, colored glass or strongly saturated reflections
can violate the assumptions. Threshold stability is not a probability of being
correct. Diagnostics explicitly contain `metric_accuracy_validated: false`.
Manual reference boundaries and physical measurements are still needed to
measure absolute error; no numerical real-image accuracy is claimed here.

Restart the Colab runtime and rerun setup/configuration, detector and subsequent
cells. Both full-frame and cropped SAHI modes use the isolated surface now.
Verify physical sample dimensions. Revalidate/re-extract training features before
treating results from an existing regression bundle as comparable: the pixel
scale and selected visible grains have changed, even though the 31-field schema
and feature ordering are unchanged.
