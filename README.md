# Urban Land Cover Analysis Pipeline

Digital Image Processing — Project | Group H — Nisarg Shah, Ansar Rahman

**Live demo:** https://nisarg-007.github.io/dip-urban-land-cover-pipeline/

A classical (non-machine-learning) image-processing pipeline that improves the visual and
structural interpretability of an urban satellite/aerial scene: correcting illumination, boosting
local contrast, denoising and sharpening structure, separating coarse land cover from fine detail
in the frequency domain, and extracting edges/corners that would support downstream segmentation.

The pipeline runs end-to-end on a **real, live 1 m/pixel NAIP aerial crop** of the I-45 (Pierce
Elevated) / US-59 interchange in downtown Houston, TX — not a toy or synthetic image — and every
tunable parameter is derived from the image's own statistics and logged numerically, rather than
hand-picked to look good.

See [`report/REPORT.md`](report/REPORT.md) for the full write-up: methodology, the exact formula
used for every parameter, results, and a validation section documenting two bugs that were caught
and fixed during development (an inverted gamma formula, and a degenerate FFT cutoff) plus an
independent review that caught and corrected a wrong quantitative claim before this was published.
The original project proposal is [`GroupH_Shah_Rahman_proposal.pdf`](GroupH_Shah_Rahman_proposal.pdf).

## What's in this repo

```
.
├── GroupH_Shah_Rahman_proposal.pdf   original project proposal
├── data/
│   └── naip_raw.png                  the exact 1024x1024, 1 m/px NAIP crop used (source below)
├── scripts/
│   └── dip_pipeline.py               single self-contained script; runs the entire pipeline
├── outputs/
│   ├── 00_original_color.png / 01_original_gray.png
│   ├── 02a_gamma_corrected.png / 02b_contrast_stretched.png       (point processing)
│   ├── 03a_clahe.png / 03b_global_hist_eq_for_comparison.png      (histogram processing)
│   ├── 04a_synthetic_noisy_input.png / 04b_median_k5.png /
│   │   04c_median_denoised_chosen.png / 04d_laplacian_sharpened.png   (spatial filtering)
│   ├── 05a_fft_magnitude_spectrum.png / 05b_radial_power_spectrum.png /
│   │   05c_fft_lowpass.png / 05d_fft_highpass.png                 (frequency-domain filtering)
│   ├── 06a_canny_edges.png / 06b_gradient_histogram.png /
│   │   06c_harris_corners.png                                      (edge & corner detection)
│   ├── 07_yolo_detections.png                                      (optional DL extension)
│   ├── 08_summary_grid.png                                         (feasibility-check figure)
│   └── metrics.json                  every derived parameter + the numbers that justify it
└── report/
    └── REPORT.md                     full methodology, results, and validation write-up
```

## Pipeline stages

| Category | Techniques | Script section |
|---|---|---|
| Point processing | Gamma correction (γ solved from the scene's own shadow-quartile mean), linear contrast stretch (2nd/98th percentile, not a fixed 0–255 assumption) | `dip_pipeline.py` §1 |
| Histogram processing | CLAHE, benchmarked quantitatively against global histogram equalization | §2 |
| Spatial filtering | Median filtering (kernel size chosen by MSE + edge-correlation search against a clean reference), Laplacian/unsharp-mask sharpening (weight chosen by a clipped-pixel budget) | §3 |
| Frequency-domain filtering | FFT low-pass / high-pass, cutoff radius chosen from the image's own radial power spectrum | §4 |
| Edge & corner detection | Canny (thresholds from the gradient-magnitude histogram via the median heuristic), Harris corners | §5 |
| Optional extension | Pretrained YOLOv8n run on the same crop, to characterize its failure mode at 1 m/pixel scale | §6 |

## Data source

`data/naip_raw.png` is a 1024×1024 px, 1 m/pixel true-color crop pulled from the **USGS NAIP
Imagery** service (National Agriculture Imagery Program, via The National Map ImageServer),
public-domain USDA/USGS aerial imagery, centered on the I-45/US-59 interchange next to the George
R. Brown Convention Center and Discovery Green in downtown Houston, TX:

```
https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage
  ?bbox=-95.377495,29.748295,-95.366905,29.757515&bboxSR=4326&imageSR=4326
  &size=1024,1024&format=png&f=image
```

## Running it, step by step

1. **Clone the repo.**
   ```bash
   git clone https://github.com/nisarg-007/dip-urban-land-cover-pipeline.git
   cd dip-urban-land-cover-pipeline
   ```

2. **Install dependencies.**
   ```bash
   pip install opencv-python-headless numpy scipy matplotlib ultralytics
   ```
   `ultralytics` (the optional YOLO extension) is the only heavy dependency — if you skip it,
   that one stage logs the import error into `metrics.json` and every other stage still runs.

3. **Run the pipeline.**
   ```bash
   python scripts/dip_pipeline.py
   ```
   The script reads the already-included `data/naip_raw.png` — no download step, no manual
   parameters to set. Internally it runs strictly in this order, each stage feeding the next:
   1. load the crop and convert to grayscale
   2. **point processing** — solve gamma from the shadow-quartile mean, then percentile-based
      linear contrast stretch
   3. **histogram processing** — CLAHE on the stretched image, benchmarked against global
      histogram equalization
   4. **spatial filtering** — inject synthetic noise and search median kernel sizes, then
      Laplacian/unsharp sharpen the clean CLAHE output
   5. **frequency-domain filtering** — FFT the original grayscale image, pick a cutoff radius
      from its radial power spectrum, produce the low-pass and high-pass results
   6. **edge & corner detection** — Canny on the sharpened image (thresholds from its gradient
      histogram), Harris corners on the same image
   7. **optional extension** — pretrained YOLOv8n on the original color crop
   8. write every intermediate image to `outputs/`, build the `08_summary_grid.png` comparison
      figure, and write `outputs/metrics.json` with every derived parameter

4. **Inspect the results.** Open `outputs/08_summary_grid.png` for the quick before/after view,
   `outputs/metrics.json` for every number, or `report/REPORT.md` for the full write-up with the
   formula and justification behind each parameter.

This was independently re-run on a second machine, from a fresh clone, to confirm the required
five categories (point processing, histogram processing, spatial filtering, frequency-domain
filtering, edge & corner detection) reproduce identical parameter values with no manual tweaking.

## Key results at a glance

- **Gamma correction**: γ = 1.245, brings the scene's darkest quartile (shadowed building faces,
  tree canopy) from a mean of 89.5/255 up to 109.1/255 (target: 110/255).
- **CLAHE vs. global histogram equalization**: CLAHE gains local contrast about evenly across dark
  and bright regions (dark/bright gain-balance ratio 0.87); global HE skews hard toward the
  already-bright band (ratio 0.66, and actually *loses* local contrast in shadow) — a genuine,
  numerically-derived case for local over global equalization on this scene.
- **Median filter**: k=3 minimizes MSE against a clean reference after injecting 2% synthetic
  salt-and-pepper noise, while best preserving edge structure.
- **Laplacian sharpening**: weight 0.05 is the largest tested that keeps hard-clipped pixels at or
  under a 5% budget, avoiding visible overshoot/ringing on the high-dynamic-range CLAHE output.
- **FFT low/high-pass**: cutoff radius of 125 cycles (on the 1024 px frame) captures 90% of the
  image's own AC power spectrum.
- **Canny**: thresholds of 65.7 / 132.4, set from the gradient-magnitude histogram, not defaults.
- **Harris**: ~62k corner-flagged pixels, concentrated on building corners, parking-lot striping,
  and interchange ramp/road junctions.
- **YOLOv8 (optional)**: 0–1 low-confidence (≤0.29) detections on this crop — the expected failure
  mode of a COCO-scale detector on ~4–5 px objects at 1 m/pixel ground sample distance, which is
  exactly why the classical Canny/Harris pipeline is the more reliable structural-feature source
  at this resolution.

Full numbers, formulas, and the validation/review process are in
[`report/REPORT.md`](report/REPORT.md).

## License

The code in this repo (`scripts/dip_pipeline.py` and everything else Group H wrote) is under the
[MIT License](LICENSE). The NAIP imagery in `data/naip_raw.png` is produced by the USDA Farm
Production and Conservation Business Center and is U.S. Government public domain (free to
use/redistribute); USDA/USGS attribution is included per their terms and is separate from the
MIT license on the code.
