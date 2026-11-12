# UMUD development reference data

Prepared 2026-11-12. The active comparison has **34 static images** and **84 directly annotated frames from one GM video**. These are development diagnostics, not an independent clinical test or a Kaggle leaderboard estimate.

## Sources and attribution

- Ritsche et al. *UMUD: a web application for easy access to musculoskeletal ultrasonography datasets*. BMC Medical Imaging 26, 139 (2026). [DOI](https://doi.org/10.1186/s12880-026-02170-0), [PubMed](https://pubmed.ncbi.nlm.nih.gov/41652557/).
- Official [UMUD OSF repository](https://osf.io/xbawc/) and [download link in the authors' application](https://github.com/PaulRitsche/UMUD/blob/ceb938082e8b8d9e70076c0adc7cea74e44e6a92/webapp/webapp.py). OSF project license: **CC BY 4.0**, verified through OSF node/license metadata. Preserve author attribution.
- [Static architecture v0.1.0](https://osf.io/hb5xj), [direct ZIP](https://osf.io/download/hb5xj/?direct=1). SHA256: `8d4c8093a737a64c39360864928f1ae65ddd2f70497dbe50ff82810173f2d48f`.
- [Dynamic architecture v0.2.0](https://osf.io/p8r2u), [direct ZIP](https://osf.io/download/p8r2u/?direct=1). SHA256: `02332b8355347534f8727e718e9185267028ac6fb01fbd3941b0a7edadeb84d5`.
- Dynamic reference methodology: van der Zee et al. *UltraTimTrack*. PeerJ Computer Science 11:e2636 (2025). [DOI](https://doi.org/10.7717/peerj-cs.2636), [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11784871/). It documents three manual observers and 84 every-second frames of this GM calf-raise video, with angle measured relative to the deep aponeurosis. The OSF README's “2025 Plos One” citation is inaccurate; do not repeat it.
- Public pretrained models: [watehell / UMUD r34 U-Net weights](https://www.kaggle.com/datasets/watehell/umud-r34-unet-weights), version 1, **CC BY-NC-SA 4.0**. These are third-party weights trained on competition masks, not new models trained by us. File hashes and source are in `checkpoints/provenance.json`.

## Static reference definition

`static_references.json` preserves each source rater's PA, FL and MT and the supplied `Scale_pixel_per_cm`; pixels/mm equals that column divided by 10. PA is interpreted as degrees and FL/MT as mm, consistent with domain convention and reference magnitudes. The original workbook does not explicitly label those three units.

The README/paper says six raters, while the workbook has R1–R7. R5 is absent from images 18–35. The declared main reference is the mean of the six complete raters **R1, R2, R3, R4, R6, R7**; the mean of all available raters is retained as a sensitivity analysis. This is an explicit analysis choice, not an asserted official consensus definition. Do not choose the averaging rule according to candidate performance.

Exclude `im_07_arch.tif`: its decoded grayscale-plus-shape hash matches historical competition training images `apo/image_0512.tif` and `fasc/image_2135.tif`. The other 34 static images have no exact match among 2,994 unique historical training image hashes. The upstream public geometry already refers to tuning on the OSF benchmark, so exclusion alone does not establish independent evaluation.

The final six source scale values were checked by ImageID: im30=8.9, im31=7.7, im32=12.6, im33=7.7, im34=7.7, im35=7.7 pixels/mm. Visible rulers corroborate their order and magnitude, including mirrored image layouts. The im30 workbook scale corresponds to the first 89-pixel 10 mm interval; fitting its entire ruler gives approximately9.06 pixels/mm. Keep the supplied static calibration to remain consistent with the manual-reference scale, and document its rounding/measurement uncertainty.

## Dynamic reference definition and scale

`dynamic_references.json` preserves the source workbook's rounded three-rater means exactly. The archive contains 168 GM and 601 TA images, but only GM frames 0–166 have labels. Even frames 0, 2, ..., 166 are directly annotated (**84 frames**); odd frames are interpolated and are excluded from the main diagnostic. No MT labels or TA labels are supplied in this workbook. GM is one participant/video; frames are not independent participants.

`dynamic_gm_scale_audit.json` derives scale from the visible ruler without using PA/FL labels. The six 10 mm tick centers are 119.5, 197.5, 274.5, 352.5, 429.5, 507.5 pixels. A least-squares ruler fit gives **7.754285714 pixels/mm**, zero-depth at 42.1 px and maximum residual 0.315 px, identically across 168 frames. The displayed depth setting is 65 mm. Horizontal use assumes isotropic raster pixels. `calibrate_dynamic_ruler.py` reproduces this dataset-specific measurement.

## Verification and limits

- Both downloaded ZIP SHA256 values match OSF metadata.
- Independent workbook re-reading reproduced static means within 1.5e-14 and copied dynamic source means with zero difference. Recomputing from already-rounded individual dynamic ratings differs from published means by at most 0.006667 mm/degrees; retain the published means.
- `competition_overlap_audit.json` compares all 804 OSF images to 309 test file hashes and the historical training hashes. The training manifest contains 3,136 retained annotation rows (1,048 apo, 2,088 fasc), so it is not a fresh full competition inventory. No dynamic exact training matches or OSF exact test-file matches were found. Exact horizontal flips of the 35 static images also produced no additional training matches.
- Exact hashes cannot exclude crops, re-encoding in the file-only test audit, adjacent frames or shared participants. Static subject IDs are absent. Unknown public checkpoint training membership and existing upstream benchmark use preclude an independent-test claim.
- Use these images only as public development data. Do not access hidden competition target labels. Reference metrics do not establish leaderboard improvement.

`prepare_reference.py` reproduces the JSON extraction; `audit_competition_overlap.py` reproduces the main overlap audit. Keep the original archives/workbooks unchanged. `validation_data_audit.json` contains provenance and limitations. The optional partial TimTrack download was stopped and is not part of this active evaluation.

## Posthoc rater-quality review

After inspection of model residuals highlighted im19 MT, an independent check of the untouched source ZIP/XLSX confirmed that `Manual_architecture!AA20` is a hardcoded R7 MT value of **80.13125**. The other five complete raters give 19.93065–20.08929. The retained six-rater mean is 30.02780258, while the six-rater median is 20.0078125. This is a major reference disagreement; it is not an extraction error, a confirmed correctable typo, or evidence by itself of an aponeurosis-boundary error. No source value or primary reference was changed.

A uniform Tukey 1.5×IQR scan of all 630 complete-rater values flagged 30 values across 22 images (PA 10, FL 6, MT 14). These are descriptive flags; six raters and small IQR can flag small differences. Other sizeable disagreements include im26 R7 FL 33.8764 versus median 68.2976, and im22 R4 PA 23.388 versus median 16.8408. Do not automatically remove flagged values.

`posthoc_median_references.json` supplies a clearly labelled **posthoc sensitivity analysis**: median of the same six complete raters for every image and every endpoint, retaining the im07 exclusion. It must be reported alongside the original mean-reference analysis, not silently substituted or treated as independent confirmation. `audit_rater_quality.py` reproduces `rater_quality_audit.json`; the audit verifies source/archive equality and that `static_references.json` was unchanged.
