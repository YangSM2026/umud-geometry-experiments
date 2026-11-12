# UMUD independent geometry experiments

**The new geometry did not improve the primary static development benchmark.**
This repository records the implementation, negative results and a narrow
exploratory follow-up. It does not claim a new trained model,
independent clinical validation, or a world-first mathematical method.

**Verified Kaggle result:** the explicitly authorized experimental submission
completed with public score **0.35433** and rank **227**, compared with the prior
scored entry's **0.69948** and previously observed rank **434**. This snapshot was
recorded at **2026-11-12 17:11:38 UTC**; it is an observation time, not an asserted
submission timestamp. See [kaggle_status.json](kaggle_status.json) and the
[leaderboard](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data/leaderboard).
The public score uses approximately one-third of the test set; the final
two-thirds remain unseen. The full public-model ensemble differs from the old
entry, so this gain cannot be attributed solely to v2 geometry.

The exact submitted CSV and immutable build receipt are in competition_run/.
The receipt preserves the earlier negative development promotion decision;
kaggle_status.json separately records the later authorized submission. The
unsubmitted previous generated candidate is retained in previous_candidate/.
All 309 experimental predictions were independently reconstructed, including
10 exact prior-triplet fallbacks; 299 rows changed. The exact prior Kaggle receipt
and independent QC are retained alongside the run.

Open [UMUD_Innovation_Review_20261113.ipynb](UMUD_Innovation_Review_20261113.ipynb)
for the executed review. All 10 code cells ran successfully, with **57 passing
tests**, recomputed metrics, checked input hashes and saved table outputs.
The 57 comprise 52 geometry/reliability/decoder/insertion tests and five
competition-builder tests. No figures are included.

## Observed results

The primary static score is the mean of MAE/6° for PA, MAE/12 mm for FL and
MAE/3 mm for MT; lower is better. It uses the unchanged six-rater mean on
34 eligible static images. This is a development-benchmark score, not a Kaggle
leaderboard score.

| Decoder | Static score | Change from baseline | Interpretation |
|---|---:|---:|---|
| Public single-pair, supported-depth baseline | 0.208018 | — | Same frozen models and per-image processing |
| Independently implemented endpoint geometry | 0.278911 | +34.08% | Worse |
| Endpoint geometry with Huber boundary refits | 0.272956 | +31.22% | Worse |
| V2 focused insertion correction | 0.217684 | +4.65% | Worse; exploratory follow-up |

On 84 directly annotated frames of **one dependent GM video**, v2 reduced PA MAE
from **0.738250° to 0.689352°** and FL MAE from **2.406689 to 2.392143 mm**.
There is no dynamic MT reference. These small changes coexist with worse static
performance and do not establish an overall gain.

V1 and v2 each produced valid measurements on all 118 eligible cases, without
fallbacks. V2 was specified after observing v1's failure. Upstream benchmark use,
incompletely audited training membership and missing static subject IDs preclude
independent-test claims. The later Kaggle evaluation is separate from this
negative development comparison and does not reverse its findings.

## Implementation and provenance ledger

| Files | Contribution and ancestry | Evidence / limit |
|---|---|---|
| endpoint_geometry.py | Independently implemented physical line/curve intersections, deep-insertion PA, endpoint FL and vertical MT | Established analytic geometry; tested, not a novel trained model |
| decoder.py | New probability-mask-to-geometry integration and optional Huber boundary refits | V1 benchmarked; worse than baseline |
| reliability.py | New source-deduplicated, model-family-aware robust aggregation and spatial fascicle selection | Invariants tested; multi-model predictive benefit not established by these single-pair benchmarks |
| insertion_correction.py | Focused v2 ablation using independent endpoint consistency while retaining public texture/FL calibration | Exploratory; mixed results |
| vendor_geometry.py, vendor_texture.py | Attributed public decoder code, with the prior supported-depth adaptation | Apache-2.0 ancestry; not claimed as newly invented |
| cache_benchmark_maps.py, evaluate_benchmarks.py, evaluate_insertion_correction.py | Frozen-checkpoint inference and paired benchmark evaluation with receipt/hash checks | No training or access to hidden competition targets |
| build_competition_candidate.py | Separate experimental competition-output construction with explicit fallback audit | Engineering tests do not establish leaderboard improvement |
| test_*.py | Analytic, equivariance, failure-handling, duplication and aggregation checks | 57 executed tests pass |
| build_innovation_review.py | Builds and executes the review from existing finalized data | Recomputes results; does not rerun model inference |

The baseline is the **original public checkpoint pair with supported-depth
per-image decoding**, not the exact unmodified V12 notebook or the six-head
competition ensemble. Neural-network weights were trained by their public
authors. The new code operates on their probability maps.

## Reference quality

The primary static reference is the mean of complete raters R1, R2, R3, R4, R6
and R7. R5 is missing for part of the dataset. im_07_arch.tif was excluded before
evaluation because its decoded image exactly matches historical training data.
Absence of other exact hashes does not exclude crops, adjacent frames or shared
participants.

The untouched workbook contains **R7 MT = 80.13125 mm for im_19**, whereas the
other five complete raters give approximately 20 mm. This is faithfully
extracted, but not a confirmed typo. The retained mean is **30.02780258 mm**;
the median is **20.0078125 mm**. No source value or primary target was altered.

The notebook reports the available-rater mean sensitivity and a clearly marked
**posthoc six-rater median sensitivity applied uniformly to all images and
targets**. Both still show worse static performance for all three candidates.
No flagged rater is automatically discarded. Static bootstrap intervals are
descriptive only; no frame-level inferential interval is calculated for the
single video.

## Reproduce the review

The notebook uses the flat local module layout and the included small
review_inputs/ JSONs, finalized CSVs/summaries, protocols and source files.
It does **not** require raw images, model weights, probability maps or the large
regenerable geometry-diagnostic JSONs.

From this directory, using an environment with the packages recorded by the
notebook:

~~~sh
python -m unittest discover -s . -p 'test_*.py' -v
python build_innovation_review.py --execute
~~~

Execution uses actual in-process IPython cells by default because the original
host forbids both TCP and Unix kernel socket binding. Each saved cell has a real
execution count and captured outputs. On a host permitting kernels, use:

~~~sh
python build_innovation_review.py --execute --engine nbclient
~~~

review_input_manifest.json pins the review dependencies;
review_execution_report.json records execution and notebook SHA-256.
results_v1_verified/ and results_v2/ are the controlling benchmark results.
The earlier unverified-cache results_v1/ is not used by the notebook.

To reproduce full image inference, separately download the licensed
[static OSF archive](https://osf.io/hb5xj),
[dynamic OSF archive](https://osf.io/p8r2u), and
[watehell checkpoint pair](https://www.kaggle.com/datasets/watehell/umud-r34-unet-weights).
Restore the image/checkpoint layout recorded by the reference JSONs and
cache_benchmark_maps.py; verify checkpoint hashes in
review_inputs/checkpoint_provenance.json. Regenerate maps and use fresh result
directories with the evaluator CLIs. Full inference also requires Torch and
segmentation-models-pytorch. This review repository does not redistribute those
images, weights or caches.

The provided downloader pins source URLs and archive SHA-256 values and rejects
unsafe archive paths. After installing the inference dependencies, a fresh
reproduction can use:

~~~sh
python download_public_inputs.py research_inputs
cp review_inputs/static_references.json review_inputs/dynamic_references.json research_inputs/
python cache_benchmark_maps.py research_inputs research_inputs/maps_reproduced
python evaluate_benchmarks.py research_inputs results_v1_reproduced --maps-dir maps_reproduced
python evaluate_insertion_correction.py research_inputs results_v2_reproduced --maps-dir maps_reproduced
~~~

Use fresh result directories; the evaluators refuse to overwrite finalized runs.

## Sources and licenses

- [UMUD, Ritsche et al., 2026](https://doi.org/10.1186/s12880-026-02170-0);
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/41652557/). OSF project data: CC BY 4.0.
- [TimTrack](https://doi.org/10.1371/journal.pone.0265752) and
  [UltraTimTrack](https://doi.org/10.7717/peerj-cs.2636) provide prior methodological
  context. UltraTimTrack is the relevant dynamic-manual-annotation reference;
  the OSF README's “2025 PLOS One” attribution is inaccurate.
- [prvsiyan V12/366298975](https://www.kaggle.com/code/prvsiyan/trying-some-stuff-with-ultrasound-geometry?scriptVersionId=366298975):
  Apache-2.0 public code ancestry, with credits to Mawra Musawwar, watehell and
  Rishabh Bafna. Preserve LICENSE_UPSTREAM and NOTICE_UPSTREAM.
- Original public checkpoint pair: **CC BY-NC-SA 4.0**. The fivefold checkpoints
  used in the earlier competition pipeline are **CC BY-NC 4.0** and are not used
  in these benchmark experiments. Code and weight licenses are distinct.

No benchmark result in this repository establishes permission to publish,
submit or replace a competition entry automatically.
