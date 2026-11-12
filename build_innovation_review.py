#!/usr/bin/env python3
"""Build and optionally execute the reproducible UMUD experiment-review notebook.

Uses finalized result CSVs and existing source modules. It does not train models,
rerun image inference, alter benchmark references, or submit to Kaggle.
Run with the supplied innovation environment: python build_innovation_review.py --execute
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from textwrap import dedent

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
NOTEBOOK = ROOT / "UMUD_Innovation_Review_20261113.ipynb"
RESULT_DIRS = ("results_v1_verified", "results_v2")
TARGETS = ("pa_deg", "fl_mm", "mt_mm")
TOLERANCES = np.array([6.0, 12.0, 3.0])
REFERENCE_FILES = (
    "static_references.json", "dynamic_references.json", "reference_verification.json",
    "rater_quality_audit.json", "posthoc_median_references.json",
    "dynamic_gm_scale_audit.json", "competition_overlap_audit.json",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_inputs():
    """Copy only existing small reference/audit JSONs; preserve their exact bytes."""
    review = ROOT / "review_inputs"
    review.mkdir(exist_ok=True)
    for name in REFERENCE_FILES:
        source = WORKSPACE / "research_validation" / name
        if source.is_file():
            shutil.copyfile(source, review / name)
        elif not (review / name).is_file():
            raise FileNotFoundError(f"Missing source and packaged review input: {name}")
    for source, name in (
        ("research_validation/maps_verified/inference_receipt.json", "inference_receipt.json"),
        ("research_validation/checkpoints/provenance.json", "checkpoint_provenance.json"),
    ):
        if (WORKSPACE / source).is_file():
            shutil.copyfile(WORKSPACE / source, review / name)
        elif not (review / name).is_file():
            raise FileNotFoundError(f"Missing source and packaged review input: {name}")
    paths = list(ROOT.glob("*.py")) + list(ROOT.glob("protocol*.json"))
    paths += list(review.glob("*.json"))
    paths += [ROOT / "LICENSE_UPSTREAM", ROOT / "NOTICE_UPSTREAM"]
    if (ROOT / "kaggle_status.json").is_file():
        paths.append(ROOT / "kaggle_status.json")
    summaries = {}
    for folder in RESULT_DIRS:
        for name in ("benchmark_predictions.csv", "benchmark_summary.json"):
            path = ROOT / folder / name
            if not path.is_file():
                raise FileNotFoundError(f"Final result missing: {path}")
            paths.append(path)
        summaries[folder] = json.loads((ROOT / folder / "benchmark_summary.json").read_text())
        for name, expected in summaries[folder]["source_sha256"].items():
            if digest(ROOT / name) != expected:
                raise ValueError(f"Recorded experiment code changed: {folder}/{name}")
        if summaries[folder].get("training") is not False:
            raise ValueError("This review expects frozen-model, no-training experiments")
    manifest = {
        "review_date": "2026-11-13",
        "scope": "Reexecute analytic tests and recompute review statistics from finalized prediction CSVs; no image inference or training.",
        "files": {str(p.relative_to(ROOT)): {"sha256": digest(p), "bytes": p.stat().st_size}
                  for p in sorted(set(paths))},
    }
    (ROOT / "review_input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summaries


def build():
    summaries = prepare_inputs()
    scores = {}
    for folder in RESULT_DIRS:
        frame = pd.read_csv(ROOT / folder / "benchmark_predictions.csv")
        for variant, group in frame[frame.dataset == "static"].groupby("variant", sort=True):
            truth = group[["truth_" + x for x in TARGETS]].to_numpy(float)
            baseline = group[["baseline_" + x for x in TARGETS]].to_numpy(float)
            prediction = group[["prediction_" + x for x in TARGETS]].to_numpy(float)
            scores[variant] = (
                float((np.abs(baseline - truth).mean(0) / TOLERANCES).mean()),
                float((np.abs(prediction - truth).mean(0) / TOLERANCES).mean()),
            )
    baseline = scores["endpoint"][0]
    assert all(abs(value[0] - baseline) < 1e-12 for value in scores.values())
    dynamic_v2 = summaries["results_v2"]["results"]["dynamic/insertion_geometry"]["metrics"]
    cells = []
    def md(text):
        cells.append(nbformat.v4.new_markdown_cell(dedent(text).strip()))
    def code(text):
        cells.append(nbformat.v4.new_code_cell(dedent(text).strip()))

    md(f"""
    # UMUD innovation review — 13 November 2026

    ## tl;dr

    **The new geometry did not improve the primary static development benchmark.**
    The normalized MAE was **{baseline:.6f}** for the public single-pair baseline,
    **{scores["endpoint"][1]:.6f}** for endpoint geometry,
    **{scores["endpoint_huber"][1]:.6f}** with Huber boundary refits, and
    **{scores["insertion_geometry"][1]:.6f}** for the focused v2 insertion correction
    (lower is better). These are development-benchmark scores, **not Kaggle scores**.

    On one dependent video, v2 reduced PA MAE from
    **{dynamic_v2["pa_deg"]["baseline_MAE"]:.6f}° to {dynamic_v2["pa_deg"]["candidate_MAE"]:.6f}°**
    and FL MAE from **{dynamic_v2["fl_mm"]["baseline_MAE"]:.6f} to {dynamic_v2["fl_mm"]["candidate_MAE"]:.6f} mm**.
    That limited result does not establish generalization. V2 was designed after
    observing v1's failure and is exploratory.

    The work independently implements physical endpoint geometry, robust boundary
    fitting, and provenance-aware robust aggregation. **No new neural network was
    trained, no world-first method is claimed, and the previous submission is not
    replaced.** This notebook reruns analytic tests and recomputes its tables from
    saved predictions. It does not rerun segmentation, access hidden competition
    labels, or submit anything.
    """)
    if (ROOT / "kaggle_status.json").is_file():
        kaggle = json.loads((ROOT / "kaggle_status.json").read_text())
        cells[0].source += (f"\n\n**Separate live Kaggle result:** public score **{kaggle['public_score']:.5f}**, "
            f"rank **{kaggle['current_rank']}**, observed **{kaggle['observed_at_utc']}**. "
            "The complete ensemble differs from the old entry; this does not isolate a gain from v2 geometry. "
            "See kaggle_status.json for the scored CSV identity and evidence. This is separate from the negative development results above.")
    md("""
    ## Context & methods

    V1 intersects observed fascicle lines with inner aponeurosis curves in
    millimetres, measures PA against the deep tangent at insertion, measures FL
    between consistent endpoints, and averages vertical MT at three fixed positions.
    A paired variant refits boundaries with Huber IRLS. Up to three spatially
    separated valid fascicles are selected with minimal extrapolation. These are
    established mathematical ideas implemented here independently, not a novel
    trained segmentation architecture.

    V2 isolates the insertion/root-consistency change while retaining public PCA,
    boundary grouping, texture fusion, FL calibration and prediction limits. It
    uses symmetric visible support because the public extractor discarded actual
    endpoint projections. Both comparisons use the **same original pretrained
    pair, per-image processing and known benchmark scale**, not the six-head
    competition ensemble or an exact reproduction of unmodified V12.

    The separate robust-aggregation module deduplicates checkpoint identities,
    controls model-family mass, and reports unavailable disagreement for a single
    surviving source. Its engineering invariants are tested here; **its multi-model
    predictive performance was not evaluated by these single-pair benchmarks**.

    ### Key assumptions

    - The primary static target remains the mean of six complete raters
      R1, R2, R3, R4, R6, R7. R5 has missing observations.
    - im_07 is excluded for an exact historical training-image overlap. The other
      34 images are still development data: subject IDs and full upstream model
      membership are unknown, and public geometry already used this benchmark.
    - The 84 directly annotated even-numbered dynamic frames belong to one video;
      interpolated frames are excluded. No dynamic MT reference exists.
    - A failed geometry estimate uses an explicitly counted whole-triplet baseline
      fallback. Failures are not silently discarded.
    - The primary score is mean(MAE_PA/6°, MAE_FL/12 mm, MAE_MT/3 mm).
      Tiny median-error and RMSE tie-breakers are recomputed separately.
    """)
    md("""
    ### Sources and attribution

    | Evidence or component | Primary source | Role / terms |
    |---|---|---|
    | UMUD data resource | [Ritsche et al., 2026](https://doi.org/10.1186/s12880-026-02170-0); [PubMed](https://pubmed.ncbi.nlm.nih.gov/41652557/) | Dataset provenance |
    | Static architecture v0.1.0 | [Authors' OSF archive](https://osf.io/hb5xj) | 35 images; 34 eligible; project CC BY 4.0 |
    | Dynamic architecture v0.2.0 | [Authors' OSF archive](https://osf.io/p8r2u) | One annotated GM sequence used here; project CC BY 4.0 |
    | Dynamic annotation method | [UltraTimTrack, PeerJ Computer Science 11:e2636](https://doi.org/10.7717/peerj-cs.2636) | Three observers and 84 directly annotated frames; the OSF README's “PLOS One” citation is inaccurate |
    | Earlier tracking methodology | [TimTrack, PLOS ONE](https://doi.org/10.1371/journal.pone.0265752) | Prior work; no claim that endpoint geometry or tracking is world-first |
    | Public decoder ancestry | [prvsiyan V12/366298975](https://www.kaggle.com/code/prvsiyan/trying-some-stuff-with-ultrasound-geometry?scriptVersionId=366298975) | Apache-2.0; retain upstream credits and notices |
    | Original frozen model pair | [watehell weights, version 1](https://www.kaggle.com/datasets/watehell/umud-r34-unet-weights) | CC BY-NC-SA 4.0; trained by public-source authors |
    | Fivefold weights in earlier submission | [prvsiyan weights](https://www.kaggle.com/datasets/prvsiyan/umud-r34-5fold-unet-weights) | CC BY-NC 4.0; **not used in these single-pair benchmarks** |

    Upstream credits also include Mawra Musawwar, watehell and Rishabh Bafna.
    LICENSE_UPSTREAM and NOTICE_UPSTREAM are retained. No raw images or model weights
    are embedded in this notebook; copied reference JSONs retain source provenance.
    """)
    md("""
    ## Data

    The following cells verify the saved input hashes before computing results.
    Opening the notebook with saved outputs requires no model weights. Rerunning
    it requires this directory's modules, result CSVs, protocols and review_inputs.
    Use the supplied Python environment with NumPy, pandas, SciPy, OpenCV,
    nbformat, nbclient and ipykernel.

    Saved cells were actually executed through in-process IPython because this
    host prohibits both TCP and Unix kernel socket binding. The builder also
    supports standard nbclient execution on hosts that permit kernel sockets.
    Large geometry diagnostics, raw images, probability maps and model weights
    are not needed to rerun this review.
    """)
    code("""
    from pathlib import Path
    import hashlib, importlib.metadata, io, json, platform, sys, unittest
    from collections import Counter
    import numpy as np
    import pandas as pd
    from IPython.display import display, Markdown

    ROOT = Path.cwd()
    if not (ROOT / "protocol.json").is_file():
        ROOT = ROOT / "umud_innovation"
    assert (ROOT / "review_input_manifest.json").is_file(), "Run from the notebook directory or its parent."
    sys.path.insert(0, str(ROOT))
    TARGETS = ("pa_deg", "fl_mm", "mt_mm")
    TOLERANCES = np.array([6.0, 12.0, 3.0])
    def sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    manifest = json.loads((ROOT / "review_input_manifest.json").read_text())
    for relative, record in manifest["files"].items():
        assert sha256(ROOT / relative) == record["sha256"], f"Changed input: {relative}"
    pd.set_option("display.max_colwidth", 72)
    pd.set_option("display.max_rows", 20)
    pd.set_option("display.max_columns", 10)
    print(f"Verified {len(manifest['files'])} source/result/reference hashes.")
    display(pd.DataFrame([
        {"Package": "Python", "Version": platform.python_version()},
        *({"Package": name, "Version": importlib.metadata.version(name)}
          for name in ("numpy", "pandas", "scipy", "nbformat", "nbclient"))
    ]).style.hide(axis="index"))
    """)
    code("""
    result_dirs = {"v1": "results_v1_verified", "v2": "results_v2"}
    frames, summaries = {}, {}
    for version, folder in result_dirs.items():
        frame = pd.read_csv(ROOT / folder / "benchmark_predictions.csv")
        summary = json.loads((ROOT / folder / "benchmark_summary.json").read_text())
        assert summary["training"] is False
        assert not frame.duplicated(["dataset", "variant", "image_id"]).any()
        for column in ("candidate_valid", "fallback_used", "baseline_valid"):
            parsed = frame[column].astype(str).str.lower().map({"true": True, "false": False})
            assert not parsed.isna().any(), f"Invalid boolean: {column}"
            frame[column] = parsed.astype(bool)
        assert (frame.candidate_valid != frame.fallback_used).all()
        for (dataset, variant), group in frame.groupby(["dataset", "variant"]):
            assert len(group) == (34 if dataset == "static" else 84)
            assert "im_07_arch.tif" not in group.image_id.to_list()
            targets = TARGETS if dataset == "static" else TARGETS[:2]
            columns = [prefix + target for prefix in ("truth_", "baseline_", "prediction_") for target in targets]
            assert np.isfinite(group[columns].to_numpy(float)).all()
            if dataset == "dynamic":
                assert group.truth_mt_mm.isna().all()
                assert all(int(name.split("_")[1].split(".")[0]) % 2 == 0 for name in group.image_id)
        for name, expected in summary["source_sha256"].items():
            assert sha256(ROOT / name) == expected, f"Experiment source mismatch: {name}"
        for name, expected in summary["input_sha256"].items():
            local = "inference_receipt.json" if name.endswith("inference_receipt.json") else name
            assert sha256(ROOT / "review_inputs" / local) == expected
        protocol = ROOT / ("protocol.json" if version == "v1" else "protocol_v2.json")
        assert sha256(protocol) == summary["protocol_sha256"]
        frames[version], summaries[version] = frame, summary
    v1base = frames["v1"].drop_duplicates(["dataset", "image_id"]).set_index(["dataset", "image_id"])
    v2base = frames["v2"].set_index(["dataset", "image_id"]).loc[v1base.index]
    np.testing.assert_allclose(v1base[["baseline_" + x for x in TARGETS]],
                               v2base[["baseline_" + x for x in TARGETS]], rtol=0, atol=1e-12)
    print("Finalized v1/v2 cohorts, source contracts, unchanged references and paired baselines verified.")
    """)
    md("""
    ### Analytic and robustness tests

    These rerun the existing tests for endpoint geometry, mask decoding, insertion
    correction, robust aggregation and fascicle selection. Passing establishes
    tested code behavior, not clinical or leaderboard accuracy.
    """)
    code("""
    suite = unittest.defaultTestLoader.discover(str(ROOT), pattern="test_*.py", top_level_dir=str(ROOT))
    def flatten(test_suite):
        for item in test_suite:
            if isinstance(item, unittest.TestSuite):
                yield from flatten(item)
            else:
                yield item
    test_counts = Counter(test.__class__.__module__ for test in flatten(suite))
    log = io.StringIO()
    test_result = unittest.TextTestRunner(stream=log, verbosity=0).run(suite)
    assert test_result.testsRun > 0 and test_result.wasSuccessful(), log.getvalue()
    display(pd.DataFrame([{"Test module": name, "Tests": count, "Result": "passed"}
                          for name, count in sorted(test_counts.items())]).style.hide(axis="index"))
    print(f"{test_result.testsRun} tests passed; failures={len(test_result.failures)}, errors={len(test_result.errors)}.")
    """)
    md("""
    ## Results

    ### Static primary analysis: retain the six-rater mean

    Every number below is recomputed from the saved per-image predictions and
    reconciled with the corresponding frozen summary. Positive deltas mean
    greater error. The baseline is repeated by experiment for traceability.
    """)
    code("""
    def metrics(prediction, truth):
        error = np.asarray(prediction, float) - np.asarray(truth, float)
        absolute = np.abs(error)
        mae = absolute.mean(axis=0)
        primary = float((mae / TOLERANCES).mean())
        tiebreakers = primary + 1e-6 * (np.median(absolute, axis=0) / TOLERANCES).mean()
        tiebreakers += 1e-9 * (np.sqrt((error ** 2).mean(axis=0)) / TOLERANCES).mean()
        return {"mae": mae, "score": primary, "with_tiebreakers": float(tiebreakers)}

    static_rows, static_details, coverage_rows = [], {}, []
    for version, frame in frames.items():
        for (dataset, variant), group in frame.groupby(["dataset", "variant"], sort=True):
            reference = summaries[version]["results"][f"{dataset}/{variant}"]
            assert int(group.candidate_valid.sum()) == reference["valid_candidates"]
            assert int(group.fallback_used.sum()) == reference["fallback_count"]
            coverage_rows.append({"Version": version, "Dataset": dataset, "Variant": variant,
                "N": len(group), "Valid": int(group.candidate_valid.sum()), "Fallbacks": int(group.fallback_used.sum())})
            if dataset != "static":
                continue
            truth = group[["truth_" + x for x in TARGETS]].to_numpy(float)
            base = metrics(group[["baseline_" + x for x in TARGETS]], truth)
            candidate = metrics(group[["prediction_" + x for x in TARGETS]], truth)
            for measured, recorded in ((base, reference["baseline"]),
                    (candidate, reference["candidate_with_explicit_fallback"])):
                np.testing.assert_allclose(measured["mae"], [recorded["MAE"][x] for x in TARGETS], atol=1e-10, rtol=0)
                assert abs(measured["score"] - recorded["primary_score"]) < 1e-10
                assert abs(measured["with_tiebreakers"] - recorded["score_with_tiebreakers"]) < 1e-10
            static_details[version, variant] = (group, base, candidate)
            static_rows.append({"Version": version, "Variant": variant, "N": len(group),
                "PA MAE (°)": candidate["mae"][0], "FL MAE (mm)": candidate["mae"][1],
                "MT MAE (mm)": candidate["mae"][2], "Primary score ↓": candidate["score"],
                "Δ vs baseline": candidate["score"] - base["score"]})
    first = next(iter(static_details.values()))[1]
    static_rows.insert(0, {"Version": "paired", "Variant": "public baseline", "N": 34,
        "PA MAE (°)": first["mae"][0], "FL MAE (mm)": first["mae"][1], "MT MAE (mm)": first["mae"][2],
        "Primary score ↓": first["score"], "Δ vs baseline": 0.0})
    static_table = pd.DataFrame(static_rows)
    display(static_table.style.hide(axis="index").format({c: "{:.6f}" for c in static_table.columns if c not in ("Version", "Variant", "N")}))
    for (version, variant), (_, base, candidate) in static_details.items():
        change = 100 * (candidate["score"] / base["score"] - 1)
        print(f"{version}/{variant}: {change:+.2f}% relative change in primary error (positive = worse).")
    """)
    md("### Coverage and explicit fallbacks")
    code("""
    display(pd.DataFrame(coverage_rows).style.hide(axis="index"))
    assert all(row["Fallbacks"] == 0 for row in coverage_rows)
    print("All 118 eligible cases per variant were valid; no candidate benefited from dropping failed cases.")
    """)
    md("""
    ### Dynamic diagnostics: one dependent video, PA and FL only

    These 84 frames are not 84 independent people. No frame-level confidence
    interval or full three-target score is reported because MT is unavailable.
    """)
    code("""
    dynamic_rows = []
    for version, frame in frames.items():
        for variant, group in frame[frame.dataset == "dynamic"].groupby("variant", sort=True):
            recorded = summaries[version]["results"][f"dynamic/{variant}"]["metrics"]
            row = {"Version": version, "Variant": variant, "Frames": len(group)}
            for target, label in (("pa_deg", "PA"), ("fl_mm", "FL")):
                truth = group["truth_" + target].to_numpy(float)
                base_error = float(np.abs(group["baseline_" + target].to_numpy(float) - truth).mean())
                error = float(np.abs(group["prediction_" + target].to_numpy(float) - truth).mean())
                assert abs(error - recorded[target]["candidate_MAE"]) < 1e-10
                assert abs(base_error - recorded[target]["baseline_MAE"]) < 1e-10
                unit = "°" if target == "pa_deg" else "mm"
                row[f"{label} baseline ({unit})"] = base_error
                row[f"{label} candidate ({unit})"] = error
                row[f"Δ {label} ({unit})"] = error - base_error
            dynamic_rows.append(row)
    dynamic_table = pd.DataFrame(dynamic_rows)
    display(dynamic_table.style.hide(axis="index").format({c: "{:.6f}" for c in dynamic_table.columns if c not in ("Version", "Variant", "Frames")}))
    """)
    md("""
    ### im_19 label disagreement: preserve the primary reference

    Inspection of residuals prompted a **posthoc** audit of im_19. The untouched
    workbook contains R7 MT = 80.13125 in Manual_architecture!AA20, whereas the
    other five complete raters are approximately 20 mm. The extraction is faithful.
    This is not a confirmed typo and no rater value was deleted or corrected.
    """)
    code("""
    static_reference = json.loads((ROOT / "review_inputs/static_references.json").read_text())
    rater_audit = json.loads((ROOT / "review_inputs/rater_quality_audit.json").read_text())
    assert rater_audit["primary_reference_modified"] is False
    assert rater_audit["original_values_retained"] is True
    im19 = next(row for row in static_reference["rows"] if row["image_id"] == "im_19_arch.tif")
    complete = ("1", "2", "3", "4", "6", "7")
    rater_table = pd.DataFrame([{"Rater": "R" + r, "PA (°)": im19["raters"][r]["PA"],
                                "FL (mm)": im19["raters"][r]["FL"], "MT (mm)": im19["raters"][r]["MT"]}
                               for r in complete])
    assert rater_table.loc[rater_table.Rater == "R7", "MT (mm)"].iloc[0] == 80.13125
    display(rater_table.style.hide(axis="index").format({c: "{:.6f}" for c in rater_table.columns if c != "Rater"}))
    values = rater_table[["PA (°)", "FL (mm)", "MT (mm)"]].to_numpy(float)
    display(pd.DataFrame([dict(zip(("Reference", "PA (°)", "FL (mm)", "MT (mm)"), ["Primary six-rater mean", *values.mean(0)])),
                          dict(zip(("Reference", "PA (°)", "FL (mm)", "MT (mm)"), ["Posthoc six-rater median", *np.median(values, axis=0)]))]
                         ).style.hide(axis="index").format({c: "{:.6f}" for c in ("PA (°)", "FL (mm)", "MT (mm)")}))
    print(f"Uniform rater scan: {sum(rater_audit['complete_six_flags_by_metric'].values())} descriptive flags "
          f"among {rater_audit['complete_six_ratings_reviewed']} values; no automatic exclusions.")
    """)
    md("""
    ### Sensitivity analyses, explicitly secondary

    The available-rater mean is a declared sensitivity analysis. The uniform
    six-rater median is **posthoc**, triggered by the im_19 disagreement and applied
    to every image and endpoint. Neither replaces the primary mean or constitutes
    independent confirmation.
    """)
    code("""
    median_reference = json.loads((ROOT / "review_inputs/posthoc_median_references.json").read_text())
    assert median_reference["primary_target_replaced"] is False
    median_lookup = {row["image_id"]: [row[x] for x in TARGETS] for row in median_reference["rows"]}
    sensitivity_rows = []
    for (version, variant), (group, base, candidate) in static_details.items():
        available = group[["sensitivity_" + x for x in TARGETS]].to_numpy(float)
        median_truth = np.array([median_lookup[name] for name in group.image_id], dtype=float)
        predictions = group[["prediction_" + x for x in TARGETS]].to_numpy(float)
        baseline_predictions = group[["baseline_" + x for x in TARGETS]].to_numpy(float)
        for label, target in (("available-rater mean", available), ("POSTHOC six-rater median", median_truth)):
            bm, cm = metrics(baseline_predictions, target), metrics(predictions, target)
            sensitivity_rows.append({"Version": version, "Variant": variant, "Reference": label,
                "Baseline score": bm["score"], "Candidate score": cm["score"],
                "Δ candidate − baseline": cm["score"] - bm["score"]})
    sensitivity_table = pd.DataFrame(sensitivity_rows)
    display(sensitivity_table.style.hide(axis="index").format({c: "{:.6f}" for c in sensitivity_table.columns if c not in ("Version", "Variant", "Reference")}))
    """)
    md("""
    ### Paired static uncertainty is descriptive

    Recompute the saved paired image bootstrap with its recorded seed and 10,000
    resamples. V1 used 97.5% intervals for two comparisons; v2 uses 95% only as an
    exploratory description. Unknown participant dependence and prior benchmark
    reuse prevent a confirmatory interpretation. Dynamic intervals are omitted.
    """)
    code("""
    interval_rows = []
    for (version, variant), (group, base, candidate) in static_details.items():
        truth = group[["truth_" + x for x in TARGETS]].to_numpy(float)
        candidate_error = np.abs(group[["prediction_" + x for x in TARGETS]].to_numpy(float) - truth)
        baseline_error = np.abs(group[["baseline_" + x for x in TARGETS]].to_numpy(float) - truth)
        delta = ((candidate_error - baseline_error) / TOLERANCES).mean(axis=1)
        recorded = summaries[version]["results"][f"static/{variant}"]["paired_primary_delta_candidate_minus_baseline"]
        key = next(k for k in recorded if k.startswith("descriptive_image_bootstrap_"))
        level = 97.5 if "97_5" in key else 95.0
        tail = (1 - level / 100) / 2
        rng = np.random.default_rng(20261113)
        bootstrap = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(axis=1)
        bounds = np.quantile(bootstrap, [tail, 1 - tail])
        np.testing.assert_allclose(bounds, recorded[key], atol=1e-10, rtol=0)
        assert abs(delta.mean() - recorded["mean"]) < 1e-10
        interval_rows.append({"Version": version, "Variant": variant, "Δ score": delta.mean(),
                              "Descriptive level (%)": level, "Lower": bounds[0], "Upper": bounds[1]})
    display(pd.DataFrame(interval_rows).style.hide(axis="index").format({c: "{:.6f}" for c in ("Δ score", "Lower", "Upper")}))
    """)
    md("""
    ## Takeaways

    1. **Do not promote the endpoint variants.** Their primary static errors exceed
       the baseline, and v1 also worsens both dynamic endpoints.
    2. **Retain v2 as a narrow exploratory result.** Its small PA/FL improvements on
       one video coexist with worse static performance; these development comparisons
       do not establish a general improvement or independent validation. The separate
       live Kaggle outcome is recorded in kaggle_status.json when present.
    3. **Keep the engineering contribution separate from efficacy.** Independent
       endpoint geometry, Huber fits, deduplication and robust aggregation now have
       executable tests, but that is not evidence of superior predictions.
    4. **Resolve reference uncertainty without changing the primary analysis.**
       The im_19 R7 disagreement motivates secondary sensitivity analysis and
       future source clarification, not a silent label correction.

    A future efficacy claim requires an independently defined evaluation cohort,
    auditable training membership, participant-level grouping, and prespecified
    selection rules. These development results did not justify automatic promotion;
    the later explicitly authorized Kaggle evaluation is recorded separately.
    """)
    md("### Reproducibility record")
    code("""
    selected_paths = [
        "results_v1_verified/benchmark_predictions.csv", "results_v1_verified/benchmark_summary.json",
        "results_v2/benchmark_predictions.csv", "results_v2/benchmark_summary.json",
        "protocol.json", "protocol_v2.json", "review_inputs/static_references.json",
        "review_inputs/inference_receipt.json",
    ]
    display(pd.DataFrame([{"Artifact": name, "SHA-256": sha256(ROOT / name)} for name in selected_paths])
            .style.hide(axis="index").set_properties(**{"text-align": "left", "word-break": "break-all"}))
    assert test_result.wasSuccessful()
    print("Review complete: all cells executed, tests passed, and displayed metrics reconciled.")
    print("No training, raw-image inference, source-reference modification, competition submission or figures were produced.")
    """)
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.update(
        kernelspec={"name": "python3", "display_name": "Python 3 (UMUD review)", "language": "python"},
        language_info={"name": "python", "version": ".".join(map(str, sys.version_info[:3]))},
        umud_review={"date": "2026-11-13", "results": list(RESULT_DIRS), "trained_new_model": False,
                     "live_kaggle_result_recorded": (ROOT / "kaggle_status.json").is_file(),
                     "benchmark_scores_are_kaggle_scores": False, "figure_policy": "Tables only; no raster figures."},
    )
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK)
    return notebook


def execute_nbclient(notebook):
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    with tempfile.TemporaryDirectory(prefix="umud-review-kernel-") as temporary:
        kernel_dir = Path(temporary) / "umud-review"
        kernel_dir.mkdir()
        (kernel_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "UMUD review", "language": "python",
        }))
        manager = KernelManager(kernel_name="umud-review", transport="ipc", ip=str(Path(temporary) / "socket"),
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary]))
        client = NotebookClient(notebook, km=manager, timeout=180,
            resources={"metadata": {"path": str(ROOT)}}, allow_errors=False)
        client.execute()


def execute_inprocess(notebook):
    """Execute real cells via IPython when host policy prohibits kernel sockets."""
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from traitlets.config import Config
    config = Config()
    config.HistoryManager.hist_file = ":memory:"
    shell = InteractiveShell.instance(config=config, user_ns={"__name__": "__main__"})
    previous_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        for number, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            count = shell.execution_count
            with capture_output(stdout=True, stderr=True, display=True) as captured:
                result = shell.run_cell(cell.source, store_history=True)
            if result.error_before_exec or result.error_in_exec:
                raise RuntimeError(f"Notebook cell {number} failed:\n{captured.stdout}\n{captured.stderr}") from (result.error_before_exec or result.error_in_exec)
            outputs = []
            for name in ("stdout", "stderr"):
                text = getattr(captured, name)
                if text:
                    outputs.append(nbformat.v4.new_output("stream", name=name, text=text))
            for output in captured.outputs:
                outputs.append(nbformat.v4.new_output("display_data", data=output.data, metadata=output.metadata))
            cell.execution_count = count
            cell.outputs = outputs
    finally:
        os.chdir(previous_cwd)
        InteractiveShell.clear_instance()


def execute(notebook, engine):
    if engine == "nbclient":
        execute_nbclient(notebook)
    else:
        execute_inprocess(notebook)
    notebook.metadata.umud_review["execution_engine"] = engine
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is not None
            assert not any(output.output_type == "error" for output in cell.outputs)
    nbformat.write(notebook, NOTEBOOK)
    report = {
        "status": "executed_without_errors",
        "execution_engine": engine,
        "execution_environment_note": "Actual in-process IPython execution; this host rejects TCP and Unix kernel socket binding." if engine == "inprocess" else "nbclient kernel execution",
        "notebook": NOTEBOOK.name, "notebook_sha256": digest(NOTEBOOK),
        "code_cells": sum(cell.cell_type == "code" for cell in notebook.cells),
        "markdown_cells": sum(cell.cell_type == "markdown" for cell in notebook.cells),
        "execution_counts": [cell.execution_count for cell in notebook.cells if cell.cell_type == "code"],
        "image_outputs": sum(mime.startswith("image/") for cell in notebook.cells
                             if cell.cell_type == "code" for output in cell.outputs
                             for mime in output.get("data", {})),
        "validation_scope": "Executed tests, independent metric recomputation, frozen-summary reconciliation, exact input hash checks; table presentation inspected separately.",
    }
    (ROOT / "review_execution_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--engine", choices=("inprocess", "nbclient"), default="inprocess")
    arguments = parser.parse_args()
    built = build()
    if arguments.execute:
        execute(built, arguments.engine)
    else:
        print(f"Built unexecuted review: {NOTEBOOK}")
