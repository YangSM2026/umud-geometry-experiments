"""Reproducible exploratory v2 evaluation; preserves v1 outputs and protocol."""

from pathlib import Path
import argparse
import hashlib
import json

import cv2
import numpy as np
import pandas as pd

from evaluate_benchmarks import TARGETS, TOLERANCES, baseline, metric
from insertion_correction import full_prediction


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(root, output, maps_dir="maps_verified"):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError("Use a new output directory to retain exploratory experiment history")
    source = Path(__file__).resolve().parent
    dependencies = [source / name for name in (
        "evaluate_insertion_correction.py", "insertion_correction.py", "evaluate_benchmarks.py",
        "endpoint_geometry.py", "vendor_geometry.py", "vendor_texture.py",
    )]
    source_hashes = {path.name: digest(path) for path in dependencies}
    protocol_path = source / "protocol_v2.json"
    protocol_hash = digest(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    receipt_path = root / maps_dir / "inference_receipt.json"
    reference_paths = [root / "static_references.json", root / "dynamic_references.json"]
    input_paths = [*reference_paths, receipt_path]
    input_hashes = {str(path.relative_to(root)): digest(path) for path in input_paths}
    receipt = json.loads(receipt_path.read_text())
    if "contract_sha256" not in receipt:
        raise ValueError("Inference cache lacks a checkpoint/source contract")
    map_records = {(row["dataset"], row["image_id"]): row for row in receipt["images"]}
    rows, diagnostics = [], []
    for dataset, reference_path in zip(("static", "dynamic"), reference_paths):
        references = json.loads(reference_path.read_text())["rows"]
        for row in references:
            if dataset == "static" and row["image_id"] == "im_07_arch.tif":
                continue
            if dataset == "dynamic" and not row["directly_manually_annotated"]:
                continue
            map_path = root / maps_dir / dataset / (row["image_id"] + ".npz")
            image_path = root / row["image_path"]
            record = map_records[dataset, row["image_id"]]
            if digest(map_path) != record["map_sha256"] or digest(image_path) != record["image_sha256"]:
                raise ValueError("Image/map differs from completed inference receipt")
            with np.load(map_path, allow_pickle=False) as cached:
                if str(cached["contract_sha256"]) != receipt["contract_sha256"] or str(cached["image_sha256"]) != record["image_sha256"]:
                    raise ValueError("Cache identity differs from inference receipt")
                apo = cached["apo"].astype(np.float32)
                fascicle = cached["fasc"].astype(np.float32)
                original_height, original_width = cached["shape"]
            map_height, map_width = apo.shape
            scale = row["scale_pixels_per_mm"] if dataset == "static" else row["scale_pixels_per_mm_from_visible_ruler"]
            mx, my = original_width / map_width / scale, original_height / map_height / scale
            gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError(f"Cannot read referenced image {row['image_id']}")
            control = baseline(apo, fascicle, mx, my, gray, scale)
            measured = full_prediction(apo, fascicle, mx, my, gray, scale)
            prediction = measured if measured["valid"] else control
            if dataset == "static":
                truth = dict(zip(TARGETS, (row["mean_complete_six_R1_R2_R3_R4_R6_R7"][key] for key in ("PA", "FL", "MT"))))
                sensitivity = dict(zip(TARGETS, (row["mean_available_R1_to_R7"][key] for key in ("PA", "FL", "MT"))))
            else:
                truth = {"pa_deg": row["pa_deg"], "fl_mm": row["fl_mm"], "mt_mm": None}
                sensitivity = truth
            rows.append({
                "dataset": dataset, "image_id": row["image_id"], "variant": "insertion_geometry",
                "candidate_valid": measured["valid"], "fallback_used": not measured["valid"], "baseline_valid": control["valid"],
                **{"truth_" + key: truth[key] for key in TARGETS},
                **{"sensitivity_" + key: sensitivity[key] for key in TARGETS},
                **{"baseline_" + key: control[key] for key in TARGETS},
                **{"prediction_" + key: prediction[key] for key in TARGETS},
                "status": measured["status"],
            })
            diagnostics.append({"dataset": dataset, "image_id": row["image_id"], "variant": "insertion_geometry", **measured})

    # Detect changes to relevant executable sources, references, or receipt mid-run.
    if source_hashes != {path.name: digest(path) for path in dependencies} or protocol_hash != digest(protocol_path):
        raise RuntimeError("Relevant source or protocol changed during evaluation")
    if input_hashes != {str(path.relative_to(root)): digest(path) for path in input_paths}:
        raise RuntimeError("Reference or inference receipt changed during evaluation")
    table = pd.DataFrame(rows)
    summary = {
        "protocol": protocol, "protocol_sha256": protocol_hash, "source_sha256": source_hashes,
        "input_sha256": input_hashes, "inference_contract_sha256": receipt["contract_sha256"],
        "results": {}, "training": False,
        "label_warning": "Exploratory post-v1 analysis; static source/participant independence incomplete; dynamic is one dependent video. No competition score was measured.",
        "baseline_description": "supported-depth original public pair; per-image, no temporal pooling; not exact original V12 or six-head competition pipeline",
    }
    for (dataset, name), data in table.groupby(["dataset", "variant"]):
        result = {
            "n": len(data), "valid_candidates": int(data.candidate_valid.sum()),
            "fallback_count": int(data.fallback_used.sum()), "baseline_valid": int(data.baseline_valid.sum()),
        }
        if dataset == "static":
            prediction = data[["prediction_" + key for key in TARGETS]].to_numpy(float)
            control = data[["baseline_" + key for key in TARGETS]].to_numpy(float)
            truth = data[["truth_" + key for key in TARGETS]].to_numpy(float)
            result.update(baseline=metric(control, truth), candidate_with_explicit_fallback=metric(prediction, truth))
            sensitivity = data[["sensitivity_" + key for key in TARGETS]].to_numpy(float)
            result["available_rater_sensitivity"] = {"baseline": metric(control, sensitivity), "candidate": metric(prediction, sensitivity)}
            matched = (data.candidate_valid & data.baseline_valid).to_numpy(bool)
            result["matched_valid_cases"] = {
                "baseline": metric(control[matched], truth[matched]), "candidate": metric(prediction[matched], truth[matched])
            } if matched.any() else None
            delta = ((abs(prediction - truth) - abs(control - truth)) / TOLERANCES).mean(1)
            rng = np.random.default_rng(20261113)
            bootstrap = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1)
            result["paired_primary_delta_candidate_minus_baseline"] = {
                "mean": float(delta.mean()), "descriptive_image_bootstrap_95pct_interval": np.quantile(bootstrap, [.025, .975]).tolist(),
                "replicates": 10000,
                "caveat": "Exploratory image-level resampling only; outcome-informed follow-up, unknown subject dependence and upstream reuse prohibit confirmatory interpretation.",
            }
        else:
            result["metrics"] = {}
            for target in ("pa_deg", "fl_mm"):
                truth = data["truth_" + target].to_numpy(float)
                result["metrics"][target] = {
                    label: float(np.mean(abs(data[prefix + target].to_numpy(float) - truth)))
                    for label, prefix in (("baseline_MAE", "baseline_"), ("candidate_MAE", "prediction_"))
                }
            result["independent_groups"] = 1
            result["confidence_interval"] = None
        summary["results"][dataset + "/" + name] = result
    output.mkdir(parents=True)
    table.to_csv(output / "benchmark_predictions.csv", index=False)
    (output / "geometry_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n")
    (output / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary["results"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("output")
    parser.add_argument("--maps-dir", default="maps_verified")
    args = parser.parse_args()
    run(args.root, args.output, args.maps_dir)
