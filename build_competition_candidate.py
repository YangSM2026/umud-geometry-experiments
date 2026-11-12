"""Build an experimental UMUD insertion-geometry CSV from existing cached maps.

This performs no model training/inference, label access, submission or promotion.
The v2 development evidence failed the overall static-reference gate. Therefore
every artifact is explicitly experimental_rejected_for_promotion.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

try:
    from . import insertion_correction as correction
    from . import vendor_geometry as vendor
except ImportError:
    import insertion_correction as correction
    import vendor_geometry as vendor

TAGS = ("test", "ens_f0", "ens_f1", "ens_f2", "ens_f3", "ens_f4")
COLS = ("pa_deg", "fl_mm", "mt_mm")
BASE = np.array([.5, .1, .1, .1, .1, .1])
STATUS = "experimental_rejected_for_promotion"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_source(path):
    path = Path(path).resolve()
    identities = [path] if path.name == "cache_identity.json" and path.is_file() else list(path.rglob("cache_identity.json"))
    candidates = [p.parent for p in identities if (p.parent / "cache").is_dir() and (p.parent / "audit" / "scale_calibration.csv").is_file()]
    if len(candidates) != 1:
        raise ValueError(f"Expected one saved-output source with cached maps, found {len(candidates)}")
    return candidates[0]


def checked_table(path, ids, required):
    frame = pd.read_csv(path)
    if not set(required).issubset(frame.columns):
        raise ValueError(f"Required columns missing: {path}")
    if frame.image_id.isna().any() or frame.image_id.duplicated().any() or set(frame.image_id) != set(ids):
        raise ValueError(f"ID mismatch: {path}")
    return frame.set_index("image_id").loc[ids].reset_index()


def fuse_head(rows, segments, prior):
    """Reuse exact prior grouping/texture; change only the segment geometry."""
    frame = pd.DataFrame(rows)
    frame["run"] = prior.run.to_numpy()
    frame["pa_tex"] = pd.to_numeric(prior.pa_tex, errors="raise").to_numpy()
    if np.isinf(frame.pa_tex.to_numpy()).any():
        raise ValueError("Infinite cached texture angle")
    for _, indexes in frame.groupby("run", sort=False).groups.items():
        pooled = np.concatenate([segments[int(i)] for i in indexes])
        pa = vendor.pa_depth_fit(pooled, .9) if len(pooled) else None
        if pa is not None:
            frame.loc[indexes, "pa_deg"] = pa
    texture = frame.groupby("run").pa_tex.transform("median")
    frame["pa_deg"] = np.where(texture.notna(), .5 * frame.pa_deg + .5 * texture, frame.pa_deg)
    sine_length = frame.mt_mm / np.sin(np.radians(frame.pa_deg.clip(5, 45)))
    frame["fl_mm"] = .75 * frame.fl_mm + .25 * sine_length
    if not np.isfinite(frame[list(COLS)].to_numpy()).all():
        raise ValueError("Non-finite geometry output before clipping")
    for column, bounds in correction.RANGES.items():
        frame[column] = frame[column].clip(*bounds)
    return frame


def combine_heads(heads, prior, allow_prior_fallback=True):
    """Shared validity weights across targets; explicitly report all-head fallback."""
    ids = prior.image_id.tolist()
    values, flags = [], []
    for tag in TAGS:
        head = heads[tag]
        if head.image_id.tolist() != ids or head.ok.isna().any():
            raise ValueError(f"Head ID/order/validity mismatch: {tag}")
        if not all(isinstance(v, (bool, np.bool_)) for v in head.ok):
            raise ValueError(f"Non-boolean validity: {tag}")
        values.append(head[list(COLS)].to_numpy(dtype=float))
        flags.append(head.ok.to_numpy(dtype=bool))
    values, flags = np.stack(values), np.stack(flags)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite head predictions")
    active = BASE[:, None] * flags
    total = active.sum(0)
    fallback = total == 0
    if fallback.any() and not allow_prior_fallback:
        raise ValueError("All new heads failed but the prior failure-aware submission is unavailable")
    weights = np.divide(active, total[None], out=np.zeros_like(active), where=total[None] > 0)
    output = prior[list(COLS)].to_numpy().copy()
    good = ~fallback
    output[good] = (values[:, good] * weights[:, good, None]).sum(0)
    result = pd.DataFrame(output, columns=COLS).round(3)
    result.insert(0, "image_id", ids)
    if not np.isfinite(result[list(COLS)]).all().all():
        raise ValueError("Non-finite final predictions")
    for col, limits in correction.RANGES.items():
        if not result[col].between(*limits).all():
            raise ValueError(f"Out-of-range final predictions: {col}")
    audit = pd.DataFrame([
        {"image_id": image_id, "head": tag, "ok": bool(flags[h, i]),
         "base_weight": float(BASE[h]), "normalized_weight": float(weights[h, i]),
         "all_new_heads_failed": bool(fallback[i]), "valid_heads": int(flags[:, i].sum()),
         "fallback": "prior_failure_aware_triplet" if fallback[i] else "none"}
        for i, image_id in enumerate(ids) for h, tag in enumerate(TAGS)
    ])
    return result, audit, fallback


def build(source_path, output_path):
    started = time.perf_counter()
    source_dir = Path(__file__).resolve().parent
    source_names = ["build_competition_candidate.py", "insertion_correction.py", "endpoint_geometry.py", "vendor_geometry.py", "vendor_texture.py"]
    if (source_dir / "competition_build_protocol.json").is_file():
        source_names.append("competition_build_protocol.json")
    sources = {name: sha256(source_dir / name) for name in source_names}
    source = find_source(source_path)
    output = Path(output_path).resolve()
    if output == source or source in output.parents:
        raise ValueError("Output must be outside the read-only saved-output source")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty experimental output directory")
    output.mkdir(parents=True, exist_ok=True)
    audit_dir = output / "audit"; audit_dir.mkdir()
    identity_path = source / "cache_identity.json"
    identity = json.loads(identity_path.read_text())
    prior_path = source / "failure_aware" / "submission.csv"
    has_failure_aware = prior_path.is_file()
    if not has_failure_aware:
        prior_path = source / "submission.csv"
    prior = pd.read_csv(prior_path)
    if list(prior.columns) != ["image_id", *COLS] or len(prior) != 309 or not prior.image_id.is_unique or prior.image_id.isna().any():
        raise ValueError("Prior must contain the exact 309 unique image rows and target columns")
    if not np.isfinite(prior[list(COLS)].to_numpy()).all():
        raise ValueError("Non-finite prior prediction")
    prior_receipt_path = prior_path.parent / ("receipt.json" if has_failure_aware else "run_receipt.json")
    prior_receipt = json.loads(prior_receipt_path.read_text())
    if prior_receipt["candidate_sha256"] != sha256(prior_path):
        raise ValueError("Prior candidate does not match its receipt")
    ids = prior.image_id.tolist()
    identity_ids = [row["image_id"] for row in identity["images"]]
    if set(identity_ids) != set(ids) or len(identity_ids) != len(ids):
        raise ValueError("Cache identity image manifest mismatch")
    scale_path = source / "audit" / "scale_calibration.csv"
    scales = checked_table(scale_path, ids, ["image_id", "sx", "sy", "layout"])
    if not np.isfinite(scales[["sx", "sy"]]).all().all() or (scales[["sx", "sy"]] <= 0).any().any():
        raise ValueError("Invalid physical scales")
    inputs = {str(p.relative_to(source)): sha256(p) for p in [identity_path, prior_path, prior_receipt_path, scale_path]}
    prior_tables = {}
    for tag in TAGS:
        old_path = source / "audit" / f"supported_{tag}_measurements.csv"
        relative = str(old_path.relative_to(source))
        inputs[relative] = sha256(old_path)
        if has_failure_aware and prior_receipt.get("input_sha256", {}).get(relative) != inputs[relative]:
            raise ValueError(f"Prior failure-aware receipt does not bind this head table: {relative}")
        old = checked_table(old_path, ids, ["image_id", "pa_tex", "run", "ok"])
        if old.ok.isna().any() or not all(isinstance(value, (bool, np.bool_)) for value in old.ok):
            raise ValueError(f"Invalid prior boolean ok flag: {relative}")
        prior_tables[tag] = old
    heads, map_hashes = {}, []
    reference_runs = None
    diagnostics = {}
    for tag in TAGS:
        old = prior_tables[tag]
        if old.run.isna().any():
            raise ValueError("Missing original content-group identifier")
        if reference_runs is None:
            reference_runs = old.run.to_numpy()
        elif not np.array_equal(reference_runs, old.run.to_numpy()):
            raise ValueError("Original grouping differs across heads")
        rows, segments, statuses = [], [], Counter()
        for i, image_id in enumerate(ids):
            map_path = source / "cache" / tag / (image_id + ".npz")
            with np.load(map_path, allow_pickle=False) as z:
                apo = z["apo"].astype(np.float32)
                fasc = z["fasc"].astype(np.float32)
                shape = z["shape"].astype(int)
            if apo.ndim != 2 or fasc.shape != apo.shape or shape.shape != (2,) or (shape <= 0).any():
                raise ValueError(f"Invalid map shape: {map_path}")
            if not all(np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1 for a in (apo, fasc)):
                raise ValueError(f"Invalid map probabilities: {map_path}")
            map_hashes.append({"path": str(map_path.relative_to(source)), "sha256": sha256(map_path)})
            height, width = apo.shape
            sx, sy = float(scales.loc[i, "sx"]), float(scales.loc[i, "sy"])
            measured = correction.measure(apo, fasc, mx=(shape[1] / width) / sx, my=(shape[0] / height) / sy)
            segments.append(measured["segs"])
            diag = measured["correction_diagnostics"]
            status = diag.get("status", "unknown")
            statuses[status] += 1
            rows.append({"image_id": image_id, **{col: measured[col] for col in COLS},
                "ok": bool(measured["ok"]), "prior_ok": bool(old.loc[i, "ok"]),
                "layout": scales.loc[i, "layout"], "geometry_status": status,
                "extracted_segments": diag.get("extracted_segment_count", 0),
                "retained_segments": diag.get("retained_segment_count", 0),
                "rejection_reasons": json.dumps(diag.get("rejection_reasons", {}), sort_keys=True)})
            if (i + 1) % 50 == 0:
                print(f"{tag}: geometry {i + 1}/309", flush=True)
        head = fuse_head(rows, segments, old)
        head.to_csv(audit_dir / f"experimental_{tag}_measurements.csv", index=False)
        heads[tag] = head
        diagnostics[tag] = dict(statuses)
    candidate, head_audit, fallback = combine_heads(heads, prior, allow_prior_fallback=has_failure_aware)
    head_audit.to_csv(output / "head_weight_audit.csv", index=False)
    pd.DataFrame(map_hashes).to_csv(output / "map_sha256.csv", index=False)
    delta = candidate[list(COLS)] - prior[list(COLS)]
    changes = candidate[["image_id"]].copy()
    changes["all_new_heads_failed"] = fallback
    changes["fallback"] = np.where(fallback, "prior_failure_aware_triplet", "none")
    for col in COLS:
        changes["prior_" + col] = prior[col]
        changes["experimental_" + col] = candidate[col]
        changes["delta_" + col] = delta[col].round(3)
    changes.to_csv(output / "changes_vs_prior.csv", index=False)
    candidate.to_csv(output / "experimental_submission.csv", index=False)
    for name, initial_hash in sources.items():
        if sha256(source_dir / name) != initial_hash:
            raise RuntimeError(f"Executable source/protocol changed during processing: {name}")
    for name, initial_hash in inputs.items():
        if sha256(source / name) != initial_hash:
            raise RuntimeError(f"Input metadata changed during processing: {name}")
    for row in map_hashes:
        if sha256(source / row["path"]) != row["sha256"]:
            raise RuntimeError(f"Probability map changed during processing: {row['path']}")
    receipt = {"status": STATUS, "promotion_allowed": False, "automatically_submitted": False,
        "method": "insertion_geometry_v2_with_frozen_upstream_calibration_texture_and_grouping",
        "rows": len(candidate), "columns": list(candidate.columns),
        "candidate_sha256": sha256(output / "experimental_submission.csv"), "prior_candidate_sha256": sha256(prior_path),
        "input_sha256": inputs, "source_sha256": sources,
        "sources_and_inputs_unchanged_at_completion": True,
        "prior_head_tables_bound_to_failure_aware_receipt": has_failure_aware,
        "checkpoint_sha256": identity.get("checkpoint_hashes", {}),
        "checkpoint_hash_provenance": "Copied from original inference cache identity; checkpoints are not loaded or rehashed in this geometry-only run.",
        "map_hash_manifest_sha256": sha256(output / "map_sha256.csv"), "map_files_read_checked_and_hashed": len(map_hashes),
        "map_hash_provenance": "Observed file SHA256 recorded during this run, with finite/range/shape checks; no previously recorded map checksum was available for comparison.",
        "valid_heads_by_model": {tag: int(frame.ok.sum()) for tag, frame in heads.items()},
        "all_new_heads_failed_images": candidate.loc[fallback, "image_id"].tolist(),
        "one_valid_head_images": int((head_audit.groupby("image_id").valid_heads.first() == 1).sum()),
        "changed_rows": int(delta.ne(0).any(axis=1).sum()), "mean_absolute_change": delta.abs().mean().to_dict(),
        "geometry_statuses": diagnostics, "same_weights_for_all_targets": True,
        "base_weights": dict(zip(TAGS, BASE.tolist())), "ground_truth_used": False,
        "new_model_training": False, "new_model_inference": False,
        "texture_rule": "Reuse each prior head pa_tex, pool by exact original run median, blend PA 0.5; preserve FL sine blend 0.25 and original clips.",
        "physical_scale_limit": "Original sx/sy retained for isolated ablation. Upstream anisotropy conflicts with organizer aspect-ratio-preservation statement; not established protocol fidelity.",
        "development_decision": "All new geometry variants failed the overall static development comparison. V2 had a small dynamic-sequence gain, insufficient for promotion; references were reused, not independent validation.",
        "accuracy_claim": "Experimental output only. No competition score, generalization gain or accuracy improvement is claimed.",
        "fallback_rule": "If all new heads fail, retain the prior failure-aware whole triplet and flag it. One-valid-head cases have no ensemble agreement evidence.",
        "elapsed_seconds": time.perf_counter() - started}
    receipt["output_sha256"] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/umud_innovation_experimental"))
    args = parser.parse_args()
    print(json.dumps(build(args.source_root, args.output_dir), indent=2))
