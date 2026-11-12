"""Discard invalid model heads and renormalize the existing UMUD ensemble.

This reads model-measurement tables, not labels or another competitor's predictions.
No new inference, training, parameter fitting or leaderboard feedback is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TAGS = ("test", "ens_f0", "ens_f1", "ens_f2", "ens_f3", "ens_f4")
TARGETS = ("pa_deg", "fl_mm", "mt_mm")
BASE_WEIGHTS = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float)
RANGES = {"pa_deg": (5, 45), "fl_mm": (30, 200), "mt_mm": (10, 50)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_validity(series: pd.Series, head: str) -> np.ndarray:
    """Parse explicit booleans; never interpret the string 'False' as truthy."""
    output = []
    for value in series:
        if isinstance(value, (bool, np.bool_)):
            output.append(bool(value))
        elif isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            output.append(value.strip().lower() == "true")
        else:
            raise ValueError(f"{head}: invalid or missing ok flag: {value!r}")
    return np.asarray(output, dtype=bool)


def combine_heads(head_tables: dict[str, pd.DataFrame], output_ids: list[str]):
    """Return candidate, original ensemble, per-head weights and invalid-head counts.

    A single validity mask and weight vector is used for all three targets.
    All-valid rows use the original arithmetic expression to preserve rounding.
    """
    if set(head_tables) != set(TAGS):
        raise ValueError("Exactly the original pair and all five fold heads are required")
    if not output_ids or len(set(output_ids)) != len(output_ids):
        raise ValueError("Output identifiers must be nonempty and unique")
    values, flags = [], []
    for tag in TAGS:
        frame = head_tables[tag]
        required = {"image_id", "ok", *TARGETS}
        if not required.issubset(frame.columns):
            raise ValueError(f"{tag}: missing required columns")
        if frame.image_id.isna().any() or frame.image_id.duplicated().any():
            raise ValueError(f"{tag}: missing or duplicate image IDs")
        if set(frame.image_id) != set(output_ids):
            raise ValueError(f"{tag}: image ID mismatch")
        frame = frame.set_index("image_id").loc[output_ids]
        numeric = frame[list(TARGETS)].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"{tag}: non-finite target value")
        values.append(numeric)
        flags.append(parse_validity(frame.ok, tag))
    values = np.stack(values)  # head, image, target
    valid = np.stack(flags)   # head, image
    active = BASE_WEIGHTS[:, None] * valid
    totals = active.sum(axis=0)
    if (totals <= 0).any():
        failed = [output_ids[i] for i in np.flatnonzero(totals <= 0)]
        raise ValueError(f"No valid model head for: {failed}")
    normalized = active / totals[None, :]
    if not np.allclose(normalized.sum(axis=0), 1., atol=1e-12):
        raise AssertionError("Normalized weights do not sum to one")
    if np.any(normalized[~valid] != 0):
        raise AssertionError("An invalid head retained a positive weight")

    baseline = .5 * values[0] + .5 * values[1:].mean(axis=0)
    combined = baseline.copy()
    affected = (~valid).any(axis=0)
    combined[affected] = (values[:, affected, :] * normalized[:, affected, None]).sum(axis=0)
    if not np.isfinite(combined).all():
        raise ValueError("Non-finite final ensemble")

    def to_frame(array):
        result = pd.DataFrame(array, columns=TARGETS).round(3)
        result.insert(0, "image_id", output_ids)
        return result

    candidate, original = to_frame(combined), to_frame(baseline)
    for column, (low, high) in RANGES.items():
        if not candidate[column].between(low, high).all():
            raise ValueError(f"Final {column} exceeds inherited output range")
    audit = pd.DataFrame([
        {"image_id": image_id, "head": tag, "ok": bool(valid[h, i]),
         "base_weight": float(BASE_WEIGHTS[h]), "active_weight": float(active[h, i]),
         "normalized_weight": float(normalized[h, i]), "invalid_heads_for_image": int((~valid[:, i]).sum())}
        for i, image_id in enumerate(output_ids) for h, tag in enumerate(TAGS)
    ])
    return candidate, original, audit, (~valid).sum(axis=0)


def run(source_dir: str | Path, output_dir: str | Path) -> dict:
    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    if output == source or output == source / "audit":
        raise ValueError("Output must be a separate directory")
    final_names = ("submission.csv", "head_weight_audit.csv", "changed_rows.csv", "receipt.json")
    if any((output / name).exists() for name in final_names):
        raise FileExistsError("Refusing to overwrite existing postprocess outputs")

    parent_path, parent_receipt_path = source / "submission.csv", source / "run_receipt.json"
    parent_hash = sha256_file(parent_path)
    parent_receipt = json.loads(parent_receipt_path.read_text())
    if parent_receipt.get("candidate_sha256") != parent_hash:
        raise ValueError("Parent candidate hash does not match its completed-run receipt")
    if parent_receipt.get("status") != "inference_completed_not_yet_scored":
        raise ValueError("Parent run is not marked as completed image inference")
    parent = pd.read_csv(parent_path)
    if list(parent.columns) != ["image_id", *TARGETS] or parent.image_id.duplicated().any():
        raise ValueError("Parent candidate schema or identifiers are invalid")
    if len(parent) != parent_receipt.get("rows") or not np.isfinite(parent[list(TARGETS)].to_numpy()).all():
        raise ValueError("Parent candidate rows or numeric values do not match its receipt")
    paths = {tag: source / "audit" / f"supported_{tag}_measurements.csv" for tag in TAGS}
    tables = {tag: pd.read_csv(path) for tag, path in paths.items()}
    candidate, reconstructed, audit, failed_counts = combine_heads(tables, parent.image_id.tolist())
    if not np.array_equal(reconstructed[list(TARGETS)].to_numpy(), parent[list(TARGETS)].to_numpy()):
        raise ValueError("Supported-head tables do not exactly reproduce the rounded parent candidate")
    delta = candidate[list(TARGETS)] - parent[list(TARGETS)]
    changed_mask = delta.ne(0).any(axis=1)
    affected_mask = failed_counts > 0
    if changed_mask[~affected_mask].any():
        raise AssertionError("An all-valid row changed")
    changed = candidate.loc[changed_mask, ["image_id"]].copy()
    changed["invalid_head_count"] = failed_counts[changed_mask.to_numpy()]
    for column in TARGETS:
        changed["parent_" + column] = parent.loc[changed_mask, column]
        changed["new_" + column] = candidate.loc[changed_mask, column]
        changed["delta_" + column] = delta.loc[changed_mask, column].round(3)

    output.mkdir(parents=True, exist_ok=True)
    pending = output / "pending"
    pending.mkdir(exist_ok=True)
    candidate.to_csv(pending / "submission.csv", index=False, lineterminator="\n")
    audit.to_csv(pending / "head_weight_audit.csv", index=False, lineterminator="\n")
    changed.to_csv(pending / "changed_rows.csv", index=False, lineterminator="\n")
    input_files = {str(path.relative_to(source)): sha256_file(path) for path in paths.values()}
    input_files["submission.csv"] = parent_hash
    input_files["run_receipt.json"] = sha256_file(parent_receipt_path)
    receipt = {
        "status": "postprocess_completed_not_yet_scored",
        "method": "validity-weighted supported-depth ensemble",
        "rows": len(candidate), "columns": list(candidate.columns),
        "parent_candidate_sha256": parent_hash,
        "candidate_sha256": sha256_file(pending / "submission.csv"),
        "input_sha256": input_files,
        "output_sha256": {name: sha256_file(pending / name) for name in final_names if name != "receipt.json"},
        "validity_rule": "A model head is usable only when its existing ok flag is explicitly True. False receives zero weight for all three targets; remaining base weights are renormalized per image. All-invalid images abort.",
        "base_weights": dict(zip(TAGS, BASE_WEIGHTS.tolist())),
        "same_weights_for_all_targets": True,
        "affected_images": int(affected_mask.sum()), "changed_rows": int(changed_mask.sum()),
        "unchanged_rows": int((~changed_mask).sum()),
        "one_valid_head_images": int((failed_counts == len(TAGS) - 1).sum()),
        "invalid_heads_by_model": {tag: int((~parse_validity(tables[tag].ok, tag)).sum()) for tag in TAGS},
        "parent_exactly_reconstructed": True,
        "all_valid_rows_unchanged": True,
        "output_order_matches_parent": candidate.image_id.tolist() == parent.image_id.tolist(),
        "ground_truth_used": False, "new_model_inference": False,
        "accuracy_claim": "No local accuracy or leaderboard improvement is claimed; no target ground truth was used.",
        "inherited_limits": "Retains parent geometry, calibration assumptions and public-leaderboard-tuned constants; validity correction does not independently validate them. Images supported by only one valid head have no ensemble agreement evidence and must not be treated as confident predictions.",
        "receipt_hash_note": "Receipt is the final completion marker; its own hash is not recursively included.",
    }
    (pending / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for name in final_names:  # receipt is promoted last as the commit marker
        if (output / name).exists():
            raise FileExistsError(f"Refusing to replace existing output: {name}")
        (pending / name).rename(output / name)
    pending.rmdir()
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.source_dir, args.output_dir), indent=2))
