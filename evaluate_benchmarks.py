"""Frozen paired development comparison. Test-set outcomes are never accessed."""
from pathlib import Path
import argparse, hashlib, json
import cv2
import numpy as np
import pandas as pd
import decoder
import vendor_geometry as old
import vendor_texture

TARGETS=("pa_deg","fl_mm","mt_mm")
TOLERANCES=np.array([6.,12.,3.])
RANGES={"pa_deg":(5,45),"fl_mm":(30,200),"mt_mm":(10,50)}

def baseline(a,f,mx,my,gray,scale):
    m=old.measure(a,f,mx,my,mt_mode="centre_minus_half",fl_mode="extend_centre",fl_factor=.9,pa_at=.9)
    if "apo" in m:
        texture=vendor_texture.texture_angle(gray,scale,scale,m["apo"])[1]
        if np.isfinite(texture):
            m["pa_deg"]=(m["pa_deg"]+texture)/2
    m["fl_mm"]=.75*m["fl_mm"]+.25*m["mt_mm"]/np.sin(np.radians(np.clip(m["pa_deg"],5,45)))
    return {k:float(np.clip(m[k],*RANGES[k])) for k in TARGETS}|{"valid":bool(m["ok"])}

def metric(pred, truth):
    e=np.asarray(pred)-np.asarray(truth)
    return {"n":len(e),"MAE":dict(zip(TARGETS,np.abs(e).mean(0).tolist())),
            "RMSE":dict(zip(TARGETS,np.sqrt((e*e).mean(0)).tolist())),
            "bias":dict(zip(TARGETS,e.mean(0).tolist())),
            "primary_score":float((np.abs(e).mean(0)/TOLERANCES).mean()),
            "score_with_tiebreakers":float((np.abs(e).mean(0)/TOLERANCES).mean()+1e-6*(np.median(np.abs(e),axis=0)/TOLERANCES).mean()+1e-9*(np.sqrt((e*e).mean(0))/TOLERANCES).mean())}

def run(root,output,maps_dir="maps_verified"):
    root,output=Path(root),Path(output)
    if (output/"benchmark_summary.json").exists():
        raise FileExistsError("Use a fresh output directory to retain experiment history")
    output.mkdir(parents=True,exist_ok=True)
    static=json.loads((root/"static_references.json").read_text())["rows"]
    dynamic=json.loads((root/"dynamic_references.json").read_text())["rows"]
    receipt_path=root/maps_dir/"inference_receipt.json"
    receipt=json.loads(receipt_path.read_text())
    if "contract_sha256" not in receipt:
        raise ValueError("Inference cache has no checkpoint/source contract")
    map_records={(r["dataset"],r["image_id"]):r for r in receipt["images"]}
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    rows=[]; diagnostics=[]
    for dataset, refs in [("static",static),("dynamic",dynamic)]:
        for row in refs:
            if dataset=="static" and row["image_id"]=="im_07_arch.tif":
                continue # exact training overlap, excluded BEFORE evaluation
            if dataset=="dynamic" and not row["directly_manually_annotated"]:
                continue
            map_path=root/maps_dir/dataset/(row["image_id"]+".npz")
            record=map_records[dataset,row["image_id"]]
            if digest(map_path)!=record["map_sha256"] or digest(root/row["image_path"])!=record["image_sha256"]:
                raise ValueError("Benchmark image/map differs from completed inference receipt")
            z=np.load(map_path,allow_pickle=False)
            if str(z["contract_sha256"])!=receipt["contract_sha256"] or str(z["image_sha256"])!=record["image_sha256"]:
                raise ValueError("Cache identity differs from inference receipt")
            a,f=z["apo"].astype(np.float32),z["fasc"].astype(np.float32)
            h0,w0=z["shape"]; h,w=a.shape
            scale=row["scale_pixels_per_mm"] if dataset=="static" else row["scale_pixels_per_mm_from_visible_ruler"]
            mx,my=w0/w/scale,h0/h/scale
            gray=cv2.imread(str(root/row["image_path"]),cv2.IMREAD_GRAYSCALE)
            control=baseline(a,f,mx,my,gray,scale)
            if dataset=="static":
                truth=dict(zip(TARGETS,[row["mean_complete_six_R1_R2_R3_R4_R6_R7"][x] for x in ("PA","FL","MT")]))
                sensitivity=dict(zip(TARGETS,[row["mean_available_R1_to_R7"][x] for x in ("PA","FL","MT")]))
            else:
                truth={"pa_deg":row["pa_deg"],"fl_mm":row["fl_mm"],"mt_mm":None}; sensitivity=truth
            for name, robust in [("endpoint",False),("endpoint_huber",True)]:
                measured=decoder.measure(a,f,mx,my,robust_boundary=robust)
                # Entire triplet falls back together; count and expose every failure.
                pred=measured if measured["valid"] else control
                rows.append({"dataset":dataset,"image_id":row["image_id"],"variant":name,
                    "candidate_valid":measured["valid"],"fallback_used":not measured["valid"],"baseline_valid":control["valid"],
                    **{"truth_"+k:truth[k] for k in TARGETS},**{"sensitivity_"+k:sensitivity[k] for k in TARGETS},
                    **{"baseline_"+k:control[k] for k in TARGETS},**{"prediction_"+k:pred[k] for k in TARGETS},
                    "status":measured["status"]})
                diagnostics.append({"dataset":dataset,"image_id":row["image_id"],"variant":name,**measured})
    table=pd.DataFrame(rows); table.to_csv(output/"benchmark_predictions.csv",index=False)
    (output/"geometry_diagnostics.json").write_text(json.dumps(diagnostics,indent=2,allow_nan=False))
    summary={"protocol":json.loads((Path(__file__).parent/"protocol.json").read_text()),"results":{},"training":False,
             "label_warning":"Static is development-only with incomplete donor/source independence; dynamic is one dependent video. No competition score was measured.",
             "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
             "protocol_sha256":hashlib.sha256((Path(__file__).parent/"protocol.json").read_bytes()).hexdigest(),
             "input_sha256":{str(p.relative_to(root)):digest(p) for p in [root/"static_references.json",root/"dynamic_references.json",receipt_path]},
             "baseline_description":"supported-depth original public pair; per-image, no temporal pooling; not exact original V12 or six-head competition pipeline"}
    for (dataset,name),data in table.groupby(["dataset","variant"]):
        result={"n":len(data),"valid_candidates":int(data.candidate_valid.sum()),"fallback_count":int(data.fallback_used.sum()),"baseline_valid":int(data.baseline_valid.sum())}
        if dataset=="static":
            pred=data[["prediction_"+k for k in TARGETS]].to_numpy(float)
            base=data[["baseline_"+k for k in TARGETS]].to_numpy(float)
            truth=data[["truth_"+k for k in TARGETS]].to_numpy(float)
            result.update(baseline=metric(base,truth),candidate_with_explicit_fallback=metric(pred,truth))
            result["available_rater_sensitivity"]={"baseline":metric(base,data[["sensitivity_"+k for k in TARGETS]]),"candidate":metric(pred,data[["sensitivity_"+k for k in TARGETS]])}
            mask=(data.candidate_valid & data.baseline_valid).to_numpy(bool)
            result["matched_valid_cases"]={"baseline":metric(base[mask],truth[mask]),"candidate":metric(pred[mask],truth[mask])} if mask.any() else None
            delta=((abs(pred-truth)-abs(base-truth))/TOLERANCES).mean(1)
            rng=np.random.default_rng(20261113)
            boots=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
            result["paired_primary_delta_candidate_minus_baseline"]={"mean":float(delta.mean()),"descriptive_image_bootstrap_97_5pct_interval":np.quantile(boots,[.0125,.9875]).tolist(),"replicates":10000,"caveat":"Bonferroni2-comparison percentile interval, image-level; unknown subject dependence and upstream benchmark reuse make it unsuitable as independent confirmatory inference."}
        else:
            result["metrics"]={}
            for target in ("pa_deg","fl_mm"):
                truth=data["truth_"+target].to_numpy(float)
                result["metrics"][target]={m:float(np.mean(abs(data[prefix+target].to_numpy(float)-truth))) for m,prefix in [("baseline_MAE","baseline_"),("candidate_MAE","prediction_")]}
            result["independent_groups"]=1
            result["confidence_interval"]=None
        summary["results"][dataset+"/"+name]=result
    (output/"benchmark_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps(summary["results"],indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root");p.add_argument("output");p.add_argument("--maps-dir",default="maps_verified");a=p.parse_args();run(a.root,a.output,a.maps_dir)
