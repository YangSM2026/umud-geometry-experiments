"""Run attributed, frozen public checkpoint pair on public benchmark images.

The inference contract (384x576 grayscale, horizontal-flip probability average)
is retained from watehell/prvsiyan; model authors retain all credit. No training.
"""
from pathlib import Path
import argparse, hashlib, json, time
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

EXPECTED = {
    "apo": "59bd01b25a3fcdfd160d82fc1532bbd59ca47a70b5b128cd2dde9dc0a1ee5a19",
    "fasc": "726aedc0914d15f9a60d2e2d319af8fa07cb857ecdeca0b2dcc3bc27d463b97c",
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(20261113)
    torch.use_deterministic_algorithms(True)
    static = json.loads((root / "static_references.json").read_text())["rows"]
    # Only manually annotated even frames; odd reference frames are interpolated.
    dynamic = json.loads((root / "dynamic_references.json").read_text())
    jobs = [("static", row["image_id"], root / row["image_path"]) for row in static]
    for row in dynamic["rows"]:
        if row["frame"] % 2 == 0:
            jobs.append(("dynamic", row["image_id"], root / row["image_path"]))
    started = time.time()
    manifest = {"checkpoint_sha256": {}, "images": [], "method": "frozen original pair; grayscale resize; horizontal flip TTA", "training": False}
    models = []
    for task in ("apo", "fasc"):
        checkpoint = root / "checkpoints" / f"{task}_resnet34_384x576_f0.pt"
        digest = sha(checkpoint)
        if digest != EXPECTED[task]:
            raise ValueError(f"Unexpected checkpoint: {task}")
        manifest["checkpoint_sha256"][task] = digest
        state = torch.load(checkpoint, weights_only=True, map_location="cpu")
        model = smp.Unet(state["encoder"], encoder_weights=None, in_channels=1, classes=1)
        model.load_state_dict({k:v.float() for k,v in state["state"].items()})
        models.append((model.eval(), tuple(state["size"])))
    contract = {"checkpoint_sha256": manifest["checkpoint_sha256"],
        "source_sha256": sha(__file__), "sizes":[list(size) for _,size in models],
        "torch":torch.__version__, "smp":smp.__version__, "numpy":np.__version__,
        "opencv":cv2.__version__, "method":manifest["method"]}
    contract_hash=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    manifest.update(contract=contract,contract_sha256=contract_hash)
    for index, (dataset, image_id, path) in enumerate(jobs):
        target = output / dataset / (image_id + ".npz")
        target.parent.mkdir(exist_ok=True)
        digest = sha(path)
        if target.exists():
            cached = np.load(target, allow_pickle=False)
            if (str(cached["image_sha256"]) != digest or "contract_sha256" not in cached
                    or str(cached["contract_sha256"]) != contract_hash):
                raise ValueError("Stale inference/image cache; use a fresh output directory")
        else:
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError(f"Unreadable image {path}")
            maps = []
            with torch.inference_mode():
                for model, (h,w) in models:
                    tensor = torch.from_numpy(cv2.resize(gray,(w,h),interpolation=cv2.INTER_AREA).astype(np.float32)/255)[None,None]
                    pred = (torch.sigmoid(model(tensor)) + torch.sigmoid(model(tensor.flip(-1))).flip(-1))/2
                    maps.append(pred[0,0].numpy().astype(np.float16))
            np.savez_compressed(target, apo=maps[0], fasc=maps[1], shape=np.array(gray.shape), image_sha256=np.array(digest),contract_sha256=np.array(contract_hash))
        manifest["images"].append({"dataset":dataset,"image_id":image_id,"image_sha256":digest,"map_sha256":sha(target)})
        print(f"{index+1}/{len(jobs)} {dataset}/{image_id} elapsed={time.time()-started:.1f}s",flush=True)
    manifest.update(seconds=time.time()-started, torch=torch.__version__, smp=smp.__version__, numpy=np.__version__, opencv=cv2.__version__)
    (output/"inference_receipt.json").write_text(json.dumps(manifest,indent=2)+"\n")

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("root"); parser.add_argument("output")
    args=parser.parse_args(); run(args.root,args.output)
