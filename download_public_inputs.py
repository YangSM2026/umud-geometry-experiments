"""Fetch checksum-pinned public research inputs; no credentials or private data.

Retain attribution: UMUD OSF data CC BY4.0; watehell model weights
CC BY-NC-SA4.0. No training is performed. Uses only published download URLs.
"""
from pathlib import Path
import argparse, hashlib, shutil, urllib.request, zipfile

SOURCES = [
 ("static", "benchmark_dataset_architecture_v0.1.0.zip", "https://osf.io/download/hb5xj/?direct=1", "8d4c8093a737a64c39360864928f1ae65ddd2f70497dbe50ff82810173f2d48f"),
 ("dynamic", "benchmark_dataset_architecture_dynamic_v0.2.0.zip", "https://osf.io/download/p8r2u/?direct=1", "02332b8355347534f8727e718e9185267028ac6fb01fbd3941b0a7edadeb84d5"),
 ("checkpoints", "watehell_v1.zip", "https://www.kaggle.com/api/v1/datasets/download/watehell/umud-r34-unet-weights?datasetVersionNumber=1", "cc20e26881e11843e4eff3adc4befd74d34c9c283e092692880929bacecff1b1"),
]

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def fetch(root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    for folder,name,url,expected in SOURCES:
        archive=root/name
        if not archive.exists():
            partial=archive.with_suffix(".partial")
            with urllib.request.urlopen(url,timeout=60) as response,partial.open("wb") as target:
                shutil.copyfileobj(response,target)
            if sha(partial)!=expected:
                raise ValueError(f"Checksum mismatch: {name}; partial file retained for inspection")
            partial.replace(archive)
        if sha(archive)!=expected:raise ValueError(f"Checksum mismatch: {name}")
        destination=root/folder;destination.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                path=(destination/member.filename).resolve()
                if not path.is_relative_to(destination.resolve()):raise ValueError("Unsafe archive path")
                if (member.external_attr>>16)&0o170000==0o120000:raise ValueError("Archive symlink rejected")
                if member.is_dir():path.mkdir(parents=True,exist_ok=True);continue
                path.parent.mkdir(parents=True,exist_ok=True)
                data=z.read(member)
                if path.exists() and path.read_bytes()!=data:raise FileExistsError(f"Existing input differs: {path}")
                if not path.exists():path.write_bytes(data)
        print("verified",name,expected)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("destination",nargs="?",default="research_inputs")
    fetch(p.parse_args().destination)
