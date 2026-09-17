import json, torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataset import SpectrogramDataset, get_train_transform
with open("data/splits/train.json") as f:
    meta = json.load(f)[:5]
ds = SpectrogramDataset(meta, transform=get_train_transform(224))
x, y = ds[0]
print("Shape:", x.shape)
print("max:", x.max(), "min:", x.min())
