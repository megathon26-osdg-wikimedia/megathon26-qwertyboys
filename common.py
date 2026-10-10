"""Shared helpers: CLIP loading (GPU if available, else CPU), embedding functions, small utilities."""
import json, os, re
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from config import *

DATA = Path("data"); IMG_DIR = DATA / "images"
DATA.mkdir(exist_ok=True); IMG_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"      # GPU when available, otherwise CPU
UA = {"User-Agent": f"CommonsCategorySuggester/0.1 (hackathon project; contact: {CONTACT_EMAIL})"}

_model = _proc = None


def load_clip():
    global _model, _proc
    if _model is None:
        if DEVICE == "cpu":
            torch.set_num_threads(os.cpu_count() or 4)
        print(f"[clip] loading {MODEL_NAME} on {DEVICE}")
        _model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
        _proc = CLIPProcessor.from_pretrained(MODEL_NAME)
    return _model, _proc


def _tensor(f):                       # newer transformers may return an output object
    return f if torch.is_tensor(f) else f.pooler_output


def _norm(f):
    return torch.nn.functional.normalize(f, dim=-1).cpu().numpy().astype("float32")


@torch.inference_mode()
def embed_images(items, bs=None, progress=False):
    """items: file paths or PIL images -> (N, d) L2-normalised float32 array."""
    model, proc = load_clip(); bs = bs or BATCH_SIZE; out = []
    batches = range(0, len(items), bs)
    if progress:
        batches = tqdm(batches, desc="embedding images", unit="batch")
    for i in batches:
        imgs = [x.convert("RGB") if isinstance(x, Image.Image) else Image.open(x).convert("RGB")
                for x in items[i:i + bs]]
        inp = proc(images=imgs, return_tensors="pt").to(DEVICE)
        out.append(_norm(_tensor(model.get_image_features(**inp))))
    return np.concatenate(out) if out else np.zeros((0, 512), "float32")


@torch.inference_mode()
def embed_texts(texts, bs=256):
    model, proc = load_clip(); out = []
    for i in range(0, len(texts), bs):
        inp = proc(text=texts[i:i + bs], padding=True, truncation=True,
                   max_length=77, return_tensors="pt").to(DEVICE)
        out.append(_norm(_tensor(model.get_text_features(**inp))))
    return np.concatenate(out) if out else np.zeros((0, 512), "float32")


def clean_title(t):
    """'File:Taj_Mahal_at-dawn.jpg' -> 'Taj Mahal at dawn'"""
    t = re.sub(r"^File:", "", t)
    t = re.sub(r"\.\w{3,4}$", "", t)
    return re.sub(r"[_\-]+", " ", t).strip()


def load_meta():
    with open(DATA / "meta.json") as f:
        return json.load(f)
