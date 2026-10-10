"""Run once: verifies GPU/CPU choice, CLIP loading, speed, FAISS and the other libraries."""
import time
from common import *
import faiss, sklearn, gradio, joblib

print("device:", DEVICE, "|", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU only")
imgs = [Image.new("RGB", (320, 240), (120, 180, 60))] * 32
embed_images(imgs[:4])                                   # warm-up
t = time.perf_counter(); z = embed_images(imgs); dt = time.perf_counter() - t
print(f"embedding shape {z.shape} | 32 images in {dt:.2f}s ({32 / dt:.1f} img/s)")
print("est. time for 6,000 images: %.1f min" % (6000 / (32 / dt) / 60))
zt = embed_texts(["a photo of a cat"])
proj = load_clip()[0].config.projection_dim
assert z.shape[1] == zt.shape[1] == proj, f"embedding dims differ: image {z.shape[1]}, text {zt.shape[1]}, expected {proj}"
print("image/text embedding dims match:", proj)
idx = faiss.IndexFlatIP(z.shape[1]); idx.add(z); print("faiss ok:", idx.ntotal)
print("all good")
