"""STAGE 7 - Efficiency numbers for the report: latency (p50/p95), throughput, index memory.
Run: python 07_benchmark.py
"""
import time
from suggester import Suggester
from common import *

S = Suggester()
test = [i for i, r in enumerate(S.recs) if r["split"] == "test" and S.keep[i]][:100]
paths = [IMG_DIR / f"{S.recs[i]['pageid']}.jpg" for i in test]
embed_images(paths[:4])                                   # warm-up
lat_embed, lat_rank = [], []
for p in paths:
    img = Image.open(p).convert("RGB")
    t = time.perf_counter(); embed_images([img]); lat_embed.append(time.perf_counter() - t)
for i in test:
    t = time.perf_counter(); S.rank(S.Z[i], S.ZT[i], k=5); lat_rank.append(time.perf_counter() - t)
t = time.perf_counter(); embed_images(paths, progress=False); thr = len(paths) / (time.perf_counter() - t)
pc = lambda a, q: 1000 * float(np.percentile(a, q))
print(f"device: {DEVICE} | model: {MODEL_NAME}")
print(f"image->embedding latency  p50 {pc(lat_embed,50):.0f} ms | p95 {pc(lat_embed,95):.0f} ms")
print(f"kNN+fusion+hierarchy      p50 {pc(lat_rank,50):.2f} ms | p95 {pc(lat_rank,95):.2f} ms")
print(f"batched embedding throughput: {thr:.1f} images/s  (~{thr * 86400:,.0f}/day)")
d = S.Z.shape[1]
print(f"index: {S.index.ntotal} vectors x {d} dims x 4 B = {S.index.ntotal * d * 4 / 1e6:.1f} MB (flat, fp32)")
print(f"scaling maths: 1e8 images x 64 B (product quantisation) = {1e8 * 64 / 1e9:.1f} GB")
