"""STAGE 3 - CLIP embeddings for images (cached), titles and category names; near-duplicate removal.
Output: data/emb.npz, data/emb_cache.npz      Run: python 03_embed.py
Re-running only embeds NEW images; changing MODEL_NAME invalidates the cache automatically.
"""
from common import *

meta = load_meta(); recs = meta["records"]; vocab = meta["vocab"]
cache_path = DATA / "emb_cache.npz"; cache = {}
if cache_path.exists():
    c = np.load(cache_path)
    if str(c["model"]) == MODEL_NAME:
        cache = dict(zip(c["ids"].tolist(), c["Z"]))
    else:
        print("model changed -> ignoring old image-embedding cache")
todo = [r for r in recs if r["pageid"] not in cache]
print(f"{len(recs)} images | {len(cache)} cached | {len(todo)} to embed on {DEVICE}")


def save_cache():
    ids = np.array(list(cache))
    np.savez(cache_path, ids=ids, Z=np.stack([cache[i] for i in ids]), model=np.array(MODEL_NAME))


for s in range(0, len(todo), 512):                 # chunked + checkpointed: safe to interrupt and resume
    chunk = todo[s:s + 512]
    new = embed_images([IMG_DIR / f"{r['pageid']}.jpg" for r in chunk], progress=True)
    for r, z in zip(chunk, new):
        cache[r["pageid"]] = z
    save_cache()

Z = np.stack([cache[r["pageid"]] for r in recs])
ZT = embed_texts([clean_title(r["title"]) for r in recs])          # title/filename text embedding
T = embed_texts([TEXT_PROMPT.format(c) for c in vocab])            # category text prototypes

split = np.array([r["split"] for r in recs])
tr = np.where(split == "train")[0]
keep = np.ones(len(recs), bool)                                    # False => near-duplicate of a train image
for s in range(0, len(recs), 2048):
    sims = Z[s:s + 2048] @ Z[tr].T
    for j in range(sims.shape[0]):
        if split[s + j] != "train" and sims[j].max() > DEDUP_THRESH:
            keep[s + j] = False
print("near-duplicates removed from val/test:", int((~keep).sum()))
np.savez(DATA / "emb.npz", Z=Z, ZT=ZT, T=T, keep=keep)
print("saved data/emb.npz", Z.shape)
