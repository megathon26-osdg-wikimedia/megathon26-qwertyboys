"""Pseudo-label unlabelled images already on disk so the whole image folder can be used.
A label is added only when TWO independent signals agree: kNN vote over labelled images AND CLIP zero-shot text match.
Pseudo-labelled images are TRAIN-only (02_prepare.py keeps them out of val/test), so evaluation stays clean.
   python relabel_by_seed.py 10 150     # first: real labels for every seed with >= 10 images
   python pseudo_label.py 50 150        # then: min images per topic, cap per topic (real labels first)"""
import json, sys, glob, random
from collections import Counter, defaultdict
from common import *

MIN = int(sys.argv[1]) if len(sys.argv) > 1 else 50
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 150
KNN_K, TAU, SHARE, TXT_P = 15, 0.05, 0.6, 0.25
RAW = DATA / "raw.jsonl"

recs = [json.loads(l) for l in open(RAW)]
real = [r for r in recs if not r.get("pseudo")]
real_ids = {r["pageid"] for r in real}
state_ids = set()
for l in open(DATA / "collect_state.jsonl"):
    state_ids |= {f["pageid"] for f in json.loads(l)["files"]}
meta = {}
for fn in sorted(glob.glob(str(DATA / "raw*.jsonl"))):
    if Path(fn).name == "raw.jsonl":
        continue
    for l in open(fn):
        r = json.loads(l); meta.setdefault(r["pageid"], r)
disk = {int(p.stem) for p in IMG_DIR.glob("*.jpg")}
cand = [i for i in sorted(disk) if i not in real_ids and i not in state_ids and i in meta and meta[i].get("timestamp")]
print(f"real labelled: {len(real)} | on disk: {len(disk)} | unlabelled candidates with metadata: {len(cand)}")

cnt = Counter(c for r in real for c in r["cats"])
classes = sorted(c for c, v in cnt.items() if v >= 10)
cidx = {c: i for i, c in enumerate(classes)}
print("target topics (>= 10 real images):", len(classes))

# ---------- embeddings (same cache file 03_embed.py uses, so nothing is embedded twice)
cache_path = DATA / "emb_cache.npz"; cache = {}
if cache_path.exists():
    c = np.load(cache_path)
    if str(c["model"]) == MODEL_NAME:
        cache = dict(zip(c["ids"].tolist(), c["Z"]))


def save_cache():
    ids = np.array(list(cache))
    np.savez(cache_path, ids=ids, Z=np.stack([cache[i] for i in ids]), model=np.array(MODEL_NAME))


def ensure(ids):
    todo = [i for i in ids if i not in cache]
    print(len(todo), "images to embed")
    for s in range(0, len(todo), 512):
        chunk = todo[s:s + 512]
        try:
            Zc = embed_images([IMG_DIR / f"{i}.jpg" for i in chunk], progress=True); ok = chunk
        except Exception:                                    # a broken image: fall back to one by one
            ok, zs = [], []
            for i in chunk:
                try:
                    zs.append(embed_images([IMG_DIR / f"{i}.jpg"])[0]); ok.append(i)
                except Exception:
                    pass
            Zc = np.stack(zs) if zs else np.zeros((0, 512), "float32")
        for i, z in zip(ok, Zc):
            cache[i] = z
        save_cache()


ensure([r["pageid"] for r in real] + cand)
cand = [i for i in cand if i in cache]

# ---------- labelled reference set
lab = [r for r in real if r["pageid"] in cache and any(c in cidx for c in r["cats"])]
L = np.stack([cache[r["pageid"]] for r in lab])
Y = np.zeros((len(lab), len(classes)), "float32")
for row, r in enumerate(lab):
    for c in r["cats"]:
        if c in cidx:
            Y[row, cidx[c]] = 1
T = embed_texts([TEXT_PROMPT.format(c) for c in classes])


def predict(Q, Lz, Ly):
    """class index per query, or -1 when the two signals do not agree confidently."""
    out = []
    for s in range(0, len(Q), 1024):
        q = Q[s:s + 1024]
        sims = q @ Lz.T
        k = min(KNN_K, sims.shape[1])
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        sv = np.take_along_axis(sims, idx, 1)
        w = np.exp((sv - sv.max(1, keepdims=True)) / TAU); w /= w.sum(1, keepdims=True)
        votes = np.einsum("bk,bkc->bc", w, Ly[idx])
        t = q @ T.T
        e = np.exp(100 * (t - t.max(1, keepdims=True))); p = e / e.sum(1, keepdims=True)
        kc, tc = votes.argmax(1), t.argmax(1)
        ok = (kc == tc) & (votes.max(1) >= SHARE) & (p.max(1) >= TXT_P)
        out.append(np.where(ok, kc, -1))
    return np.concatenate(out)


# ---------- how good is this rule? test it on real images whose labels we know (10% held out)
perm = np.random.RandomState(0).permutation(len(lab)); h, rest = perm[:max(1, len(perm) // 10)], perm[max(1, len(perm) // 10):]
ph = predict(L[h], L[rest], Y[rest]); acc = ph >= 0
hit = Y[h][np.arange(len(h)), np.maximum(ph, 0)] > 0
print(f"PSEUDO-LABEL CHECK on {len(h)} held-out real images: accepted {acc.mean():.0%}, precision of accepted {hit[acc].mean():.1%}")

# ---------- label the unlabelled candidates
pseudo = []
if cand:
    pc = predict(np.stack([cache[i] for i in cand]), L, Y)
    pseudo = [dict(meta[i], cats=[classes[c]], pseudo=True) for i, c in zip(cand, pc) if c >= 0]
print(f"pseudo-labelled {len(pseudo)} of {len(cand)} unlabelled images")

# ---------- combine, enforce MIN images per topic, cap at CAP (real labels first)
allr = {r["pageid"]: r for r in real + pseudo}
tot = Counter(c for r in allr.values() for c in r["cats"])
keep = {c for c, v in tot.items() if v >= MIN}
rm, pm = defaultdict(list), defaultdict(list)
for pid, r in allr.items():
    for c in r["cats"]:
        if c in keep:
            (pm if r.get("pseudo") else rm)[c].append(pid)
rng, chosen = random.Random(0), defaultdict(set)
for c in keep:
    a, b = sorted(rm[c]), sorted(pm[c]); rng.shuffle(a); rng.shuffle(b)
    for pid in (a + b)[:CAP]:
        chosen[pid].add(c)
out = []
for pid, r in allr.items():
    if pid in chosen:
        r["cats"] = sorted(chosen[pid]); out.append(r)
with open(RAW, "w") as fh:
    for r in out:
        fh.write(json.dumps(r) + "\n")
fin = Counter(c for r in out for c in r["cats"])
print(f"FINAL: {len(out)} images ({sum(1 for r in out if r.get('pseudo'))} pseudo-labelled) | {len(fin)} topics, "
      f"{min(fin.values())} to {max(fin.values())} images each")
print("dropped (< %d):" % MIN, sorted((v, c) for c, v in tot.items() if v < MIN))
