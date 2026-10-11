"""Relabel collected images with the broad SEED topics (Dogs, Cats, Taj Mahal, ...) instead of fine-grained Commons categories.
Original raw.jsonl is kept as data/raw_fine_labels.jsonl.   Run: python relabel_by_seed.py"""
import json, shutil
from common import *

state, raw, bak = DATA / "collect_state.jsonl", DATA / "raw.jsonl", DATA / "raw_fine_labels.jsonl"
if raw.exists() and not bak.exists():
    shutil.copy(raw, bak)
seedset = set(SEEDS)
recs = {}
for line in open(state):
    d = json.loads(line)
    for f in d["files"]:
        if not (IMG_DIR / f"{f['pageid']}.jpg").exists():
            continue
        r = recs.setdefault(f["pageid"], dict(f, cats=set()))
        r["cats"].add(d["seed"])
        r["cats"] |= {c for c in f.get("cats", []) if c in seedset}     # keep overlaps between seed topics
with open(raw, "w") as fh:
    for r in recs.values():
        r["cats"] = sorted(r["cats"]); fh.write(json.dumps(r) + "\n")
from collections import Counter
cnt = Counter(c for r in recs.values() for c in r["cats"])
print(len(recs), "images |", len(cnt), "seed categories | with >= 12 images:", sum(v >= 12 for v in cnt.values()))
print("largest:", cnt.most_common(8))
print("smallest:", cnt.most_common()[-8:])
