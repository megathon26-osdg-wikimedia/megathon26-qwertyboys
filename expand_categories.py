"""Targeted expansion: fetch MORE images for the categories already in the vocabulary.
  python expand_categories.py --target 70 --minutes 60 --workers 6   -> data/raw_extra.jsonl (resumable)
  python expand_categories.py --merge                                -> rebuilds data/raw.jsonl (original kept in data/raw_before_expand.jsonl)
"""
import json, time, random, argparse, threading, shutil
from concurrent.futures import ThreadPoolExecutor
from category_fetch import fetch_files, download
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--target", type=int, default=70)
ap.add_argument("--minutes", type=float, default=60)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--merge", action="store_true")
args = ap.parse_args()
EXTRA, DONE = DATA / "raw_extra.jsonl", DATA / "expand_done.txt"
lock = threading.Lock()

if args.merge:
    base, bak = DATA / "raw.jsonl", DATA / "raw_before_expand.jsonl"
    if not bak.exists():
        shutil.copy(base, bak)
    recs = {}
    for l in open(bak):
        r = json.loads(l); recs[r["pageid"]] = r
    n0 = len(recs)
    if EXTRA.exists():
        for l in open(EXTRA):
            r = json.loads(l)
            if (IMG_DIR / f"{r['pageid']}.jpg").exists():
                recs.setdefault(r["pageid"], r)
    with open(base, "w") as f:
        for r in recs.values():
            f.write(json.dumps(r) + "\n")
    print(f"merged: {n0} original + {len(recs) - n0} new = {len(recs)} records in data/raw.jsonl")
    raise SystemExit

vocab = load_meta()["vocab"]
have = {json.loads(l)["pageid"] for l in open(DATA / "raw.jsonl")}
if EXTRA.exists():
    have |= {json.loads(l)["pageid"] for l in open(EXTRA)}
done = set(open(DONE).read().split("\n")) if DONE.exists() else set()
todo = [c for c in vocab if c not in done]
random.Random(0).shuffle(todo)                       # even coverage if the time limit hits
deadline = time.time() + args.minutes * 60
print(f"{len(vocab)} categories | {len(done)} already done | target {args.target} files per category")


def work(cat):
    if time.time() > deadline:
        return 0
    try:
        files = fetch_files(cat, args.target)
    except Exception as e:
        tqdm.write(f"skip {cat} -> {e}"); return 0
    new = [dict(f, cats=sorted(f["cats"])) for f in files
           if f["pageid"] not in have and f.get("thumburl") and f.get("mime") in ("image/jpeg", "image/png")]
    good = [f for f in new if download(f)]
    with lock:
        have.update(f["pageid"] for f in good)
        with open(EXTRA, "a") as fh:
            for f in good:
                fh.write(json.dumps(f) + "\n")
        with open(DONE, "a") as fh:
            fh.write(cat + "\n")
    return len(good)


with ThreadPoolExecutor(args.workers) as ex:
    total = sum(tqdm(ex.map(work, todo), total=len(todo), desc="categories"))
print("new images downloaded this run:", total)
print("when finished (or stopped), run:  python expand_categories.py --merge")
