"""STAGE 2 - Clean labels, TEMPORAL split, build vocabulary (+ category parents for the hierarchy step).
Output: data/meta.json        Run: python 02_prepare.py   (add --no-parents to skip the API calls)
"""
import argparse, time, requests
from collections import Counter
from common import *

API = "https://commons.wikimedia.org/w/api.php"
ap = argparse.ArgumentParser(); ap.add_argument("--no-parents", action="store_true")
args = ap.parse_args()
MAINT = re.compile("|".join(MAINT_PATTERNS), re.I)


def fetch_parents(cats):
    out = {}
    for i in range(0, len(cats), 50):
        params = dict(action="query", format="json", prop="categories", cllimit="max",
                      titles="|".join("Category:" + c for c in cats[i:i + 50]), clshow="!hidden")
        while True:
            r = requests.get(API, params=params, headers=UA, timeout=30).json()
            for p in r.get("query", {}).get("pages", {}).values():
                t = p["title"].removeprefix("Category:")
                out.setdefault(t, []).extend(c["title"].removeprefix("Category:")
                                             for c in p.get("categories", []))
            if "continue" not in r:
                break
            params.update(r["continue"])
        time.sleep(0.3)
    return out


raw = [json.loads(l) for l in open(DATA / "raw.jsonl")]
missing = [r for r in raw if not (IMG_DIR / f"{r['pageid']}.jpg").exists()]
raw = [r for r in raw if (IMG_DIR / f"{r['pageid']}.jpg").exists()]
if missing:
    print(f"WARNING: dropped {len(missing)} records without a downloaded image")

raw.sort(key=lambda r: r["timestamp"])                    # TEMPORAL split: oldest -> train, newest -> test
n = len(raw); a = int(n * TRAIN_FRAC); b = int(n * (TRAIN_FRAC + VAL_FRAC))
for i, r in enumerate(raw):
    r["split"] = "train" if i < a else "val" if i < b else "test"

# vocabulary is counted on TRAIN ONLY (no information from the future)
all_cnt = Counter(c for r in raw if r["split"] == "train" for c in set(r["cats"]))
dropped = Counter({c: k for c, k in all_cnt.items() if MAINT.search(c)})
cnt = Counter({c: k for c, k in all_cnt.items() if not MAINT.search(c)})
vocab = sorted(c for c, k in cnt.items() if k >= N_MIN)
vs = set(vocab)

recs = []
for r in raw:
    labs = [c for c in r["cats"] if c in vs]
    if labs:
        recs.append(dict(pageid=r["pageid"], title=r["title"], timestamp=r["timestamp"],
                         user=r.get("user"), labels=labs, split=r["split"]))

parents = {}
if not args.no_parents and vocab:
    print("fetching category parents (2 levels) ...")
    try:
        frontier = vocab
        for _ in range(2):
            new = fetch_parents([c for c in frontier if c not in parents])
            parents.update(new)
            frontier = sorted({p for ps in new.values() for p in ps} - set(parents))
    except Exception as e:
        print("WARNING: parent fetch failed (", e, ") -> hierarchy suppression will be a no-op")

json.dump(dict(records=recs, vocab=vocab, parents=parents), open(DATA / "meta.json", "w"))
print(f"\n{len(recs)} images | {len(vocab)} categories | splits {dict(Counter(r['split'] for r in recs))}")
print("\n>>> CHECK THESE: most frequent categories kept (should be real content):")
print("   ", [f"{c} ({k})" for c, k in cnt.most_common(25)])
print(">>> categories REMOVED by MAINT_PATTERNS (should be junk; if real content is here, edit config.py):")
print("   ", [f"{c} ({k})" for c, k in dropped.most_common(20)])
