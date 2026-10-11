"""Recover images downloaded by expand_categories.py by assigning each to the seed topic its Commons category came from.
Only unambiguous matches are used. Appends to data/collect_state.jsonl.   Run once: python recover_extra.py"""
import json
from collections import defaultdict, Counter
from common import *

STATE, EXTRA = DATA / "collect_state.jsonl", DATA / "raw_extra.jsonl"
MAINT = re.compile("|".join(MAINT_PATTERNS), re.I)
on_disk = lambda p: (IMG_DIR / f"{p}.jpg").exists()
cat2seeds, have = defaultdict(set), set()
for line in open(STATE):
    d = json.loads(line)
    for f in d["files"]:
        have.add(f["pageid"])
        for c in f.get("cats", []):
            if not MAINT.search(c):
                cat2seeds[c].add(d["seed"])
unique = {c: next(iter(s)) for c, s in cat2seeds.items() if len(s) == 1}      # category -> its single seed
by_seed, ambiguous, nomatch = defaultdict(list), 0, 0
for line in open(EXTRA):
    r = json.loads(line)
    if r["pageid"] in have or not on_disk(r["pageid"]):
        continue
    seeds = {unique[c] for c in r.get("cats", []) if c in unique}
    if len(seeds) == 1:
        by_seed[next(iter(seeds))].append(r)
    elif seeds:
        ambiguous += 1
    else:
        nomatch += 1
with open(STATE, "a") as fh:
    for s, fs in by_seed.items():
        fh.write(json.dumps({"seed": s, "files": fs}) + "\n")
print("recovered:", sum(len(v) for v in by_seed.values()), "images for", len(by_seed), "seeds |",
      "ambiguous (skipped):", ambiguous, "| no seed match (skipped):", nomatch)
print("biggest gains:", Counter({s: len(v) for s, v in by_seed.items()}).most_common(8))
