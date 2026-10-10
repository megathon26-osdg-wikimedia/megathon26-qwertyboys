"""STAGE 1A (v4) - parallel, time-boxed, resumable.
  python 01_collect.py --minutes 40     collect for up to 40 min, then build data/raw.jsonl
  python 01_collect.py --finish         skip listing, just build raw.jsonl from what is saved
"""
import json, time, random, argparse, threading, requests
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from common import *

API = "https://commons.wikimedia.org/w/api.php"
STATE = DATA / "collect_state.jsonl"
ap = argparse.ArgumentParser()
ap.add_argument("--minutes", type=float, default=40)
ap.add_argument("--workers", type=int, default=5)
ap.add_argument("--finish", action="store_true")
args = ap.parse_args()
lock = threading.Lock()


def api(params):
    for attempt in range(5):
        r = requests.get(API, params=params, headers=UA, timeout=30)
        if r.status_code == 429:
            time.sleep(3 * (attempt + 1)); continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("rate limited")


def fetch_files(cat, limit):
    params = dict(action="query", format="json", generator="categorymembers",
                  gcmtitle=f"Category:{cat}", gcmtype="file", gcmlimit=50,
                  prop="imageinfo|categories", iiprop="url|timestamp|size|mime|user",
                  iiurlwidth=330, cllimit="max", clshow="!hidden")
    pages = {}
    while len(pages) < limit:
        r = api(params)
        for pid, p in r.get("query", {}).get("pages", {}).items():
            d = pages.setdefault(pid, {"pageid": int(pid), "title": p["title"], "cats": set()})
            if "imageinfo" in p:
                ii = p["imageinfo"][0]
                d.update(thumburl=ii.get("thumburl"), timestamp=ii["timestamp"],
                         mime=ii["mime"], user=ii.get("user"))
            for c in p.get("categories", []):
                d["cats"].add(c["title"].removeprefix("Category:"))
        if "continue" not in r:
            break
        params.update(r["continue"]); time.sleep(0.05)
    return list(pages.values())[:limit]


def subcats(cat, n):
    r = api(dict(action="query", format="json", list="categorymembers",
                 cmtitle=f"Category:{cat}", cmtype="subcat", cmlimit=n))
    return [m["title"].removeprefix("Category:") for m in r.get("query", {}).get("categorymembers", [])]


def collect(seed):
    out, seen, frontier = {}, set(), [seed]
    for depth in range(SUBCAT_DEPTH + 1):
        nxt = []
        for cat in frontier:
            if cat in seen or len(out) >= MAX_PER_SEED:
                continue
            seen.add(cat)
            try:
                for f in fetch_files(cat, PER_LEAF):
                    out[f["pageid"]] = f
                if depth < SUBCAT_DEPTH:
                    nxt += subcats(cat, SUBCATS_PER_NODE)
            except Exception as e:
                tqdm.write(f"skip {cat} -> {e}")
            time.sleep(0.05)
        random.shuffle(nxt)
        frontier = nxt
    return list(out.values())[:MAX_PER_SEED]


def download(rec):
    out = IMG_DIR / f"{rec['pageid']}.jpg"
    if out.exists():
        return True
    for attempt in range(3):
        try:
            r = requests.get(rec["thumburl"], headers=UA, timeout=30)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1)); continue
            r.raise_for_status()
            Image.open(BytesIO(r.content)).convert("RGB").save(out, quality=92)
            return True
        except Exception:
            time.sleep(1)
    return False


done = {}
if STATE.exists():
    for line in open(STATE):
        d = json.loads(line); done[d["seed"]] = d["files"]
print(f"{len(done)} seeds already finished")
deadline = time.time() + args.minutes * 60
todo = [] if args.finish else [s for s in SEEDS if s not in done]


def work(seed):
    if time.time() > deadline:
        return
    try:
        files = collect(seed)
    except Exception as e:
        tqdm.write(f"skip {seed} -> {e}"); return
    files = [dict(f, cats=sorted(f["cats"])) for f in files
             if f.get("thumburl") and f.get("mime") in ("image/jpeg", "image/png")]
    for f in files:
        download(f)
    with lock:
        done[seed] = files
        with open(STATE, "a") as fh:
            fh.write(json.dumps({"seed": seed, "files": files}) + "\n")
    tqdm.write(f"  {seed}: {len(files)} files | seeds done: {len(done)}/{len(SEEDS)}")


with ThreadPoolExecutor(args.workers) as ex:
    list(tqdm(ex.map(work, todo), total=len(todo), desc="seeds"))

recs = list({f["pageid"]: f for fs in done.values() for f in fs}.values())
missing = [r for r in recs if not (IMG_DIR / f"{r['pageid']}.jpg").exists()]
print(len(recs), "files listed;", len(missing), "images still to download")
with ThreadPoolExecutor(4) as ex:
    list(tqdm(ex.map(download, missing), total=len(missing)))
n = 0
with open(DATA / "raw.jsonl", "w") as f:
    for r in recs:
        if (IMG_DIR / f"{r['pageid']}.jpg").exists():
            f.write(json.dumps(r) + "\n"); n += 1
print("saved", n, "records to data/raw.jsonl")
