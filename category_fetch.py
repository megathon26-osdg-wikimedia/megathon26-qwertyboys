"""Fetch images for ONE Commons category (plus its sub-categories) for the demo's 'learn a new category' feature."""
import time, random, requests
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from common import *

API = "https://commons.wikimedia.org/w/api.php"


def api(params):
    for attempt in range(5):
        r = requests.get(API, params=params, headers=UA, timeout=30)
        if r.status_code == 429:
            time.sleep(3 * (attempt + 1)); continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Wikimedia is rate-limiting us - try again in a minute")


def normalize(name):
    name = name.strip().removeprefix("Category:").replace("_", " ").strip()
    return name[:1].upper() + name[1:]


def resolve(name):
    """-> (commons_category_title or None, note). Exact match first, else the closest search hit."""
    r = api(dict(action="query", format="json", titles=f"Category:{name}"))
    page = next(iter(r["query"]["pages"].values()))
    if "missing" not in page:
        return name, ""
    r = api(dict(action="query", format="json", list="search", srsearch=name,
                 srnamespace=14, srlimit=1))
    hits = r.get("query", {}).get("search", [])
    if not hits:
        return None, f"No Commons category found for '{name}'."
    best = hits[0]["title"].removeprefix("Category:")
    return best, f"'{name}' is not a Commons category, so I used the closest match: '{best}'."


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


def list_files(cat, total, depth=2, per_node=20, per_leaf=40):
    out = {f["pageid"]: f for f in fetch_files(cat, total)}
    seen, frontier = {cat}, [cat]
    for _ in range(depth):
        if len(out) >= total:
            break
        nxt = []
        for c in frontier:
            try:
                nxt += [s for s in subcats(c, per_node) if s not in seen]
            except Exception:
                pass
        random.shuffle(nxt); frontier = []
        for s in nxt[:60]:
            if len(out) >= total:
                break
            seen.add(s); frontier.append(s)
            try:
                for f in fetch_files(s, min(per_leaf, total - len(out))):
                    out[f["pageid"]] = f
            except Exception:
                pass
    return list(out.values())[:total]


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


def fetch_category(cat, total=200, progress=lambda frac, text: None, workers=4):
    """-> list of records {pageid, title, cats, ...} whose thumbnail is saved in data/images/."""
    progress(0.0, f"Listing files in Category:{cat}")
    files = [f for f in list_files(cat, total)
             if f.get("thumburl") and f.get("mime") in ("image/jpeg", "image/png")]
    if not files:
        return []
    ok = []
    with ThreadPoolExecutor(workers) as ex:
        for i, g in enumerate(ex.map(download, files), 1):      # consumed in the main thread
            ok.append(g)
            if i % 10 == 0 or i == len(files):
                progress(0.3 + 0.7 * i / len(files), f"Downloading images {i}/{len(files)}")
    return [dict(f, cats=sorted(f["cats"])) for f, g in zip(files, ok) if g]
