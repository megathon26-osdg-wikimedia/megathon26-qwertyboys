"""Fetch images for ONE Commons category (plus sub-categories) for the demo's 'learn a new category' feature (v2)."""
import time, random, requests
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from common import *

API = "https://commons.wikimedia.org/w/api.php"
_last = {"t": 0.0}


def api(params):
    for attempt in range(6):
        wait = 0.15 - (time.time() - _last["t"])
        if wait > 0:
            time.sleep(wait)
        _last["t"] = time.time()
        r = requests.get(API, params=params, headers=UA, timeout=30)
        if r.status_code in (429, 503):
            try:
                time.sleep(float(r.headers.get("Retry-After", 3 * (attempt + 1))))
            except ValueError:
                time.sleep(3 * (attempt + 1))
            continue
        r.raise_for_status()
        j = r.json()
        if "error" in j:
            raise RuntimeError(j["error"].get("info", "API error"))
        return j
    raise RuntimeError("Wikimedia is rate-limiting us - wait a minute and try again")


def normalize(name):
    name = name.strip().removeprefix("Category:").replace("_", " ").strip()
    return name[:1].upper() + name[1:]


def cat_info(titles):
    """{category: (n_files, n_subcats)} for categories that exist."""
    out = {}
    for i in range(0, len(titles), 40):
        r = api(dict(action="query", format="json", prop="categoryinfo",
                     titles="|".join("Category:" + t for t in titles[i:i + 40])))
        for p in r.get("query", {}).get("pages", {}).values():
            if "missing" in p or "invalid" in p:
                continue
            ci = p.get("categoryinfo", {})
            out[p["title"].removeprefix("Category:")] = (ci.get("files", 0), ci.get("subcats", 0))
    return out


def resolve(name):
    """-> (commons_category or None, note). Tries the exact name, singular/plural, and search hits; prefers one with content."""
    cands = [name]
    cands += [name[:-1]] if name.lower().endswith("s") else [name + "s", name + "es"]
    r = api(dict(action="query", format="json", list="search", srsearch=name, srnamespace=14, srlimit=5))
    cands += [h["title"].removeprefix("Category:") for h in r.get("query", {}).get("search", [])]
    cands = list(dict.fromkeys(cands))
    info = cat_info(cands)
    if not info:
        return None, f"No Commons category found for '{name}'."
    size = lambda c: info[c][0] + 3 * info[c][1]
    ok = [c for c in cands if c in info and size(c) > 0]
    if not ok:
        return None, f"A category for '{name}' exists but has no files or sub-categories."
    best = name if (name in ok and info[name][0] >= 10) else max(ok, key=size)
    note = "" if best == name else f"Using the Commons category '{best}' for '{name}' ({info[best][0]} files, {info[best][1]} sub-categories)."
    return best, note


def list_ids(cat, total, depth=3, max_cats=80, per_leaf=60):
    """Breadth-first walk: file page ids from the category, then its sub-categories. -> (ids, categories_visited, errors)"""
    ids, seen, frontier, visited, errors = [], {cat}, [cat], 0, 0
    for d in range(depth + 1):
        nxt = []
        for c in frontier:
            if len(ids) >= total or visited >= max_cats:
                break
            visited += 1
            try:
                r = api(dict(action="query", format="json", list="categorymembers", cmtitle=f"Category:{c}",
                             cmtype="file", cmlimit=min(total if d == 0 else per_leaf, 500), cmprop="ids|title"))
                have = set(ids)
                ids += [m["pageid"] for m in r["query"]["categorymembers"] if m["pageid"] not in have]
            except Exception:
                errors += 1
            if d < depth:
                try:
                    r = api(dict(action="query", format="json", list="categorymembers", cmtitle=f"Category:{c}",
                                 cmtype="subcat", cmlimit=50))
                    nxt += [m["title"].removeprefix("Category:") for m in r["query"]["categorymembers"]
                            if m["title"].removeprefix("Category:") not in seen]
                except Exception:
                    errors += 1
        random.shuffle(nxt)
        frontier = list(dict.fromkeys(nxt))
        seen.update(frontier)
    return ids[:total], visited, errors


def file_info(ids):
    pages = {}
    for i in range(0, len(ids), 50):
        params = dict(action="query", format="json", pageids="|".join(map(str, ids[i:i + 50])),
                      prop="imageinfo|categories", iiprop="url|timestamp|mime|user", iiurlwidth=330,
                      cllimit="max", clshow="!hidden")
        while True:
            r = api(params)
            for pid, p in r.get("query", {}).get("pages", {}).items():
                d = pages.setdefault(int(pid), {"pageid": int(pid), "title": p.get("title", ""), "cats": set()})
                if "imageinfo" in p:
                    ii = p["imageinfo"][0]
                    d.update(thumburl=ii.get("thumburl"), timestamp=ii.get("timestamp"),
                             mime=ii.get("mime"), user=ii.get("user"))
                for c in p.get("categories", []):
                    d["cats"].add(c["title"].removeprefix("Category:"))
            if "continue" not in r:
                break
            params.update(r["continue"])
    return list(pages.values())


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
    """-> records {pageid, title, cats, ...} whose thumbnail is saved in data/images/. Details in fetch_category.last_info."""
    progress(0.0, f"Listing files in Category:{cat}")
    ids, visited, errors = list_ids(cat, total)
    fetch_category.last_info = f"Listed {len(ids)} files ({visited} categories visited, {errors} request errors)."
    if not ids:
        if errors:
            fetch_category.last_info += " Errors usually mean Wikimedia is rate-limiting; wait a minute and retry."
        return []
    progress(0.2, "Reading file details")
    files = [f for f in file_info(ids) if f.get("thumburl") and f.get("mime") in ("image/jpeg", "image/png")]
    if not files:
        fetch_category.last_info += " None were JPEG/PNG images."
        return []
    ok = []
    with ThreadPoolExecutor(workers) as ex:
        for i, g in enumerate(ex.map(download, files), 1):
            ok.append(g)
            if i % 10 == 0 or i == len(files):
                progress(0.3 + 0.7 * i / len(files), f"Downloading images {i}/{len(files)}")
    return [dict(f, cats=sorted(f["cats"])) for f, g in zip(files, ok) if g]
