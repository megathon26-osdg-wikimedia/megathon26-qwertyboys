"""STAGE 1B - Feed YOUR OWN dataset instead of (or in addition to) Commons.

Mode 1 - CSV:      python 01b_import_local.py --csv labels.csv --images my_images/
   CSV header:     filename,title,categories,timestamp,user
     filename      path relative to --images
     title         optional (defaults to the file name)
     categories    one or more labels separated by |      e.g.  Charminar|Hyderabad|Mosques in India
     timestamp     optional ISO time (2024-05-01T12:00:00Z); defaults to the file's modified time.
                   It drives the temporal train/test split, so make it the real upload/capture date if you can.
     user          optional uploader/photographer (kept for grouping)
Mode 2 - folders:  python 01b_import_local.py --folders my_dataset/
   my_dataset/<category name>/*.jpg   (one label per image = the folder name)
Add --append to add to an existing data/raw.jsonl instead of replacing it.
"""
import argparse, csv, json, os
from datetime import datetime, timezone
from common import *

START_ID = 10_000_000
EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iso(ts, path):
    if ts:
        dt = datetime.fromisoformat(ts.strip().replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


ap = argparse.ArgumentParser()
ap.add_argument("--csv"); ap.add_argument("--images", default=".")
ap.add_argument("--folders"); ap.add_argument("--append", action="store_true")
args = ap.parse_args()
if not args.csv and not args.folders:
    ap.error("give --csv/--images or --folders")

raw_path = DATA / "raw.jsonl"
existing = [json.loads(l) for l in open(raw_path)] if args.append and raw_path.exists() else []
next_id = max([START_ID] + [r["pageid"] + 1 for r in existing])

items = []   # (path, title, cats, timestamp, user)
if args.csv:
    for row in csv.DictReader(open(args.csv, newline="", encoding="utf-8")):
        p = Path(args.images) / row["filename"]
        cats = [c.strip() for c in row["categories"].split("|") if c.strip()]
        items.append((p, row.get("title") or p.stem, cats, row.get("timestamp"), row.get("user")))
else:
    for d in sorted(Path(args.folders).iterdir()):
        if d.is_dir():
            for p in sorted(d.rglob("*")):
                if p.suffix.lower() in EXT:
                    items.append((p, p.stem, [d.name], None, None))

out, bad = list(existing), 0
for p, title, cats, ts, user in tqdm(items, desc="importing"):
    try:
        img = Image.open(p).convert("RGB"); img.thumbnail((512, 512))
        img.save(IMG_DIR / f"{next_id}.jpg", quality=92)
    except Exception as e:
        bad += 1; print("skip", p, e); continue
    out.append(dict(pageid=next_id, title=f"File:{title}.jpg", cats=cats,
                    timestamp=iso(ts, p), user=user, mime="image/jpeg"))
    next_id += 1
with open(raw_path, "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(f"imported {len(out) - len(existing)} images ({bad} skipped) -> {raw_path} ({len(out)} total)")
