"""Demo UI: upload -> ranked suggestions + neighbour images -> accept/reject.
Optional new category: learned silently in a background thread while the model keeps serving requests.
Run: python app.py   (http://127.0.0.1:7860)"""
import time, random, threading, inspect, traceback, html as _html
import gradio as gr
from suggester import Suggester
from category_fetch import normalize, resolve, fetch_category
from common import *


def preflight():
    """Stop with a clear message if the data files were built from different runs."""
    meta, e, probs = load_meta(), np.load(DATA / "emb.npz"), []
    if e["Z"].shape[0] != len(meta["records"]):
        probs.append(f"emb.npz has {e['Z'].shape[0]} image vectors but meta.json has {len(meta['records'])} records")
    if e["T"].shape[0] != len(meta["vocab"]):
        probs.append(f"emb.npz has {e['T'].shape[0]} category vectors but meta.json has {len(meta['vocab'])} categories")
    if (DATA / "head.npz").exists() and np.load(DATA / "head.npz")["W"].shape[0] != len(meta["vocab"]):
        probs.append("head.npz was trained for a different category list")
    if probs:
        raise SystemExit("\n!! DATA FILES ARE OUT OF SYNC:\n  - " + "\n  - ".join(probs) +
                         "\n\nFix:  rm -f data/head.npz data/fusion.joblib && python 03_embed.py && python run_pipeline.py --from 4 --tag fix\n")


preflight()
S = Suggester(load_user=True)
LOCK = threading.Lock()        # guards the model state and the CLIP embedder (held only briefly)
LEARNING = set()
LEARN_N = 200                  # images fetched per new category


def slug(name):
    return re.sub(r"\W+", "_", name)[:60]


def log(rec):
    print("[learn]", rec)
    with open(DATA / "learn_log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def bg_learn(name, upload_item):
    t0 = time.time()
    try:
        cat, _ = resolve(name)
        if not cat:
            log(dict(name=name, status="no commons category found")); return
        recs = fetch_category(cat, LEARN_N)
        if not recs:
            log(dict(name=name, commons=cat, status="no images", detail=getattr(fetch_category, "last_info", ""))); return
        items = []
        for i in range(0, len(recs), 16):                       # small chunks: foreground requests can interleave
            chunk = recs[i:i + 16]
            with LOCK:
                Zc = embed_images([IMG_DIR / f"{r['pageid']}.jpg" for r in chunk])
            items += [(r["pageid"], r["title"], z, r["cats"]) for r, z in zip(chunk, Zc)]
        random.Random(0).shuffle(items)
        nh = int(0.15 * len(items)) if len(items) >= 10 else 0
        held, train = items[:nh], items[nh:]
        with LOCK:
            S.add_category(name, train)
            acc = S.self_check(name, np.stack([z for _, _, z, _ in held])) if held else None
            S.add_category(name, held)
            fa = S.false_alarm(name)
        S.save_user_cat(name, items + [upload_item])
        log(dict(name=name, commons=cat, status="learned", images=len(items), self_check=acc,
                 false_alarm=fa, seconds=round(time.time() - t0, 1)))
    except Exception as e:
        log(dict(name=name, status="error", error=repr(e)))
    finally:
        LEARNING.discard(name)


def cards(picks):
    """Ranked suggestion cards with confidence bars (inline styles, works in light and dark)."""
    if not picks:
        return ""
    rows = []
    for i, (c, s) in enumerate(picks, 1):
        pct = max(0.0, min(1.0, float(s))) * 100
        rows.append(
            f'<div style="margin:8px 0;"><div style="display:flex;justify-content:space-between;font-size:14px;">'
            f'<span><b>{i}.</b> {_html.escape(c)}</span><span style="opacity:.7">{pct:.0f}%</span></div>'
            f'<div style="height:7px;border-radius:6px;background:rgba(128,128,128,.22);margin-top:4px;">'
            f'<div style="height:7px;border-radius:6px;width:{max(pct, 2):.0f}%;background:#6366f1;"></div></div></div>')
    return ('<div style="border:1px solid rgba(128,128,128,.3);border-radius:12px;padding:10px 16px;">'
            + "".join(rows) + "</div>")


def run(img, title, newcat):
    try:
        if img is None:
            return "", gr.update(choices=[], value=[]), [], None, "Upload an image first."
        name = normalize(newcat) if (newcat or "").strip() else ""
        if name:
            with LOCK:                                           # instant: text prototype + the uploaded image
                up = (-int(time.time() * 1000), "user upload", embed_images([img])[0], [])
                S.add_category(name, [up])
            if name not in LEARNING and not (DATA / "user_cats" / f"{slug(name)}.npz").exists():
                LEARNING.add(name)
                threading.Thread(target=bg_learn, args=(name, up), daemon=True).start()
        with LOCK:
            picks, neigh, z = S.suggest(img, title or "", k=8)
        labels = [f"{c}  ({s:.2f})" for c, s in picks]
        gallery = []
        for pid, t, sim, labs in neigh:
            p = IMG_DIR / f"{pid}.jpg"
            if p.exists():
                gallery.append((str(p), f"similarity {sim:.2f} | {', '.join(labs[:3])}"))
        return cards(picks), gr.update(choices=labels, value=[]), gallery, {"z": z, "shown": [c for c, _ in picks]}, ""
    except Exception as e:
        traceback.print_exc()
        return "", gr.update(choices=[], value=[]), [], None, f"**Something went wrong:** `{type(e).__name__}: {e}`  (full details in the terminal)"


def feedback(accepted, st):
    if not st:
        return "Run a suggestion first."
    acc = [a.rsplit("  (", 1)[0] for a in (accepted or [])]
    rejected = [c for c in st["shown"] if c not in acc]          # rejects = hard negatives
    if acc:
        with LOCK:
            S.add(st["z"], acc)                                  # index grows immediately; no retraining
    with open(DATA / "feedback.jsonl", "a") as f:
        f.write(json.dumps(dict(ts=time.time(), accepted=acc, rejected=rejected)) + "\n")
    return f"**Saved.** {len(acc)} accepted (added to the index), {len(rejected)} rejected (logged as hard negatives)."


# ---------------------------------------------------------------- UI
THEME = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate", font=["Inter", "system-ui", "sans-serif"])
CSS = """
.gradio-container {max-width: 1150px !important; margin: auto !important;}
#hero {background: linear-gradient(135deg, #1e293b, #334155); padding: 22px 26px; border-radius: 14px; margin-bottom: 14px;}
#hero h1 {margin: 0 0 4px 0; font-size: 26px; color: #ffffff;}
#hero p {margin: 0; color: #cbd5e1; font-size: 14px;}
#hero .chips span {display: inline-block; background: rgba(255,255,255,.14); border-radius: 999px;
                   padding: 3px 12px; margin: 12px 8px 0 0; font-size: 12px; color: #e2e8f0;}
#foot {color: #64748b; font-size: 12px; text-align: center; margin-top: 10px;}
"""
_ui = dict(theme=THEME, css=CSS)
if "theme" in inspect.signature(gr.Blocks.__init__).parameters:
    blocks_kw, launch_kw = _ui, {}          # older Gradio: theme/css are Blocks arguments
else:
    blocks_kw, launch_kw = {}, _ui          # newer Gradio: theme/css are launch arguments

HERO = f"""
<div id="hero">
  <h1>Commons Category Suggester</h1>
  <p>Upload an image and get ranked Wikimedia Commons category suggestions. A human accepts or rejects every one.</p>
  <div class="chips">
    <span>Encoder: {MODEL_NAME.split('/')[-1]}</span>
    <span>{len(S.vocab)} categories</span>
    <span>{S.index.ntotal:,} indexed images</span>
    <span>Device: {DEVICE.upper()}</span>
  </div>
</div>
"""

with gr.Blocks(title="Commons Category Suggester", **blocks_kw) as demo:
    gr.HTML(HERO)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            gr.Markdown("### 1. Your image")
            img = gr.Image(type="pil", label="Image", height=320)
            title = gr.Textbox(label="Title or description (optional)", placeholder="e.g. Sunset over the old fort")
            newcat = gr.Textbox(label="Category missing? Add one (optional)", placeholder="e.g. Hawa Mahal")
            btn = gr.Button("Suggest categories", variant="primary", size="lg")
            status = gr.Markdown()
        with gr.Column(scale=6):
            gr.Markdown("### 2. Suggested categories")
            res = gr.HTML()
            chk = gr.CheckboxGroup(label="Mark the correct ones, then submit")
            fb = gr.Button("Submit feedback", size="lg")
    gr.Markdown("### 3. Why these suggestions? Most similar categorised images")
    gallery = gr.Gallery(label="Nearest neighbours in the index", columns=6, height=240)
    gr.HTML('<div id="foot">Suggestions only. Nothing is edited automatically, and the system does not identify individuals.</div>')
    state = gr.State()
    btn.click(run, [img, title, newcat], [res, chk, gallery, state, status])
    fb.click(feedback, [chk, state], [status])

if __name__ == "__main__":
    demo.launch(**launch_kw)
