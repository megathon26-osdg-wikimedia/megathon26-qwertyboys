"""STAGE 8 - Demo UI: upload -> ranked suggestions + neighbour images (explanation) -> accept/reject,
plus: teach the model a NEW category, fetched from Wikimedia on the spot.
Run: python app.py      (opens http://127.0.0.1:7860)
"""
import time, random
import gradio as gr
from suggester import Suggester
from category_fetch import normalize, resolve, fetch_category
from common import *

S = Suggester(load_user=True)


def run(img, title):
    if img is None:
        return gr.update(choices=[], value=[]), [], None
    picks, neigh, z = S.suggest(img, title or "", k=8)
    labels = [f"{c}  ({s:.2f})" for c, s in picks]
    gallery = []
    for pid, t, sim, labs in neigh:
        p = IMG_DIR / f"{pid}.jpg"
        if p.exists():
            gallery.append((str(p), f"sim {sim:.2f} | {', '.join(labs[:3])}"))
    return gr.update(choices=labels, value=[]), gallery, {"z": z, "shown": [c for c, _ in picks]}


def feedback(accepted, st):
    if not st:
        return "Run a suggestion first."
    acc = [a.rsplit("  (", 1)[0] for a in (accepted or [])]
    rejected = [c for c in st["shown"] if c not in acc]          # rejects = hard negatives
    if acc:
        S.add(st["z"], acc)                                      # index grows immediately; no retraining
    with open(DATA / "feedback.jsonl", "a") as f:
        f.write(json.dumps(dict(ts=time.time(), accepted=acc, rejected=rejected)) + "\n")
    return f"Saved: {len(acc)} accepted (added to the index), {len(rejected)} rejected (logged as hard negatives)."


def learn(img, name, n, progress=gr.Progress()):
    name = (name or "").strip()
    if not name:
        return "Type a category name first."
    progress(0.02, desc="Looking the category up on Commons")
    try:
        cat, note = resolve(normalize(name))
        if cat is None:
            return note
        recs = fetch_category(cat, int(n), progress=lambda f, t: progress(0.05 + 0.6 * f, desc=t))
    except Exception as e:
        return f"Could not fetch from Wikimedia: {e}"
    if not recs:
        return f"{note} No usable images were found in Category:{cat}."
    progress(0.7, desc=f"Embedding {len(recs)} images")
    Z = embed_images([IMG_DIR / f"{r['pageid']}.jpg" for r in recs])
    items = [(r["pageid"], r["title"], z, r["cats"]) for r, z in zip(recs, Z)]
    random.Random(0).shuffle(items)
    nh = int(0.15 * len(items)) if len(items) >= 10 else 0
    held, train = items[:nh], items[nh:]
    progress(0.9, desc="Adding to the index")
    S.add_category(cat, train)
    acc = S.self_check(cat, np.stack([z for _, _, z, _ in held])) if held else None
    S.add_category(cat, held)
    fa = S.false_alarm(cat)
    user = []
    if img is not None:                                          # the uploaded image is an example too
        user = [(-int(time.time() * 1000), "user upload", embed_images([img])[0], [])]
        S.add_category(cat, user)
    S.save_user_cat(cat, items + user)
    msg = f"{note}\n\n" if note else ""
    msg += f"**Learned `{cat}`**: {len(items)} images from Wikimedia" + (" + your upload" if user else "") + " added to the index."
    if acc is not None:
        msg += f"\n\nSanity check: {acc:.0%} of {len(held)} held-out images of this category got it in their top 5."
    if fa == fa:
        msg += f"\n\nFalse-alarm check: this category was the #1 suggestion for {fa:.0%} of held-out images from other categories (lower is better)."
    return msg


with gr.Blocks(title="Commons Category Suggester") as demo:
    gr.Markdown("## Wikimedia Commons category suggester\nSuggestions only: a human accepts or rejects each one.")
    with gr.Row():
        with gr.Column():
            img = gr.Image(type="pil", label="Upload an image")
            title = gr.Textbox(label="Title / description (optional)")
            btn = gr.Button("Suggest categories", variant="primary")
            gr.Markdown("**Missing category?** Teach it. The model fetches images for it from Wikimedia.")
            newcat = gr.Textbox(label="New category (optional)", placeholder="e.g. Hawa Mahal")
            nimg = gr.Slider(50, 800, value=200, step=50, label="Images to fetch from Wikimedia")
            learn_btn = gr.Button("Learn this category from Wikimedia")
            learn_status = gr.Markdown()
        with gr.Column():
            chk = gr.CheckboxGroup(label="Suggested categories (tick the correct ones)")
            fb = gr.Button("Submit feedback")
            status = gr.Markdown()
    gallery = gr.Gallery(label="Why? Most similar categorised images", columns=3, height=300)
    state = gr.State()
    btn.click(run, [img, title], [chk, gallery, state])
    fb.click(feedback, [chk, state], [status])
    learn_btn.click(learn, [img, newcat, nimg], [learn_status]).then(run, [img, title], [chk, gallery, state])

if __name__ == "__main__":
    demo.launch()
