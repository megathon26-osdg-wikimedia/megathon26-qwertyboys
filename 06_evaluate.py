"""STAGE 6 - Evaluate on the temporal TEST split. P/R/F1@k, macro recall by head/torso/tail, ablations.
Run:  python 06_evaluate.py --tag baseline        (results saved to data/results_<tag>.json)
      python 06_evaluate.py --tag degraded --degrade   (blur/low-res/JPEG robustness test)
"""
import argparse, csv, io, random
from collections import Counter, defaultdict
from suggester import Suggester
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--degrade", action="store_true"); ap.add_argument("--tag", default="run")
args = ap.parse_args()


def degrade(img):
    w, h = img.size
    img = img.resize((max(w // 4, 16), max(h // 4, 16))).resize((w, h))      # low resolution + blur
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=30); buf.seek(0)        # heavy JPEG
    return Image.open(buf).convert("RGB")


S = Suggester()
test = [i for i, r in enumerate(S.recs) if r["split"] == "test" and S.keep[i]]
if not test:
    raise SystemExit("No test images after deduplication.")
Z = S.Z[test]
if args.degrade:
    Z = embed_images([degrade(Image.open(IMG_DIR / f"{S.recs[i]['pageid']}.jpg")) for i in test], progress=True)

freq = Counter(c for _, _, labs in S.ref for c in labs)
order = sorted(range(len(S.vocab)), key=lambda c: -freq[c]); n = len(order)
bucket = {c: ("head" if j < n / 3 else "torso" if j < 2 * n / 3 else "tail") for j, c in enumerate(order)}
configs = [("knn", False), ("txt", False), ("meta", False), ("fusion", False), ("fusion", True)]
if S.W is not None:
    configs.insert(2, ("cls", False))

print(f"model {MODEL_NAME} | test images {len(test)} | categories {n} | degrade={args.degrade} "
      f"| head={'yes' if S.W is not None else 'no'} | fusion={'fitted' if S.fusion is not None else 'fallback weights'}\n")
results = {"_meta": dict(model=MODEL_NAME, n_test=len(test), n_categories=n, degrade=args.degrade)}
review = []
for mode, hier in configs:
    P, R, F1 = defaultdict(list), defaultdict(list), defaultdict(list)
    hit = defaultdict(lambda: [0, 0])                       # category -> [hits@max_k, occurrences]
    kmax = max(EVAL_KS)
    for row, i in enumerate(test):
        truth = set(S.recs[i]["labels"])
        preds, _ = S.rank(Z[row], S.ZT[i], k=kmax, mode=mode, hier=hier)
        names = [p for p, _ in preds]
        for k in EVAL_KS:
            inter = len(set(names[:k]) & truth)
            p, r = inter / k, inter / len(truth)
            P[k].append(p); R[k].append(r); F1[k].append(2 * p * r / (p + r) if p + r else 0.0)
        for c in truth:
            hit[c][1] += 1; hit[c][0] += c in names
        if (mode, hier) == ("fusion", True):
            review.append((S.recs[i]["title"], "; ".join(sorted(truth)), "; ".join(names)))
    tag = f"{mode}{'+hier' if hier else ''}"
    res = {}
    print(f"== {tag}")
    for k in EVAL_KS:
        res.update({f"P@{k}": float(np.mean(P[k])), f"R@{k}": float(np.mean(R[k])), f"F1@{k}": float(np.mean(F1[k]))})
        print(f"  k={k}: P {res[f'P@{k}']:.3f}  R {res[f'R@{k}']:.3f}  F1 {res[f'F1@{k}']:.3f}")
    macro = defaultdict(list)
    for c, (h, t) in hit.items():
        macro[bucket[S.cidx[c]]].append(h / t)
    allm = [v for vs in macro.values() for v in vs]
    res[f"macro_recall@{kmax}"] = float(np.mean(allm))
    for b in ("head", "torso", "tail"):
        if macro[b]:
            res[f"recall@{kmax}_{b}"] = float(np.mean(macro[b]))
    print(f"  macro-recall@{kmax} {res[f'macro_recall@{kmax}']:.3f} | " +
          " ".join(f"{b}:{np.mean(macro[b]):.3f}" for b in ("head", "torso", "tail") if macro[b]) + "\n")
    results[tag] = res

json.dump(results, open(DATA / f"results_{args.tag}.json", "w"), indent=2)
random.seed(0)
with open(DATA / "review_sample.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["title", "true_labels", "suggested_top5", "reviewer_correct_count"])
    w.writerows(random.sample(review, min(100, len(review))))
print(f"saved data/results_{args.tag}.json and data/review_sample.csv (hand-check rows for the 'reviewed' score)")
