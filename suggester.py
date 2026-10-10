"""Core engine: kNN + text prototype + supervised head + title scorer -> fusion -> hierarchy."""
import faiss, joblib
from common import *

MODES = {"knn": 0, "cls": 1, "txt": 2, "meta": 3}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Suggester:
    def __init__(self, K=None, tau=None, load_user=False):
        self.K, self.tau = K or KNN_K, tau or KNN_TAU
        self.meta = load_meta(); self.recs = self.meta["records"]; self.vocab = self.meta["vocab"]
        self.cidx = {c: i for i, c in enumerate(self.vocab)}
        e = np.load(DATA / "emb.npz")
        self.Z, self.ZT, self.T, self.keep = e["Z"], e["ZT"], e["T"], e["keep"]
        tr = [i for i, r in enumerate(self.recs) if r["split"] == "train"]
        self.index = faiss.IndexFlatIP(self.Z.shape[1])       # exact; swap for HNSW/IVFPQ at millions of images
        self.index.add(np.ascontiguousarray(self.Z[tr]))
        self.ref = [(self.recs[i]["pageid"], self.recs[i]["title"],
                     [self.cidx[c] for c in self.recs[i]["labels"]]) for i in tr]
        self.anc = self._ancestors()
        from collections import Counter
        self.cnt = Counter(); [self.cnt.update(r[2]) for r in self.ref]
        self.n_orig = len(self.vocab)
        self.med = float(np.median([self.cnt[c] for c in range(self.n_orig) if self.cnt[c]] or [1]))
        self.W = self.b = self.fusion = None
        if (DATA / "head.npz").exists():
            h = np.load(DATA / "head.npz"); self.W, self.b = h["W"], h["b"]
        if (DATA / "fusion.joblib").exists():
            self.fusion = joblib.load(DATA / "fusion.joblib")
        if load_user:                       # only the demo loads user-taught categories; evaluation stays clean
            self._load_user_cats()

    def _ancestors(self):
        par, anc = self.meta.get("parents", {}), {}
        for c in self.vocab:
            seen, stack = set(), list(par.get(c, []))
            while stack:
                p = stack.pop()
                if p in seen:
                    continue
                seen.add(p); stack.extend(par.get(p, []))
            anc[self.cidx[c]] = {self.cidx[p] for p in seen if p in self.cidx}
        return anc

    # ---------------- component scorers ----------------
    def knn(self, z):
        k = min(self.K, self.index.ntotal)
        sims, ids = self.index.search(np.ascontiguousarray(z[None], dtype="float32"), k)
        sims, ids = sims[0], ids[0]
        w = np.exp((sims - sims.max()) / self.tau); w /= w.sum()      # softmax-weighted vote
        s = np.zeros(len(self.vocab), "float32")
        for wi, i in zip(w, ids):
            for c in self.ref[i][2]:
                s[c] += wi
        for c in range(self.n_orig, len(self.vocab)):
            if s[c] > 0:
                s[c] *= min(1.0, (self.med / max(self.cnt[c], 1)) ** KNN_PRIOR_ALPHA)
        return s, list(zip(ids.tolist(), sims.tolist()))

    def feats(self, z, zt=None):
        """(C, 4) feature matrix [knn, cls, txt, meta] for every category."""
        s_knn, nb = self.knn(z)
        s_cls = sigmoid(self.W @ z + self.b) if self.W is not None else np.zeros_like(s_knn)
        s_txt = self.T @ z
        s_meta = self.T @ zt if zt is not None else np.zeros_like(s_knn)
        return np.stack([s_knn, s_cls, s_txt, s_meta], 1).astype("float32"), nb

    def candidates(self, F, n=None):
        n = n or CANDIDATES_PER_SCORER
        cols = [j for j in range(F.shape[1]) if F[:, j].std() > 0]   # skip unused scorers (e.g. no title)
        return np.unique(np.concatenate([np.argsort(-F[:, j])[:n] for j in cols])) if cols else np.arange(len(F))

    def combine(self, F, mode="fusion"):
        if mode in MODES:
            return F[:, MODES[mode]]
        cand = self.candidates(F)
        out = np.full(len(F), -1.0, "float32")
        if self.fusion is not None:
            out[cand] = self.fusion.predict_proba(F[cand])[:, 1]
        else:
            wk, wc, wt = FALLBACK_WEIGHTS
            e = np.exp(100 * (F[:, 2] - F[:, 2].max())); soft = e / e.sum()
            out[cand] = (wk * F[:, 0] + wc * F[:, 1] + wt * soft)[cand]
        return out

    # ---------------- ranking + hierarchy ----------------
    def rank(self, z, zt=None, k=5, mode="fusion", hier=True):
        F, nb = self.feats(z, zt)
        s = self.combine(F, mode)
        order = np.argsort(-s)[:max(k * 3, 15)]
        ranked = [(int(c), float(s[c])) for c in order if s[c] > -1]
        if hier:   # most-specific convention: drop a category if one of its descendants is also proposed
            names = {c for c, _ in ranked}
            drop = set().union(*[self.anc.get(c, set()) & names for c, _ in ranked])
            ranked = [(c, sc) for c, sc in ranked if c not in drop]
        return [(self.vocab[c], sc) for c, sc in ranked[:k]], nb

    def suggest(self, img, title="", k=8):
        z = embed_images([img])[0]
        zt = embed_texts([clean_title(title)])[0] if title.strip() else None
        picks, nb = self.rank(z, zt, k)
        neigh = [(self.ref[i][0], self.ref[i][1], sim, [self.vocab[c] for c in self.ref[i][2]])
                 for i, sim in nb[:6]]
        return picks, neigh, z

    # ---------------- continual learning: no retraining ----------------
    def add(self, z, cats, pageid=-1, title="user upload"):
        self.index.add(np.ascontiguousarray(z[None], dtype="float32"))
        labs = [self.cidx[c] for c in cats if c in self.cidx]
        self.ref.append((pageid, title, labs)); self.cnt.update(labs)

    # ---------------- NEW CATEGORY: learned from Wikimedia, no retraining ----------------
    def add_category(self, name, items):
        """items: list of (pageid, title, z, extra_commons_categories).
        Adds the category (text prototype + head row) if new, and adds the examples to the kNN index."""
        if name not in self.cidx:
            t = embed_texts([TEXT_PROMPT.format(name)])[0]
            self.cidx[name] = len(self.vocab); self.vocab.append(name)
            self.T = np.vstack([self.T, t[None]]).astype("float32")
            if self.W is not None:                      # same text-initialisation the head started from
                self.W = np.vstack([self.W, (t * HEAD_INIT_SCALE)[None]]).astype("float32")
                self.b = np.append(self.b, HEAD_INIT_BIAS).astype("float32")
            self.anc[self.cidx[name]] = set()
        c = self.cidx[name]
        pos = {r[0]: i for i, r in enumerate(self.ref) if r[0] >= 0}
        added = 0
        for pid, title, z, extra in items:
            if pid >= 0 and pid in pos:                 # already indexed: just attach the new label
                labs = self.ref[pos[pid]][2]
                if c not in labs:
                    labs.append(c); self.cnt[c] += 1
                continue
            self.index.add(np.ascontiguousarray(z[None], dtype="float32"))
            labs = [c] + [self.cidx[x] for x in extra if x in self.cidx and x != name]
            self.ref.append((pid, title, labs)); self.cnt.update(labs); added += 1
        return added

    def self_check(self, name, Zheld):
        """Fraction of held-out images of the new category that get it in their top-5 (sanity check, not a benchmark)."""
        if not len(Zheld):
            return float("nan")
        hits = sum(name in [n for n, _ in self.rank(z, None, k=5, hier=False)[0]] for z in Zheld)
        return hits / len(Zheld)

    def save_user_cat(self, name, items):
        d = DATA / "user_cats"; d.mkdir(exist_ok=True)
        slug = re.sub(r"\W+", "_", name)[:60]
        np.savez(d / f"{slug}.npz", name=np.array(name), ids=np.array([i[0] for i in items]),
                 titles=np.array([i[1] for i in items]), Z=np.stack([i[2] for i in items]),
                 extra=np.array([json.dumps(list(i[3])) for i in items]))

    def _load_user_cats(self):
        d = DATA / "user_cats"
        for p in (sorted(d.glob("*.npz")) if d.exists() else []):
            f = np.load(p)
            if f["Z"].shape[1] != self.Z.shape[1]:
                print("skipping", p.name, "(made with a different encoder)"); continue
            items = [(int(i), str(t), z, json.loads(str(e)))
                     for i, t, z, e in zip(f["ids"], f["titles"], f["Z"], f["extra"])]
            self.add_category(str(f["name"]), items)

    def false_alarm(self, name, k=1, n=300):
        """How often the new category lands in the top-k for held-out images of OTHER categories (lower is better)."""
        idx = [i for i, r in enumerate(self.recs) if r["split"] != "train" and self.keep[i]][:n]
        if not idx:
            return float("nan")
        return sum(name in [x for x, _ in self.rank(self.Z[i], self.ZT[i], k=k, hier=False)[0]] for i in idx) / len(idx)
