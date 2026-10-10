"""STAGE 5 - Calibrated stacking: logistic regression over [knn, cls, txt, meta] fitted on the VAL split.
Output: data/fusion.joblib       Run: python 05_fit_fusion.py   (re-run after any change to scorers/head)
"""
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from suggester import Suggester
from common import *

S = Suggester(); X, y = [], []
for i, r in enumerate(S.recs):
    if r["split"] != "val" or not S.keep[i]:
        continue
    F, _ = S.feats(S.Z[i], S.ZT[i]); cand = S.candidates(F)
    Y = np.zeros(len(S.vocab)); Y[[S.cidx[c] for c in r["labels"]]] = 1
    X.append(F[cand]); y.append(Y[cand])
if not X:
    raise SystemExit("No validation images - collect more data or lower N_MIN / change the split in config.py")
X, y = np.concatenate(X), np.concatenate(y)
print("fusion training rows:", X.shape, "| positives:", int(y.sum()))
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X, y)
print("standardised coefficients [knn, cls, txt, meta]:", model[-1].coef_.round(2))
joblib.dump(model, DATA / "fusion.joblib")
print("saved data/fusion.joblib")
