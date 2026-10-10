"""Compare saved evaluation runs:  python compare_runs.py [--config fusion+hier]"""
import argparse, glob, json
ap = argparse.ArgumentParser(); ap.add_argument("--config", default="fusion+hier"); a = ap.parse_args()
keys = ["P@1", "P@3", "F1@5", "R@5", "macro_recall@5", "recall@5_tail"]
print(f"{'run':<22}" + "".join(f"{k:>16}" for k in keys))
for fn in sorted(glob.glob("data/results_*.json")):
    r = json.load(open(fn)); c = r.get(a.config)
    if c:
        print(f"{fn[13:-5]:<22}" + "".join(f"{c.get(k, float('nan')):>16.3f}" for k in keys))
