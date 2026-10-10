"""Run stages in order:  python run_pipeline.py --from 2 [--to 6]
Stages: 2 prepare | 3 embed | 4 head | 5 fusion | 6 evaluate   (stage 1 / 1b are run by hand)"""
import argparse, subprocess, sys
STAGES = {2: "02_prepare.py", 3: "03_embed.py", 4: "04_train_head.py", 5: "05_fit_fusion.py", 6: "06_evaluate.py"}
ap = argparse.ArgumentParser(); ap.add_argument("--from", dest="a", type=int, default=2)
ap.add_argument("--to", dest="b", type=int, default=6); ap.add_argument("--tag", default="run"); x = ap.parse_args()
for n in range(x.a, x.b + 1):
    cmd = [sys.executable, STAGES[n]] + (["--tag", x.tag] if n == 6 else [])
    print("\n=== stage", n, "->", " ".join(cmd)); 
    if subprocess.run(cmd).returncode:
        sys.exit(f"stage {n} failed")
