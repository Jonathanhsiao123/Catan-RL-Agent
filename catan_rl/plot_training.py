"""Plot training curves from metrics.csv files (headless-safe, writes PNG).

Usage:
  python -m catan_rl.plot_training checkpoints/seed0 checkpoints/seed1 --out curves.png
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(run_dir):
    path = os.path.join(run_dir, "metrics.csv")
    rows = list(csv.DictReader(open(path)))
    return {
        k: [float(r[k]) for r in rows]
        for k in ("steps", "winrate", "entropy", "policy_loss", "value_loss")
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="run directories containing metrics.csv")
    p.add_argument("--out", default="curves.png")
    args = p.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [
        ("winrate", "win rate (rolling 100 eps)", axes[0][0]),
        ("entropy", "policy entropy", axes[0][1]),
        ("policy_loss", "policy loss", axes[1][0]),
        ("value_loss", "value loss", axes[1][1]),
    ]
    for run in args.runs:
        m = load(run)
        label = os.path.basename(os.path.normpath(run))
        for key, _, ax in panels:
            ax.plot(m["steps"], m[key], label=label, alpha=0.85)
    for key, title, ax in panels:
        ax.set_title(title)
        ax.set_xlabel("agent steps")
        ax.grid(alpha=0.3)
        if key == "winrate":
            ax.axhline(0.25, ls="--", c="gray", lw=1, label="random baseline (25%)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
