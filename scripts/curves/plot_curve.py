from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_curves(csv_path: str, out_dir: str | None = None, show: bool = False) -> tuple[str, str]:
    df = pd.read_csv(csv_path)
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(csv_path), "plots")
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(csv_path))[0]

    plt.figure(figsize=(8, 5))
    plt.plot(df["slip"], df["torque_nm"], "-o", markersize=3)
    plt.xlabel("Slip")
    plt.ylabel("Torque (Nm)")
    plt.title(f"Torque vs Slip - {base}")
    plt.grid(True)
    torque_png = os.path.join(out_dir, f"{base}_torque.png")
    plt.tight_layout()
    plt.savefig(torque_png)
    if show:
        plt.show()
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["slip"], df["current_a"], "-o", markersize=3)
    plt.xlabel("Slip")
    plt.ylabel("Current (A)")
    plt.title(f"Current vs Slip - {base}")
    plt.grid(True)
    current_png = os.path.join(out_dir, f"{base}_current.png")
    plt.tight_layout()
    plt.savefig(current_png)
    if show:
        plt.show()
    plt.close()

    return torque_png, current_png


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Guimaraes curve CSVs")
    parser.add_argument("--csv", "-c", required=True, help="Path to curve CSV (outputs/curves/*.csv)")
    parser.add_argument("--out", "-o", help="Output directory for PNGs (defaults to <csv-folder>/plots)")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    torque_png, current_png = plot_curves(args.csv, out_dir=args.out, show=args.show)
    print("Saved:", torque_png)
    print("Saved:", current_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())