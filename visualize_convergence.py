"""
visualize_convergence.py
=========================
Runs the binary genetic algorithm (GA_Bin.py) and visualizes how the
cost of the box girder bridge decreases generation over generation.

Anyone who runs this script gets:
    1. Live progress printed to the console.
    2. A static convergence plot (best cost, mean cost, ± std-dev band)
       and a bar chart of the final optimal design variables, saved as
       a PNG.
    3. An animated GIF of the same two panels evolving generation by
       generation, so you can *watch* the population converge.

Usage
-----
    python visualize_convergence.py

Optional flags
---------------
    --preset {full,fixed_h_lv}
                  design-space configuration                (default: full)
                  "full"       = Resultado N6 (thesis headline result,
                                 all 5 variables searched freely)
                  "fixed_h_lv" = Resultado N7 (Es/Ei/Ew pinned at
                                 constructive minimums, only H and Lv
                                 searched)
    --psize N     population size            (default: 5000)
    --ngener N    number of generations       (default: 50)
    --probmut P   mutation probability [%]    (default: 2)
    --seed N      RNG seed, or "none" for a   (default: 42)
                  non-reproducible run
    --no-show     save the plots but don't open a window
                  (useful when running on a server / CI)
    --no-animate  skip the animated GIF (faster; static PNG only)
    --gif-fps N   animation playback speed    (default: 8)

Examples
--------
    # Quick run for a fast preview
    python visualize_convergence.py --psize 500 --ngener 30

    # Full run matching the thesis's headline result (Resultado N6)
    python visualize_convergence.py --psize 5000 --ngener 50

    # Reproduce the thesis's Resultado N7 instead
    python visualize_convergence.py --preset fixed_h_lv

Dependencies
------------
    numpy, matplotlib (Pillow, a matplotlib dependency, is used for GIF export)
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from GA_Bin import run_ga, PRESETS


def parse_args():
    p = argparse.ArgumentParser(description="Visualize GA cost convergence")
    p.add_argument("--preset", type=str, default="full", choices=list(PRESETS),
                    help="Design-space configuration (thesis Resultado N6 or N7)")
    p.add_argument("--psize", type=int, default=5000, help="Population size")
    p.add_argument("--ngener", type=int, default=50, help="Number of generations")
    p.add_argument("--probmut", type=float, default=2, help="Mutation probability [%%]")
    p.add_argument("--seed", type=str, default="42", help="RNG seed, or 'none'")
    p.add_argument("--no-show", action="store_true", help="Don't open plot windows")
    p.add_argument("--no-animate", action="store_true", help="Skip the animated GIF")
    p.add_argument("--gif-fps", type=int, default=8, help="Animation frames per second")
    p.add_argument("--outfile", type=str, default="convergence_plot.png",
                    help="Output PNG filename")
    p.add_argument("--gif-outfile", type=str, default="convergence_animation.gif",
                    help="Output GIF filename")
    return p.parse_args()


def _annotation_text(history: dict) -> str:
    thesis_cost = PRESETS[history["preset"]]["thesis_cost"]
    thesis_name = PRESETS[history["preset"]]["thesis_result"]
    best = history["best"]
    reduction = best[0] - best[-1]
    pct = 100 * reduction / best[0] if best[0] else 0
    diff_pct = 100 * (best[-1] - thesis_cost) / thesis_cost
    return (f"Best cost reduced by {reduction:,.0f} ARS/m ({pct:.2f}%)\n"
            f"vs. thesis {thesis_name}: {thesis_cost:,.0f} ARS/m ({diff_pct:+.2f}%)")


def plot_convergence(history: dict, outfile: str, show: bool = True):
    """
    Build a two-panel static figure:
        Top    — best & mean cost per generation, with ± std-dev band
        Bottom — final optimal design variables as a bar chart
    """
    gens  = history["generation"]
    best  = history["best"]
    mean  = history["mean"]
    std   = history["std"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1.fill_between(gens, mean - std, mean + std, color="tab:blue",
                      alpha=0.15, label="Mean ± std dev")
    ax1.plot(gens, mean, color="tab:orange", linestyle="--", linewidth=1.5,
              label="Mean cost")
    ax1.plot(gens, best, color="tab:blue", linewidth=2.2, marker="o",
              markersize=3, label="Best cost")

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Cost [ARS / m]")
    ax1.set_title("Genetic algorithm convergence — box girder bridge cost "
                   f"(preset: {history['preset']})")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.ticklabel_format(style="plain", axis="y")

    y_top = max(mean[0], (mean + std).max())
    ax1.annotate(
        _annotation_text(history),
        xy=(gens[-1], best[-1]),
        xytext=(gens[len(gens) // 2], y_top),
        arrowprops=dict(arrowstyle="->", color="gray", connectionstyle="arc3,rad=0.15"),
        fontsize=9, color="dimgray", ha="center",
    )

    names = history["var_names"]
    vals  = history["best_vars"]
    bars = ax2.bar(names, vals, color="teal")
    ax2.set_ylabel("Value [m]")
    ax2.set_title("Optimal design variables found")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.4f}",
                  ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"\nPlot saved to: {os.path.abspath(outfile)}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def animate_convergence(history: dict, outfile: str, fps: int = 8):
    """
    Build an animated GIF of the same two panels, revealed generation by
    generation, so the convergence is visible as motion rather than a
    single static curve.
    """
    gens  = history["generation"]
    best  = history["best"]
    mean  = history["mean"]
    std   = history["std"]
    names = history["var_names"]
    vars_hist = history["best_vars_hist"]
    ngener = len(gens)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [2, 1]}
    )

    # ── Static axis setup (limits fixed up front so the animation doesn't jump) ──
    y_min = min(best.min(), (mean - std).min())
    y_max = max(mean[0], (mean + std).max())
    pad = 0.05 * (y_max - y_min)
    ax1.set_xlim(gens[0], gens[-1])
    ax1.set_ylim(max(0, y_min - pad), y_max + pad)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Cost [ARS / m]")
    ax1.set_title("Genetic algorithm convergence — box girder bridge cost "
                   f"(preset: {history['preset']})")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.ticklabel_format(style="plain", axis="y")

    mean_line, = ax1.plot([], [], color="tab:orange", linestyle="--", linewidth=1.5, label="Mean cost")
    best_line, = ax1.plot([], [], color="tab:blue", linewidth=2.2, marker="o", markersize=3, label="Best cost")
    ax1.fill_between([], [], [], color="tab:blue", alpha=0.15, label="Mean ± std dev")  # legend entry only
    ax1.legend(loc="upper right")
    gen_text = ax1.text(0.02, 0.95, "", transform=ax1.transAxes, va="top", fontsize=10,
                         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
    band_holder = {"artist": None}

    ax2.set_ylim(0, vars_hist.max() * 1.25)
    ax2.set_ylabel("Value [m]")
    ax2.set_title("Best design variables found so far")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    bars = ax2.bar(names, np.zeros(len(names)), color="teal")
    bar_labels = [ax2.text(i, 0, "", ha="center", va="bottom", fontsize=9) for i in range(len(names))]

    def update(frame):
        i = frame + 1
        if band_holder["artist"] is not None:
            band_holder["artist"].remove()
        band_holder["artist"] = ax1.fill_between(
            gens[:i], (mean - std)[:i], (mean + std)[:i], color="tab:blue", alpha=0.15)
        mean_line.set_data(gens[:i], mean[:i])
        best_line.set_data(gens[:i], best[:i])
        gen_text.set_text(f"Generation {gens[i-1]}/{ngener}\nBest: {best[i-1]:,.0f} ARS/m")

        for bar, v, lbl in zip(bars, vars_hist[i - 1], bar_labels):
            bar.set_height(v)
            lbl.set_position((bar.get_x() + bar.get_width() / 2, v))
            lbl.set_text(f"{v:.4f}")

        return [band_holder["artist"], mean_line, best_line, gen_text, *bars, *bar_labels]

    anim = FuncAnimation(fig, update, frames=ngener, blit=False)
    plt.tight_layout()

    anim.save(outfile, writer=PillowWriter(fps=fps),
              savefig_kwargs={"facecolor": "white"})
    print(f"Animation saved to: {os.path.abspath(outfile)}")
    plt.close(fig)


def main():
    args = parse_args()
    seed = None if args.seed.lower() == "none" else int(args.seed)

    print("-" * 70)
    print("  Running GA + visualizing cost convergence")
    print(f"  preset={args.preset}  psize={args.psize}  ngener={args.ngener}  "
          f"probmut={args.probmut}%  seed={seed}")
    print("-" * 70)

    history = run_ga(
        psize=args.psize,
        ngener=args.ngener,
        probmut=args.probmut,
        seed=seed,
        preset=args.preset,
        verbose=True,
    )

    plot_convergence(history, outfile=args.outfile, show=not args.no_show)

    if not args.no_animate:
        animate_convergence(history, outfile=args.gif_outfile, fps=args.gif_fps)


if __name__ == "__main__":
    main()
