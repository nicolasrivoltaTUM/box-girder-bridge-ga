"""
GA_Bin.py
=========
Python conversion of GA_Bin.m (MATLAB)

Binary Genetic Algorithm for minimisation of the cost function of a
post-tensioned multicellular box girder bridge.

Original: GA_Bin.m — Nicolás Ignacio Rivolta (2023)
Converted to Python by: Claude (Anthropic)

Usage
-----
    python GA_Bin.py

Dependencies
------------
    numpy, matplotlib
"""

import math
import time
import numpy as np

from f_costo_viga import f_costo_viga, decode_chromosome

L = 30.0  # bridge span [m], fixed case study parameter (see f_costo_viga.py)

# ------------------------------------------------------------------ #
#  DESIGN-SPACE PRESETS
#
#  The thesis (Ch. 7, "Resultados y conclusiones") runs the GA under
#  several configurations. These two are the ones with published,
#  independently-verifiable results:
#
#  "full"       -- Resultado N6, "Seccion optima de la viga cajon": all
#                  five variables searched freely within their full
#                  mechanical + constructive bounds. This is the thesis's
#                  headline result -- a $443,951/m design, a ~20% cost
#                  reduction vs. the traditional hand-calculated section
#                  ($552,340/m). Converges by generation ~40.
#  "fixed_h_lv" -- Resultado N7: Es, Ei and Ew are pinned at their
#                  code-minimum constructive values (0.20, 0.20, 0.30 m)
#                  and only the section height H and cantilever length Lv
#                  are searched. Demonstrates using the GA to pre-size
#                  just the two variables an engineer has the most
#                  freedom over. Published result: $504,479/m.
# ------------------------------------------------------------------ #
PRESETS: dict[str, dict] = {
    "full": {
        "lim": [
            [0.175, 0.35],    # Es  top slab thickness       [m]
            [0.14,  0.35],    # Ei  bottom slab thickness     [m]
            [L / 30, L / 10], # H   total section height      [m]
            [0.25,  0.50],    # Ew  web thickness             [m]
            [1.00,  2.50],    # Lv  cantilever length         [m]
        ],
        "thesis_result": "Resultado N6",
        "thesis_cost": 443_951.0,
    },
    "fixed_h_lv": {
        "lim": [
            [0.20, 0.201],    # Es  fixed at 0.20 m
            [0.20, 0.201],    # Ei  fixed at 0.20 m
            [L / 30, L / 10], # H   total section height      [m]
            [0.30, 0.301],    # Ew  fixed at 0.30 m
            [0.00, 2.50],     # Lv  cantilever length         [m]
        ],
        "thesis_result": "Resultado N7",
        "thesis_cost": 504_479.0,
    },
}
VAR_NAMES = ["Es", "Ei", "H", "Ew", "Lv"]


# ------------------------------------------------------------------ #
#  HELPER: initial population
# ------------------------------------------------------------------ #
def init_pop(nvars: int, psize: int, rng: np.random.Generator) -> np.ndarray:
    """
    Create a random binary population.

    Returns
    -------
    Pop : (psize, nvars) array of 0/1 uint8
    """
    return rng.integers(0, 2, size=(psize, nvars), dtype=np.int8)


# ------------------------------------------------------------------ #
#  HELPER: roulette-wheel selection (MATLAB faithful translation)
# ------------------------------------------------------------------ #
def roulette_selection(Sol: np.ndarray, psize: int, nelit: int,
                        rng: np.random.Generator) -> np.ndarray:
    """
    Return indices (into original population) of (psize - nelit) selected
    individuals using the MATLAB roulette logic (walk backwards through the
    cumulative probability vector).

    NOTE: In the original code the *sorted-descending* fitness values are
    used to build probabilities, so the best individuals get the lowest
    probability — this is intentional for a maximisation problem where the
    GA was later flipped to minimisation via large penalties.  We keep the
    same logic here for fidelity.
    """
    Sum  = Sol.sum()
    Prob = np.cumsum(Sol / Sum)   # cumulative probabilities

    Selec = np.empty(psize - nelit, dtype=int)
    pos = psize - 1
    I   = 0
    while I < psize - nelit:
        if Prob[pos] > rng.random():
            Selec[I] = pos
            I += 1
        pos = pos - 1 if pos > 0 else psize - 1
    return Selec


# ------------------------------------------------------------------ #
#  CORE: run_ga — reusable GA routine, returns full history
# ------------------------------------------------------------------ #
def run_ga(psize: int = 5000, ngener: int = 50, probmut: float = 2,
           nelit_pct: float = 2, seed: int | None = 42,
           preset: str = "full", verbose: bool = True) -> dict:
    """
    Run the binary genetic algorithm and return the full convergence
    history, so callers (scripts, notebooks, visualisation tools) can
    do whatever they like with it — print it, plot it, save it to CSV.

    Parameters
    ----------
    psize     : population size
    ngener    : number of generations
    probmut   : mutation probability [%]
    nelit_pct : elite percentage [%]
    seed      : RNG seed (None for a non-reproducible run)
    preset    : design-space configuration, one of PRESETS.keys()
                ("full" = Resultado N6, "fixed_h_lv" = Resultado N7)
    verbose   : print per-generation progress to stdout

    Returns
    -------
    dict with keys:
        'generation'   : np.ndarray, generation numbers (1..ngener)
        'best'         : np.ndarray, best (lowest) cost per generation
        'mean'         : np.ndarray, mean cost per generation
        'std'          : np.ndarray, std-dev of cost per generation
        'best_vars_hist' : np.ndarray (ngener, nvars), best-so-far design
                           variables at each generation (for animation)
        'best_individual' : np.ndarray, best chromosome found
        'best_cost'    : float, best cost found overall
        'best_vars'    : np.ndarray, decoded design variables of best_individual
        'var_names'    : list of str, names of the design variables
        'lim'          : np.ndarray, variable bounds used
        'preset'       : str, preset name used
        'elapsed_sec'  : float, wall-clock run time
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose from {list(PRESETS)}")

    funct = f_costo_viga
    ngen  = 5

    lim = np.array(PRESETS[preset]["lim"])
    var_names = VAR_NAMES

    ncro    = 2**6
    nvars   = ncro * ngen
    nelit   = math.ceil(nelit_pct / 100 * psize)

    rng = np.random.default_rng(seed)

    # Precompute which bit columns belong to odd/even genes
    # (for odd-generation single-point crossover — MATLAB croF / croM logic)
    croF = []   # columns for genes 1, 3, 5 (0-indexed: 0, 2, 4)
    croM = []
    for icro in range(ngen):
        start = icro * ncro
        cols  = list(range(start, start + ncro))
        if icro % 2 == 0:       # MATLAB: mod(icro,2)==1 → 1-indexed odd
            croF.extend(cols)
        else:
            croM.extend(cols)
    croF = np.array(croF, dtype=int)
    croM = np.array(croM, dtype=int)

    # ── INITIAL POPULATION ──────────────────────────────────────── #
    Pop = init_pop(nvars, psize, rng)

    # ── CONVERGENCE HISTORY ─────────────────────────────────────── #
    best_hist = np.empty(ngener)
    mean_hist = np.empty(ngener)
    std_hist  = np.empty(ngener)
    best_vars_hist = np.empty((ngener, ngen))
    BestInd = None
    best_cost_ever = np.inf

    t0 = time.time()

    # ── GENERATIONAL LOOP ───────────────────────────────────────── #
    for igen in range(1, ngener + 1):

        # Evaluate objective function for every individual
        Sol = np.array([funct(Pop[ip], lim, ngen, ncro) for ip in range(psize)])

        # Sort descending (MATLAB `sort(Sol,'descend')`)
        ix      = np.argsort(Sol)[::-1]
        Sol     = Sol[ix]
        Pop_sorted = Pop[ix]

        Best = Sol[-1]
        Mean = Sol.mean()
        Std  = Sol.std()

        # Track the best-ever individual (elitism should make this monotonic,
        # but we guard explicitly in case elitism settings ever change)
        if Best < best_cost_ever:
            best_cost_ever = Best
            BestInd = Pop_sorted[-1].copy()

        if verbose:
            print(f"Gen {igen:3d}/{ngener}  |  Best: {Best:,.2f}  |  Mean: {Mean:,.2f}  |  "
                  f"Std: {Std:,.2f}  |  t={time.time()-t0:.1f}s")

        best_hist[igen - 1] = best_cost_ever
        mean_hist[igen - 1] = Mean
        std_hist[igen - 1]  = Std
        best_vars_hist[igen - 1] = decode_chromosome(BestInd, lim, ngen, ncro)

        # Elite: keep best nelit individuals (end of sorted array = lowest Sol)
        Elit = Pop_sorted[-nelit:]

        # ── SELECTION (roulette) ──────────────────────────────── #
        Selec = roulette_selection(Sol, psize, nelit, rng)
        Pop   = np.vstack([Pop_sorted[Selec], Elit])

        # ── CROSSOVER ────────────────────────────────────────── #
        n_pairs = round(rng.random() * (psize - nelit) * 0.5)
        non_elite_idx = list(range(psize - nelit))

        for _ in range(n_pairs):
            ind1 = int(rng.integers(0, psize - nelit))
            ind2 = ind1
            while ind2 == ind1:
                ind2 = int(rng.integers(0, psize - nelit))

            IndM = Pop[ind1].copy()
            IndF = Pop[ind2].copy()

            if igen % 2 == 0:
                # Even generation: uniform mask crossover
                mask = rng.integers(0, 2, size=nvars, dtype=np.int8)
                Pop[ind1, mask == 1] = IndF[mask == 1]
                Pop[ind2, mask == 0] = IndM[mask == 0]
            else:
                # Odd generation: single-point (gene-aware) crossover
                Pop[ind1, croF] = IndF[croF]
                Pop[ind2, croM] = IndM[croM]

        # ── MUTATION (bit-flip) ───────────────────────────────── #
        for ind in range(psize - nelit):
            if rng.random() <= probmut / 100:
                ivar = int(rng.integers(0, nvars))
                Pop[ind, ivar] = 1 - Pop[ind, ivar]

    elapsed = time.time() - t0
    best_vars = decode_chromosome(BestInd, lim, ngen, ncro)
    thesis_cost = PRESETS[preset]["thesis_cost"]

    if verbose:
        print("\n" + "=" * 70)
        print("  OPTIMAL SOLUTION")
        print("=" * 70)
        print(f"\n  Best Cost : {best_cost_ever:,.2f} ARS/m"
              f"   (thesis {PRESETS[preset]['thesis_result']}: {thesis_cost:,.0f} ARS/m, "
              f"{100 * (best_cost_ever - thesis_cost) / thesis_cost:+.2f}%)\n")
        for name, val in zip(var_names, best_vars):
            print(f"    {name:10s}: {val:.6f} m")
        print(f"\n  Elapsed time: {elapsed:.1f}s")
        print("-" * 70)

    return {
        "generation": np.arange(1, ngener + 1),
        "best": best_hist,
        "mean": mean_hist,
        "std": std_hist,
        "best_vars_hist": best_vars_hist,
        "best_individual": BestInd,
        "best_cost": float(best_cost_ever),
        "best_vars": best_vars,
        "var_names": var_names,
        "lim": lim,
        "preset": preset,
        "elapsed_sec": elapsed,
    }


# ------------------------------------------------------------------ #
#  MAIN — run the GA standalone (no plotting; see visualize_convergence.py)
# ------------------------------------------------------------------ #
def main():
    print("-" * 70)
    print("         AG Bin - Binary Genetic Algorithm")
    print("  Heuristic optimisation of the multicellular box girder bridge")
    print("  Nicolas Ignacio Rivolta (2023) -- Universidad Nacional del Nordeste")
    print("-" * 70)

    history = run_ga(psize=5000, ngener=50, probmut=2, nelit_pct=2, seed=42, preset="full")
    return history


if __name__ == "__main__":
    main()
