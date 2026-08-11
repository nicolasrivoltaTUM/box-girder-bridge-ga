# MATLAB → Python conversion notes

## "Optimización de un puente de sección cajón multicelular a partir de algoritmos genéticos" — Nicolás Rivolta (2023)

This document records how `GA_Bin.py` / `f_costo_viga.py` relate to the
original MATLAB scripts (`GA_Bin.m` / `f_CostoViga.m`, reproduced in full in
the thesis appendix, `thesis.pdf` pp. 269-299), including two bugs that were
found and fixed while validating the port against the thesis's own published
results (Chapter 7).

## Files

| File | Original | Description |
|---|---|---|
| `GA_Bin.py` | `GA_Bin.m` | Binary genetic algorithm main loop |
| `f_costo_viga.py` | `f_CostoViga.m` | Structural objective function |

## Validation method

Rather than trust a line-by-line read of ~1,500 lines of MATLAB, each
candidate fix was checked by plugging the thesis's own published optimal
design variables (Tabla VII.17, VII.20 — Resultado N6 and N7) directly into
`f_costo_viga.py` (bypassing the GA, via a chromosome that decodes to exact
target values) and comparing the computed cost to the published figure.

## Two bugs found and fixed

### 1. Inverted compression-stress check

The allowable compressive stress constants (`fadm_comp_1`, `fadm_comp_2`) are
positive magnitudes (e.g. `0.6 * fc'c`), while compressive stresses are
negative by this function's sign convention. The original port compared
`stress < adm_comp` — true for virtually any compressive stress — so every
candidate design failed this check and picked up the $1,000,000 penalty.
Verified against the MATLAB source (`f_CostoViga.m` line ~903):

```matlab
if (fcs_1 < 0)
    if (fcs_1 <= fadm_comp_1)
        F_Penaliz(1) = 1;   % pass
    else
        F_Penaliz(1) = 0;   % fail
    end
```

Fixed to `stress > adm_comp` in `f_costo_viga.py`, matching this logic.
Symptom before the fix: a full 5,000-individual x 50-generation GA run never
found a feasible solution — every run's reported "best" cost was almost
exactly $1,000,000 above the thesis's published figures.

### 2. Third design variable: total height vs. void height

The thesis's own design-variable and restriction tables (e.g. Tabla V.1,
Tabla VII.17) define X3 as the **total section height H**, bounded by the
classic L/30-L/10 span-to-depth serviceability rule. The MATLAB appendix
code instead decodes `X(3)` into a variable named `H_hueco` (void height)
and derives `H = H_hueco + Es + Ei`.

Reproducing the thesis's published numbers only works if X3 is read as the
total height H directly, with the void height derived as
`H_hueco = H - Es - Ei` — the reverse of what the appendix code does. This
is almost certainly a variable-naming leftover from an earlier version of
the script (the console output even prints `X(3)` under the label
`H_hueco`), not an intentional modeling choice — the thesis's minimum-depth
guideline (L/30-L/10) is a standard total-depth rule, not a void-depth rule,
and treating X3 as H_hueco directly makes the L/30-L/10 bound assign
unrealistic total section heights (H = H_hueco + ~0.3-0.5 m).

`f_costo_viga.py` decodes X3 as `H` and derives `H_hueco` internally.

### Result after both fixes

Both fixes together reproduce the thesis's Chapter 7 headline numbers to
within half a percent, using the exact bounds and GA parameters (population
5,000, 50 generations) from the corresponding thesis results:

| | This port | Thesis (Tabla VII.18 / VII.21) | Diff |
|---|---|---|---|
| Resultado N6 ("full" preset) | $444,054.74/m | $443,951/m | +0.02% |
| Resultado N7 ("fixed_h_lv" preset) | $504,675.86/m | $504,479/m | +0.04% |

## Other MATLAB → Python translation notes

### Binary encoding / decoding
**MATLAB:**
```matlab
Ind = strrep(num2str(Ind), ' ', '');
X(ix) = lim(ix,1) + (lim(ix,2)-lim(ix,1)) / (2^ncro-1) * bin2dec(Ind(...));
```
**Python:**
```python
bit_str = "".join(str(int(b)) for b in ind)
X[i] = lim[i,0] + (lim[i,1]-lim[i,0]) / (2**ncro-1) * int(segment, 2)
```

### Sorting
**MATLAB:** `[Sol,ix] = sort(Sol,'descend')`
**Python:** `ix = np.argsort(Sol)[::-1]; Sol = Sol[ix]`

### Random numbers
**MATLAB:** `rand` -> uniform [0,1], `round(rand*N)` -> int
**Python:** `rng.random()`, `rng.integers(0, N+1)` via a seeded `np.random.default_rng`

### 1-indexed to 0-indexed
All MATLAB 1-based indices are converted to 0-based Python indices, e.g.
`Pop(ix(psize),:)` -> `Pop_sorted[-1]`.

### Penalty logic
`Funcion_Pen` is set to either 1,000,000 or 0 depending on whether any
verification failed, then **added** (not multiplied) into the final cost
expression — reproduced as an additive penalty in `f_costo_viga.py`.

### SLS penalty flags collapsing onto one shared index (kept, not "fixed")
In the MATLAB source, all four SLS stress checks (top/bottom fibre, states 1
and 2) — and every later secondary/constructive check (bar spacing, stirrup
spacing, cantilever deflection, slab ratio, web slenderness) — write to the
*same* `F_Penaliz(1)` array slot, each overwriting the last. Only the final
check evaluated in the source's execution order actually determines that
slot's value; the other three SLS results are computed but immediately
discarded. `f_costo_viga.py` reproduces this exactly (all four `_check_stress`
calls plus every secondary check target `F_Penaliz[0]`), since deviating
from it changes which candidate designs are considered feasible and was
found empirically to move the reproduced results further from the thesis's
published figures, not closer.

### Performance
The Python loop over `psize` individuals per generation is slower than
MATLAB's vectorized/JIT'd equivalent; a full 5,000 x 50 run still completes
in under 10 seconds on a modern laptop, so no further optimization was done.
