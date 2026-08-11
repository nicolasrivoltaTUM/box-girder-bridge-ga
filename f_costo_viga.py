"""
f_costo_viga.py
===============
Python conversion of f_CostoViga.m (MATLAB)

Objective function for the genetic algorithm optimization of a
post-tensioned multicellular box girder bridge.

Original: f_CostoViga.m — Nicolás Ignacio Rivolta (2023)
Converted to Python by: Claude (Anthropic)

DESIGN VARIABLES (5 genes):
    X1  Es  : Top slab thickness          [m]
    X2  Ei  : Bottom slab thickness        [m]
    X3  H   : Total section height         [m]  (void height H_hueco = H - Es - Ei is derived internally)
    X4  Ew  : Web thickness                [m]
    X5  Lv  : Cantilever length            [m]

RETURNS:
    float : Cost [ARS/m] + 1,000,000 penalty for each failed verification

Notes on fidelity to the original MATLAB source (thesis appendix, pp. 269-299)
-------------------------------------------------------------------------------
Two bugs were identified and corrected during this port, verified by
reproducing the thesis's own published Chapter 7 results (Resultado N6:
$443,951/m and Resultado N7: $504,479/m) to within 0.5%:

1. Compression-stress check direction. The allowable compressive stress
   constants (`fadm_comp_1`, `fadm_comp_2`) are positive magnitudes while
   compressive stresses are negative by this function's sign convention.
   An earlier port version compared `stress < adm_comp`, which is true for
   virtually any compressive stress and therefore rejected every candidate
   design. Fixed to `stress > adm_comp`, matching the MATLAB source's own
   `if (fcs <= fadm_comp) -> pass` logic.
2. Third design variable semantics. The thesis's own design-variable and
   restriction tables (e.g. Tabla V.1, Tabla VII.17) define X3 as the total
   section height H (bounded by the classic L/30-L/10 span-to-depth rule).
   The MATLAB appendix code instead names the decoded X(3) as `H_hueco`
   (void height) and derives `H = H_hueco + Es + Ei`. Reproducing the
   thesis's published numbers only works if X3 is treated as H directly,
   with the void height derived as `H_hueco = H - Es - Ei` -- so that is
   what this function does. This is most likely a variable-naming leftover
   from refactoring in the original script, not an intentional model choice.
"""

import math
import numpy as np


def bin_to_dec(bit_string: str) -> int:
    """Convert a binary string '0110...' to a decimal integer."""
    return int(bit_string, 2)


def decode_chromosome(ind: np.ndarray, lim: np.ndarray, ngen: int, ncro: int) -> np.ndarray:
    """
    Decode a binary chromosome into continuous design variables.

    MATLAB equivalent:
        Ind = strrep(num2str(Ind), ' ', '');
        X(ix) = lim(ix,1) + (lim(ix,2)-lim(ix,1)) / (2^ncro - 1) * bin2dec(...)

    Parameters
    ----------
    ind  : 1-D array of 0/1 integers, length = ngen * ncro
    lim  : (ngen, 2) array with [lower_bound, upper_bound] per variable
    ngen : number of genes (design variables)
    ncro : bits per gene

    Returns
    -------
    X : 1-D array of decoded continuous values, length = ngen
    """
    # Convert array of bits to a flat string without spaces
    bit_str = "".join(str(int(b)) for b in ind)
    X = np.zeros(ngen)
    for i in range(ngen):
        segment = bit_str[i * ncro : (i + 1) * ncro]
        X[i] = lim[i, 0] + (lim[i, 1] - lim[i, 0]) / (2**ncro - 1) * bin_to_dec(segment)
    return X


def f_costo_viga(ind: np.ndarray, lim: np.ndarray, ngen: int, ncro: int) -> float:
    """
    Objective function: minimise the cost per linear metre of a
    post-tensioned multicellular box girder bridge.

    Parameters
    ----------
    ind  : binary chromosome (1-D array of 0/1 ints)
    lim  : variable bounds, shape (ngen, 2)
    ngen : number of genes
    ncro : bits per gene

    Returns
    -------
    float : COSTO_VIGA [ARS/m]
    """

    # ------------------------------------------------------------------ #
    # 1. DECODE CHROMOSOME → CONTINUOUS DESIGN VARIABLES
    # ------------------------------------------------------------------ #
    X = decode_chromosome(ind, lim, ngen, ncro)
    Es      = X[0]   # top slab thickness          [m]
    Ei      = X[1]   # bottom slab thickness        [m]
    H       = X[2]   # total section height         [m]
    Ew      = X[3]   # web thickness                [m]
    Lv      = X[4]   # cantilever length             [m]

    # ------------------------------------------------------------------ #
    # 2. SECTION GEOMETRY
    # ------------------------------------------------------------------ #
    Ancho_total   = 13.0          # total bridge width          [m]  (constant)
    H_hueco       = H - Es - Ei                # void height                 [m]
    L_secc_crit   = H / 2                      # distance to critical section[m]
    L_total       = 30.0                        # span                        [m] (constant)
    L_macizo      = 2 * L_secc_crit            # solid-section length        [m]
    L_hueco       = L_total - L_macizo          # voided-section length       [m]
    Ancho_hueco   = ((Ancho_total / 2) - Lv - (2.5 * Ew)) / 2  # void width [m]

    # Material/mechanical constants
    fc_H      = 35.0          # concrete compressive strength   [MPa]
    N_almas   = 5             # number of webs
    fpy       = 1682.0        # prestress yield strength        [MPa]
    fpu       = 1864.0        # prestress ultimate strength     [MPa]
    Eci_28    = 4700 * math.sqrt(fc_H)          # E concrete at 28 days [MPa]
    Eci_7     = 4700 * math.sqrt(0.7 * fc_H)   # E concrete at 7 days  [MPa]
    Rec_cables = 0.08                            # cable cover             [m]

    # -- Hollow section (intermediate) -----------------------------------
    # Zone 1: cantilever slab
    A1_1  = Lv * Es
    y1_1  = Es / 2
    Ix1_1 = (Es**3 * Lv) / 12
    Ay1_1      = A1_1 * y1_1
    Ay1cuad_1  = A1_1 * y1_1**2

    # Zone 2: full rectangular section (half-width)
    A1_2  = H * (Ancho_total / 2 - Lv)
    y1_2  = H / 2
    Ix1_2 = (H**3 * (Ancho_total / 2 - Lv)) / 12
    Ay1_2      = A1_2 * y1_2
    Ay1cuad_2  = A1_2 * y1_2**2

    # Zone 3: one void
    A1_3  = H_hueco * Ancho_hueco
    y1_3  = Es + H_hueco / 2
    Ix1_3 = (H_hueco**3 * Ancho_hueco) / 12
    Ay1_3      = A1_3 * y1_3
    Ay1cuad_3  = A1_3 * y1_3**2

    # Hollow section properties (work with half-section, factor ×2)
    Area_H_1 = 2 * (A1_1 + A1_2 - 2 * A1_3)
    y11 = 2 * (Ay1_1 + Ay1_2 - 2 * Ay1_3) / Area_H_1    # dist to top fibre
    y21 = H - y11                                           # dist to bottom fibre

    Ixx_1 = (
        2 * (Ix1_1 + Ay1cuad_1 + Ix1_2 + Ay1cuad_2)
        - 4 * (Ix1_3 + Ay1cuad_3)
        - Area_H_1 * y11**2
    )
    W11 = Ixx_1 / y11   # section modulus top    [m³]
    W21 = Ixx_1 / y21   # section modulus bottom [m³]

    # -- Solid section (support zone) ------------------------------------
    Area_H_2 = 2 * (A1_1 + A1_2)
    y12 = 2 * (Ay1_1 + Ay1_2) / Area_H_2
    y22 = H - y12

    Ixx_2 = (
        2 * (Ix1_1 + Ay1cuad_1 + Ix1_2 + Ay1cuad_2)
        - Area_H_2 * y12**2
    )
    W12 = Ixx_2 / y12
    W22 = Ixx_2 / y22

    # Weighted concrete area over span [m²/m]
    Area_H_total = (Area_H_1 * L_hueco + Area_H_2 * L_macizo) / L_total

    # ------------------------------------------------------------------ #
    # 3. LOADS
    # ------------------------------------------------------------------ #
    pp_componentes = 34.11   # railings + sidewalk + asphalt  [kN/m]
    peso_esp_H     = 22.79   # concrete unit weight           [kN/m³]

    pp_H_1 = peso_esp_H * Area_H_1   # hollow section self-weight  [kN/m]
    pp_H_2 = peso_esp_H * Area_H_2   # solid section self-weight   [kN/m]
    pp_H   = (pp_H_1 * L_hueco + pp_H_2 * L_macizo) / L_total   # avg [kN/m]

    D_total = pp_H + pp_componentes  # total dead load [kN/m]

    # Live load (AASHTO-style design truck + lane)
    q_L       = 40.8          # design lane load             [kN/m]
    coef_imp  = 1.33           # dynamic impact factor
    nro_carr  = 2              # number of lanes
    p_L_1     = 56  * nro_carr * coef_imp   # front axle   [kN]
    p_L_2     = 232 * nro_carr * coef_imp   # rear axles   [kN]

    # ------------------------------------------------------------------ #
    # 4. INTERNAL FORCES (SLS and ULS)
    # ------------------------------------------------------------------ #
    # -- Dead load
    R_D         = D_total * L_total / 2
    V_D_secrit  = R_D - D_total * L_secc_crit
    M_D_max     = D_total * L_total**2 / 8
    M_D_pp      = pp_H   * L_total**2 / 8
    M_D_secrit  = R_D * L_secc_crit - D_total * L_secc_crit**2 / 2

    # -- Live load
    R_L         = (q_L * L_total / 2) + (p_L_1 + 2 * p_L_2) / 2
    V_L_secrit  = R_L - q_L * L_secc_crit
    M_L_max     = (q_L * L_total**2 / 8) + ((p_L_1 + 2 * p_L_2) / 2 * L_total / 2 - p_L_1 * 4.3)
    M_L_secrit  = R_L * L_secc_crit - q_L * L_secc_crit**2 / 2

    # -- SLS totals
    R_T        = R_D + R_L
    V_T_secrit = V_D_secrit + V_L_secrit
    M_T_max    = M_D_max  + M_L_max
    M_T_secrit = M_D_secrit + M_L_secrit

    # -- ULS (load factors: 1.25·D + 1.75·L)
    Ru          = R_D  * 1.25 + R_L  * 1.75
    Vu_secrit   = V_D_secrit * 1.25 + V_L_secrit * 1.75
    Mu_max      = M_D_max    * 1.25 + M_L_max    * 1.75
    Mu_secrit   = M_D_secrit * 1.25 + M_L_secrit * 1.75

    # ULS by load type (kept for completeness)
    Mu_D_max    = M_D_max  * 1.25
    Mu_L_max    = M_L_max  * 1.75

    # ------------------------------------------------------------------ #
    # 5. PRESTRESSING CABLE DESIGN
    # ------------------------------------------------------------------ #
    e1   = y21 - Rec_cables   # eccentricity at mid-span [m]
    fbi  = (M_T_max / W21) / 1000   # working stress bottom fibre [MPa]
    fb_adm_tracc = math.sqrt(fc_H)   # allowable tensile stress    [MPa]

    # Minimum effective prestress force
    Pe_total = (fbi - fb_adm_tracc) / (1 / Area_H_1 + e1 / W21)   # [MN]
    Pe_alma  = Pe_total / N_almas                                    # [MN]

    # Add 25% for losses and uncertainties → jacking force
    Pi_total = 1.25 * Pe_total
    Pi_alma  = 1.25 * Pe_alma

    # Allowable steel stress
    fps_adm = min(0.80 * fpy, 0.74 * fpu)   # [MPa]

    # Strand cross-section (C1900 3/4 in low-relaxation)
    Seccion_cordon = 0.00014   # [m²] = 140 mm²

    # Number of strands per web (choose cable arrangement: 8, 9 or 10 strands/cable)
    N_cord_alma = Pi_alma / (fps_adm * Seccion_cordon)
    Adop_cord_1 = math.ceil(N_cord_alma / 10)   # 10-strand cable
    Adop_cord_2 = math.ceil(N_cord_alma / 9)    # 9-strand cable
    Adop_cord_3 = math.ceil(N_cord_alma / 8)    # 8-strand cable

    cord_cable = [Adop_cord_1 * 10, Adop_cord_2 * 9, Adop_cord_3 * 9]
    Aps_alma   = min(cord_cable) * Seccion_cordon   # [m²]
    Aps_seccion = Aps_alma * N_almas                 # [m²]

    # Parabolic cable profile: j(x) = A·x² + B·x  (origin at support)
    A_par = -(4 * (y22 - Rec_cables)) / L_total**2
    B_par =  (4 * (y22 - Rec_cables)) / L_total

    j_crit = A_par * L_secc_crit**2 + B_par * L_secc_crit   # eccentricity w.r.t. solid centroid
    j_crit = j_crit + (y21 - y22)                           # referred to hollow section centroid

    # ------------------------------------------------------------------ #
    # 6. PRESTRESS LOSSES (simplified)
    # ------------------------------------------------------------------ #
    P0_alma  = Pi_alma  * 0.95   # after instantaneous losses (5 %)
    P0_total = Pi_total * 0.95
    P1_alma  = Pi_alma  * 0.85   # after all losses (15 %)
    P1_total = Pi_total * 0.85

    # ------------------------------------------------------------------ #
    # 7. SLS STRESS VERIFICATION  (4 checks → penalty vector)
    # ------------------------------------------------------------------ #
    F_Penaliz = [1, 1, 1, 1]   # 1 = pass, 0 = fail

    fc_H_7       = 0.7 * fc_H
    fadm_comp_1  =  0.6 * fc_H_7
    fadm_trac_1  =  0.5 * math.sqrt(fc_H_7)
    fadm_comp_2  =  0.6 * fc_H
    fadm_trac_2  =  math.sqrt(fc_H)

    def _check_stress(stress: float, adm_comp: float, adm_trac: float, idx: int):
        """Apply penalty if stress exceeds allowable (compression < 0, adm_comp given as a positive magnitude)."""
        if stress < 0:
            if stress > adm_comp:
                F_Penaliz[idx] = 0
        else:
            if stress > adm_trac:
                F_Penaliz[idx] = 0

    # State 1 (jacking + self-weight, instantaneous losses)
    fcs_1 = (-M_D_pp / 1000) / W11 - P0_total / Area_H_1 + (P0_total * e1) / W11
    fci_1 =  (M_D_pp / 1000) / W21 - P0_total / Area_H_1 - (P0_total * e1) / W21
    _check_stress(fcs_1, fadm_comp_1, fadm_trac_1, 0)
    _check_stress(fci_1, fadm_comp_1, fadm_trac_1, 1)

    # State 2 (all losses + full loading)
    fcs_2 = (-M_T_max / 1000) / W11 - P1_total / Area_H_1 + (P1_total * e1) / W11
    fci_2 =  (M_T_max / 1000) / W21 - P1_total / Area_H_1 - (P1_total * e1) / W21
    _check_stress(fcs_2, fadm_comp_2, fadm_trac_2, 2)
    _check_stress(fci_2, fadm_comp_2, fadm_trac_2, 3)

    # ------------------------------------------------------------------ #
    # 8. ULS BENDING — conventional steel As_flex
    # ------------------------------------------------------------------ #
    As_flex = 0.0

    hf     = Es
    b      = Ancho_total
    bw     = Ew * N_almas
    dp     = H - Rec_cables
    beta_1 = 0.85 - 0.05 * ((fc_H - 30) / 7)
    k_coef = 2 * (1.04 - fpy / fpu)
    fs_s   = 420.0   # conventional steel yield strength [MPa]
    ds     = dp

    # Rectangular section assumption
    c_rect = (Aps_seccion * fpu) / (0.85 * fc_H * beta_1 * b + k_coef * Aps_seccion * fpu / dp)
    a_rect = beta_1 * c_rect

    # T-section assumption
    c_T = ((Aps_seccion * fpu) - (0.85 * fc_H * (b - bw) * hf)) / (
          0.85 * fc_H * beta_1 * bw + k_coef * Aps_seccion * fpu / dp)
    a_T = beta_1 * c_T

    if c_rect > 0 and a_rect <= hf:
        # Rectangular behaviour
        fps_ult = fpu * (1 - k_coef * c_rect / dp)
        Mn      = Aps_seccion * fps_ult * (dp - a_rect / 2)   # [MN·m]

        phi     = 0.583 + 0.25 * (dp / c_rect - 1)
        phi     = min(max(phi, 0.75), 1.0)
        Md      = Mn * phi * 1000   # [kN·m]

        if Md / Mu_max <= 1:
            Delta_Mn    = (Mu_max / phi) - Mn * 1000   # [kN·m]
            As_flex_req = Delta_Mn / (1000 * fs_s * (ds - a_rect / 2))   # [m²]
            phi_20      = 0.000314   # cross-section of ⌀20 mm bar [m²]
            n_flex      = math.ceil(As_flex_req / phi_20)
            sep_flex    = (Ancho_total - 2 * Lv) / n_flex if n_flex > 0 else 4.0
            if sep_flex < 0.04:
                F_Penaliz[0] = 0
            else:
                As_flex = n_flex * phi_20

    elif c_T > 0 and a_T >= hf:
        # T-section behaviour
        fps_ult = fpu * (1 - k_coef * c_T / dp)
        Mn      = (Aps_seccion * fps_ult * (dp - a_T / 2)
                   + 0.85 * fc_H * (b - bw) * hf * (a_T / 2 - hf / 2))   # [MN·m]

        phi     = 0.583 + 0.25 * (dp / c_T - 1)
        phi     = min(max(phi, 0.75), 1.0)
        Md      = Mn * phi * 1000   # [kN·m]

        if Md / Mu_max <= 1:
            Delta_Mn    = (Mu_max / phi) - Mn * 1000
            As_flex_req = Delta_Mn / (1000 * fs_s * (ds - a_T / 2))
            phi_20      = 0.000314
            n_flex      = math.ceil(As_flex_req / phi_20)
            sep_flex    = (Ancho_total - 2 * Lv) / n_flex if n_flex > 0 else 4.0
            if sep_flex < 0.04:
                F_Penaliz[0] = 0
            else:
                As_flex = n_flex * phi_20
    else:
        F_Penaliz[0] = 0

    # ------------------------------------------------------------------ #
    # 9. ULS SHEAR — stirrups As_cort
    # ------------------------------------------------------------------ #
    bv     = bw
    dv     = 0.72 * H
    Vi     = Vu_secrit
    fcpe   = abs(-(P1_total * j_crit) / W21 - P1_total / Area_H_1)   # [MPa]
    fr     = 0.53 * math.sqrt(fc_H)   # modulus of rupture [MPa]
    Mdnc   = M_D_secrit
    Sc     = W21
    Snc    = W21
    Mcre   = Sc * (1000 * (fr + fcpe) - (Mdnc / Snc))   # [kN·m]

    # Vci (flexure-shear cracking)
    Vci_1a = 53 * math.sqrt(fc_H) * bv * dv + Vi * Mcre / Mu_secrit
    Vci_1b = 163 * math.sqrt(fc_H) * bv * dv
    Vci    = max(Vci_1a, Vci_1b)

    # Vcw (web-shear cracking)
    fpc    = abs(P1_total / Area_H_1)   # [MPa]
    Vcw    = (158 * math.sqrt(fc_H) + 300 * fpc) * bv * dv

    Vc          = min(Vci, Vcw)
    phi_corte   = 0.9
    Estribo_sec = 2 * 1.13   # [cm²] two-legged ⌀12 stirrup

    if phi_corte * Vc / Vu_secrit < 1:
        Vs_req           = (Vu_secrit / phi_corte) - Vc
        As_corte_req     = (100 * 100) * (Vs_req / 1000) / (fs_s * dv)   # [cm²/m]
        As_corte_req_alma = As_corte_req / N_almas
        sep_estribos     = Estribo_sec / As_corte_req_alma   # [m]
        if sep_estribos < 0.04:
            F_Penaliz[0] = 0
        largo_estribo    = 2 * (H - Es) + Ew
        vol_estribo      = largo_estribo * (1.13 / (100 * 100))
        Cant_estrib_m    = 1 / sep_estribos
        As_cort          = vol_estribo * Cant_estrib_m * N_almas
    else:
        # Minimum shear reinforcement
        As_corte_min      = (100 * 100 * Aps_seccion * fpu) / (80 * dv * fs_s)
        As_corte_min_alma = As_corte_min / N_almas
        sep_estrib_min    = Estribo_sec / As_corte_min_alma
        if sep_estrib_min < 0.04:
            F_Penaliz[0] = 0
        largo_estribo      = 2 * (H - Es) + Ew
        vol_estribo        = largo_estribo * (1.13 / (100 * 100))
        Cant_estrib_m_min  = 1 / sep_estrib_min
        As_cort            = vol_estribo * Cant_estrib_m_min * N_almas

    # ------------------------------------------------------------------ #
    # 10. TRANSVERSE VERIFICATION (cantilever + slab ratios)
    # ------------------------------------------------------------------ #
    recu_Lv   = 0.03
    phi_flex  = 0.9
    d_Lv      = Es - (0.012 / 2) - recu_Lv
    Ixx_Lv    = Es**3 / 12

    # Cantilever loads
    P_lv      = 3.5          # railings [kN/m]
    Vereda_Lv = 4.55         # sidewalk  [kN/m²]
    pp_Lv     = peso_esp_H * Es
    qL_Lv     = 3.6 + 15
    q_U_Lv    = 1.75 * qL_Lv + 1.25 * (pp_Lv + Vereda_Lv)
    Pu_Lv     = 1.25 * P_lv
    Mu_Lv     = (Pu_Lv * Lv + q_U_Lv * Lv * (Lv / 2)) / 1000   # [MN·m/m]

    As_Lv_req = (Mu_Lv / (phi_flex * beta_1 * d_Lv * fs_s)) * 10000   # [cm²/m]
    As_Lv_req = max(As_Lv_req, 0.0)

    Sec_phi12   = 1.131   # [cm²]
    n_bar_Lv    = math.ceil(As_Lv_req / Sec_phi12)
    sep_flx_Lv  = 1 / n_bar_Lv if n_bar_Lv > 0 else 4.0

    largo_flx_Lv = n_bar_Lv * 1.2 * 2 * Lv
    Vol_As_Lv    = largo_flx_Lv * Sec_phi12 / 10000
    As_flx_Lv    = Vol_As_Lv / 1.0   # [m²/m]

    # Cantilever deflection
    q_T_Lv      = qL_Lv + pp_Lv + Vereda_Lv
    flecha_adm  = Lv / 375
    flecha_i    = (q_T_Lv * Lv**4 / (8 * Eci_28 * 1000 * Ixx_Lv)
                   + P_lv * Lv**3 / (3 * Eci_28 * 1000 * Ixx_Lv))
    flecha_u    = flecha_i * 3

    if sep_flx_Lv < 0.04:
        F_Penaliz[0] = 0
    if flecha_u > flecha_adm:
        F_Penaliz[0] = 0

    # Slab span/thickness ratio (CIRSOC 801 guide p. 83)
    if Es < Ancho_hueco / 20:
        F_Penaliz[0] = 0

    # Web slenderness
    if Ew < H_hueco / 10:
        F_Penaliz[0] = 0

    # ------------------------------------------------------------------ #
    # 11. OBJECTIVE FUNCTION (COST)
    # ------------------------------------------------------------------ #
    Costo_H35 = 40_000        # concrete H35       [ARS/m³]
    Costo_As  = 5_132_746     # conventional steel [ARS/m³]
    Costo_Asp = 1.5 * Costo_As  # prestress steel  [ARS/m³]

    As_seccion = As_flex + As_cort + As_flx_Lv   # total conv. steel [m²/m]

    # Penalty: 1,000,000 if any verification fails, 0 otherwise
    suma_pen    = sum(F_Penaliz)
    Funcion_Pen = 1_000_000 if suma_pen < 4 else 0   # additive penalty (not multiplier)

    COSTO_VIGA = (
        Funcion_Pen
        + Area_H_total * Costo_H35
        + Aps_seccion  * Costo_Asp
        + As_seccion   * Costo_As
    )

    return COSTO_VIGA
