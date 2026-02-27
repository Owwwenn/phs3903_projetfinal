"""
Code généré avec LLM après avoir calculé l'expression symboliquement avec Wolfram.

Hessienne de l'énergie potentielle de Coulomb avec rotations quaternioniques.

Wolfram calcule d²U/d(coord)² via des termes du type:
  3*Abs(u)^2 * Deriv[1][Abs](u)^2 / r^5
  - Deriv[1][Abs](u)^2 / r^3
  - Abs(u) * Deriv[2][Abs](u) / r^3

Simplifications clés (u réel, hors singularité):
  - Deriv[1][Abs](u)   = sign(u)       => sign(u)^2 = 1
  - Deriv[2][Abs](u)   = 2*DiracDelta(u) = 0  (hors singularité)
  - Abs(u)^2 * 1 = u^2

Donc chaque groupe en y_k par exemple vaut:
  3*yk^2 / r^5 - 1/r^3 - 0  =  (3*yk^2 - r^2) / r^5

C'est exactement le tenseur de Maxwell / terme dipolaire de Coulomb:
  d²(1/r)/d(xk)^2 = (3*xk^2 - r^2) / r^5

Ce fichier calcule la trace (laplacien) et les composantes diagonales xx, yy, zz
de la hessienne, en réutilisant les mêmes 9 vecteurs que force_coulomb.py
et potential_coulomb.py.
"""

import numpy as np


def _hessian_diag(dx, dy, dz):
    """
    Retourne (Hxx, Hyy, Hzz) = dérivées secondes de 1/r par rapport à x, y, z.
    Formule: d²(1/r)/dxi² = (3*xi^2 - r^2) / r^5
    """
    r2  = dx**2 + dy**2 + dz**2
    r5  = r2**2.5
    Hxx = (3*dx**2 - r2) / r5
    Hyy = (3*dy**2 - r2) / r5
    Hzz = (3*dz**2 - r2) / r5
    return Hxx, Hyy, Hzz


def compute_hessian_diag(x1, y1, z1, x2, y2, z2,
                         a1, b1, c1, d1,
                         a2, b2, c2, d2,
                         roh, rom, qh, qm, t, eps0):
    """
    Retourne les composantes diagonales (d²U/dx², d²U/dy², d²U/dz²)
    de la hessienne de l'énergie potentielle.

    Les 9 vecteurs déplacement sont identiques à ceux de potential_coulomb.py.
    """

    ct = np.cos(t / 2)
    st = np.sin(t / 2)

    dx12 = x1 - x2
    dy12 = y1 - y2
    dz12 = z1 - z2

    # =========================================================
    # VECTEURS DEPLACEMENT (identiques à potential_coulomb.py)
    # =========================================================

    # --- Terme qm^2 ---
    dx_m = (2*a1*c1 - 2*a2*c2 + 2*b1*d1 - 2*b2*d2) * rom + dx12
    dy_m = (2*a1*b1 - 2*a2*b2 - 2*c1*d1 + 2*c2*d2) * rom - dy12
    dz_m = (a1**2 - a2**2 - b1**2 + b2**2 - c1**2 + c2**2 + d1**2 - d2**2) * rom + dz12

    # --- Termes qh*qm : rot1 sur 2 ---
    rx1_cos = 2*(a1*c1 + b1*d1) * roh
    rx1_sin = (a1**2 + b1**2 - c1**2 - d1**2) * roh
    ry1_cos = 2*(a1*b1 - c1*d1) * roh
    ry1_sin = 2*(b1*c1 + a1*d1) * roh
    rz1_cos = (a1**2 - b1**2 - c1**2 + d1**2) * roh
    rz1_sin = 2*(a1*c1 - b1*d1) * roh

    bx2 = 2*a2*c2*rom + 2*b2*d2*rom
    by2 = 2*a2*b2*rom - 2*c2*d2*rom
    bz2 = (-a2**2 + b2**2 + c2**2 - d2**2) * rom

    dxA1p = bx2 - dx12 - rx1_cos*ct + rx1_sin*st
    dyA1p = by2 + dy12 - ry1_cos*ct - ry1_sin*st
    dzA1p = bz2 + dz12 + rz1_cos*ct + rz1_sin*st

    dxA1m = bx2 - dx12 - rx1_cos*ct - rx1_sin*st
    dyA1m = by2 + dy12 - ry1_cos*ct + ry1_sin*st
    dzA1m = bz2 + dz12 + rz1_cos*ct - rz1_sin*st

    # --- Termes qh*qm : rot2 sur 1 ---
    rx2_cos = 2*(a2*c2 + b2*d2) * roh
    rx2_sin = (a2**2 + b2**2 - c2**2 - d2**2) * roh
    ry2_cos = 2*(a2*b2 - c2*d2) * roh
    ry2_sin = 2*(b2*c2 + a2*d2) * roh
    rz2_cos = (a2**2 - b2**2 - c2**2 + d2**2) * roh
    rz2_sin = 2*(a2*c2 - b2*d2) * roh

    bx1 = 2*a1*c1*rom + 2*b1*d1*rom
    by1 = 2*a1*b1*rom - 2*c1*d1*rom
    bz1 = (a1**2 - b1**2 - c1**2 + d1**2) * rom

    dxA2p = bx1 + dx12 - rx2_cos*ct - rx2_sin*st
    dyA2p = -by1 + dy12 - ry2_cos*ct + ry2_sin*st
    dzA2p = bz1 + dz12 - rz2_cos*ct - rz2_sin*st

    dxA2m = bx1 + dx12 - rx2_cos*ct + rx2_sin*st
    dyA2m = -by1 + dy12 - ry2_cos*ct - ry2_sin*st
    dzA2m = bz1 + dz12 - rz2_cos*ct + rz2_sin*st

    # --- Termes qh^2 : rotations combinées ---
    rx12_cos   =  2*(a1*c1 - a2*c2 + b1*d1 - b2*d2) * roh
    rx12_sin_pp = (a1**2 + a2**2 + b1**2 + b2**2 - c1**2 - c2**2 - d1**2 - d2**2) * roh
    rx12_sin_pm = (a1**2 - a2**2 + b1**2 - b2**2 - c1**2 + c2**2 - d1**2 + d2**2) * roh

    ry12_cos   =  2*(a1*b1 - a2*b2 - c1*d1 + c2*d2) * roh
    ry12_sin_pp = 2*(b1*c1 + b2*c2 + a1*d1 + a2*d2) * roh
    ry12_sin_pm = 2*(b1*c1 - b2*c2 + a1*d1 - a2*d2) * roh

    rz12_cos   =  (a1**2 - a2**2 - b1**2 + b2**2 - c1**2 + c2**2 + d1**2 - d2**2) * roh
    rz12_sin_pp = 2*(a1*c1 + a2*c2 - b1*d1 - b2*d2) * roh
    rz12_sin_pm = 2*(a1*c1 - a2*c2 - b1*d1 + b2*d2) * roh

    dxB1 = dx12 + rx12_cos*ct - rx12_sin_pp*st
    dyB1 = dy12 - ry12_cos*ct - ry12_sin_pp*st
    dzB1 = dz12 + rz12_cos*ct - rz12_sin_pp*st

    dxB2 = dx12 + rx12_cos*ct + rx12_sin_pp*st
    dyB2 = dy12 - ry12_cos*ct + ry12_sin_pp*st
    dzB2 = dz12 + rz12_cos*ct + rz12_sin_pp*st

    dxB3 = dx12 + rx12_cos*ct - rx12_sin_pm*st
    dyB3 = dy12 - ry12_cos*ct - ry12_sin_pm*st
    dzB3 = dz12 + rz12_cos*ct - rz12_sin_pm*st

    dxB4 = dx12 + rx12_cos*ct + rx12_sin_pm*st
    dyB4 = dy12 - ry12_cos*ct + ry12_sin_pm*st
    dzB4 = dz12 + rz12_cos*ct + rz12_sin_pm*st

    # =========================================================
    # CALCUL DES COMPOSANTES DIAGONALES DE LA HESSIENNE
    # d²(1/r)/dxi² = (3*xi^2 - r^2) / r^5
    # =========================================================

    def H(dx, dy, dz):
        return _hessian_diag(dx, dy, dz)

    # qm^2
    Hxx_mm, Hyy_mm, Hzz_mm = H(dx_m, dy_m, dz_m)

    # qh*qm (4 termes)
    Hxx_A1p, Hyy_A1p, Hzz_A1p = H(dxA1p, dyA1p, dzA1p)
    Hxx_A1m, Hyy_A1m, Hzz_A1m = H(dxA1m, dyA1m, dzA1m)
    Hxx_A2p, Hyy_A2p, Hzz_A2p = H(dxA2p, dyA2p, dzA2p)
    Hxx_A2m, Hyy_A2m, Hzz_A2m = H(dxA2m, dyA2m, dzA2m)

    # qh^2 (4 termes)
    Hxx_B1, Hyy_B1, Hzz_B1 = H(dxB1, dyB1, dzB1)
    Hxx_B2, Hyy_B2, Hzz_B2 = H(dxB2, dyB2, dzB2)
    Hxx_B3, Hyy_B3, Hzz_B3 = H(dxB3, dyB3, dzB3)
    Hxx_B4, Hyy_B4, Hzz_B4 = H(dxB4, dyB4, dzB4)

    # Sommes pondérées
    def total(Hmm, HA1p, HA1m, HA2p, HA2m, HB1, HB2, HB3, HB4):
        return (
            qm**2 * Hmm
            + qh*qm * (HA1p + HA1m + HA2p + HA2m)
            + qh**2  * (HB1  + HB2  + HB3  + HB4 )
        ) / (4 * eps0 * np.pi)

    d2U_dx2 = total(Hxx_mm,
                    Hxx_A1p, Hxx_A1m, Hxx_A2p, Hxx_A2m,
                    Hxx_B1, Hxx_B2, Hxx_B3, Hxx_B4)

    d2U_dy2 = total(Hyy_mm,
                    Hyy_A1p, Hyy_A1m, Hyy_A2p, Hyy_A2m,
                    Hyy_B1, Hyy_B2, Hyy_B3, Hyy_B4)

    d2U_dz2 = total(Hzz_mm,
                    Hzz_A1p, Hzz_A1m, Hzz_A2p, Hzz_A2m,
                    Hzz_B1, Hzz_B2, Hzz_B3, Hzz_B4)

    laplacian = d2U_dx2 + d2U_dy2 + d2U_dz2

    return d2U_dx2, d2U_dy2, d2U_dz2, laplacian


# =========================================================
# EXEMPLE D'UTILISATION
# =========================================================
if __name__ == "__main__":
    eps0 = 8.854187817e-12

    Hxx, Hyy, Hzz, lap = compute_hessian_diag(
        x1=1.0, y1=0.0, z1=0.0,
        x2=0.0, y2=0.0, z2=0.0,
        a1=1.0, b1=0.0, c1=0.0, d1=0.0,
        a2=1.0, b2=0.0, c2=0.0, d2=0.0,
        roh=0.1, rom=0.1,
        qh=1e-9, qm=1e-9,
        t=0.5,
        eps0=eps0
    )

    print(f"d²U/dx² = {Hxx:.6e} J/m²")
    print(f"d²U/dy² = {Hyy:.6e} J/m²")
    print(f"d²U/dz² = {Hzz:.6e} J/m²")
    print(f"Laplacien = {lap:.6e} J/m²")
    # Note: pour une charge ponctuelle isolée, le laplacien devrait être ≈ 0 (hors source)