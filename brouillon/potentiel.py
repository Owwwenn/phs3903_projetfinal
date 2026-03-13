"""
Code généré avec LLM après avoir calculé l'expression symboliquement avec Wolfram.

Énergie potentielle de Coulomb entre deux charges avec rotations quaternioniques.

U = (1 / (4 * eps0 * pi)) * (qm^2/r_mm + qh*qm * sum(1/r_hm) + qh^2 * sum(1/r_hh))

Simplifications appliquées:
  1. cos(t/2) et sin(t/2) calculés une seule fois.
  2. Sous-termes de rotation précalculés et réutilisés.
  3. Les 9 distances sont calculées une seule fois (partagées avec force_coulomb.py).
  4. Abs(u)^2 = u^2 pour tout réel => on supprime tous les Abs().
  5. Factorisation finale par 1/(4*eps0*pi).

Structure:
  - 1 terme  qm^2  : distance "midpoint rom"
  - 4 termes qh*qm : rotations de 1 sur 2 et de 2 sur 1 (±sin)
  - 4 termes qh^2  : rotations combinées (±sin_pp, ±sin_pm)
"""

import numpy as np


def compute_potential(x1, y1, z1, x2, y2, z2,
                      a1, b1, c1, d1,
                      a2, b2, c2, d2,
                      roh, rom, qh, qm, t, eps0):

    ct = np.cos(t / 2)
    st = np.sin(t / 2)

    # =========================================================
    # PRECALCUL DES SOUS-TERMES (réutilisés dans force aussi)
    # =========================================================

    dx12 = x1 - x2
    dy12 = y1 - y2
    dz12 = z1 - z2

    # --- Terme qm^2 : déplacement "rom" ---
    dx_m = (2*a1*c1 - 2*a2*c2 + 2*b1*d1 - 2*b2*d2) * rom + dx12
    dy_m = (2*a1*b1 - 2*a2*b2 - 2*c1*d1 + 2*c2*d2) * rom - dy12
    dz_m = (a1**2 - a2**2 - b1**2 + b2**2 - c1**2 + c2**2 + d1**2 - d2**2) * rom + dz12

    r_mm = np.sqrt(dx_m**2 + dy_m**2 + dz_m**2)

    # --- Termes qh*qm : rotation de 1 agissant sur 2 ---
    # Amplitudes des composantes de rotation pour particule 1
    rx1_cos  =  2*(a1*c1 + b1*d1) * roh
    rx1_sin  =  (a1**2 + b1**2 - c1**2 - d1**2) * roh
    ry1_cos  =  2*(a1*b1 - c1*d1) * roh
    ry1_sin  =  2*(b1*c1 + a1*d1) * roh
    rz1_cos  =  (a1**2 - b1**2 - c1**2 + d1**2) * roh
    rz1_sin  =  2*(a1*c1 - b1*d1) * roh

    # Base "rom" de la particule 2
    bx2 = 2*a2*c2*rom + 2*b2*d2*rom
    by2 = 2*a2*b2*rom - 2*c2*d2*rom
    bz2 = (-a2**2 + b2**2 + c2**2 - d2**2) * rom

    # 2 configurations ±sin pour rot1 vue par 2
    dxA1p = bx2 - dx12 - rx1_cos * ct + rx1_sin * st
    dyA1p = by2 + dy12 - ry1_cos * ct - ry1_sin * st
    dzA1p = bz2 + dz12 + rz1_cos * ct + rz1_sin * st

    dxA1m = bx2 - dx12 - rx1_cos * ct - rx1_sin * st
    dyA1m = by2 + dy12 - ry1_cos * ct + ry1_sin * st
    dzA1m = bz2 + dz12 + rz1_cos * ct - rz1_sin * st

    # Amplitudes des composantes de rotation pour particule 2
    rx2_cos  =  2*(a2*c2 + b2*d2) * roh
    rx2_sin  =  (a2**2 + b2**2 - c2**2 - d2**2) * roh
    ry2_cos  =  2*(a2*b2 - c2*d2) * roh
    ry2_sin  =  2*(b2*c2 + a2*d2) * roh
    rz2_cos  =  (a2**2 - b2**2 - c2**2 + d2**2) * roh
    rz2_sin  =  2*(a2*c2 - b2*d2) * roh

    # Base "rom" de la particule 1
    bx1 = 2*a1*c1*rom + 2*b1*d1*rom
    by1 = 2*a1*b1*rom - 2*c1*d1*rom
    bz1 = (a1**2 - b1**2 - c1**2 + d1**2) * rom

    # 2 configurations ±sin pour rot2 vue par 1
    dxA2p = bx1 + dx12 - rx2_cos * ct - rx2_sin * st
    dyA2p = -by1 + dy12 - ry2_cos * ct + ry2_sin * st
    dzA2p = bz1 + dz12 - rz2_cos * ct - rz2_sin * st

    dxA2m = bx1 + dx12 - rx2_cos * ct + rx2_sin * st
    dyA2m = -by1 + dy12 - ry2_cos * ct - ry2_sin * st
    dzA2m = bz1 + dz12 - rz2_cos * ct + rz2_sin * st

    def dist3(dx, dy, dz):
        return np.sqrt(dx**2 + dy**2 + dz**2)

    r_hm1p = dist3(dxA1p, dyA1p, dzA1p)
    r_hm1m = dist3(dxA1m, dyA1m, dzA1m)
    r_hm2p = dist3(dxA2p, dyA2p, dzA2p)
    r_hm2m = dist3(dxA2m, dyA2m, dzA2m)

    # --- Termes qh^2 : rotations combinées ---
    # Coefficients combinés 1+2
    rx12_cos  =  2*(a1*c1 - a2*c2 + b1*d1 - b2*d2) * roh
    rx12_sin_pp = (a1**2 + a2**2 + b1**2 + b2**2 - c1**2 - c2**2 - d1**2 - d2**2) * roh
    rx12_sin_pm = (a1**2 - a2**2 + b1**2 - b2**2 - c1**2 + c2**2 - d1**2 + d2**2) * roh

    ry12_cos  =  2*(a1*b1 - a2*b2 - c1*d1 + c2*d2) * roh
    ry12_sin_pp = 2*(b1*c1 + b2*c2 + a1*d1 + a2*d2) * roh
    ry12_sin_pm = 2*(b1*c1 - b2*c2 + a1*d1 - a2*d2) * roh

    rz12_cos  =  (a1**2 - a2**2 - b1**2 + b2**2 - c1**2 + c2**2 + d1**2 - d2**2) * roh
    rz12_sin_pp = 2*(a1*c1 + a2*c2 - b1*d1 - b2*d2) * roh
    rz12_sin_pm = 2*(a1*c1 - a2*c2 - b1*d1 + b2*d2) * roh

    # 4 configurations B
    dxB1 = dx12 + rx12_cos * ct - rx12_sin_pp * st
    dyB1 = dy12 - ry12_cos * ct - ry12_sin_pp * st
    dzB1 = dz12 + rz12_cos * ct - rz12_sin_pp * st

    dxB2 = dx12 + rx12_cos * ct + rx12_sin_pp * st
    dyB2 = dy12 - ry12_cos * ct + ry12_sin_pp * st
    dzB2 = dz12 + rz12_cos * ct + rz12_sin_pp * st

    dxB3 = dx12 + rx12_cos * ct - rx12_sin_pm * st
    dyB3 = dy12 - ry12_cos * ct - ry12_sin_pm * st
    dzB3 = dz12 + rz12_cos * ct - rz12_sin_pm * st

    dxB4 = dx12 + rx12_cos * ct + rx12_sin_pm * st
    dyB4 = dy12 - ry12_cos * ct + ry12_sin_pm * st
    dzB4 = dz12 + rz12_cos * ct + rz12_sin_pm * st

    r_hh1 = dist3(dxB1, dyB1, dzB1)
    r_hh2 = dist3(dxB2, dyB2, dzB2)
    r_hh3 = dist3(dxB3, dyB3, dzB3)
    r_hh4 = dist3(dxB4, dyB4, dzB4)

    # =========================================================
    # ENERGIE POTENTIELLE
    # =========================================================
    U = (
        qm**2 / r_mm
        + qh * qm * (1/r_hm1p + 1/r_hm1m + 1/r_hm2p + 1/r_hm2m)
        + qh**2  * (1/r_hh1  + 1/r_hh2  + 1/r_hh3  + 1/r_hh4 )
    ) / (4 * eps0 * np.pi)

    return U


# =========================================================
# EXEMPLE D'UTILISATION
# =========================================================
if __name__ == "__main__":
    eps0 = 8.854187817e-12

    U = compute_potential(
        x1=1.0, y1=0.0, z1=0.0,
        x2=0.0, y2=0.0, z2=0.0,
        a1=1.0, b1=0.0, c1=0.0, d1=0.0,
        a2=1.0, b2=0.0, c2=0.0, d2=0.0,
        roh=0.1, rom=0.1,
        qh=1e-9, qm=1e-9,
        t=0.5,
        eps0=eps0
    )

    print(f"U = {U:.6e} J")