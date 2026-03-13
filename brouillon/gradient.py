"""
Code généré avec LLM après avoir calculé l'expression symboliquement avec Wolfram.

Force de Coulomb entre deux charges avec rotations quaternioniques.

Le vecteur de force F = (Fx, Fy, Fz) / (4 * eps0 * pi)

Paramètres:
  - x1,y1,z1, x2,y2,z2 : positions de base des deux charges
  - a1,b1,c1,d1 : quaternion de rotation 1
  - a2,b2,c2,d2 : quaternion de rotation 2
  - roh : rayon (rotation "half-step")
  - rom : rayon (rotation "midpoint")
  - qh  : charge "half"
  - qm  : charge "mid"
  - t   : paramètre angulaire
  - eps0: permittivité du vide

Simplifications appliquées:
  1. Tous les sous-termes réutilisés sont précalculés une seule fois.
  2. Pour x réel, Abs(x) * Derivative[1][Abs](x) = x  (i.e. signe(x)*|x| = x)
     => Abs(u) * sign(u) = u, donc Abs(u)*Deriv[Abs](u) = u directement.
  3. Les normes 3D sont regroupées par famille (4 variantes pour chaque signe de Sin/Cos).
  4. La division par (4*eps0*pi) est factorisée à la fin.
"""

import numpy as np

def sign(x):
    """Signe de x (retourne 0 si x==0)."""
    return np.sign(x)

def compute_forces(x1, y1, z1, x2, y2, z2,
                   a1, b1, c1, d1,
                   a2, b2, c2, d2,
                   roh, rom, qh, qm, t, eps0):

    ct = np.cos(t / 2)
    st = np.sin(t / 2)

    # =========================================================
    # PRECALCUL DES TERMES DE ROTATION (réutilisés partout)
    # =========================================================

    # --- Termes "rom" (midpoint rotation) ---
    dx_m = (2*a1*c1 - 2*a2*c2 + 2*b1*d1 - 2*b2*d2) * rom + (x1 - x2)
    dy_m = (2*a1*b1 - 2*a2*b2 - 2*c1*d1 + 2*c2*d2) * rom - (y1 - y2)
    dz_m = (a1**2 - a2**2 - b1**2 + b2**2 - c1**2 + c2**2 + d1**2 - d2**2) * rom + (z1 - z2)

    norm_m = (dx_m**2 + dy_m**2 + dz_m**2)**1.5

    # --- Termes "roh" dépendant de (a1,b1,c1,d1) et (a2,b2,c2,d2) ---
    # Composantes de rotation pour la particule 1 agissant sur 2
    rx1_cos =  2*(a1*c1 + b1*d1) * roh
    rx1_sin_p = (a1**2 + b1**2 - c1**2 - d1**2) * roh
    ry1_cos =  2*(a1*b1 - c1*d1) * roh
    ry1_sin_p = 2*(b1*c1 + a1*d1) * roh
    rz1_cos = (a1**2 - b1**2 - c1**2 + d1**2) * roh
    rz1_sin_p = 2*(a1*c1 - b1*d1) * roh

    # Composantes de rotation pour la particule 2 agissant sur 1
    rx2_cos =  2*(a2*c2 + b2*d2) * roh
    rx2_sin_p = (a2**2 + b2**2 - c2**2 - d2**2) * roh
    ry2_cos =  2*(a2*b2 - c2*d2) * roh
    ry2_sin_p = 2*(b2*c2 + a2*d2) * roh
    rz2_cos = (a2**2 - b2**2 - c2**2 + d2**2) * roh
    rz2_sin_p = 2*(a2*c2 - b2*d2) * roh

    # Base positions
    bx2_m = 2*a2*c2*rom + 2*b2*d2*rom   # base x pour particule 2 en rom
    by2_m = 2*a2*b2*rom - 2*c2*d2*rom
    bz2_m = -(a2**2*rom) + b2**2*rom + c2**2*rom - d2**2*rom

    bx1_m = 2*a1*c1*rom + 2*b1*d1*rom
    by1_m = 2*a1*b1*rom - 2*c1*d1*rom
    bz1_m = a1**2*rom - b1**2*rom - c1**2*rom + d1**2*rom

    dx12 = x1 - x2
    dy12 = y1 - y2
    dz12 = z1 - z2

    # =========================================================
    # FAMILLE A : termes qh*qm (rotation de 1 vue par 2)
    # Variante +sin et -sin pour chaque axe
    # =========================================================

    # -- Vecteurs déplacement pour les 4 configurations (±cos déjà fixé, ±sin varie)
    # Config A1: rot1 avec +sin, vue de 2
    dxA1p = bx2_m - dx12 - rx1_cos * ct + rx1_sin_p * st   # x: 2ac2*rom + 2bd2*rom - x12 - rx1_cos*ct + rx1_sin*st
    dyA1p = by2_m + dy12 - ry1_cos * ct - ry1_sin_p * st
    dzA1p = bz2_m + dz12 + rz1_cos * ct + rz1_sin_p * st

    # Config A1: rot1 avec -sin
    dxA1m = bx2_m - dx12 - rx1_cos * ct - rx1_sin_p * st
    dyA1m = by2_m + dy12 - ry1_cos * ct + ry1_sin_p * st
    dzA1m = bz2_m + dz12 + rz1_cos * ct - rz1_sin_p * st

    # -- Vecteurs déplacement pour rot2 vue par 1
    # Config A2: +sin
    dxA2p = bx1_m + dx12 - rx2_cos * ct - rx2_sin_p * st
    dyA2p = -by1_m + dy12 - ry2_cos * ct + ry2_sin_p * st
    dzA2p = bz1_m + dz12 - rz2_cos * ct - rz2_sin_p * st

    # Config A2: -sin
    dxA2m = bx1_m + dx12 - rx2_cos * ct + rx2_sin_p * st
    dyA2m = -by1_m + dy12 - ry2_cos * ct - ry2_sin_p * st
    dzA2m = bz1_m + dz12 - rz2_cos * ct + rz2_sin_p * st

    def norm3(dx, dy, dz):
        return (dx**2 + dy**2 + dz**2)**1.5

    # =========================================================
    # FAMILLE B : termes qh^2 (rotation combinée 1 et 2)
    # =========================================================
    # Combinaisons de coefficients pour la rotation combinée
    rx12_cos = 2*(a1*c1 - a2*c2 + b1*d1 - b2*d2) * roh
    rx12_sin_pp = (a1**2 + a2**2 + b1**2 + b2**2 - c1**2 - c2**2 - d1**2 - d2**2) * roh
    rx12_sin_pm = (a1**2 - a2**2 + b1**2 - b2**2 - c1**2 + c2**2 - d1**2 + d2**2) * roh

    ry12_cos = 2*(a1*b1 - a2*b2 - c1*d1 + c2*d2) * roh
    ry12_sin_pp = 2*(b1*c1 + b2*c2 + a1*d1 + a2*d2) * roh
    ry12_sin_pm = 2*(b1*c1 - b2*c2 + a1*d1 - a2*d2) * roh

    rz12_cos = (a1**2 - a2**2 - b1**2 + b2**2 - c1**2 + c2**2 + d1**2 - d2**2) * roh
    rz12_sin_pp = 2*(a1*c1 + a2*c2 - b1*d1 - b2*d2) * roh
    rz12_sin_pm = 2*(a1*c1 - a2*c2 - b1*d1 + b2*d2) * roh

    # 4 configurations B (±sin_pp, ±sin_pm)
    # B1: -sin_pp, -sin_pm (note: Cos term stays, only Sin sign changes)
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

    # =========================================================
    # CALCUL DES FORCES
    # Simplification clé: Abs(u)*Deriv[Abs](u) = u (pour u réel)
    # Donc chaque terme Abs(u)*Deriv[Abs](u) / norm^(3/2) = u / norm^(3/2)
    # =========================================================

    # --- Fx ---
    # Terme qm^2
    Fx_mm = -(qm**2 * dx_m) / norm_m

    # Termes qh*qm (4 contributions)
    Fx_hm = qh * qm * (
        dxA1p / norm3(dxA1p, dyA1p, dzA1p)
        - dxA1m / norm3(dxA1m, dyA1m, dzA1m)
        - dxA2p / norm3(dxA2p, dyA2p, dzA2p)
        - dxA2m / norm3(dxA2m, dyA2m, dzA2m)
    )

    # Termes qh^2 (4 contributions)
    Fx_hh = -(qh**2) * (
        dxB1 / norm3(dxB1, dyB1, dzB1)
        + dxB2 / norm3(dxB2, dyB2, dzB2)
        + dxB3 / norm3(dxB3, dyB3, dzB3)
        + dxB4 / norm3(dxB4, dyB4, dzB4)
    )

    Fx = (Fx_mm + Fx_hm + Fx_hh) / (4 * eps0 * np.pi)

    # --- Fy ---
    Fy_mm = (qm**2 * dy_m) / norm_m

    Fy_hm = qh * qm * (
        - dyA1p / norm3(dxA1p, dyA1p, dzA1p)
        - dyA1m / norm3(dxA1m, dyA1m, dzA1m)
        + dyA2p / norm3(dxA2p, dyA2p, dzA2p)
        - dyA2m / norm3(dxA2m, dyA2m, dzA2m)
    )

    Fy_hh = -(qh**2) * (
        dyB3 / norm3(dxB3, dyB3, dzB3)
        + dyB4 / norm3(dxB4, dyB4, dzB4)
        + dyB1 / norm3(dxB1, dyB1, dzB1)
        + dyB2 / norm3(dxB2, dyB2, dzB2)
    )

    Fy = (Fy_mm + Fy_hm + Fy_hh) / (4 * eps0 * np.pi)

    # --- Fz ---
    Fz_mm = -(qm**2 * dz_m) / norm_m

    Fz_hm = qh * qm * (
        - dzA1m / norm3(dxA1m, dyA1m, dzA1m)
        - dzA1p / norm3(dxA1p, dyA1p, dzA1p)   # note: signe à vérifier selon Wolfram
        - dzA2p / norm3(dxA2p, dyA2p, dzA2p)
        - dzA2m / norm3(dxA2m, dyA2m, dzA2m)
    )

    Fz_hh = -(qh**2) * (
        dzB2 / norm3(dxB2, dyB2, dzB2)
        + dzB1 / norm3(dxB1, dyB1, dzB1)
        + dzB3 / norm3(dxB3, dyB3, dzB3)
        + dzB4 / norm3(dxB4, dyB4, dzB4)
    )

    Fz = (Fz_mm + Fz_hm + Fz_hh) / (4 * eps0 * np.pi)

    return np.array([Fx, Fy, Fz])


# =========================================================
# EXEMPLE D'UTILISATION
# =========================================================
if __name__ == "__main__":
    eps0 = 8.854187817e-12

    # Quaternions unitaires (exemple: rotation identité)
    a1, b1, c1, d1 = 1.0, 0.0, 0.0, 0.0
    a2, b2, c2, d2 = 1.0, 0.0, 0.0, 0.0

    F = compute_forces(
        x1=1.0, y1=0.0, z1=0.0,
        x2=0.0, y2=0.0, z2=0.0,
        a1=a1, b1=b1, c1=c1, d1=d1,
        a2=a2, b2=b2, c2=c2, d2=d2,
        roh=0.1, rom=0.1,
        qh=1e-9, qm=1e-9,
        t=0.5,
        eps0=eps0
    )

    print(f"Fx = {F[0]:.6e} N")
    print(f"Fy = {F[1]:.6e} N")
    print(f"Fz = {F[2]:.6e} N")