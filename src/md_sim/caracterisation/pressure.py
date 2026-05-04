def get_pressure(N, T, L_box, virial):
    kB = 0.001987   # kcal/(mol·K)
    V = L_box[0] * L_box[1] * L_box[2]
    P = (N * kB * T + virial / 3.0) / V
    return P