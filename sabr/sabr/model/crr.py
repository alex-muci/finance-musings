"""
Cox-Ross-Rubinstein american option valuation (vectorised-style)
"""
import math
import numpy as np


# cp, F, S, K, t, sig, r, n=101, American=True
def crr(cp, F, K, T, r, sigma, n=101, American=True):
    """ Cox-Ross-Rubinstein European option valuation.
    """
    dt = T / n  # length of time interval
    df = math.exp(-r * dt)  # discount per interval

    # Binomial Parameters
    u = math.exp(sigma * math.sqrt(dt))  # up movement
    d = 1. / u  # down movement
    q = (math.exp(r * dt) - d) / (u - d)

    # vectorisation
    mu = np.arange(n + 1)
    mu = np.resize(mu, (n + 1, n + 1))  # matrix
    md = np.transpose(mu)
    mu = u ** (mu - md)
    md = d ** md
    Stck = F * mu * md

    # call or put
    V = np.maximum(Stck - K, 0) * cp    # value matrix

    if American:
        h = np.maximum(Stck - K, 0) * cp
        C = np.zeros((n + 1, n + 1), dtype=np.float)  # continuation values
        ex = np.zeros((n + 1, n + 1), dtype=np.float)  # exercise matrix

    z = 0
    for t in range(n - 1, -1, -1):  # backwards iteration
        if American:
            C[0:n - z, t] = (q * V[0:n - z, t + 1] +
                             (1 - q) * V[1:n - z + 1, t + 1]) * df
            V[0:n - z, t] = np.where(h[0:n - z, t] > C[0:n - z, t],
                                     h[0:n - z, t], C[0:n - z, t])
            ex[0:n - z, t] = np.where(h[0:n - z, t] > C[0:n - z, t], 1, 0)
        else:   # European
            V[0:n - z, t] = (q * V[0:n - z, t + 1] +
                             (1 - q) * V[1:n - z + 1, t + 1]) * df

        z += 1

    value = V[0, 0]
    delta = (V[1, 1] - V[1, 0]) / (F * (u - d))

    return value, delta
