"""
Leisen-Reimer american option valuation and greeks

TODO: vectorise implementation in lropt (currently is C++ style)
"""

from math import exp, log, sqrt


def lropt(cp, F, S, K, t, sig, r, n=101, American=True):
    """
    Leisen-Reimer AMERICAN option value and delta
     (for European options change bool to 'American=False')
     r is the ctsly compounded interest rate to t
     n is the number of steps (must be an odd number)
    """

    dt = t / float(n)
    df = exp(-r * dt)
    cumfac = log(F/S) / n
    cumfac = exp(cumfac)

    vt = sig * sqrt(t)
    d1 = (log(F / K) + 0.5 * (vt**2)) / vt
    d2 = d1 - vt
    pdm = pz(n, d2)
    pdp = pz(n, d1)

    # Binomial Parameters
    u = cumfac * pdp / pdm              # up move
    D = (cumfac - pdm * u) / (1 - pdm)  # down move
    p = pdm

    f = [[0 for x in range(n+1)] for y in range(n+1)]
    for j in range(n+1):
        f[n][j] = max(cp * (-K + S * (u ** j) * (D ** (n - j))), 0)

    for i in range(n-1, -1, -1):
        for j in range(i+1):
            f[i][j] = df * (p * f[i + 1][j + 1] + (1 - p) * f[i + 1][j])
            if American:
                f[i][j] = max(f[i][j],
                              cp * (-K + S * (u ** j) * (D ** (i - j))))

    value = f[0][0]
    delta = (f[1][1] - f[1][0]) / (S * (u - D))

    return value, delta


def lropt_gamma(cp, f, s, x, t, v, r, n=101, blip_BPS=10.0):

    # derivative wrt spot
    blip = blip_BPS * s / 10000.0

    # delta up and down
    b = lropt(cp, f, s, x, t, v, r, n)[0]
    b_up = lropt(cp, f, s + blip, x, t, v, r, n)[0]
    b_dn = lropt(cp, f, s - blip, x, t, v, r, n)[0]

    # gamma
    bg = (b_up - 2.0 * b + b_dn) / (blip ** 2)
    bg *= (s / 100.0)

    return bg


def lropt_vega(cp, f, s, x, t, v, r, n=101, blip_BPS=100.0):

    # derivative wrt spot
    blip = 0.5 * blip_BPS / 10000.0

    b = lropt(cp, f, s, x, t, v, r, n)[0]
    b_up = lropt(cp, f, s, x, t, v + blip, r, n)[0]
    b_dn = lropt(cp, f, s, x, t, v - blip, r, n)[0]

    # vega
    vega = (b_up - b_dn)

    # volgamma
    # bg = (b_up - 2.0 * b + b_dn) / (blip ** 2);

    return vega


def lropt_theta(cp, f, s, x, dte, v, r, n=101, blip_BPS=100.0):
    """
    Theta in forward space: returns undiscounted value
     takes days to expiry, not yearfrac
     returns decay (i.e. negative)
    """

    t = float(dte) / 365.25
    b = lropt(cp, f, s, x, t, v, r, n)[0]

    t1 = (float(dte) - 1.0)/365.25
    b1 = lropt(cp, f, s, x, t1, v, r, n)[0]

    theta = (b1 - b)

    return theta


def pz(n, z):
    expterm = (z / (n + 1 / 3)) ** 2
    expterm = exp(-expterm * (n + 1 / 6))
    nxtterm = 0.5 + _sign(z) * 0.5 * sqrt(1 - expterm)
    return nxtterm


def _sign(x):
    """ returns sign function (as float)
        if x is complex then use numpy.sign()
    """
    sgn_int = x and (1, -1)[x < 0]
    return 1.0 * sgn_int