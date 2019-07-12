"""
Miscellaneous (static) utility functions
- Black (un-disconted) price for calls and puts
- numerical greeks

TODO:
 - insert analytical greeks & write test analytical vs. numerical
 - insert r and zero it, BUT insert it
 - implied vol: Peter Jackel 2 iteration algo + NUMBA
"""

from math import log, exp, sqrt  # or  from numpy import log, exp, sqrt
from scipy.optimize import brentq  # NB: 'fsolve' is for n-dim root finding
from scipy.stats import norm

N = norm.cdf


def black(cp, f, x, t, v, r):
    """
    Black price

    :parameters
    cp is +1 for calls and -1 for puts
    f is the forward price
    x is the strikes
    t is the maturity, in year_fract
    v is the implied volatility, e.g. 0.15
    r is the int rate
    """
    df = exp(-r * t)
    vt = cp * v * sqrt(t)  # voltime formulation
    d1 = (log(f/x) + 0.5 * (vt**2)) / vt
    d2 = d1 - vt
    return cp * (f * N(d1) - x * N(d2)) * df


def implied_vol(cp, f, x, t, r, price):
    """
    Returns the Black implied vol

    """
    # checks
    intrinsic = abs(max(f-x if cp > 0 else x-f, 0.0))
    if price < intrinsic:
        raise Exception("Price is below intrinsic value.")
    upper_bound = f if cp > 0 else x
    if price >= upper_bound:
        raise Exception("Price is above the upper bound.")

    fnct = lambda vol: price - black(cp, f, x, t, vol, r)

    # returns a zero of f in [a, b]: f must be continuous and [a, b] sign-changing
    return brentq(fnct, a=1e-12, b=5.,
                  xtol=1e-10, rtol=1e-10, maxiter=300, full_output=False)


#   #################################   #
#   #   #   NUMERICAL Greeks    #   #   #
#   #################################   #
def black_delta(cp, f, x, t, v, r, blip_bps=10.):
    """
    BlackDelta in forward space: returns undiscounted value

    :param: blip_bps is in bps (10. means 10 bps) of the forward
    """
    # corner cases at maturity
    if t == 0.0:
        if f == x:
            return {1: 0.5, -1: -0.5}[cp]
        elif f > x:
            return {1: 1.0, -1: 0.0}[cp]
        else:
            return {1: 0.0, -1: -1.0}[cp]
    else:

        blip = blip_bps * f / 10000.0

        return (black(cp, f + blip, x, t, v, r) -
                black(cp, f - blip, x, t, v, r)) / (2 * blip)


def black_gamma(cp, f, x, t, v, r, blip_bps=10.):
    """
    BlackGamma in forward space: returns undiscounted value
    expressed as percentage of the forward

    :param: blip_bps is in bps (10. means 10 bps) of the forward
    """
    # corner case at maturity: gamma explodes if ATM
    if t == 0.0:
        return float("inf") if f == x else 0.0
    else:

        blip = blip_bps * f / 10000.0

        gamma = (black(cp, f + blip, x, t, v, r) -
                 2.0 * black(cp, f, x, t, v, r) +
                 black(cp, f - blip, x, t, v, r)) / (blip ** 2)
        gamma *= (f / 100.0)

        return gamma


def black_vega(cp, f, x, t, v, r, blip_bps=100.0):
    """
    Vega in forward space (1pc vol bump, centered)

    :param blip_bps is in bps, e.g. 100.0
    """
    # i.e. 1% of vol
    blip = blip_bps / 10000.0

    return (black(cp, f, x, t, v + blip, r) -
            black(cp, f, x, t, v - blip, r)) / 2.


def black_theta(cp, f, x, dte, v, r):
    """
    Theta in forward space: returns undiscounted value
    takes days-to-expiry, not yearfrac
    returns decay (i.e. negative)
    """
    t = float(dte) / 365.25

    # corner case on maturity day
    if t <= 1. / 365.25:
        return black(cp, f, x, 0.00001, v, r) - black(cp, f, x, t, v, r)

    t_1 = (float(dte) - 1.)/365.25
    return black(cp, f, x, t_1, v, r) - black(cp, f, x, t, v, r)


#   #################################   #
#   #   #   ANALYTICAL Greeks   #   #   #
#   #################################   #

def black_delta_an(cp, f, x, t, v, r):
    """
    BlackDelta, analytical version
    """
    d1, d2 = d12(f, x, t, v)
    pass


def black_gamma_an(cp, f, x, t, v, r):
    """
    BlackGamma, analytical version
    """
    pass


def black_vega_an(cp, f, x, t, v, r):
    """
    Vega , analytical version
    """
    pass


def black_theta_an(cp, f, x, dte, v, r):
    """
    Theta, analytical version
    """
    pass


#   #################################   #
def d12(U, x, t, v, r_q=0.):
    """
    Black & Scholes d1 and d2

    :parameters
    U is the underlying
    x is the strikes
    t is the maturity, in year_fract
    v is the implied volatility, e.g. 0.15
    r_q is the int rate less dividend yield, for Black is zero
    """
    v_sqrt = v * sqrt(t)  # voltime formulation
    d1 = (log(U/x) + 0.5 * (v_sqrt**2) + r_q * t) / v_sqrt
    d2 = d1 - v_sqrt
    return d1, d2
