"""
Sample vol surface fitting using sabr
First version: RG 12/12/17

NB: for COMMAND LINE
To run from the command line, run it as a module (from 'analytics'):
    C:/Users/.../analytics:> python -m analytics.samples.sabr_run
rather than the usual python analytics/samples/sabr_run.py
"""

import analytics.utils as utils
import analytics.calibration.sabrfit as sabrfit
import analytics.model.bs as bs


def main():

    f = 2661.35     # fwd as of 11/12/17 close
    t = 0.1807      # 45 days to expiry (dte)
    r = 0.
    inip =[-0.25, 0.8, 0.14]     # wake-up for vol surface

    filename = r"vols.txt"  # r"analytics\samples\vols.txt"
    strikes, vols = utils.readvols(filename)

    # basic check for congruence
    if len(strikes) != len(vols):
        raise Exception("strikes & vols arrays do not match up!")

    # initializer for vol surface fitting
    s = sabrfit.Sabrfit(f, t, strikes, vols)

    # UNCONSTRAINED: vol surface parametrization: returns tuple (p[], gof)
    pout_u = s.sabrp_u(inip)
    print(pout_u[0], pout_u[1])

    # CONSTRAINED
    inip =[-0.25, 0.8, 0.14]    # re-set
    pout = s.sabrp(inip)
    print(pout[0], pout[1])

    # #test vol greeks
    # blip_bps = 1.0 #blipping spot
    # blipvol_bps = 10.0 #blipping vol
    # inivol = pout[0][2]
    # sg = sabr_greeks.SabrGreeks(stk, t, pout[0], blip_bps, blipvol_bps)
    # v = sg.volderiv(f, inivol)
    # print(v)

    # test black values
    stk = 2595.0
    vfit = s.fittedvol(f, stk, pout[0])
#   print(vfit) # prints fitted vol for a user defined strike
    cp = -1
    b = bs.black(cp, f, stk, t, vfit, r)
    bd = bs.black_delta(cp, f, stk, t, vfit, r)
    bg = bs.black_gamma(cp, f, stk, t, vfit, r)
    bv = bs.black_vega(cp, f, stk, t, vfit, r)
    dt = t * 365.25
    bt = bs.black_theta(cp, f, stk, dt, vfit, r)


    # old style
    print("black=%.2f, delta=%.2f and gamma=%.2f, vega=%.2f, theta=%.2f"
          % (b, bd, bg, bv, bt))

    # new style
    print("black={:.2f}, delta={:.2f} and gamma={:.2f}, vega={:.2f}, theta={}"
          .format(b, bd, bg, bv, bt))


    # plot the fit
    # s.plotvols(pout_u[0])
    s.plotvols(pout[0])


if __name__ == '__main__':
    main()
