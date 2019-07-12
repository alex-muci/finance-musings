"""
Implements the SABR stochatic vol model (Hagan et al 2003)

SABR model (alpha: vol of vol in [>0], beta in [0,1], rho in [-1, 1]):
  df_t = vol_t f_t^beta dW_t         f_0 = fwd
  dvol_t = alpha vol_t dB_t          vol_0 = inivol (and alpha = vol of vol)
  and d<W, B>_t = rho dt

Calibration done via a constrained NM (see analytics.calibration.constrNM)

TODO: check VOL formula below, insert vol greeks,
      DONE: [error trapping, boundary checking (when f=one of the stks)]
"""
import numpy as np
from scipy.optimize import minimize
from analytics.calibration.constrNM import constrNM
import matplotlib.pyplot as plt


class Sabrfit:
    '''
     Returns the fitted sabr parameters {rho, alpha, inivol}
     -> which parametrises the vol surface
    '''
    def __init__(self, fwd, tenor, strikes, vols, beta=1.0):
        """
        strikes and vols are arrays
        NB: we work with forwards and not spot, i.e. F = S * exp((r-q) * t)
        """

        self.fwd = fwd
        self.tenor = tenor
        self.strikes = strikes
        self.vols = vols
        self.beta = beta

        self.initial_params = [-0.25, 0.8, 0.14]  # =[rho, alpha, ini_vol {, beta=1.0}]

    def lossfn(self, p):
        """
        Loss function for d-s minimization
        """
        asum = 0.0
        for i, strike in enumerate(self.strikes):
            fitvol = Sabrfit.vol(self.fwd, strike, self.tenor,
                                 p[0], p[1], p[2], self.beta)
            asum += (fitvol - self.vols[i])**2

        return asum

    def sabrp(self, inip=None):
        """
        Downhill simplex CONSTRAINED minimization for parameter vector p
        :returns the tuple(p[], )
        inip=[rho, alpha, inivol {, beta=1.0}], alpha is the vol of vol
        """
        if inip is None:
            inip = self.initial_params
        lb = [-1.0, 1e-5, 1e-5]
        ub = [1.0, None, None]  # None means infinity here
        res = constrNM(self.lossfn, inip, lb, ub, xtol=1e-8, maxiter=400)
        gof = self.gof(res['xopt'])
        return res['xopt'], gof

    # ###### UNCONSTRAINED/ ##### #
    def sabrp_u(self, inip=None):
        """
        Downhill simplex UN-CONSTRAINED minimization for parameter vector p
        :returns the tuple(p[], gof)
        """
        if inip is None:
            inip = self.initial_params
        res = minimize(self.lossfn, inip, method='nelder-mead',
                       options={'xtol': 1e-8, 'disp': True})
        gof = self.gof(res.x)
        return res.x, gof
    # ###### /UNCONSTRAINED ##### #

    def gof(self, p):
        """
        Helper for sabrp_u()->returns the max error (abs) in vol terms in fit
        e.g. 0.5 = max difference between actual and fitted vols is 0.5
        """
        maxdiff = 0.0
        for i in range(len(self.strikes)):
            fitvol = Sabrfit.vol(self.fwd, self.strikes[i], self.tenor,
                                 p[0], p[1], p[2], self.beta)
            diff = abs(fitvol - self.vols[i])
            if diff > maxdiff:
                maxdiff = diff

        # return rounded to 2 digits
        mdiff = round(maxdiff * 100.0, 2)
        return mdiff

    def plotvols(self, p):
        """
        Plot actual & fitted vols vs. strikes
        """
        npoints = len(self.strikes)
        x = self.strikes
        y1 = self.vols
        y2 = [0] * npoints
        y2 = [self.fittedvol(self.fwd, vol, p) for vol in x]

        plt.figure(figsize=(10, 5))
        plt.title('Actual vs Fitted Vols')
        plt.xlabel('Strikes')
        plt.ylabel('red: Actual vols, blue: Fitted vols')
        plt.grid(True)
        plt.xlim(min(x) - 10, max(x) + 10)
        plt.ylim(min(y1) - 0.005, max(y1) + 0.005)

        plt.plot(x, y1, color="red", label="Actual vols")
        plt.plot(x, y2, color="blue", label="Fitted vols")
        plt.legend()

        plt.show()

    def fittedvol(self, fwd, stk, p):
        """
        Instance method for fitted vol

        :params:
        fwd and stk: floats
        fitted vol surface params p[]
        tenor NOT allowed as input as we're fitting time slices

        should require time-varying p[] even if interpolated from a few knots
        """
        fitvol = Sabrfit.vol(fwd, stk, self.tenor, p[0], p[1], p[2])
        return fitvol

    @staticmethod
    def vol(fwd, stk, opttenor, rho, alpha, inivol, beta=1.0):
        """ Returns (one) model vol, given sabr parameters (and single strike)
        """
        skewness = (1.0 - beta)
        if fwd != stk:
            otmness = (fwd / stk)
            zedval = Sabrfit.z(fwd, stk, alpha, inivol, beta)
            xval = Sabrfit.x(zedval, rho)
            mult = skewness * np.log(otmness)

            multTerm = (1.0 + (mult**2) / 24.0 + (mult ** 4) / 1920.0)
            mt = ((fwd * stk)**(skewness * 0.5))
            multTerm *= mt
            multTerm = (inivol * zedval / xval) / multTerm

            # bracketed terms
            mult = skewness * inivol
            t1 = (mult**2) / ((fwd * stk)**skewness) / 24.0
            t2 = (rho * beta * alpha * inivol) / mt / 4.0
            t3 = (2.0 - 3.0 * (rho**2)) * (alpha**2) / 24.0

            sigma = multTerm * (1.0 + (t1 + t2 + t3) * opttenor)

        else:
            f_beta = fwd ** skewness
            f_two_beta = fwd ** (2. - 2 * beta)
            sigma = ((inivol / f_beta) *
                     (1 + opttenor * ((skewness**2 / 24. * inivol**2 /
                      f_two_beta) + (0.25 * rho * beta *
                      alpha * inivol / f_beta) +
                      ((2. - 3 * rho ** 2) / 24. * alpha ** 2))))

        if sigma is None:
            return 0.0
        else:
            return sigma

    @staticmethod
    def x(z, rho):
        """ Helper method for vol()
        """
        current = (np.sqrt(1.0 - 2.0 * rho * z + z * z) + (z - rho))/(1.0-rho)
        return np.log(current)

    @staticmethod
    def z(fwd, stk, alpha, inivol, beta=1.0):
        """ Helper method for vol()
        """
        exponent = (1.0 - beta) * 0.5
        return np.log(fwd/stk) * (alpha / inivol) * ((fwd * stk)**exponent)
