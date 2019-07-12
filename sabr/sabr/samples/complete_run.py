"""
INCOMPLETE ** INCOMPLETE ** INCOMPLETE
Sample vol surface fitting using SABR:
 1. read raw prices for [SPX, ...] options (using .csv file in analytics.data from Raj) &
    get raw implied vols
 2. TODO: get implied vol spline for each liquid maturity (e.g. LIQUID: 4 or more prices) &
    calibrate SABR params from implied vol spline (using constrained version of the Nelder-Mead optimization) &
    interpolate SABR params for non-LIQUID maturities
 3. check initial raw prices vs. calibrated model prices
"""

import os
import numpy as np
import analytics.utils as utils
import pandas as pd
import analytics.calibration.sabrfit as sabrfit
import analytics.model.bs as bs


def complete_sample():

    # insert tickers (.csv files in /.data)
    underlyings = ['spx', ]

    data_folder = utils.DEFAULT_data_dir
    sabr_params = {}  # {tckr: sabr_param}

    for tckr in underlyings:

        tckr_fullpath = os.path.join(data_folder, "%s.csv" % tckr)

        # 1. get raw data and calculate implieds
        #    df.columns = 'Expires', 'Strikes', 'Prices', 'CP', 'implied_vols'
        mat, f, df_tckr = utils.get_data(tckr_fullpath)

        # 2. for each maturity, SABR calibration
        #
        mats = set(df_tckr['Expires'])
        mats = np.sort(list(mats))
        sabr = {}

        for T in mats:
            # TODO: create multi-index pd rather than this fudge ?
            df_mat = df_tckr[df_tckr['Expires'] == T]

            strikes = df_mat['Strikes'].values
            vols = df_mat['implied_vols'].values

            # TODO: spline before calibration
            if len(vols) > 4:
                # CONSTRAINED: vol surface parametrization: returns tuple (p[], gof)
                sabr[T], _ = sabrfit.Sabrfit(f, mat[T], strikes, vols).sabrp()
            else:  # do not even try to fit
                _arr = np.zeros(3, dtype=float); _arr.fill(np.nan)
                sabr[T], _ = _arr, 0.

        sabr_params[tckr] = pd.DataFrame.from_dict(sabr, orient='index')
        sabr_params[tckr].columns = ['rho', 'alpha', 'ini_vol']

        # interpolate missing maturities
        '''
            ...
        sabr_params[tckr]['rho'].interpolate(method='spline', order=3).plot()
        sabr_params[tckr]['rho'].interpolate(method='time').plot()
            ...
        '''

        # 3. check initial raw prices vs. calibrated model prices
        #


if __name__ == '__main__':
    complete_sample()
