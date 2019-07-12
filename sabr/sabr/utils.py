"""
Miscellaneous (static) utility functions:
 - dates: year_fract and days_between
 - reading files

TODO: in get_data() insert i. checks (volume and strikes), int_rates & divs, prices vs. bid or ask, ...
"""
import os
# from munch import munchify
import pandas_datareader as pdr
import datetime as dt
import numpy as np
import pandas as pd

# for the 3D plot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.pylab import cm  # color map
import scipy as sp

from analytics.trading_calendars import USTradingCalendar
from analytics.model.bs import implied_vol


#   #################################   #
def days_between(d_end, d_start=0,
                 dates_string=False, trading_calendar=True):
    """ :returns the no of (business) days between 2 dates (d1 and d2)
        :params dates are datetimes, if  string then insert dates_string=True
    """
    # converting from string to datetime
    if dates_string:
        d2 = dt.datetime.strptime(d_end, "%Y-%m-%d")
        d1 = dt.datetime.today() if d_start == 0 else dt.datetime.strptime(d_start, "%Y-%m-%d")
    else:  # already datetime
        d2 = d_end; d1 = dt.datetime.today() if d_start == 0 else d_start

    assert d2 >= d1

    if trading_calendar:
        _holidays = USTradingCalendar().holidays(d1, d2).tolist()
        delta = np.busday_count(d1, d2, holidays=_holidays)
    else:
        delta = (d2 - d1).days
    return delta


def year_frac(d_end, d_start=0,
              dates_string=False, trading_calendar=True):
    """ :returns year fraction between 2 (business) dates
        :params dates are datetimes, if  string then insert dates_string=True
    """
    delta_days = days_between(d_end, d_start, dates_string, trading_calendar)
    year = 252. if trading_calendar else 365.25

    return delta_days / year


def add_days(d, n):
    """add n calendar days to d
       :param d is a string (not a datetime)
    """
    d = dt.datetime.strptime(d, "%Y-%m-%d")
    d = d + dt.timedelta(days=n)
    return d


def sign(x):
    """ :returns sign function (as float)
        if x is complex then use numpy.sign()
    """
    sgn_int = x and (1, -1)[x < 0]
    return 1.0 * sgn_int


#   #################################   #
#   Folders dir for DATA and OUT(puts)... in one place for ease of future potential changes
def get_updir_wfld(folder_string):
    analytics_top_dir = os.path.dirname(os.path.dirname(__file__))  # parent dir of current one
    return os.path.join(analytics_top_dir, folder_string)  # ~/folder

DEFAULT_data_dir = get_updir_wfld("data")
DEFAULT_out_dir = get_updir_wfld("out")

DEFAULT_samples_dir = os.path.join(get_updir_wfld("analytics"), "samples")

# # better , but trying to avoid extra dependencies (from munch import munchify)
# e.g. DEFAULT_data becomes DEFAULT.DATA_DIR
# DEFAULT = munchify({
#    "DATA_DIR": get_updir_wfld("data"),
#    "OUT_DIR": get_updir_wfld("out"),
#    "SAMPLE_DIR": os.path.join(get_updir_wfld("analytics"), "samples")
# })
#   #################################   #


def readvols(filename):
    """
    reads a textfile with 2 columns (strikes, vols)
    :return tuple (strikes, vols), each is a list
    """
    with open(filename) as textFile:
        lines = [line.split() for line in textFile]
    stks = [float(i[0]) for i in lines]
    vols = [float(i[1]) for i in lines]
    return stks, vols


# TODO: since '_trading_calendar=True' makes it slower -> get list of year_frac in one go
def get_data(filenamefull, calculate_iv=True, _trading_calendar=False,
             # volume_threshold=1, strike_threshold=False,  # NOT implemented yet
             int_rate=0., dividend_rate=0.,
             print_debug=False):
    """
    :param filenamefull: filename with full path
    :param calculate_iv: calculate implieds (bool)
    :param _trading_calendar:  business days as default for year_frac calcs
    :param int_rate: 0.0 as default
    :param dividend_rate: 0.0 as default

    :param print_debug: for debug purposes only

    :return 1. get raw option data from .csv, 2. calculate implied vols 3. add to the df
               df.columns = 'Expires', 'Strikes', 'Prices', 'CP', 'implied_vols'
             4. rtrn : fwd and df

      [tkr].csv file:
      - col of values:  'Expires', 'Strikes', 'Prices', 'CP[+1 call and -1 for put]'
      - single value: 'ValueDate', 'Underlying', 'Spot', 'ATM strike', 'Interest Rate'
    """
    # 1. get raw data
    df = pd.read_csv(filenamefull, header=0, parse_dates=True,  # index_col=0, dayfirst=True,
                     # names=('Expires', 'Strikes', 'Prices', 'CP', # etc.)
                     )
    # debug
    df_initial_rows = df.shape[0]   # or = len(df.index) # but slower

    # 2. Filtering dataframe (checks), [remember to reset index before adding imp vol]
    df = df[df['Prices'] > 0.01]  # df.reset_index(drop=True)
    # debug
    df_row_nonzero_prices = df.shape[0]

    # 3. get single useful values
    value_date = pd.to_datetime(df['ValueDate'][0])
    spot = df['Spot'][0]
    # and then drop them all ...
    df.drop(['ValueDate', 'Underlying', 'Spot', 'ATM strike', 'Interest Rate'], axis=1, inplace=True)

    # finally prepare some data
    df['Expires'] = pd.to_datetime(df['Expires'], infer_datetime_format=True)
    #
    cp = df['CP']
    prices = df['Prices']     # .values           # transforms it into np.array
    strikes = df['Strikes']   # " "  "            # "     "       "       "
    expiries = df['Expires']

    assert len(prices) == len(strikes)
    assert len(strikes) == len(expiries)

    # 4. calculate implieds
    if calculate_iv:

        sigmas = []; mat = {}
        for cp, price, strike, expiry in zip(cp, prices, strikes, expiries):

            f = spot    # for now
            r_q = int_rate - dividend_rate

            try:
                t = year_frac(expiry, value_date, trading_calendar=_trading_calendar)
                mat[expiry] = t  # TODO: to be removed, just for simple testing ...
                sigma = implied_vol(cp, f, strike, t, r_q, price)
                sigmas.append(sigma)
            except:
                sigma = 0.
                sigmas.append(sigma)

    df['implied_vols'] = pd.Series(sigmas)

    # filtering: only consistent (i.e. non-zero sigma) prices
    df = df[df['implied_vols'] > 0.001];  # df.reset_index(drop=True)
    # debug
    df_row_nonzero_implieds = df.shape[0]

    if print_debug == True:
        print("rows at start: {}, rows with non zero prices: {}, final rows with consistent (non-zero vol) prices: {}"
              .format(df_initial_rows, df_row_nonzero_prices, df_row_nonzero_implieds))

    # TODO: it should rtrns just df --> just fugding for simple testing ...
    return mat, spot, df


def get_options_from_yahoo(tkr):
    """
    :param ticker: underlying, e.g. APPL
    """
    tkr_options = pdr.data.Options(tkr, 'yahoo')  # [needs data.Options from pandas_datareader]
    data = tkr_options.get_all_data()
    return data


def plot_3d_surf(inputs, labels,
                 mesh_p=300, interp_method='linear'):
    """
    :rtype: object
    :param inputs: list of 3 arrays -> x, y, z
    :param labels: list of strings -> labels for x, y, z
    :return: plotting 3d vol surface (not just data points)
    """
    x, y, z = inputs
    xl, yl, zl = labels

    fig = plt.figure(figsize=(11, 8))
    # ax = fig.gca(projection='3d')
    ax = Axes3D(fig, azim=-29, elev=50)

    # creates a rectangular grid out of x and y arrays; linspace() returns evenly spaced numbers in defined interval
    xx, yy = np.meshgrid(sp.linspace(min(x), max(x), mesh_p), sp.linspace(min(y), max(y), mesh_p))
    # interpolates between the mesh data points
    zz = sp.interpolate.griddata(np.array([x, y]).T, np.array(z), (xx, yy), method=interp_method)

    surf = ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm)  # , rstride=1, cstride=1,  linewidth=0.5, antialiased=True, cmap=cm.coolwarm)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_zlabel(zl)

    ax.contour(xx, yy, zz)

    # fig.colorbar(surf, shrink=0.5, aspect=5)

    plot_3d(x, y, z, fig, ax)    # to plot data points on top too

    plt.show()


def plot_3d(x, y, z, _fig=None, _ax=None):
    """
    Generic 3d plot - plot only the data points (as opposed to a surface plot)
    """
    # if ... else for embedding it into another pic
    fig = plt.figure() if _fig is None else _fig
    ax = Axes3D(fig, azim=-29, elev=50) if _ax is None else _ax

    ax.plot(x, y, z, 'o')

    if _fig is None:
        plt.xlabel("x")
        plt.ylabel("y")
    plt.show()


#   #################################   #
if __name__ == "__main__":

    import analytics.utils as utils
    data_folder = utils.DEFAULT_data_dir
    tckr = 'spx'
    tckr_fullpath = os.path.join(utils.DEFAULT_data_dir, "%s.csv" % tckr)
    _, _, df = get_data(tckr_fullpath, print_debug=True, _trading_calendar=True)
    print(df.head())
