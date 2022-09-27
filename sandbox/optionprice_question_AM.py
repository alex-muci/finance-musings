from datetime import datetime, timedelta
from scipy.stats import norm
from math import log, exp, sqrt
from typing import Tuple

N = norm.cdf
"""
TODO:
    Implement the below methods `get_price` & `get_delta`.
    
    They should calculate and return the Black76 option price (as given by equations [1] & [2]) and delta 
    (as given by equations [5] & [10]) in
    https://www.glynholton.com/notes/black_1976/
    Further info at https://en.wikipedia.org/wiki/Black_model
    (NB - pay close attention to the sign of call and put deltas)

    Feel free to use scipy.stats.norm (already imported) in order to calc CDF where appropriate.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
    
    Other than scipy.stats.norm, you may only use modules available in the Python Standard Library for version 3.7.
    
    You may edit the __init__ & _calc_maturity methods if needed.
"""


class EuropeanOptionOnFuture:
    """
    A class used to represent a European Option on a Futures contract and to calculate price and risk measures,
    as given by the Black76 model

    Methods
    _______
    get_price(futures_price: float, current_time: datetime) -> float
        Returns the discounted option price
    get_delta(futures_price: float, current_time: datetime) -> float
        Returns the discounted option delta
    """

    def __init__(self, strike_price: float, expiry_date: datetime, vol: float, discount_rate: float, option_type: str):
        """
        :param strike_price: the strike price of the option (x)
        :param expiry_date: the expiration date of the option
        :param vol: the implied volatility with which the option is to be priced (sigma)
        :param discount_rate: the interest rate
        :param option_type: either 'c' or 'p' to represent whether the option is a call or put
        """
        self._strike_price = strike_price
        self._expiry_date = expiry_date
        self._vol = vol
        self._discount_rate = discount_rate
        self._option_type = option_type

        self.cp = {'c': 1., 'p': -1.}.get(option_type.lower())  # get more compact Black76 vs. usual "if type=='c'..."
        if self.cp is None:
            raise ValueError("'option_type' must be either 'p' or 'c' (incl. capital letters)")

    def get_price(self, futures_price: float, current_time: datetime) -> float:
        """
        Returns the discounted Black76 option price

        :param futures_price: reference price for the option's underlying futures contract
        :param current_time: the time at which to calculate the option's price
        :return: the option price at the input futures price & time
        """
        cp = self.cp
        df, d1, d2 = self._df_d12(futures_price, current_time)
        return cp * (futures_price * N(cp * d1) - self._strike_price * N(cp * d2)) * df

    def get_delta(self, futures_price: float, current_time: datetime) -> float:
        """
        Returns the discounted Black76 analytic option delta

        :param futures_price: reference price for the option's underlying futures contract
        :param current_time: the time at which to calculate the option's delta
        :return: the option delta at the input futures price & time
        """
        cp = self.cp
        df, d1, _ = self._df_d12(futures_price, current_time)
        return cp * N(cp * d1) * df

    def _calc_maturity(self, current_time: datetime) -> float:
        """Returns the time to maturity (in years) of the option, using a calendar days / 365 model

        :param current_time: the time at which to calculate the option's maturity
        :return: the time remaining until the expiration of the option
        """
        return (self._expiry_date - current_time).days / 365

    def _df_d12(self, futures_price: float, current_time: datetime) -> Tuple[float, float, float]:
        """
        Helper: Returns discount factors and Black d1 and d2 functions

        :param current_time: the time at which to calculate the option's maturity
        :return: the time remaining until the expiration of the option
        """
        f = futures_price
        if f <= 0.:
            raise ValueError("'futures_price' must be positive!")

        x = max(self._strike_price, 1e-12)  # floor it for safety - better a descriptor in __init__
        t = max(self._calc_maturity(current_time), 1e-12)
        v = max(self._vol, 1e-12)
        r = self._discount_rate

        df = exp(-r * t)  # could be a different method, but save some calcs here

        v_sqrt = v * sqrt(t)
        d1 = (log(f / x) + 0.5 * (v_sqrt ** 2)) / v_sqrt
        d2 = d1 - v_sqrt

        return df, d1, d2


if __name__ == '__main__':
    # Simple test case.
    # Feel free to add a few more.

    x = 100  # Strike price
    sig = 0.5  # Volatility
    expiry = datetime(2022, 11, 1, 12, 0, 0)  # Expiration date
    r = 0.01  # Discount rate
    opt_type = 'c'  # Option type

    opt = EuropeanOptionOnFuture(x, expiry, sig, r, opt_type)

    f = 100  # Test case underlying price
    curr_time = datetime(2021, 11, 1, 12, 0, 0)  # Test case current time

    price = opt.get_price(f, curr_time)
    delta = opt.get_delta(f, curr_time)

    # Check price & delta
    print(f"Test price = {price}")
    assert abs(price - 19.544836) <= 0.0001
    print(f"Test delta = {delta}")
    assert abs(delta - 0.59273) <= 0.0001

    # Call/put delta simple check
    assert EuropeanOptionOnFuture(x, expiry, sig, r, 'c').get_delta(f, curr_time) > 0
    assert EuropeanOptionOnFuture(x, expiry, sig, r, 'p').get_delta(f, curr_time) < 0

    #   ################# #################
    # Check price/delta with volatility=0 (and discount_rate=0) -> call==fwd and delta ~=1
    opt_zerovol, diff = EuropeanOptionOnFuture(x, expiry, 0., 0., opt_type), 3.
    price_zerovol, delta_zerovol = opt_zerovol.get_price(x+diff, curr_time), opt_zerovol.get_delta(x+diff, curr_time)
    # print(f"Option with zero vol (and rate), price: {price_zerovol} (i.e. diff: {diff}) with delta {delta_zerovol}.")
    assert abs(price_zerovol - 3.) < 0.00001
    assert abs(delta_zerovol - 1.) < 0.00001

    # more tests
    forward = 0.034
    strike = 0.050
    r = 0.0
    time_to_expiry = 2.0
    volatility = 0.20
    expiry = curr_time + timedelta(days=365*2)

    call = EuropeanOptionOnFuture(strike, expiry, volatility, r, 'c')
    put = EuropeanOptionOnFuture(strike, expiry, volatility, r, 'p')

    assert round(call.get_price(forward, curr_time)*1000, 4) == 0.4599
    assert round(put.get_price(forward, curr_time)*10, 4) == 0.1646
    assert round(call.get_delta(forward, curr_time), 4) == 0.1108
    assert round(put.get_delta(forward, curr_time), 4) == -0.8892

    # one more
    forward = 101.0
    strike = 102.0
    expiry = curr_time + timedelta(days=int(365*0.5))
    r = .01
    volatility = 0.2
    assert round(EuropeanOptionOnFuture(strike, expiry, volatility, r, 'p').get_price(forward, curr_time), 4) == 6.1968

    # capital letter for option_type
    assert round(EuropeanOptionOnFuture(strike, expiry, volatility, r, 'P').get_price(forward, curr_time), 4) == 6.1968

    # # assertion error: futures_price < 0
    # import pytest
    # with pytest.raises(ValueError):
    #    call.get_price(-100, curr_time)
    #    # EuropeanOptionOnFuture(strike, expiry, volatility, r, 'blah').get_price(forward, curr_time)
