""" test date functions
"""

from analytics.utils import *
import pytest as pt


def test_dates():

    end = "2018-1-17"    # Monday 15-1-18 was USMartinLutherKingJr holiday
    start = "2018-1-11"

    bus_days = days_between(end, start, dates_string=True)
    calendar_days = days_between(end, start, dates_string=True, trading_calendar=False)

    year_frac_bus = year_frac(end, start, dates_string=True)
    year_fract_cal = year_frac(end, start, dates_string=True, trading_calendar=False)

    assert bus_days == 3
    assert calendar_days == 6

    assert year_frac_bus == pt.approx(0.011905, abs=1e-6)   # 0.01190476190476190.. = bus_days/252.
    assert year_fract_cal == pt.approx(0.016427, abs=1e-6)  # 0.01642710472279261.. = calendar_days/365.25

if __name__=='__main__':
    test_dates()