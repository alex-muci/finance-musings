"""
Define the US trading calendar using pandas.tseries.holiday

TODO: Eurex calendar
"""

from pandas.tseries.holiday import (AbstractHolidayCalendar,    # inherit from this to create your calendar
                                    Holiday, nearest_workday,   # to custom some holidays
                                    #
                                    USMartinLutherKingJr,       # already defined holidays
                                    USPresidentsDay,            # "     "   "   "   "   "
                                    GoodFriday,
                                    USMemorialDay,              # "     "   "   "   "   "
                                    USLaborDay,
                                    USThanksgivingDay           # "     "   "   "   "   "
                                    )


class USTradingCalendar(AbstractHolidayCalendar):
    rules = [
      Holiday('NewYearsDay', month=1, day=1, observance=nearest_workday),
      USMartinLutherKingJr,
      USPresidentsDay,
      GoodFriday,
      USMemorialDay,
      Holiday('USIndependenceDay', month=7, day=4, observance=nearest_workday),
      USLaborDay,
      USThanksgivingDay,
      Holiday('Christmas', month=12, day=25, observance=nearest_workday)
    ]

