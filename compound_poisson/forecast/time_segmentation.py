"""For splitting an array of dates into divisions, eg every year or every summer

How to use or implement:
    An array of consecutive dates (assuming there) is passed into the
        constructor
    Use the method __iter__() in an interator, each iteration return a
        two-array, containing the time used to represent the division (eg 1st
        January 2000 to represent the year 2000) and a slice object pointing to
        the elements of the time_array which correspond to the division of the
        current iteration
"""

import datetime
from dateutil import relativedelta

class TimeSegmentator(object):

    def __init__(self, time_array):
        self.time_array = time_array

    def get_time_array(self):
        time_array = []
        for date, index in self:
            time_array.append(date)
        return time_array

    def __iter__(self):
        raise NotImplementedError

class AllInclusive(TimeSegmentator):
    #returns the index of the entire test set

    def __init__(self, time_array):
        super().__init__(time_array)

    def __iter__(self):
        yield (None, slice(0, len(self.time_array)))


class YearSegmentator(TimeSegmentator):
    #assumes the data always start on the 1st January

    def __init__(self, time_array):
        super().__init__(time_array)

    def __iter__(self):
        index_start = 0
        for i, time in enumerate(self.time_array):
            if i>0:
                if time.month == 12 and time.day == 31:
                    year = self.time_array[index_start].year
                    year = datetime.date(year, 1, 1)
                    index = slice(index_start, i+1)
                    yield (year, index)
                    index_start = i+1

class SeasonSegmentator(TimeSegmentator):
    """For segmentating seasons, assuming they are 3 months long

    Attributes:
        start_month: integer, month of the start of the season, inclusive,
            to be used for datetime
        start_day: integer, day of the start of the season, inclusive,
            to be used for datetime
        end_month: integer, month of the end of the season, exclusive,
            to be used for datetime
        end__day: integer, day of the end of the season, exclusive,
            to be used for datetime
    """

    def __init__(self, time_array, start_month, start_day, end_month, end_day):
        super().__init__(time_array)
        self.start_month = start_month
        self.start_day = start_day
        self.end_month = end_month
        self.end_day = end_day

    def get_time_for_season(self, index):
        #represent the season with the date in the middle of the season
        #index is the pointer to the start of the season
        date = self.time_array[index]
        delta = relativedelta.relativedelta(months=1, days=14)
        return date + delta

    def __iter__(self):
        index_start = 0
        is_in_season = False
        delta = relativedelta.relativedelta(days=1)
        for i, time in enumerate(self.time_array):
            if is_in_season:
                time_add_one = time + delta
                if (time_add_one.month == self.end_month
                    and time_add_one.day == self.end_day):
                    is_in_season = False
                    season_date = self.get_time_for_season(index_start)
                    index = slice(index_start, i+1)
                    yield (season_date, index)
                    index_start = None
            else:
                if (time.month == self.start_month
                    and time.day == self.start_day):
                    is_in_season = True
                    index_start = i

class SpringSegmentator(SeasonSegmentator):
    def __init__(self, time_array):
        super().__init__(time_array, 3, 1, 6, 1)

class SummerSegmentator(SeasonSegmentator):
    def __init__(self, time_array):
        super().__init__(time_array, 6, 1, 9, 1)

class AutumnSegmentator(SeasonSegmentator):
    def __init__(self, time_array):
        super().__init__(time_array, 9, 1, 12, 1)

class WinterSegmentator(SeasonSegmentator):
    def __init__(self, time_array):
        super().__init__(time_array, 12, 1, 3, 1)
