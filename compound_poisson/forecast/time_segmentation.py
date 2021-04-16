"""For splitting an array of dates into divisions, eg every year or every summer

How to use or implement:
    An array of consecutive dates (assumed here) is passed into the
        constructor
    Use the method __iter__() in an interator, each iteration return a
        two-array, containing the date used to represent the division (eg 1st
        January 2000 to represent the year 2000) and a slice object pointing to
        the elements of the time_array which correspond to the division of the
        current iteration

Abstract base class: TimeSegmentator
TimeSegmentator <- AllInclusive
                <- YearSegmentator
                <- SpringSegmentator
                <- SummerSegmentator
                <- AutumnSegmentator
                <- WinterSegmentator
"""

import datetime


class TimeSegmentator(object):
    """Abstract class for splitting array of dates into divisions

    Only complete divisions (eg complete calander years) are considered

    Attributes:
        time_array: array of date objects, assumed to be consecutive
        start_month: integer, month of the start of the season, inclusive,
            to be used for datetime
        start_day: integer, day of the start of the season, inclusive,
            to be used for datetime
        end_month: integer, month of the end of the season, exclusive,
            to be used for datetime
        end_day: integer, day of the end of the season, exclusive,
            to be used for datetime
    """

    def __init__(self, time_array, start_month, start_day, end_month, end_day):
        self.time_array = time_array
        self.start_month = start_month
        self.start_day = start_day
        self.end_month = end_month
        self.end_day = end_day

    def get_date_for_segement(self, index):
        """Return datetime to represent a division

        Args:
            index: slice object pointing to time_array
        Return:
            date object: half way point of division
        """
        date_start = self.time_array[index.start]
        date_end = self.time_array[index.stop-1]
        delta = (date_end - date_start) / 2
        return date_start + delta

    def get_time_array(self):
        """Return array of all the date time representation of each division
        """
        time_array = []
        for date, index in self:
            time_array.append(date)
        return time_array

    def __iter__(self):
        """Yield a two-array, containing the date used to represent the
            division and a slice object pointing to the elements of the
            time_array
        """
        index_start = 0
        is_in_season = False
        delta = datetime.timedelta(1)  # delta of 1 day
        for i, time in enumerate(self.time_array):
            if is_in_season:
                time_add_one = time + delta
                if (time_add_one.month == self.end_month
                        and time_add_one.day == self.end_day):
                    is_in_season = False
                    index = slice(index_start, i+1)
                    season_date = self.get_date_for_segement(index)
                    yield (season_date, index)
                    index_start = None
            else:
                if (time.month == self.start_month
                        and time.day == self.start_day):
                    is_in_season = True
                    index_start = i


class AllInclusive(TimeSegmentator):
    """Dummy implementation which treats the entire time_array as one division
    """

    def __init__(self, time_array):
        super().__init__(time_array, None, None, None, None)

    # override
    def __iter__(self):
        yield (None, slice(0, len(self.time_array)))


class YearSegmentator(TimeSegmentator):
    """Divide the time_array into calander years
    """

    def __init__(self, time_array):
        super().__init__(time_array, 1, 1, 1, 1)

    # override
    def get_date_for_segement(self, index):
        """Return datetime to represent a division

        Args:
            index: slice object pointing to time_array
        Return:
            date object: Janauary 1st of that year
        """
        year = self.time_array[index.start].year
        year = datetime.date(year, 1, 1)
        return year


class Q12Segmentator(TimeSegmentator):

    def __init__(self, time_array):
        super().__init__(time_array, 1, 1, 7, 1)


class Q34Segmentator(TimeSegmentator):

    def __init__(self, time_array):
        super().__init__(time_array, 7, 1, 1, 1)


class SpringSegmentator(TimeSegmentator):

    def __init__(self, time_array):
        super().__init__(time_array, 3, 1, 6, 1)


class SummerSegmentator(TimeSegmentator):

    def __init__(self, time_array):
        super().__init__(time_array, 6, 1, 9, 1)


class AutumnSegmentator(TimeSegmentator):

    def __init__(self, time_array):
        super().__init__(time_array, 9, 1, 12, 1)


class WinterSegmentator(TimeSegmentator):

    def __init__(self, time_array):
        super().__init__(time_array, 12, 1, 3, 1)
