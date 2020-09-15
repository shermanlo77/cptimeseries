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

class TimeSegmentator(object):

    def __init__(self, time_array):
        self.time_array = time_array

    def __iter__(self):
        raise NotImplementedError

class YearSegmentator(TimeSegmentator):

    def __init__(self, time_array):
        super().__init__(time_array)

    def __iter__(self):
        #key: years #value: index of times with that year
        index_start = 0
        for i, time in enumerate(self.time_array):
            if i>0:
                if time.month == 12 and time.day == 31:
                    year = self.time_array[index_start].year
                    year = datetime.date(year, 1, 1)
                    index = slice(index_start, i+1)
                    yield (year, index)
                    index_start = i+1
