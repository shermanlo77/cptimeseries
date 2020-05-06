import dataset

LAT = (43, 56)
LONG = (55, 74)
WEEK_TRAINING_RANGE = [0, 7]
WEEK_TEST_RANGE = [7, 14]

class IsleOfManTraining(dataset.ana.AnaDual10Training):

    def __init__(self):
        super().__init__()

    def load_data(self):
        super().load_data()
        self.crop(LAT, LONG)

class IsleOfManTest(dataset.ana.AnaDual10Test):

    def __init__(self):
        super().__init__()

    def load_data(self):
        super().load_data()
        self.crop(LAT, LONG)

class IsleOfManWeekTraining(IsleOfManTraining):

    def __init__(self):
        super().__init__()

    def load_data(self):
        super().load_data()
        self.trim(WEEK_TRAINING_RANGE)

class IsleOfManWeekTest(IsleOfManTraining):

    def __init__(self):
        super().__init__()

    def load_data(self):
        super().load_data()
        self.trim(WEEK_TEST_RANGE)
