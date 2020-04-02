import dataset

LAT = (43, 56)
LONG = (55, 74)
TRAINING_RANGE = [0, 3653]
TEST_RANGE = [3653, 4018]

class IsleOfMan(dataset.AnaDualExample1):

    def __init__(self):
        super().__init__()

    def load_data(self):
        super().load_data()
        self.crop(LAT, LONG)

class IsleOfManTraining(IsleOfMan):

    def __init__(self):
        super().__init__()

    def load_data(self):
        super().load_data()
        self.trim(TRAINING_RANGE)

class IsleOfManTest(IsleOfMan):

    def __init__(self):
        super().__init__()

    def load_data(self):
        super().load_data()
        self.trim(TEST_RANGE)
