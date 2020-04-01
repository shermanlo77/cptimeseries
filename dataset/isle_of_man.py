import dataset

class IsleOfMan(dataset.AnaInterpolate1):

    def __init__(self):
        super().__init__()

    def load_data(self):
        super().load_data()
        lat = (43, 56)
        long = (55, 74)
        self.crop(lat, long)

class IsleOfManTraining(IsleOfMan):

    def __init__(self):
        super().__init__()

    def load_data(self):
        super().load_data()
        training_range = [0, 3653]
        self.trim(training_range)

class IsleOfManTest(IsleOfMan):

    def __init__(self):
        super().__init__()

    def load_data(self):
        super().load_data()
        test_range = [3653, 4018]
        self.trim(test_range)
