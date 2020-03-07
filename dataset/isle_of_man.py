import dataset

class IsleOfMan(dataset.Ana_1):
    def __init__(self):
        super().__init__()
        lat = (43, 56)
        long = (55, 74)
        self.crop(lat, long)

class IsleOfManTraining(IsleOfMan):
    def __init__(self):
        super().__init__()
        training_range = [0, 3653]
        self.trim(training_range)

class IsleOfManTest(IsleOfMan):
    def __init__(self):
        super().__init__()
        test_range = [3653, 4018]
        self.trim(test_range)
