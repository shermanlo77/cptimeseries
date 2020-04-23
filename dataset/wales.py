import dataset

LAT = (55, 75)
LONG = (56, 82)

class WalesTraining(dataset.AnaDual10Training):

    def __init__(self):
        super().__init__()

    def load_data(self):
        super().load_data()
        self.crop(LAT, LONG)

class WalesTest(dataset.AnaDual10Test):

    def __init__(self):
        super().__init__()

    def load_data(self):
        super().load_data()
        self.crop(LAT, LONG)
        self.trim([0, 365])
