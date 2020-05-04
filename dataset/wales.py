import dataset

LAT = (55, 75)
LONG = (56, 82)

class Wales10Training(dataset.ana.AnaDual10Training):

    def __init__(self):
        super().__init__()

    def load_data(self):
        super().load_data()
        self.crop(LAT, LONG)

class Wales10Test(dataset.ana.AnaDual10Test):

    def __init__(self):
        super().__init__()

    def load_data(self):
        super().load_data()
        self.crop(LAT, LONG)

class Wales1Training(dataset.ana.AnaDual1Training):

    def __init__(self):
        super().__init__()

    def load_data(self):
        super().load_data()
        self.crop(LAT, LONG)

class Wales1Test(dataset.ana.AnaDual1Test):

    def __init__(self):
        super().__init__()

    def load_data(self):
        super().load_data()
        self.crop(LAT, LONG)
