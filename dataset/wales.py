import dataset

LAT = (55, 75)
LONG = (56, 82)


class WalesTraining(dataset.ana.AnaDualTraining):

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


class Wales2Training(dataset.ana.AnaDual2Training):

    def __init__(self):
        super().__init__()

    def load_data(self):
        super().load_data()
        self.crop(LAT, LONG)


class Wales5Training(dataset.ana.AnaDual5Training):

    def __init__(self):
        super().__init__()

    def load_data(self):
        super().load_data()
        self.crop(LAT, LONG)


class Wales10Training(dataset.ana.AnaDual10Training):

    def __init__(self):
        super().__init__()

    def load_data(self):
        super().load_data()
        self.crop(LAT, LONG)


class WalesTest(dataset.ana.AnaDualTest):

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
