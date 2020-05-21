import dataset

class CardiffTraining(dataset.location.Location):

    def __init__(self):
        super().__init__()

    def load_data(self):
        self.load_data_from_city(dataset.ana.AnaDualTraining(), "Cardiff")

class CardiffTest(dataset.location.Location):

    def __init__(self):
        super().__init__()

    def load_data(self):
        self.load_data_from_city(dataset.ana.AnaDualTest(), "Cardiff")

class CardiffTrainingHalf(dataset.location.Location):

    def __init__(self):
        super().__init__()

    def load_data(self):
        self.load_data_from_city(dataset.ana.AnaDual10Test(), "Cardiff")
