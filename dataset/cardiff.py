import dataset
from dataset import location

class CardiffTraining(location.Location):

    def __init__(self):
        super().__init__()

    def load_data(self):
        self.load_data_from_city(dataset.ana.AnaDualTraining(), "Cardiff")

class CardiffTest(location.Location):

    def __init__(self):
        super().__init__()

    def load_data(self):
        self.load_data_from_city(dataset.ana.AnaDualTest(), "Cardiff")

class CardiffTrainingHalf(location.Location):

    def __init__(self):
        super().__init__()

    def load_data(self):
        self.load_data_from_city(dataset.ana.AnaDual10Test(), "Cardiff")
