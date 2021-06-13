import dataset
from dataset import location


class CardiffTraining(location.Location):

    def __init__(self):
        super().__init__()

    def load_data(self):
        self.load_data_from_city(dataset.ana.AnaDualTraining(), "Cardiff")


class Cardiff1Training(location.Location):

    def __init__(self):
        super().__init__()

    def load_data(self):
        self.load_data_from_city(dataset.ana.AnaDual1Training(), "Cardiff")


class Cardiff2Training(location.Location):

    def __init__(self):
        super().__init__()

    def load_data(self):
        self.load_data_from_city(dataset.ana.AnaDual2Training(), "Cardiff")


class Cardiff5Training(location.Location):

    def __init__(self):
        super().__init__()

    def load_data(self):
        self.load_data_from_city(dataset.ana.AnaDual5Training(), "Cardiff")


class Cardiff10Training(location.Location):

    def __init__(self):
        super().__init__()

    def load_data(self):
        self.load_data_from_city(dataset.ana.AnaDual10Training(), "Cardiff")


class CardiffTest(location.Location):

    def __init__(self):
        super().__init__()

    def load_data(self):
        self.load_data_from_city(dataset.ana.AnaDualTest(), "Cardiff")
