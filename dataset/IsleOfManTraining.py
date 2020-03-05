from .IsleOfMan import IsleOfMan

class IsleOfManTraining(IsleOfMan):
    
    def __init__(self):
        super().__init__()
        training_range = [0, 3653]
        self.trim(training_range)
