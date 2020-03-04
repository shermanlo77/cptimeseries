from .Ana_1 import Ana_1

class IsleOfMan(Ana_1):
    def __init__(self):
        super().__init__()
        lat = (43, 57)
        long = (55, 75)
        
        for key, model_field in self.model_field.items():
            self.model_field[key] = model_field[
                :, lat[0]:lat[1], long[0]:long[1]]
        self.rain = self.rain[:, lat[0]:lat[1], long[0]:long[1]]
        
        for key in self.topography.keys():
            self.topography[key] = self.topography[key][
                lat[0]:lat[1], long[0]:long[1]]
            self.topography_normalise[key] = self.topography_normalise[key][
                lat[0]:lat[1], long[0]:long[1]]
