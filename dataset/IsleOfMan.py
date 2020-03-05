from .Ana_1 import Ana_1

class IsleOfMan(Ana_1):
    def __init__(self):
        super().__init__()
        lat = (43, 56)
        long = (55, 74)
        self.crop(lat, long)
