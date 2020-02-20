import os
import pathlib

import joblib

from .Data import Data

class Ana_1(Data):
    def __init__(self):
        super().__init__()
        path_here = pathlib.Path(__file__).parent.absolute()
        self.copy_from(joblib.load(os.path.join(path_here, "ana_input_1.gz")))
