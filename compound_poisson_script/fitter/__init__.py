import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

from .FitterMcmc import FitterMcmc
from .FitterSlice import FitterSlice
from .FitterHyperSlice import FitterHyperSlice
