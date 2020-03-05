import numpy as np

from .Target import Target

class TargetZ(Target):
    """Wrapper Target class for the latent variables z
    
    Attributes:
        time_series: TimeSeries object being wrapped around
        z_array_before: copy of time_series.z_array when save_state called
    """
    
    def __init__(self, time_series):
        super().__init__()
        self.time_series = time_series
        self.z_array_before = None
    
    def get_n_dim(self):
        return len(self.time_series)
    
    def get_state(self):
        return self.time_series.z_array
    
    def update_state(self, state):
        self.time_series.z_array = state
        self.time_series.update_all_cp_parameters()
    
    def get_log_likelihood(self):
        return self.time_series.get_joint_log_likelihood()
    
    def get_log_target(self):
        return self.get_log_likelihood()
    
    def save_state(self):
        self.z_array_before = self.time_series.z_array.copy()
    
    def revert_state(self):
        self.update_state(self.z_array_before)
