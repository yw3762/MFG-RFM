import torch


class ErrorTracker1D:
    def __init__(self, residual, normalization, boundary, l1_update_error, n_pts=1000):
        self.errors = {
            'residual': residual,
            'normalization': normalization,
            'boundary': boundary,
            'l1-update-error': l1_update_error,
            'fd-residual': torch.empty(n_pts),
        }

    def __setitem__(self, key, value):
        self.errors[key] = value

    def __getitem__(self, key):
        return self.errors[key]
