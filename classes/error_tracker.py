
class ErrorTracker1D:
    def __init__(self, residual, normalization, boundary, l1_error):
        self.errors = {
            'residual': residual,
            'normalization': normalization,
            'boundary': boundary,
            'l1-error': l1_error
        }
