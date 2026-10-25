from .als import ALS

class CustomALS(ALS):
    def make_target(self, ratings):
        return (ratings - 1.0) / 4.0
