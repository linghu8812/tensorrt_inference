from .Model import Model


class ClassRes:
    def __init__(self, classes, prob):
        super().__init__()
        self.classes = classes
        self.prob = prob


class Classification(Model):
    def __init__(self, config):
        super().__init__(config)
