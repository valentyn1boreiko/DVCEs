class MixedSampleDataAugmentation():
    def __init__(self, loss):
        self.loss = loss

    def apply_mix(self, x):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.apply_mix(*args, **kwargs)

