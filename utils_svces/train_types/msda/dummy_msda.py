from .mixed_sample_data_augmentation import MixedSampleDataAugmentation

class DummyMSDA(MixedSampleDataAugmentation):
    def __init__(self):
        super().__init__(None)

    def apply_mix(self, x):
        return x