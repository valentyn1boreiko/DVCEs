from .big_transfer import KNOWN_MODELS
import numpy as np
import os

BIG_TRANSFER_MODEL_DIR = 'BigTransfer/'

def build_model_big_transfer(model_name, num_classes, pretrained=True):
    model = KNOWN_MODELS[model_name](head_size=num_classes, zero_head=True)
    if pretrained:
        model.load_from(np.load(os.path.join(BIG_TRANSFER_MODEL_DIR, f"{model_name}.npz")))


    return model, model_name