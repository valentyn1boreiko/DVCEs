from utils_svces.model_normalization import *
import os
import yaml
import argparse
import sys
import utils_svces.models.model_factory_32 as factory_32
import utils_svces.models.model_factory_224 as factory_224

models_dict = {
    "cifar10": Cifar10Wrapper,
    "restrictedimagenet": RestrictedImageNetWrapper,
    "cifar100": Cifar100Wrapper,
    "funduskaggle": FundusKaggleWrapper_clahe_v2_new_qual_eval_drop1,  # FundusKaggleWrapper_clahe_v2_new_qual_eval, #FundusKaggleWrapper_raw_v2_new_qual_eval_artifacts_green_circles_blue_squares, #FundusKaggleWrapper_clahe_v2_new_qual_eval, #FundusKaggleWrapper_clahe_v2_new_qual_eval,#FundusKaggleWrapper_raw_clahe_v2, #FundusKaggleWrapper, #FundusKaggleWrapperBackgroundSubtracted, #FundusKaggleWrapper,
    "funduskaggle_background_sub": FundusKaggleWrapper,  # FundusKaggleWrapperBackgroundSubtracted}
    "oct": OCTWrapper_first1000,
}

factory_dict = {32: factory_32, 224: factory_224}


