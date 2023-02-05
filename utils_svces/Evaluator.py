from functools import partial

import matplotlib.colors as mcolors
import timm

from blended_diffusion.utils_blended.model_normalization import ResizeAndMeanWrapper
from utils_svces.load_trained_model import load_model as load_model_ratio

try:
    pass
except Exception as err:
    print(str(err))
import torch
from utils_svces.temperature_wrapper import TemperatureWrapper
from utils_svces.config import (
    models_dict,
    Evaluator_model_names_cifar10,
    Evaluator_model_names_imagenet1000,
    descr_args_generate,
    loader_all,
    temperature_scaling_dl_dict,
    Evaluator_model_names_funduskaggle,
    Evaluator_model_names_oct,
    full_dataset_dict,
)

from robustbench import load_model as load_model_benchmark
from utils_svces.model_normalization import IdentityWrapper, NormalizationWrapper

interpolation_to_int = {
    "nearest": 1,
    "bilinear": 2,
    "bicubic": 3,
    # For PIL compatibility
    "box": 4,
    "hamming": 5,
    "lanczos": 6,
}


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {"red": [], "green": [], "blue": []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict["red"].append([item, r1, r2])
            cdict["green"].append([item, g1, g2])
            cdict["blue"].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap("CustomMap", cdict)


c = mcolors.ColorConverter().to_rgb


class Evaluator(object):
    def __init__(self, args, config, kwargs, dataloader=None):
        self.args = args
        self.config = config
        self.kwargs = kwargs
        self.dataloader = dataloader

    def load_model(
        self,
        model_id,
        return_preloaded_models=False,
        use_temperature=True,
        prewrapper=IdentityWrapper,
    ):
        dataset = self.config.data.dataset_for_scorenet
        bs = self.config.sampling.batch_size
        folder = self.config.sampling.model_description.folder
        model_descr_args = {}
        folders = {}
        model_loaders = {}

        if dataset.lower() in ["cifar10", "tinyimages"]:
            type_ = Evaluator_model_names_cifar10[model_id]
        elif dataset.lower() in ["imagenet1000"]:
            type_ = Evaluator_model_names_imagenet1000[model_id]
        elif dataset.lower() in ["funduskaggle"]:
            type_ = Evaluator_model_names_funduskaggle[model_id]
        elif dataset.lower() in ["oct"]:
            type_ = Evaluator_model_names_oct[model_id]
        else:
            raise ValueError("Not implemented!")
        device = self.args.device

        if "nonrobust" not in type_:
            descr_args = vars(self.config.sampling.model_description).copy()
            descr_args["device"] = self.args.device
            descr_args["dataset"] = dataset
            descr_args["type"] = type_

            if self.args.model_epoch_num is not None:
                descr_args["checkpoint"] = self.args.model_epoch_num

            if "benchmark" in type_:
                # Overwrite ratio loader with one of the benchmarks loaders from https://github.com/RobustBench/robustbench or from
                # https://github.com/yaircarmon/semisup-adv
                model_name = type_.split("-")[1]
                is_Madry = "Madry" in model_name
                is_Microsoft = "Microsoft" in model_name
                is_Anon1s_small_radius = "Anon1small_radius" in model_name
                is_MaxNets = "Max:" in model_name
                is_Anon1_finetuning = "Anon1:finetuning_experimental" in model_name
                is_SAM = "SAM_experimental" in model_name
                is_XCIT = "xcit" in model_name

                is_experimental = (
                    is_XCIT or "improved" in model_name or "experimental" in model_name
                )

                if descr_args["dataset"].lower() in [
                    "cifar10",
                    "tinyimages",
                    "imagenet1000",
                    "funduskaggle",
                    "oct",
                ]:
                    # model_name in Gowal2020Uncovering_extra, Gowal2020Uncovering, Wu2020Adversarial
                    # add randomized smoothing!
                    if not is_experimental and len(type_.split("-")) == 3:
                        threat_model = type_.split("-")[2]
                    else:
                        threat_model = None

                    descr_args = descr_args_generate(
                        threat_model=threat_model,
                        is_experimental=is_experimental,
                        model_name=model_name,
                        project_folder=self.args.project_folder,
                    )
                    if is_Madry:
                        arguments = model_name.split("_")
                        descr_args["norm"] = arguments[1]
                        descr_args["device"] = device
                        assert (
                            arguments[2] == "improved" or arguments[2] == "experimental"
                        ), 'not a correct type of Madry model (only "improved" OR "experimental" are allowed)!'
                        descr_args["improved"] = arguments[2] == "improved"
                        descr_args["num_pretrained_epochs"] = (
                            None if len(arguments) < 5 else arguments[4]
                        )
                        if "_eps_" in model_name:
                            assert len(arguments) == 5, "Broken filename!"
                            descr_args["epsilon_finetuned"] = arguments[4]
                    elif is_Microsoft:
                        arguments = model_name.split(",")
                        descr_args["model_arch"] = arguments[0].split("Microsoft")[1]
                        descr_args["norm"] = arguments[2]
                        descr_args["epsilon"] = arguments[4]
                    elif is_Anon1s_small_radius:
                        descr_args["eps"] = model_name.split(":")[0]
                    elif is_MaxNets:
                        arguments = type_.split(",")
                        descr_args["dataset_name"] = dataset.lower()
                        descr_args["arch"] = arguments[1]
                        descr_args["model_name_id"] = arguments[2]
                        descr_args["num_classes"] = len(self.config.data.class_labels)
                        descr_args["img_size"] = self.config.data.image_size
                    elif is_Anon1_finetuning:
                        arguments = type_.split(",")
                        # Currently is only for 224x224 models!
                        descr_args["dataset_name"] = dataset.lower()
                        descr_args["arch"] = arguments[1].lower()
                        descr_args["model_name_id"] = arguments[2]
                        descr_args["num_classes"] = len(self.config.data.class_labels)
                    elif is_XCIT:
                        descr_args["model_name"] = model_name.split(",")[0]
                        descr_args["model_path"] = model_name.split(",")[1]

                    type_ = "-".join(type_.split("-")[1:])

                    load_model_final = (
                        load_model_benchmark
                        if not is_experimental
                        else models_dict[
                            "Madry"
                            if is_Madry
                            else "Anon1_small_radius_experimental"
                            if is_Anon1s_small_radius
                            else "Microsoft"
                            if is_Microsoft
                            else "Max"
                            if is_MaxNets
                            else "Anon1:finetuning"
                            if is_Anon1_finetuning
                            else "SAM"
                            if is_SAM
                            else "XCITrobust"
                            if is_XCIT
                            else model_name
                        ]
                    )
                else:
                    raise ValueError(
                        "Benchmark robust models are only available for CIFAR10!"
                    )

            else:
                if "_feature_model" in type_:
                    print("Loading feature comparison model")
                    descr_args["model_params"] = ["return_feature_map", True]
                    descr_args["type"] = type_.split("_")[0]

                load_model_final = load_model_ratio

        else:
            # ToDo: improve to remore the 'improved' suffix
            model_name = type_.split("_")[0] if "_" in type_ else type_
            descr_args = descr_args_generate(
                is_experimental=True,
                pretrained=(model_name == "ResNet50IN1000" or "timm" in type_),
                project_folder=self.args.project_folder,
            )
            if "BiT" in model_name or "ViT" in model_name or "CLIP" in type_:
                descr_args["model_name"] = model_name
                descr_args["class_labels"] = self.config.data.class_labels
                descr_args["dataset"] = dataset.lower()
                load_model_final = models_dict[model_name]
            elif "ViT" in model_name:
                descr_args["device"] = device
                load_model_final = models_dict[model_name]
            elif "timm" in type_:
                # ImageNet models used, with respective normalization
                model_name = type_.split(",")[1]
                print("timm model used is", model_name)
                def load_model_final(**kwargs):
                    return timm.create_model(**kwargs)
                cfg = timm.create_model(model_name, pretrained=True).default_cfg
                assert cfg["input_size"][1] == cfg["input_size"][2]
                print("loading cfg from timm", cfg)
                prewrapper = partial(
                    ResizeAndMeanWrapper,
                    size=cfg["input_size"][1],
                    interpolation=interpolation_to_int[cfg["interpolation"]],
                    mean=torch.tensor(cfg["mean"]),
                    std=torch.tensor(cfg["std"]),
                )
                print('prewrapper loaded')

                descr_args["model_name"] = model_name
            elif model_name == "ResNet50IN1000":
                load_model_final = models_dict[model_name]
            else:
                raise ValueError("Model is not implemented.")

        print("device is", device)
        print("device_ids are", self.args.device_ids)
        print("img_size is", self.config.data.image_size)
        print("dataset is", dataset)
        use_wrapper = True

        def load_model(x, loader, type_, folder):
            return TemperatureWrapper(loader_all(use_wrapper, device, loader, x, self.args.device_ids, prewrapper=prewrapper), T=None if not use_temperature else TemperatureWrapper.compute_temperature(loader_all(use_wrapper, device, loader, x, self.args.device_ids, prewrapper=prewrapper), temperature_scaling_dl_dict, bs, device=device, type_=type_, folder=folder, dataset=dataset, img_size=self.config.data.image_size, project_folder=self.args.project_folder, data_folder=self.args.data_folder, loader_full_dataset=full_dataset_dict))

        model_loaders[type_] = load_model_final
        model_descr_args[type_] = descr_args
        folders[type_] = folder
        print("model_descr_args", model_descr_args)

        if return_preloaded_models:
            return type_, model_loaders[type_], model_descr_args[type_], folders[type_]
        else:
            return [
                load_model(
                    model_descr_args[type_], model_loaders[type_], type_, folders[type_]
                )
                for type_ in model_descr_args.keys()
            ][0]
