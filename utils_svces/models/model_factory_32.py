from utils_svces.models.models_32x32.resnet import ResNet50, ResNet18, ResNet34
from utils_svces.models.models_32x32.fixup_resnet import fixup_resnet20, fixup_resnet56
from utils_svces.models.models_32x32.wide_resnet import (
    WideResNet28x2,
    WideResNet28x10,
    WideResNet28x20,
    WideResNet34x20,
    WideResNet40x10,
    WideResNet70x16,
    WideResNet34x10,
)
from timm.models.factory import create_model
from utils_svces.models.models_32x32.pyramid import aa_PyramidNet


def try_number_conversion(s):
    try:
        value = float(s)
        return value
    except ValueError:
        return s


def parse_params(params_list):
    params = {}

    if params_list is not None:
        assert len(params_list) % 2 == 0
        for i in range(len(params_list) // 2):
            key = params_list[2 * i]
            value = params_list[2 * i + 1]
            value = try_number_conversion(value)
            params[key] = value

        print(params)

    return params


def build_model(model_name, num_classes, model_params=None):
    model_name = model_name.lower()
    model_config = parse_params(model_params)

    img_size = 32
    if model_name == "resnet18":
        model = ResNet18(num_classes=num_classes)
        model_name = "ResNet18"
    elif model_name == "resnet34":
        model = ResNet34(num_classes=num_classes)
        model_name = "ResNet34"
    elif model_name == "resnet50":
        model = ResNet50(num_classes=num_classes)
        model_name = "ResNet50"
    elif model_name == "fixup_resnet20":
        model = fixup_resnet20(num_classes=num_classes)
        model_name = "FixupResNet20"
    elif model_name == "fixup_resnet56":
        model = fixup_resnet56(num_classes=num_classes)
        model_name = "FixupResNet56"
    elif model_name == "shakedrop_pyramid":
        model = aa_PyramidNet(depth=110, alpha=270, num_classes=num_classes)
        model_name = "ShakedropPyramid"
    elif model_name == "shakedrop_pyramid272":
        model = aa_PyramidNet(depth=272, alpha=200, num_classes=num_classes)
        model_name = "ShakedropPyramid272"
    elif model_name == "wideresnet28x2":
        model = WideResNet28x2(num_classes=num_classes, **model_config)
        model_name = "WideResNet28x2"
    elif model_name == "wideresnet28x10":
        model = WideResNet28x10(num_classes=num_classes)
        model_name = "WideResNet28x10"
    elif model_name == "wideresnet28x20":
        model = WideResNet28x20(num_classes=num_classes)
        model_name = "WideResNet28x20"
    elif model_name == "wideresnet34x10":
        model = WideResNet34x10(num_classes=num_classes, **model_config)
        model_name = "WideResNet34x10"
    elif model_name == "wideresnet34x20":
        model = WideResNet34x20(num_classes=num_classes)
        model_name = "WideResNet34x20"
    elif model_name == "wideresnet40x10":
        model = WideResNet40x10(num_classes=num_classes)
        model_name = "WideResNet40x10"
    elif model_name == "wideresnet70x16":
        model = WideResNet70x16(num_classes=num_classes)
        model_name = "WideResNet70x16"
    elif model_name == "vit-b16":
        model = create_model(
            "vit_base_patch16_224_in21k", num_classes=num_classes, pretrained=True
        )
        model_name = "ViT-B16"
        img_size = 224
    elif model_name == "vit-b32":
        model = create_model(
            "vit_base_patch32_224_in21k", num_classes=num_classes, pretrained=True
        )
        model_name = "ViT-B32"
        img_size = 224
    else:
        print(f"Net {model_name} not supported")
        raise NotImplementedError()

    return model, model_name, model_config, img_size
