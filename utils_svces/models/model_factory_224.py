from timm.models.factory import create_model

def build_model(model_name, num_classes, **kwargs):
    model_name = model_name.lower()
    if model_name == 'sslresnext50':
        model_name = 'SSLResNext50'
        model = create_model('ssl_resnext50_32x4d', num_classes=num_classes, **kwargs)
        config = dict(name=model_name, **kwargs)
    elif model_name == 'resnet50':
        model_name = 'ResNet50'
        model = create_model('resnet50', num_classes=num_classes, **kwargs)
        config = dict(name=model_name, **kwargs)
    elif model_name == 'tresnetm':
        model_name = 'TResNet-M'
        model = create_model('tresnet_m', num_classes=num_classes, **kwargs)
        config = dict(name=model_name, **kwargs)
    elif model_name == 'seresnext26t':
        model_name = 'SE-ResNeXt-26-T'
        model = create_model('seresnext26t_32x4d', num_classes=num_classes, **kwargs)
        config = dict(name=model_name, **kwargs)
    elif model_name == 'seresnext50':
        model_name = 'SE-ResNeXt-50'
        model = create_model('seresnext50_32x4d', num_classes=num_classes, **kwargs)
        config = dict(name=model_name, **kwargs)
    else:
        print(f'Net {model_name} not supported')
        raise NotImplemented()

    return model, model_name, config