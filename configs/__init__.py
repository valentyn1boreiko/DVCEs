import argparse
import os

import yaml

__all__ = ['get_config']

def get_config(args):
    config = dict2namespace(setdefault(_get_raw_config(args.config), _get_raw_config("default.yml")))

    for key in args.__dict__:
        if key in ['timestep_respacing', 'skip_timesteps',
                'lp_custom', 'lp_custom_value', 'lpips_sim_lambda', 'l2_sim_lambda', 'range_lambda', 'l1_sim_lambda',
                'deg_cone_projection',
                'clip_guidance_lambda', 'classifier_lambda',
                'num_imgs', 'batch_size',
                'seed',
                'classifier_type', 'second_classifier_type', 'third_classifier_type', 'method']:
            if int(getattr(args, key)) != -1:
                setattr(config, key, getattr(args, key))
        else:
            setattr(config, key, getattr(args, key))

    config.config = 'imagenet1000.yml'

    if config.use_blended:
        print('using blended')
    else:
        print('not using blended')

    return config


def _get_raw_config(name):
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, name), 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    return yaml_dict


def setdefault(config, default):
    print('config is', config, 'default is', default)
    for x in default:
        v = default.get(x)
        if isinstance(v, dict) and x in config:
            setdefault(config.get(x), v)
        else:
            config.setdefault(x, v)
    return config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace