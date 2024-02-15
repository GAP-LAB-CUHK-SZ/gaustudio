models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(config):
    if isinstance(config, str):
        name = config
        config = {}
    else:
        name = config.get('name')

    if not name:
        raise ValueError('Model name is required')

    if name not in models:
        raise ValueError(f'Unknown model: {name}')
    print(name)
    model = models[name](config)
    return model

from . import vanilla_sg, mip_sg, scaffold_sg