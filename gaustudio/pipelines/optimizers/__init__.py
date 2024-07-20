optimizers = {}

def register(name):
    def decorator(cls):
        optimizers[name] = cls
        return cls
    return decorator


def make(config):
    if isinstance(config, str):
        name = config
        config = {}
    else:
        name = config.get('name')

    if not name:
        raise ValueError('Optimizer name is required')

    if name not in optimizers:
        raise ValueError(f'Unknown optimizer: {name}')
    model = optimizers[name](config)
    return model

from . import general_optimizer