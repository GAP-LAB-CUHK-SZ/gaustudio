initlalizers = {}

def register(name):
    def decorator(cls):
        initlalizers[name] = cls
        return cls
    return decorator


def make(config):
    if isinstance(config, str):
        name = config
        config = {}
    else:
        name = config.get('name')

    if not name:
        raise ValueError('Initlalizer name is required')

    if name not in initlalizers:
        raise ValueError(f'Unknown initlalizer: {name}')
    print(name)
    model = initlalizers[name](config)
    return model

from . import colmap, loftr, gaussiansky, depth