renderers = {}


def register(name):
    def decorator(cls):
        renderers[name] = cls
        return cls
    return decorator


def make(config):
    if isinstance(config, str):
        name = config
        config = {}
    else:
        name = config.get('name')

        
    if not name:
        raise ValueError('Renderer name is required')

    if name not in renderers:
        raise ValueError(f'Unknown renderer: {name}')
        
    renderer = renderers[name](config)
    return renderer

from . import vanilla_renderer, mip_renderer, scaffold_renderer, surfel_renderer