renderers = {}


def register(name):
    def decorator(cls):
        renderers[name] = cls
        return cls
    return decorator


def make(name, config):
    renderer = renderers[name](config)
    return renderer

from . import vanilla_renderer, mip_renderer, scaffold_renderer