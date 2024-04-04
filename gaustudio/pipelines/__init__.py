pipelines = {}


def register(name):
    def decorator(cls):
        pipelines[name] = cls
        return cls
    return decorator


def make(config):
    if isinstance(config, str):
        name = config
        config = {}
    else:
        name = config.get('name')

    if not name:
        raise ValueError('Pipeline name is required')

    if name not in pipelines:
        raise ValueError(f'Unknown Pipeline: {name}')

    model = pipelines[name](config)
    return model
