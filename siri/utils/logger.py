from siri.global_config import GlobalConfig as cfg


def lprint_(obj, x, debug=False):
    if debug and not cfg.debug: return

    buff = ''

    if obj is not None:
        if isinstance(obj, str):
            buff += f"[{obj}] "
        elif hasattr(obj, '__class__'):
            if hasattr(obj.__class__, '__name__'):
                buff += f"[{obj.__class__.__name__}] "

    if isinstance(x, str):
        buff += x
    else:
        buff += f"{str(x)} type={str(type(x))}"
    return buff

def lprint(obj, x, debug=False):
    buff = lprint_(obj, x, debug=debug)
    if buff is not None: print(buff)


def print_obj(obj):
    for attr in dir(obj):
        if not attr.startswith('__'):
            value = getattr(obj, attr)
            print(f"{attr}: {value}")