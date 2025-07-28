is_jax_enabled = False  # Default setting

def set_jax_enabled(value: bool):
    global is_jax_enabled
    is_jax_enabled = value

    from fftlog.config import set_jax_enabled as fftlog_set_jax_enabled
    fftlog_set_jax_enabled(is_jax_enabled)

    reload_modules()
    return

def get_jax_enabled():
    return is_jax_enabled

def reload_modules(): # https://stackoverflow.com/a/54369021
    from types import ModuleType
    import sys, importlib
    
    def deep_reload(m: ModuleType, lib='pybird'):
        name = m.__name__  # get the name that is used in sys.modules
        name_ext = name + '.'  # support finding sub modules or packages
        def compare(loaded: str): return (loaded == name) or loaded.startswith(name_ext)
        all_mods = tuple(sys.modules)  # prevent changing iterable while iterating over it
        sub_mods = filter(compare, all_mods)
        for pkg in sorted(sub_mods, key=lambda item: item.count('.'), reverse=True):
            if pkg != '%s.config' % lib: importlib.reload(sys.modules[pkg])  # reload packages, beginning with the most deeply nested
        return

    import pybird; deep_reload(pybird) # reload all modules of pybird
    import fftlog; deep_reload(fftlog, lib='fftlog')
    return