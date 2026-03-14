"""
This library provides a decorator that fulfills a similar but simpler
functionality to that of the `doer` library. This is mainly intended to be
used to store frequently used things that may be slow to calculate, like
Hamiltonians, propagators, etc., for large dimensions. In practice, I use it
quite indiscriminately.
"""
import os
import inspect, functools, glob

import numpy as np


def store(path, overwrite_name=False):
    def decorator(func):
        if overwrite_name is False:
            func_name = func.__name__
        else:
            func_name = overwrite_name
        argspec = inspect.getfullargspec(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # generate filename

            if argspec.defaults is not None:
                kwarg_defaults = {**dict(zip(argspec.args[-len(argspec.defaults):],
                                              argspec.defaults))}
            else:
                kwarg_defaults = {}
            params = {**kwarg_defaults, **dict(zip(argspec.args, args)),
                      **kwargs,}
            # important that actual args come after defaults

            for k, v in params.copy().items():
                if isinstance(v, dict):
                    d = params.pop(k)
                    params.update(d)
                if callable(v): # if argument is a function
                    params[k] = v.__name__
            params = dict(params.items())
            params = {k:params[k] for k in sorted(params.keys())} # sort params

            with np.printoptions(legacy='1.25'): # avoid np.float64() in values
                filename = path + f'{func_name} '
                filename += " ".join(f"{k}={v}" for k, v in params.items())

            # there used to be an `if' to handle different saving formats
            load_func = np.load
            save_func = np.save
            format = 'npy'
            filename += f'.{format}'

            # if the file already exists, load it
            files = list(glob.iglob(path+f'/*.{format}'))
            if filename in files:
                out = load_func(filename)
                print(f'Loaded: {filename}')
                return out
            else:
                print(f'Not found: {filename}')

            out = func(*args, **kwargs)

            if not os.path.isdir(path): # make path before saving
                os.makedirs(path)
            save_func(filename, out)
            print(f'Saved: {filename}')
            return out
        return wrapper
    return decorator
