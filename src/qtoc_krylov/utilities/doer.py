"""
This library provides a class that allows for automatically checking whether a
given calculation has already been made and its results stored. If it has, then
simply load the data. If it hasn't proceed to calculate and store.

Files are stored named with a hash code. A look up table is used to link a
given calculation to the correct hash.
"""

import os
import json
import inspect
import hashlib
from copy import deepcopy
from functools import partial

import numpy as np


def _ignore_save(data, ignore_save):
    if not ignore_save is None:
        ignore = [i for i in range(len(data)) if i not in ignore_save]
        out = np.empty(len(data), dtype=object)
        out[:] = data
        out = out[ignore]
        if out.shape == (1,):
            out = out[0]
    else:
        out = data
    return out


def _ignore_data(data, ignore_data):
    if not ignore_data is None:
        if isinstance(data, tuple):
            out = [data[i] for i in range(len(data)) if i not in ignore_data]
            if len(out) == 1:
                out = out[0]
            return out
        ignore = [i for i in range(len(data)) if i not in ignore_data]
        return data[ignore]
    return data


def get_hash(doer):
    try:
        with open(doer.path+'table.json', 'r') as table:
            dic = json.load(table)
            hash = dic[doer.get_infostring()]
    except (FileNotFoundError, KeyError):
        hash = None
    return hash


def delete_entry(doer):
    try:
        with open(doer.path+'table.json', 'r+') as table:
            dic = json.load(table)
            infostring = doer.get_infostring()
            del dic[infostring]
        with open(doer.path+'table.json', 'w') as table:
            json.dump(dic, table, indent=4, separators=(', ', ': '))
            print(f'Deleted entry from table: {infostring}')
    except (FileNotFoundError, KeyError):
        ...


def load_data(doer):
    infostring = doer.get_infostring()
    hash = get_hash(doer)

    filename = f'{doer.path}{hash}.npy'
    try:
        data = np.load(filename, allow_pickle=True)
        print(f'Data found: {infostring}')
    except FileNotFoundError:
        print(f'Data NOT found: {infostring}')
        return None
    return data


def delete_data(doer):
    infostring = doer.get_infostring()
    hash = get_hash(doer)

    filename = f'{doer.path}{hash}.npy'
    try:
        os.remove(filename)
        print(f'Removed data: {filename}')
    except FileNotFoundError:
        ...


def save_data(data, doer):
    data = _ignore_save(data, doer.ignore_save)

    infostring = doer.get_infostring()
    hash = hashlib.sha256(infostring.encode('UTF-8')).hexdigest()

    filename = f'{doer.path}{hash}.npy'
    if not os.path.isdir(doer.path): # make path before saving
        os.makedirs(doer.path)
    np.save(filename, data)

    try:
        with open(doer.path+'table.json', 'r+') as table:
            dic = json.load(table)
            dic[infostring] = hash
            table.seek(0)
            json.dump(dic, table, indent=4, separators=(', ', ': '))
    except FileNotFoundError:
        with open(doer.path+'table.json', 'w') as table:
            dic = {}
            dic[infostring] = hash
            json.dump(dic, table, indent=4, separators=(', ', ': '))

    print(f'Data saved: {infostring}')
    return data


def _get_params_partial(partial_func):
    """
    Get the total arguments that are set from the function itself and from
    its partial.
    """
    f_spec = inspect.getfullargspec(partial_func.func)
    p_spec = inspect.getfullargspec(partial_func)

    p_spec_def = p_spec.defaults
    if p_spec_def is None:
        p_defaults = {}
    else:
        p_defaults = dict(zip(p_spec.args[-len(p_spec_def):], p_spec_def))

    p_spec_kwdef = p_spec.kwonlydefaults
    if p_spec_kwdef is None:
        p_kwdefaults = {}
    else:
        p_kwdefaults = p_spec_kwdef

    # these are not the real defaults, but those defaulted by partial
    f_defaults = dict(zip(f_spec.args[:len(partial_func.args)],
                          partial_func.args))

    p_params = {**f_defaults, **p_defaults, **p_kwdefaults}
    return p_params


class Doer:
    def __init__(self, func, alias=None,
                 args=None, fake_args=None, ignore_args=None,
                 ignore_save=None, ignore_out=None, ignore_load=None,
                 path=None):
        if args is None:
            args = {}
        if fake_args is None:
            fake_args = {}
        if ignore_args is None:
            ignore_args = []
        elif not isinstance(ignore_args, list):
            ignore_args = [ignore_args]

        self.func = func
        if hasattr(func, '__wrapped__'):
            # argspec is lost when function is wrapped
            self.spec = inspect.getfullargspec(getattr(func, '__wrapped__'))
        else:
            self.spec = inspect.getfullargspec(func)
        self.name = func.__name__
        if alias is None:
            self.alias = self.name
        else:
            self.alias = alias
        self.args = args
        self.fake_args = fake_args # are not passed, only appear in infostring
        self.ignore_args = ignore_args # are passed, don't appear in infostring

        self.ignore_save = ignore_save # dont save some part of output
        self.ignore_out = ignore_out # dont return some part of output
        self.ignore_load = ignore_load # dont return some part of loaded output
        self.path = path

    def set_args(self, **kwargs):
        self.args = {**self.args, **kwargs}

    def set_fakeargs(self, **kwargs):
        self.fake_args = {**self.fake_args, **kwargs}

    def set_ignoreargs(self, args):
        if isinstance(args, list):
            self.ignore_args += args
        else:
            self.ignore_args += [args]

    def get_infostring(self):
        argspec = self.spec

        if argspec.defaults is not None:
            kwarg_defaults = {**dict(zip(argspec.args[-len(argspec.defaults):],
                                          argspec.defaults))}
        else:
            kwarg_defaults = {}
        params = {**kwarg_defaults, **self.args, **self.fake_args}
        # NOTE: important that actual args come after defaults

        for p in self.ignore_args:
            params.pop(p)

        for k, v in params.copy().items():
            if isinstance(v, dict):
                d = params.pop(k)
                params.update(d)
            if callable(v): # if argument is a function
                if isinstance(v, functools.partial):
                    params_partial = _get_params_partial(v)
                    params[k] = v.func.__name__ + ' '
                    params[k] = '--' + params[k]
                    params[k] += ' '.join(f"{kk}={vv}" for kk, vv in params_partial.items())
                    params[k] += '--'
                else:
                    params[k] = v.__name__
            if isinstance(v, Doer): # if argument is a Doer
                params[k] = '--' + v.get_infostring() + '--'
        params = dict(params.items())
        params = {k:params[k] for k in sorted(params.keys())} # sort params

        with np.printoptions(legacy='1.25'): # avoid np.float64() in values
            infostring = f'{self.alias} '
            infostring += " ".join(f"{k}={v}" for k, v in params.items())
        return infostring

    def getit(self):
        return self.func(**self.args)

    def _doit(self):
        doit_args = self.args.copy()
        for k, v in doit_args.items():
            if isinstance(v, Doer):
                doit_args[k] = v.getit()
        return self.func(**doit_args)

    def doit(self, load=True, save=True, replace=False):
        if load:
            loaded_data = load_data(self)
            if not loaded_data is None:
                data = loaded_data
            elif save:
                data = self._doit()
                save_data(data, self)
            else:
                raise FileNotFoundError(f'Data NOT found: {self.get_infostring()}')
            return _ignore_data(data, self.ignore_load)
        else:
            data = self._doit()
            if save and replace:
                delete_entry(self)
                delete_data(self)
                save_data(data, self)
            if save and (not replace):
                print(f'Not saving (replace is set to False): {self.get_infostring()}')
            return _ignore_data(data, self.ignore_out)

    def copy(self):
        return deepcopy(self)
