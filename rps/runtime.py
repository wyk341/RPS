"""Configuration and checkpoint utilities for the RPS training entry point."""
import copy
from pathlib import Path
import runpy

import torch


class Config(dict):
    def __init__(self, values=None, **kwargs):
        super().__init__()
        for key, value in dict(values or {}, **kwargs).items():
            self[key] = value

    @classmethod
    def _convert(cls, value):
        if isinstance(value, dict) and not isinstance(value, cls):
            return cls(value)
        if isinstance(value, list):
            return [cls._convert(x) for x in value]
        if isinstance(value, tuple):
            return tuple(cls._convert(x) for x in value)
        return value

    def __setitem__(self, name, value):
        super().__setitem__(name, self._convert(value))

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        self[name] = value

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def to_dict(self):
        def plain(value):
            if isinstance(value, dict):
                return {k: plain(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(plain(v) for v in value)
            return value
        return plain(self)

    @classmethod
    def fromfile(cls, filename):
        namespace = runpy.run_path(str(Path(filename).resolve()))
        if not isinstance(namespace.get('config'), dict):
            raise ValueError('The configuration file must define a config dictionary.')
        return cls(namespace['config'])


class DataContainer:
    def __init__(self, data):
        self.data = data


def load_checkpoint(model, filename, map_location='cpu', strict=True):
    checkpoint = torch.load(str(filename), map_location=map_location)
    state = checkpoint.get('state_dict', checkpoint)
    if not isinstance(state, dict):
        raise ValueError('Expected a model state dictionary.')
    state = {key.removeprefix('module.') if hasattr(key, 'removeprefix')
             else key[7:] if key.startswith('module.') else key: value
             for key, value in state.items()}
    model.load_state_dict(state, strict=strict)
    return checkpoint
