import copy
import os
import numpy as np
import torch
from rps.runtime import DataContainer

def unwrap_img(x):
    if isinstance(x, DataContainer):
        x = x.data
    if isinstance(x, (list, tuple)):
        x = x[0]
    if isinstance(x, torch.Tensor) and x.dim() == 3:
        x = x.unsqueeze(0)
    return x


def unwrap_metas(x):
    if isinstance(x, DataContainer):
        x = x.data
    if isinstance(x, (list, tuple)):
        return x[0] if len(x) > 0 and isinstance(x[0], list) else list(x)
    if isinstance(x, dict):
        return [x]
    return x


def capture_training_rng():
    import random
    return {'schema_version': 1, 'python': random.getstate(), 'numpy': np.
        random.get_state(), 'torch_cpu': torch.get_rng_state().clone(),
        'torch_cuda': [state.clone() for state in torch.cuda.
        get_rng_state_all()] if torch.cuda.is_available() else [],
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.
        is_available() else 0}


def _validated_numpy_rng_state(state):
    if not isinstance(state, tuple) or len(state) != 5:
        raise ValueError('Invalid NumPy RNG state')
    probe = np.random.RandomState(0)
    probe.set_state(state)
    return state


def _validated_torch_rng_state(state, label):
    if (not torch.is_tensor(state) or state.dtype != torch.uint8 or state.
        ndim != 1 or state.device.type != 'cpu'):
        raise ValueError('Invalid {} RNG state'.format(label))
    return state


def restore_training_rng(state):
    import random
    expected = {'schema_version', 'python', 'numpy', 'torch_cpu',
        'torch_cuda', 'cuda_device_count'}
    if not isinstance(state, dict) or set(state) != expected or state[
        'schema_version'] != 1:
        raise ValueError('Invalid training RNG state')
    python_probe = random.Random(0)
    python_probe.setstate(state['python'])
    _validated_numpy_rng_state(state['numpy'])
    cpu_state = _validated_torch_rng_state(state['torch_cpu'], 'Torch CPU')
    torch.Generator().set_state(cpu_state)
    cuda_states = state['torch_cuda']
    count = state['cuda_device_count']
    actual_count = torch.cuda.device_count() if torch.cuda.is_available(
        ) else 0
    if type(count) is not int or count < 0 or not isinstance(cuda_states, list
        ) or len(cuda_states) != count or count != actual_count:
        raise ValueError(
            'CUDA RNG device count mismatch: checkpoint={}, current={}'.
            format(count, actual_count))
    for index, cuda_state in enumerate(cuda_states):
        _validated_torch_rng_state(cuda_state, 'CUDA {}'.format(index))
        torch.Generator(device='cuda:{}'.format(index)).set_state(cuda_state)
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(cpu_state)
    if cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)


def _loader_identity(loader):
    sampler = loader.sampler
    return {'dataset_length': len(loader.dataset), 'loader_length': len(
        loader), 'sampler_length': len(sampler), 'sampler_type': type(
        sampler).__module__ + '.' + type(sampler).__qualname__,
        'batch_size': loader.batch_size, 'drop_last': bool(loader.drop_last
        ), 'num_workers': int(loader.num_workers)}


def _capture_sampler_state(sampler):
    import copy
    from torch.utils.data import SequentialSampler
    if hasattr(sampler, 'state'):
        return {'kind': 'numpy_state_attribute', 'value': copy.deepcopy(
            _validated_numpy_rng_state(sampler.state))}
    if callable(getattr(sampler, 'state_dict', None)) and callable(getattr(
        sampler, 'load_state_dict', None)):
        return {'kind': 'state_dict', 'value': copy.deepcopy(sampler.
            state_dict())}
    if isinstance(sampler, SequentialSampler):
        return {'kind': 'sequential', 'value': None}
    raise ValueError(
        'Sampler does not expose an exact epoch-boundary state: {}'.format(
        type(sampler).__qualname__))


def capture_loader_states(named_loaders):
    if not isinstance(named_loaders, dict) or not all(isinstance(name, str) for
        name in named_loaders):
        raise ValueError('Loaders must be a dictionary with string names')
    loaders = {}
    for name, loader in named_loaders.items():
        if loader is None:
            loaders[name] = None
            continue
        identity = _loader_identity(loader)
        if identity['num_workers'] != 0:
            raise ValueError(
                'Exact epoch-boundary resume requires num_workers=0')
        generator = getattr(loader, 'generator', None)
        loaders[name] = {'identity': identity, 'generator': generator.
            get_state().clone() if generator is not None else None,
            'sampler': _capture_sampler_state(loader.sampler)}
    return {'schema_version': 1, 'loaders': loaders}


def restore_loader_states(named_loaders, state):
    import copy
    if not isinstance(state, dict) or set(state) != {'schema_version',
        'loaders'} or state['schema_version'] != 1 or not isinstance(state[
        'loaders'], dict) or not isinstance(named_loaders, dict) or set(
        named_loaders) != set(state['loaders']):
        raise ValueError(
            'Invalid loader state or loader name mismatch')
    for name, loader in named_loaders.items():
        saved = state['loaders'][name]
        if loader is None or saved is None:
            if loader is not None or saved is not None:
                raise ValueError('Loader presence mismatch: {}'.format(name))
            continue
        if not isinstance(saved, dict) or set(saved) != {'identity',
            'generator', 'sampler'}:
            raise ValueError('Invalid loader state: {}'.format(name))
        identity = _loader_identity(loader)
        if identity['num_workers'] != 0 or saved['identity'] != identity:
            raise ValueError('Loader configuration mismatch: {}'.format(name))
        generator = getattr(loader, 'generator', None)
        if (generator is None) != (saved['generator'] is None):
            raise ValueError('Loader generator presence mismatch: {}'.
                format(name))
        if generator is not None:
            torch.Generator().set_state(_validated_torch_rng_state(saved[
                'generator'], 'loader {}'.format(name)))
        current = _capture_sampler_state(loader.sampler)
        sampler_state = saved['sampler']
        if not isinstance(sampler_state, dict) or set(sampler_state) != {'kind'
            , 'value'} or sampler_state['kind'] != current['kind']:
            raise ValueError('Sampler state kind mismatch: {}'.format(name))
        if sampler_state['kind'] == 'numpy_state_attribute':
            _validated_numpy_rng_state(sampler_state['value'])
        elif sampler_state['kind'] == 'sequential' and sampler_state['value'
            ] is not None:
            raise ValueError('Invalid sequential sampler state: {}'.format(
                name))
    for name, loader in named_loaders.items():
        if loader is None:
            continue
        saved = state['loaders'][name]
        if loader.generator is not None:
            loader.generator.set_state(saved['generator'])
        sampler_state = saved['sampler']
        if sampler_state['kind'] == 'numpy_state_attribute':
            loader.sampler.state = copy.deepcopy(sampler_state['value'])
        elif sampler_state['kind'] == 'state_dict':
            loader.sampler.load_state_dict(copy.deepcopy(sampler_state[
                'value']))


def _save_ckpt_full(model_student, model_teacher, optimizer, scaler,
    scheduler, save_path, meta=None, global_iter=None, training_state=None):
    ckpt = {'state_dict': model_student.state_dict(), 'teacher_state_dict':
        model_teacher.state_dict() if model_teacher is not None else None,
        'optimizer': optimizer.state_dict() if optimizer is not None else
        None, 'scaler': scaler.state_dict() if scaler is not None else None,
        'scheduler': scheduler.state_dict() if scheduler is not None else
        None, 'global_iter': int(global_iter) if global_iter is not None else
        0, 'meta': meta or {}}
    if training_state is not None:
        ckpt['training_state'] = training_state
    import tempfile
    destination = os.path.abspath(os.fspath(save_path))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w+b', prefix=os.path.
            basename(destination) + '.tmp-', dir=os.path.dirname(
            destination), delete=False) as handle:
            temporary = handle.name
            torch.save(ckpt, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
