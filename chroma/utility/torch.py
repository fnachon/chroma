# Copyright Generate Biomedicines, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for device-aware torch execution."""

from __future__ import annotations

import warnings
from typing import Any, Callable

import torch

_MPS_OP_SUPPORT_CACHE = {}


def _iter_tensors(obj: Any):
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, tuple):
        for item in obj:
            yield from _iter_tensors(item)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_tensors(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _iter_tensors(item)


def _op_cache_key(op: Callable[..., Any], args: Any, kwargs: Any):
    tensor_signature = tuple(
        (str(tensor.dtype), tensor.ndim)
        for tensor in _iter_tensors((args, kwargs))
        if tensor.device.type == "mps"
    )
    return (getattr(op, "__module__", ""), getattr(op, "__qualname__", repr(op)), tensor_signature)


def _find_mps_device(obj: Any) -> torch.device | None:
    if torch.is_tensor(obj):
        return obj.device if obj.device.type == "mps" else None
    if isinstance(obj, tuple):
        for item in obj:
            device = _find_mps_device(item)
            if device is not None:
                return device
        return None
    if isinstance(obj, list):
        for item in obj:
            device = _find_mps_device(item)
            if device is not None:
                return device
        return None
    if isinstance(obj, dict):
        for item in obj.values():
            device = _find_mps_device(item)
            if device is not None:
                return device
        return None
    return None


def _move_mps_tensors(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device) if obj.device.type == "mps" else obj
    if isinstance(obj, tuple):
        return tuple(_move_mps_tensors(item, device) for item in obj)
    if isinstance(obj, list):
        return [_move_mps_tensors(item, device) for item in obj]
    if isinstance(obj, dict):
        return {key: _move_mps_tensors(value, device) for key, value in obj.items()}
    return obj


def _move_tensors(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(_move_tensors(item, device) for item in obj)
    if isinstance(obj, list):
        return [_move_tensors(item, device) for item in obj]
    if isinstance(obj, dict):
        return {key: _move_tensors(value, device) for key, value in obj.items()}
    return obj


def _is_mps_not_implemented_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "mps" in message and (
        "not implemented" in message
        or "not currently implemented" in message
        or "not supported" in message
        or "unsupported" in message
    )


def _is_mps_cpu_fallback_warning(warning: Warning) -> bool:
    message = str(warning).lower()
    return "mps backend" in message and "fall back to run on the cpu" in message


def _call_on_cpu(
    op: Callable[..., Any],
    args: Any,
    kwargs: Any,
    mps_device: torch.device,
) -> Any:
    cpu_args = _move_mps_tensors(args, torch.device("cpu"))
    cpu_kwargs = _move_mps_tensors(kwargs, torch.device("cpu"))
    result = op(*cpu_args, **cpu_kwargs)
    return _move_tensors(result, mps_device)


def call_with_explicit_mps_cpu_fallback(
    op: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Run an op on CPU when given MPS tensors, then move the result back."""

    mps_device = _find_mps_device((args, kwargs))
    if mps_device is None:
        return op(*args, **kwargs)

    key = _op_cache_key(op, args, kwargs)
    if _MPS_OP_SUPPORT_CACHE.get(key) is False:
        return _call_on_cpu(op, args, kwargs, mps_device)
    if _MPS_OP_SUPPORT_CACHE.get(key) is True:
        return op(*args, **kwargs)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message=".*MPS backend.*fall back to run on the CPU.*",
            category=UserWarning,
        )
        try:
            result = op(*args, **kwargs)
        except UserWarning as exc:
            if not _is_mps_cpu_fallback_warning(exc):
                raise
            _MPS_OP_SUPPORT_CACHE[key] = False
            return _call_on_cpu(op, args, kwargs, mps_device)

    _MPS_OP_SUPPORT_CACHE[key] = True
    return result


def call_with_mps_cpu_fallback(op: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run an op on the current device and retry on CPU if MPS rejects it."""

    mps_device = _find_mps_device((args, kwargs))
    if mps_device is None:
        return op(*args, **kwargs)

    key = _op_cache_key(op, args, kwargs)
    if _MPS_OP_SUPPORT_CACHE.get(key) is False:
        return _call_on_cpu(op, args, kwargs, mps_device)

    try:
        result = op(*args, **kwargs)
    except Exception as exc:
        if not _is_mps_not_implemented_error(exc):
            raise
        _MPS_OP_SUPPORT_CACHE[key] = False
        return _call_on_cpu(op, args, kwargs, mps_device)

    _MPS_OP_SUPPORT_CACHE[key] = True
    return result
