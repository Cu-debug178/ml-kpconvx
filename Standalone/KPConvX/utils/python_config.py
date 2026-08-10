#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

"""Pointcept-style declarative Python config overlays for KPConvX."""

import ast
import copy
import hashlib
import importlib.util
import os
import sys
import uuid
from collections.abc import Mapping
from types import ModuleType


CONFIG_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(__file__), '..', 'configs')
)
BASE_KEY = '_base_'
DELETE_KEY = '_delete_'
PROVENANCE_KEYS = frozenset(
    ('config_file', 'config_sources', 'config_options')
)


class PythonConfigError(ValueError):
    """Raised when a Python config file is missing or invalid."""


def _clone_config_value(value):
    """Recursively clone mapping configs without relying on EasyDict deepcopy."""

    if isinstance(value, Mapping):
        cloned = {
            key: _clone_config_value(item)
            for key, item in value.items()
        }
        try:
            return type(value)(cloned)
        except (TypeError, ValueError):
            return cloned
    if isinstance(value, list):
        return [_clone_config_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_config_value(item) for item in value)
    return copy.deepcopy(value)


def resolve_python_config(config_path, config_root=CONFIG_ROOT):
    """Resolve a config from the current directory or the project config root."""

    if not config_path or not str(config_path).strip():
        raise PythonConfigError('Python config path must not be empty.')

    raw_path = os.path.expanduser(os.fspath(config_path))
    path_variants = [raw_path]
    if not raw_path.endswith('.py'):
        path_variants.append(raw_path + '.py')

    candidates = []
    for path in path_variants:
        if os.path.isabs(path):
            candidates.append(path)
        else:
            candidates.append(os.path.abspath(path))
            candidates.append(os.path.abspath(os.path.join(config_root, path)))

    checked = []
    for candidate in candidates:
        candidate = os.path.realpath(candidate)
        if candidate in checked:
            continue
        checked.append(candidate)
        if os.path.isfile(candidate):
            if not candidate.endswith('.py'):
                raise PythonConfigError(
                    'Python config must use the .py extension: {:s}'.format(candidate)
                )
            return candidate

    raise PythonConfigError(
        'Python config not found. Checked: {:s}'.format(', '.join(checked))
    )


def _resolve_base_config(base_path, child_path):
    if not isinstance(base_path, (str, os.PathLike)):
        raise PythonConfigError(
            '{:s} entries in {:s} must be paths.'.format(BASE_KEY, child_path)
        )

    base_path = os.path.expanduser(os.fspath(base_path))
    if not base_path.endswith('.py'):
        base_path += '.py'
    if not os.path.isabs(base_path):
        base_path = os.path.join(os.path.dirname(child_path), base_path)
    base_path = os.path.realpath(base_path)

    if not os.path.isfile(base_path):
        raise PythonConfigError(
            'Base Python config not found: {:s} (referenced by {:s})'.format(
                base_path, child_path
            )
        )
    return base_path


def _load_config_module(config_path):
    module_name = '_kpconvx_python_config_' + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    if spec is None or spec.loader is None:
        raise PythonConfigError(
            'Unable to create an import specification for {:s}.'.format(config_path)
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        source = spec.loader.get_source(module_name)
        if source is None:
            raise PythonConfigError(
                'Unable to read Python config source: {:s}'.format(config_path)
            )
        code = compile(source, config_path, 'exec')
        exec(code, module.__dict__)
    except Exception as error:
        raise PythonConfigError(
            'Failed to execute Python config {:s}: {}'.format(config_path, error)
        ) from error
    finally:
        sys.modules.pop(module_name, None)
    return module


def _module_config(module):
    values = {}
    for name, value in module.__dict__.items():
        if name == BASE_KEY:
            values[name] = value
        elif name.startswith('_'):
            continue
        elif isinstance(value, ModuleType) or callable(value):
            continue
        else:
            values[name] = value
    return values


def _overlay_parameter_paths(overlays):
    paths = set()
    for _, overlay in overlays:
        for section_name, section_override in overlay.items():
            if isinstance(section_override, Mapping):
                for key in section_override:
                    if key != DELETE_KEY:
                        paths.add((str(section_name), str(key)))
            else:
                paths.add((str(section_name),))
    return paths


def _format_parameter_paths(paths):
    return ', '.join('.'.join(path) for path in sorted(paths))


def _load_config_chain(config_path, active_paths=()):
    config_path = os.path.realpath(config_path)
    if config_path in active_paths:
        cycle = active_paths + (config_path,)
        raise PythonConfigError(
            'Circular Python config inheritance: {:s}'.format(' -> '.join(cycle))
        )

    module_values = _module_config(_load_config_module(config_path))
    base_files = module_values.pop(BASE_KEY, [])
    if isinstance(base_files, (str, os.PathLike)):
        base_files = [base_files]
    elif not isinstance(base_files, (list, tuple)):
        raise PythonConfigError(
            '{:s} in {:s} must be a path or a list of paths.'.format(
                BASE_KEY, config_path
            )
        )

    overlays = []
    inherited_paths = set()
    next_active_paths = active_paths + (config_path,)
    for base_file in base_files:
        resolved_base = _resolve_base_config(base_file, config_path)
        base_overlays = _load_config_chain(resolved_base, next_active_paths)
        base_paths = _overlay_parameter_paths(base_overlays)
        conflicts = inherited_paths & base_paths
        if conflicts:
            raise PythonConfigError(
                'Sibling base configs referenced by {:s} override the same '
                'parameter(s): {:s}'.format(
                    config_path, _format_parameter_paths(conflicts)
                )
            )
        inherited_paths.update(base_paths)
        overlays.extend(base_overlays)
    overlays.append((config_path, module_values))
    return overlays


def _merge_mapping(override, base):
    """Deep-merge one mapping, honoring Pointcept's ``_delete_`` marker."""

    replace = bool(override.get(DELETE_KEY, False))
    clean_override = {
        key: value for key, value in override.items() if key != DELETE_KEY
    }
    if replace or not isinstance(base, Mapping):
        result = {}
    else:
        result = copy.deepcopy(dict(base))

    for key, value in clean_override.items():
        if isinstance(value, Mapping) and key in result:
            result[key] = _merge_mapping(value, result[key])
        elif isinstance(value, Mapping):
            result[key] = _merge_mapping(value, {})
        else:
            result[key] = copy.deepcopy(value)
    return result


def _apply_overlay(cfg, overlay, config_path):
    if not isinstance(overlay, Mapping):
        raise PythonConfigError(
            'Python config {:s} must contain section dictionaries.'.format(config_path)
        )

    for section_name, section_override in overlay.items():
        if section_name not in cfg:
            raise PythonConfigError(
                'Python config introduced unknown section: {:s}'.format(section_name)
            )
        if not isinstance(cfg[section_name], Mapping):
            raise PythonConfigError(
                'Base config section {:s} is not mapping-like.'.format(section_name)
            )
        if not isinstance(section_override, Mapping):
            raise PythonConfigError(
                'Section {:s} in {:s} must be a dictionary.'.format(
                    section_name, config_path
                )
            )
        if DELETE_KEY in section_override:
            raise PythonConfigError(
                '{:s} cannot replace the complete {:s} section; replace an individual '
                'dictionary-valued parameter instead.'.format(DELETE_KEY, section_name)
            )

        for key, value in section_override.items():
            if key not in cfg[section_name]:
                raise PythonConfigError(
                    'Python config introduced unknown parameter: {:s}.{:s}'.format(
                        section_name, str(key)
                    )
                )
            if section_name == 'exp' and key in PROVENANCE_KEYS:
                raise PythonConfigError(
                    'Python config provenance parameter exp.{:s} is managed '
                    'automatically.'.format(str(key))
                )
            current_value = cfg[section_name][key]
            if isinstance(value, Mapping):
                cfg[section_name][key] = _merge_mapping(value, current_value)
            else:
                cfg[section_name][key] = copy.deepcopy(value)


def _display_config_path(config_path, config_root):
    config_path = os.path.realpath(config_path)
    config_root = os.path.realpath(config_root)
    try:
        common_path = os.path.commonpath((config_path, config_root))
    except ValueError:
        common_path = ''
    if common_path == config_root:
        return os.path.relpath(config_path, config_root)
    return config_path


def _config_source_record(config_path, config_root):
    digest = hashlib.sha256()
    with open(config_path, 'rb') as config_file:
        for chunk in iter(lambda: config_file.read(1024 * 1024), b''):
            digest.update(chunk)
    return {
        'path': _display_config_path(config_path, config_root),
        'sha256': digest.hexdigest(),
    }


def _parse_option_value(raw_value):
    lowered = raw_value.lower()
    if lowered == 'true':
        return True
    if lowered == 'false':
        return False
    if lowered in ('none', 'null'):
        return None
    try:
        return ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        return raw_value


def parse_config_options(option_strings):
    """Parse Pointcept-style ``section.key=value`` command-line options."""

    parsed = {}
    for option in option_strings or ():
        if '=' not in option:
            raise PythonConfigError(
                'Config option must use section.key=value syntax: {:s}'.format(option)
            )
        full_key, raw_value = option.split('=', 1)
        full_key = full_key.strip()
        if len(full_key.split('.')) != 2 or any(
            not part for part in full_key.split('.')
        ):
            raise PythonConfigError(
                'Config option must target one existing section.key: {:s}'.format(
                    full_key
                )
            )
        if full_key in parsed:
            raise PythonConfigError(
                'Config option was provided more than once: {:s}'.format(full_key)
            )
        parsed[full_key] = _parse_option_value(raw_value.strip())
    return parsed


def apply_config_options(cfg, option_strings):
    """Apply strict generic CLI overrides to a deep copy of ``cfg``."""

    parsed = parse_config_options(option_strings)
    updated_cfg = _clone_config_value(cfg)
    for full_key, value in parsed.items():
        section_name, key = full_key.split('.')
        if section_name not in updated_cfg or key not in updated_cfg[section_name]:
            raise PythonConfigError(
                'Config option targets an unknown parameter: {:s}'.format(full_key)
            )
        if section_name == 'exp' and key in PROVENANCE_KEYS:
            raise PythonConfigError(
                'Config provenance parameter {:s} is managed automatically.'.format(
                    full_key
                )
            )
        updated_cfg[section_name][key] = copy.deepcopy(value)

    updated_cfg.exp.config_options = list(option_strings or ())
    return updated_cfg


def apply_python_config(cfg, config_path, config_root=CONFIG_ROOT):
    """Apply a declarative Python config chain to a deep copy of ``cfg``.

    The existing dataset ``my_config()`` object remains the implicit root base.
    Each config may add relative ``_base_`` files and override existing values
    with section dictionaries such as ``model = dict(...)``. The original
    object is never mutated when loading or validation fails.
    """

    if not isinstance(cfg, Mapping):
        raise PythonConfigError('Base config must be a mapping-like object.')

    resolved_path = resolve_python_config(config_path, config_root=config_root)
    overlays = _load_config_chain(resolved_path)
    updated_cfg = _clone_config_value(cfg)
    for overlay_path, overlay in overlays:
        _apply_overlay(updated_cfg, overlay, overlay_path)
    updated_cfg.exp.config_file = _display_config_path(resolved_path, config_root)
    updated_cfg.exp.config_sources = [
        _config_source_record(source_path, config_root)
        for source_path, _ in overlays
    ]
    return updated_cfg, resolved_path
