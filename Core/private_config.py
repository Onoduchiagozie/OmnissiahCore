"""Load optional machine settings without storing credentials in the repository."""
import copy
import json
import os
from pathlib import Path


def merge(base, overrides):
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def private_path():
    selected = os.environ.get('OMNISSIAH_PRIVATE_CONFIG')
    return Path(selected if selected else Path.home() / '.config' / 'omnissiahcore' / 'settings.json').expanduser().resolve()


def load_settings(defaults_path):
    raw = json.loads(Path(defaults_path).read_text(encoding='utf-8-sig'))
    location = private_path()
    if location.exists():
        private = json.loads(location.read_text(encoding='utf-8-sig'))
        for name, value in private.get('environment', {}).items():
            if not isinstance(name, str) or not isinstance(value, str):
                raise ValueError('Private environment entries must be strings.')
            os.environ.setdefault(name, value)
        raw = merge(raw, {k: v for k, v in private.items() if k != 'environment'})
    elif os.environ.get('OMNISSIAH_PRIVATE_CONFIG'):
        raise FileNotFoundError('The explicitly selected private settings file is missing.')
    data_root = os.environ.get('OMNISSIAH_DATA_DIR') or raw.get('data_root')
    if data_root:
        data_root = Path(os.path.expandvars(data_root)).expanduser()
        if not data_root.is_absolute():
            data_root = location.parent / data_root
        raw['data_root'] = str(data_root.resolve())
    return raw
