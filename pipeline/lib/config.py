"""YAML configuration loading, environment expansion and stage hashing."""
import copy
import hashlib
import json
import os
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

from pipeline.lib import REPO

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
REQUIRED = ("run_name", "seed", "output_root", "dataset")


def _expand(value):
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, str):
        def replace(match):
            name = match.group(1)
            if name not in os.environ:
                raise ValueError(f"Config references unset environment variable ${{{name}}}")
            return os.environ[name]
        return _ENV.sub(replace, value)
    return value


def _merge(base, override):
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_raw(path):
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if "extends" in raw:
        raw = _merge(_load_raw(Path(path).parent / raw.pop("extends")), raw)
    return raw


def load_config(path, overrides=None):
    """Load a YAML config; ``extends: other.yaml`` merges a base config first."""
    path = Path(path).resolve()
    cfg = _expand(_merge(_load_raw(path), overrides or {}))
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")
    root = Path(cfg["output_root"]).expanduser().resolve()
    if cfg["dataset"]["name"] != "synthetic" and (root == REPO or REPO in root.parents):
        # Real participant-derived outputs must stay in approved private storage.
        raise ValueError("output_root must be outside the Git repository for real data")
    cfg["output_root"] = str(root)
    cfg["config_path"] = str(path)
    return cfg


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def code_digest(paths):
    """Hash source files so a code change invalidates dependent caches."""
    h = hashlib.sha256()
    files = []
    for p in paths:
        p = Path(p)
        files.extend(sorted(p.rglob("*.py")) if p.is_dir() else [p])
    for f in sorted(set(files)):
        h.update(str(f.relative_to(REPO)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def package_versions(names=("numpy", "pandas", "scipy", "scikit-learn", "praat-parselmouth")):
    out = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "unavailable"
    return out


def stage_key(cfg, sections, upstream=None, code=(), extra=None):
    """Cache key: selected config sections + upstream stage keys + code + versions."""
    payload = {
        "sections": {s: cfg.get(s) for s in sections},
        "upstream": upstream or {},
        "code": code_digest(code) if code else None,
        "versions": package_versions(),
        "extra": extra,
    }
    return digest(payload), payload
