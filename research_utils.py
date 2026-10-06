"""Configuration and provenance shared by the research CLI."""
import hashlib
import json
import platform
import subprocess
from pathlib import Path
import numpy as np
import yaml


def read_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError('Config must be a mapping')
    return config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))


def provenance(config, paths=()):
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    dirty = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
    return {'config': config, 'git_commit': git.stdout.strip(),
            'git_dirty': bool(dirty.stdout.strip()), 'python': platform.python_version(),
            'numpy': np.__version__,
            'code_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(Path('.').rglob('*.py'))
                            if 'artifacts' not in p.parts and '.venv' not in p.parts},
            'input_sha256': {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}}
