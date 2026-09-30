"""Paths and atomic artifacts shared by the half-read analysis stages."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile

REPO = Path(__file__).resolve().parents[1]
DEPLOY = Path(os.environ.get('HALF_READ_DEPLOY_ROOT', REPO / 'data/half_read')).resolve()
RESULTS = Path(os.environ.get('HALF_READ_OUTPUT_ROOT', DEPLOY)).resolve()


@contextmanager
def atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Retain the extension for writers that infer the output format from it.
    fd, name = tempfile.mkstemp(prefix=f'.{path.stem}.', suffix=f'.tmp{path.suffix}', dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def cache_receipt(inputs, outputs):
    return {'input_sha256': {str(Path(p).resolve()): digest(p) for p in inputs},
            'output_sha256': {Path(p).name: digest(p) for p in outputs}}


def reuse_cache(manifest_path, inputs, outputs):
    manifest_path = Path(manifest_path)
    paths = [manifest_path, *map(Path, outputs)]
    if not any(p.exists() for p in paths):
        return False
    if not all(p.is_file() for p in paths):
        raise ValueError(f'{manifest_path}: incomplete cache; use a fresh output directory')
    saved = json.loads(manifest_path.read_text())
    expected = cache_receipt(inputs, outputs)
    for key, hashes in expected.items():
        if key not in saved or saved[key] != hashes:
            raise ValueError(f'{manifest_path}: {key} mismatch; use a fresh output directory')
    print(f'Reusing verified cache: {manifest_path} ({len(outputs)} outputs)', flush=True)
    return True
