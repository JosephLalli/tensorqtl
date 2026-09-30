"""Interrupted writes and stale analysis caches must never appear complete."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from half_read_io import atomic_path, cache_receipt, reuse_cache


def test_interrupted_write_preserves_previous_result(tmp_path):
    output = tmp_path / 'result.tsv'
    output.write_text('complete\n')
    with pytest.raises(RuntimeError, match='interrupted'):
        with atomic_path(output) as temporary:
            temporary.write_text('partial')
            raise RuntimeError('interrupted')
    assert output.read_text() == 'complete\n'
    assert list(tmp_path.iterdir()) == [output]
    with atomic_path(output) as temporary:
        temporary.write_text('replacement\n')
    assert output.read_text() == 'replacement\n'


@pytest.mark.parametrize('changed', ['input', 'source', 'output', 'missing_output'])
def test_cache_checks_inputs_code_and_outputs(tmp_path, changed, capsys):
    source, data, output = [tmp_path / name for name in ('code.py', 'data.tsv', 'result.tsv')]
    manifest = tmp_path / 'manifest.json'
    assert not reuse_cache(manifest, [source, data], [output])
    source.write_text('original code')
    data.write_text('original input')
    output.write_text('complete result')
    manifest.write_text(json.dumps(cache_receipt([source, data], [output])))
    assert reuse_cache(manifest, [source, data], [output])
    assert 'Reusing verified cache' in capsys.readouterr().out
    if changed == 'missing_output':
        output.unlink()
    else:
        {'input': data, 'source': source, 'output': output}[changed].write_text('changed')
    with pytest.raises(ValueError, match='incomplete cache|mismatch'):
        reuse_cache(manifest, [source, data], [output])
