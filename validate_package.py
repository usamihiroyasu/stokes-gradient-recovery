"""Check the distributed package without running scientific calculations."""
from pathlib import Path
import ast
import hashlib
import json

root = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def verify(manifest):
    count = 0
    for line in (root / manifest).read_text().splitlines():
        if not line.strip():
            continue
        expected, name = line.split('  ', 1)
        p = root / name
        assert p.is_file() and not p.is_symlink(), name
        assert digest(p) == expected, 'Hash mismatch: ' + name
        count += 1
    return count

if __name__ == '__main__':
    files = verify('MANIFEST.sha256')
    frozen = verify('FROZEN_FILES.sha256')
    syntax = 0
    for p in root.rglob('*.py'):
        ast.parse(p.read_text(), filename=str(p.relative_to(root)))
        syntax += 1
    print(json.dumps({'status': 'PASS', 'manifest_files': files,
                      'unchanged_scientific_files': frozen,
                      'python_sources_parsed': syntax,
                      'scientific_calculations_executed': False}, indent=2))
