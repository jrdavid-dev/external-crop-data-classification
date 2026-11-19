"""Consolidate legacy storage without rewriting immutable scientific receipts."""
import argparse
import os
from pathlib import Path
from datetime import datetime, timezone

from crop_pipeline.common import save_json
from crop_pipeline.windows.seasons import file_sha256

LEGACY_ROOTS = ('sen4agrinet_work', 'notebooks/data_collection/sen4agrinet_work')
DESTINATION = 'data/pipeline_work'


def inventory(roots):
    """Preflight every entry before the first move; reject ambiguous destinations."""
    seen, records = {}, []
    for root in roots:
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f'Expected an original storage directory: {root}')
        for path in sorted(root.rglob('*')):
            if path.is_symlink() or not (path.is_file() or path.is_dir()):
                raise ValueError(f'Unsupported storage entry: {path}')
            relative = path.relative_to(root).as_posix()
            if relative in seen and (path.is_file() or seen[relative] == 'file'):
                raise ValueError(f'Storage collision: {relative}')
            seen[relative] = 'file' if path.is_file() else 'directory'
            if path.is_file():
                records.append({'original': str(path), 'relative': relative,
                                'bytes': path.stat().st_size, 'sha256': file_sha256(path)})
    return records


def merge(source, destination):
    """Rename intact subtrees on the same filesystem; never overwrite a file."""
    if not destination.exists():
        source.rename(destination)
        return
    if not source.is_dir() or not destination.is_dir():
        raise FileExistsError(destination)
    for child in list(source.iterdir()):
        merge(child, destination / child.name)
    source.rmdir()  # Only an empty directory, after its children have moved.


def migrate(repository, destination=DESTINATION):
    repository = Path(repository).resolve()
    target = repository / destination
    roots = [repository / name for name in LEGACY_ROOTS]
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if any(root.stat().st_dev != target.parent.stat().st_dev for root in roots):
        raise ValueError('Migration requires same-filesystem renames')
    print('Hashing original files before migration...', flush=True)
    records = inventory(roots)
    run = repository / 'data/storage_migrations' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir(parents=True)
    manifest = run / 'manifest.json'
    data = {'destination': str(target), 'legacy_roots': [str(r) for r in roots],
            'files': records, 'status': 'planned'}
    save_json(manifest, data)
    print(f'Preflight complete: {len(records)} files, no collisions. Moving...', flush=True)
    roots[0].rename(target)
    roots[0].symlink_to(os.path.relpath(target, roots[0].parent), target_is_directory=True)
    merge(roots[1], target)
    roots[1].symlink_to(os.path.relpath(target, roots[1].parent), target_is_directory=True)
    print('Verifying all relocated bytes and compatibility paths...', flush=True)
    for record in records:
        path = target / record['relative']
        if path.stat().st_size != record['bytes'] or file_sha256(path) != record['sha256']:
            raise ValueError(f'Migrated file checksum mismatch: {path}')
        if not Path(record['original']).samefile(path):
            raise ValueError(f'Historical path does not resolve: {record["original"]}')
    data['status'] = 'verified'
    save_json(manifest, data)
    print(f'Migration verified. Recovery inventory: {manifest}', flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path.cwd())
    args = parser.parse_args()
    migrate(args.repository)
