"""Explicitly partial Sen4AgriNet reference discovery for bounded training batches."""
import hashlib
import json
import sqlite3
from collections import defaultdict, deque
from pathlib import Path

from crop_pipeline.common import save_json
from crop_pipeline.sources import sen4agrinet as references
from crop_pipeline.windows.seasons import file_sha256


def ordered_patches(records, seed=42):
    """Seeded round robin across observation-year/tile strata; independent of cache state."""
    rank = lambda value: hashlib.sha256(f'{seed}|{value}'.encode()).hexdigest()
    strata = defaultdict(list)
    for record in records:
        parts = Path(record['filename']).parts
        strata[(str(record.get('year', parts[1])), str(record.get('tile', parts[2])))].append(record)
    queues = deque(deque(sorted(strata[key], key=lambda r: rank(r['filename'])))
                   for key in sorted(strata, key=lambda key: rank(str(key))))
    while queues:
        queue = queues.popleft()
        yield queue.popleft()
        if queue:
            queues.append(queue)


def eligible_counts(database):
    # Match adaptive sampling's physical identity across repeated observation years.
    physical = defaultdict(set)
    for (serialized,) in database.execute(
            'SELECT feature FROM fragments WHERE parcel_group NOT IN (SELECT parcel_group FROM excluded)'):
        properties = json.loads(serialized)['properties']
        if properties.get('source_split', 'train') != 'train':
            continue
        key = (str(properties.get('country', '')), str(properties['parcel_id']))
        physical[key].add(properties['crop'])
    return {crop: sum(crops == {crop} for crops in physical.values()) for crop in references.CROP_CODES}


def prepare(manifest, reference_root, existing_root, parcels_per_crop=250, seed=42,
            download_enabled=True, scan_enabled=True, target_crops=None,
            max_patches=None, cached_only=False):
    """Stop at the per-crop physical-parcel budget, or explicitly report catalogue exhaustion.

    Whole patches are checked before counting. Unseen patches are not claimed to have
    been checked for conflicting labels; this is a training subset, not a full audit.
    """
    if type(parcels_per_crop) is not int or parcels_per_crop <= 0:
        raise ValueError('parcels_per_crop must be a positive integer')
    target_crops = tuple(sorted(references.CROP_CODES if target_crops is None else target_crops))
    if not target_crops or set(target_crops) - references.CROP_CODES.keys():
        raise ValueError('Provide at least one known target crop')
    if max_patches is not None and (type(max_patches) is not int or max_patches <= 0):
        raise ValueError('max_patches must be a positive integer or None')
    checkpoint_folder = references.run_folder(reference_root, manifest)
    configuration = {'version': 1, 'revision': manifest['revision'], 'seed': seed,
                     'parcels_per_crop': parcels_per_crop, 'target_crops': list(target_crops),
                     'max_patches': max_patches, 'cached_only': cached_only,
                     'manifest_sha256': hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                     'helper_sha256': file_sha256(Path(__file__))}
    key = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:16]
    folder = checkpoint_folder / 'bounded' / key
    completed_path = folder / 'completed_inventory.json'
    if completed_path.exists():
        audit = json.loads(completed_path.read_text())
        if audit.get('configuration') != configuration or audit.get('status') != 'complete':
            raise ValueError('Bounded reference inventory configuration changed')
        for entry in audit['inventories']:
            if file_sha256(Path(entry['path'])) != entry['sha256']:
                raise ValueError('Bounded reference inventory changed; rebuild its inventory')
        print(f"Reusing bounded reference inventory: {audit['files_scanned']} patches.")
        return audit
    # A verified full inventory is already enough; never transfer it again.
    if (checkpoint_folder / 'completed_inventory.json').exists():
        return references.load_completed_inventory(checkpoint_folder, manifest)
    if not scan_enabled:
        raise FileNotFoundError('No complete dataset-wide reference discovery or bounded reference inventory; enable rebuild_field_inventory')
    folder.mkdir(parents=True, exist_ok=True)
    transfer = {'network_bytes': 0, 'patches_reused_from_checkpoints': 0, 'patches_processed': 0}
    scanned = []
    with sqlite3.connect(folder / 'consolidation.sqlite') as database:
        database.execute('DROP TABLE IF EXISTS fragments')
        database.execute('CREATE TABLE fragments (parcel_group TEXT, crop TEXT, year INTEGER, mixed INTEGER, feature TEXT)')
        database.execute('CREATE INDEX group_index ON fragments(parcel_group)')
        database.execute('CREATE TEMP VIEW excluded AS SELECT parcel_group FROM fragments GROUP BY parcel_group HAVING COUNT(DISTINCT crop)>1 OR MAX(mixed)=1')
        counts = {crop: 0 for crop in references.CROP_CODES}
        for record in ordered_patches(manifest['files'], seed):
            if max_patches is not None and len(scanned) >= max_patches:
                break
            # Cache-only mode must never call the remote retrieval path.
            if cached_only:
                path = references.checkpoint_path(checkpoint_folder, record)
                if not path.exists():
                    continue
                try:
                    result = json.loads(path.read_text())
                except (ValueError, OSError):
                    continue
                if not references.valid_checkpoint(result, record):
                    continue
                result = {**result, 'reused_checkpoint': True, 'network_bytes_this_run': 0}
            else:
                result = references.process_patch(record, manifest, checkpoint_folder, existing_root,
                                                   download_enabled, False, reference_only=True)
            references.index_checkpoint(database, checkpoint_folder, record)
            scanned.append(record)
            transfer['network_bytes'] += result.get('network_bytes_this_run', 0)
            transfer['patches_reused_from_checkpoints'] += int(result.get('reused_checkpoint', False))
            transfer['patches_processed'] += 1
            counts = eligible_counts(database)
            print(f"Reference subset: {len(scanned)}/{len(manifest['files'])} patches checked; "
                  + ', '.join(f'{crop} {counts[crop]}/{parcels_per_crop}' for crop in target_crops), flush=True)
            save_json(folder / 'scan_progress.json', {'configuration': configuration, 'files_scanned': len(scanned),
                      'eligible_physical_parcels': counts, 'transfer': transfer})
            if all(counts[crop] >= parcels_per_crop for crop in target_crops):
                break
        if not scanned:
            raise ValueError('No verified reference patches available. Enable scanning or finish at least one reference patch first.')
        subset = {**manifest, 'files': scanned, 'scope': 'bounded_training_reference_subset'}
        audit = references.write_year_inventories(database, folder, subset)
    audit.update(configuration=configuration, catalogue_files=len(manifest['files']),
                 scanned_files=[record['filename'] for record in scanned],
                 eligible_physical_parcels=counts,
                 target_parcels_per_crop=parcels_per_crop,
                 stop_reason=('parcel_targets_reached' if all(counts[crop] >= parcels_per_crop for crop in target_crops)
                              else 'patch_cap_reached' if max_patches is not None and len(scanned) >= max_patches
                              else 'cached_checkpoints_exhausted' if cached_only and len(scanned) < len(manifest['files'])
                              else 'catalogue_exhausted'),
                 target_crops=list(target_crops), max_patches=max_patches, cached_only=cached_only,
                 deficits={crop: max(0, parcels_per_crop-count) for crop, count in counts.items() if crop in target_crops},
                 coverage_note='Only scanned patches checked for label ambiguity; not a full-release audit.')
    save_json(folder / 'last_scan_transfer.json', transfer)
    save_json(completed_path, audit)
    return audit
