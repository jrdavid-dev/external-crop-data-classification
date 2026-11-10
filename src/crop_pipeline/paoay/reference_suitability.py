"""Separate, editable reference context; never infer dated crop labels from notes."""

import csv
from pathlib import Path

from crop_pipeline.paoay import references

AREA_TYPES = {'unreviewed', 'single_crop', 'rotation_or_mixed', 'seedlings_or_flowers', 'non_target_crop'}
TEMPORAL_STATES = {'unreviewed', 'dated_evidence_available', 'calendar_only', 'timing_unknown'}
COLUMNS = ['reference_id', 'name', 'area_type', 'temporal_evidence', 'evidence_source', 'review_notes']


def prepare_review(reference_handoff):
    """Create a checklist once, then validate and display preserved manual edits."""
    bundle = references.load_reference_bundle(reference_handoff)
    records = [r for r in bundle['records'] if r['reference_id'] != bundle['manifest']['roi_id']]
    path = bundle['folder'].parent.parent/'reviews'/(bundle['manifest']['bundle_id']+'_suitability.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows({'reference_id': r['reference_id'], 'name': r['name'],
                              'area_type': 'unreviewed', 'temporal_evidence': 'unreviewed',
                              'evidence_source': '', 'review_notes': ''} for r in records)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != COLUMNS:
            raise ValueError('Suitability checklist columns changed; existing edits preserved')
        rows = list(reader)
    expected = {r['reference_id']: r['name'] for r in records}
    if len(rows) != len(expected) or {r['reference_id'] for r in rows} != set(expected):
        raise ValueError('Suitability checklist must contain each reference area once; edits preserved')
    for row in rows:
        if (row['name'] != expected[row['reference_id']] or row['area_type'] not in AREA_TYPES
                or row['temporal_evidence'] not in TEMPORAL_STATES):
            raise ValueError('Unknown suitability category or changed reference name; edits preserved')
    return path, rows
