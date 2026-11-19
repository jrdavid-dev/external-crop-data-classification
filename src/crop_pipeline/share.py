"""Export/import completed imagery for offline teammates: python -m crop_pipeline.share."""
import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from crop_pipeline.common import save_json
from crop_pipeline.preprocessing_workflow import paths
from crop_pipeline.sources import adaptive, jecam_transfer as transfer, supplemental, test_cohort
from crop_pipeline.windows.seasons import file_sha256


def cohorts():
    return {
        **{s: (s, adaptive.DEFAULT_POLICY) for s in sorted(test_cohort.DEVELOPMENT)},
        'jecam': (test_cohort.TEST, test_cohort.TEST_POLICY),
        'carrot_potato_supplement': (test_cohort.TEST, test_cohort.CARROT_POTATO_SUPPLEMENT_POLICY),
        'cabbage_supplement': (test_cohort.TEST, test_cohort.CABBAGE_SUPPLEMENT_POLICY),
        **{s: (s, supplemental.policy_for(s)) for s in supplemental.SPECS},
    }


def export(repository, destination):
    """Publish a new, self-contained directory; existing exports are never overwritten."""
    _, report, _ = paths(repository)
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    inventory, entries = [], []
    with TemporaryDirectory(dir=destination.parent) as temp:
        staging = Path(temp) / 'package'
        staging.mkdir()
        for cohort, (source, policy) in cohorts().items():
            rounds = transfer.campaign_rounds(report, source, policy)
            states = [transfer.read(p).get('status', 'unknown') for p in sorted(rounds.glob('*/*/state.json'))
                      if p.parent.joinpath('selection.json').is_file()
                      and transfer.read(p.parent / 'selection.json').get('source') == source]
            try:
                bundles = transfer.export_completed(report, staging / cohort, source, policy=policy)
            except FileNotFoundError:
                bundles = []
            inventory.append({'cohort': cohort, 'completed_bundles': len(bundles),
                              'unassessed_batches': sum(s not in {'completed', 'disabled', 'skipped'} for s in states)})
            for folder in bundles:
                bundle = transfer.verify_bundle(folder, source)
                entries.append({'cohort': cohort, 'path': folder.relative_to(staging).as_posix(),
                                'bundle_sha256': file_sha256(folder / 'bundle.json'),
                                'parcel_groups': len(bundle['selection']['groups'])})
        if not entries:
            raise FileNotFoundError('No completed local imagery batches available')
        save_json(staging / 'package.json', {'version': 1, 'bundles': entries, 'inventory': inventory})
        guide = Path(__file__).resolve().parents[2] / 'docs/sharing_imagery.md'
        if guide.exists():
            shutil.copyfile(guide, staging / 'START_HERE.md')
        research = Path(__file__).resolve().parents[2] / 'docs/research_basis.md'
        if research.exists():
            shutil.copyfile(research, staging / 'RESEARCH.md')
        staging.rename(destination)
    return destination


def verify(folder):
    folder = Path(folder).expanduser().resolve()
    data = transfer.read(folder / 'package.json')
    if data['version'] != 1 or not data['bundles']:
        raise ValueError('Unsupported or empty imagery package')
    for entry in data['bundles']:
        source, policy = cohorts()[entry['cohort']]
        bundle_folder = transfer.checked_path(folder, entry['path'])
        if file_sha256(bundle_folder / 'bundle.json') != entry['bundle_sha256']:
            raise ValueError('Bundle metadata checksum mismatch')
        bundle = transfer.verify_bundle(bundle_folder, source)
        if bundle.get('policy_key') != adaptive.policy_key(policy):
            raise ValueError('Bundle cohort policy mismatch')
    return folder, data


def import_package(repository, folder):
    """Verify the whole payload before registering any batch; preserve local selections."""
    folder, data = verify(folder)
    _, report, _ = paths(repository)
    # Preflight in a disposable copy of receiver selection/receipt metadata so a
    # later cohort conflict cannot leave earlier cohorts partially imported.
    campaign_root = report.parent / adaptive.COLLECTION_NAME
    with TemporaryDirectory() as temp:
        trial = Path(temp) / 'inventory'
        if campaign_root.exists():
            for p in campaign_root.glob('*/rounds/*/*/*.json'):
                if p.name in {'selection.json', 'completed.json'}:
                    target = trial.parent / adaptive.COLLECTION_NAME / p.relative_to(campaign_root)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(p, target)
        # Import helper stages payloads too: this also checks all manifest contracts.
        for entry in data['bundles']:
            source, policy = cohorts()[entry['cohort']]
            transfer.import_completed(trial, folder / entry['path'], source, policy=policy)
    imported = []
    for entry in data['bundles']:
        source, policy = cohorts()[entry['cohort']]
        imported.extend(transfer.import_completed(report, folder / entry['path'], source, policy=policy))
    return imported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['export', 'verify', 'import'])
    parser.add_argument('folder', type=Path, help='Portable package directory (extract ZIP first)')
    parser.add_argument('--repository', type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.action == 'export':
        result = export(args.repository, args.folder)
    elif args.action == 'verify':
        result = verify(args.folder)[1]['inventory']
    else:
        result = import_package(args.repository, args.folder)
    print(result)


if __name__ == '__main__':
    main()
