"""Local cabbage window audit across dedicated test campaigns, without publication."""

import json
from pathlib import Path
from tempfile import mkdtemp

from crop_pipeline.common import save_json
from crop_pipeline.preprocessing_workflow import collection_signature, paths
from crop_pipeline.sources import adaptive, pilot_sources, supplemental, test_cohort
from crop_pipeline.windows import external, seasons
from crop_pipeline.windows.workflow import WindowAnalysis


def discover(report):
    """Include known and additional saved dedicated-test policies; verify receipts."""
    policies = [test_cohort.TEST_POLICY, test_cohort.CABBAGE_SUPPLEMENT_POLICY,
                test_cohort.CARROT_POTATO_SUPPLEMENT_POLICY,
                *[supplemental.policy_for(source) for source in supplemental.SPECS]]
    for path in sorted((Path(report).parent / adaptive.COLLECTION_NAME).glob('*/policy.json')):
        policy = json.loads(path.read_text())
        if policy.get('test_source'):
            policies.append(policy)
    result, seen = [], set()
    for policy in policies:
        key = adaptive.policy_key(policy)
        if key in seen:
            continue
        seen.add(key)
        campaign = adaptive.root(report, policy)
        manifests, states = [], []
        for batch in sorted(campaign.glob('rounds/*/*')):
            if not batch.is_dir():
                continue
            state = adaptive.read_json(batch / 'state.json', {}).get('status', 'not_started')
            receipt = batch / 'completed.json'
            states.append({'batch': str(batch), 'status': state, 'has_receipt': receipt.is_file()})
            # A running/retrying batch is not a stable completed input.
            if state != 'completed' or not receipt.is_file():
                continue
            for record in json.loads(receipt.read_text())['handoffs']:
                path = Path(record['path'])
                if seasons.file_sha256(path) != record['sha256']:
                    raise ValueError(f'Registered test handoff changed: {path}')
                manifests.append(path)
        sources = pilot_sources.load_sources(list(dict.fromkeys(manifests))) if manifests else []
        result.append({'cohort': policy.get('cohort', 'jecam'), 'campaign': key,
                       'sources': sources, 'batches': states,
                       'status': 'completed_inputs_available' if sources else 'no_completed_inputs'})
    return result


def summarize(cohort_rows):
    """Count unique capped locations, with separate full and matched recipes."""
    output = []
    pools = {**cohort_rows, 'all_test_cohorts': {
        recipe: [row for data in cohort_rows.values() for row in data[recipe]]
        for recipe in ('fixed', 'aligned')}}
    for cohort, data in pools.items():
        indexed = {name: {external.pixel_key(row): row for row in data[name]}
                   for name in ('fixed', 'aligned')}
        common = indexed['fixed'].keys() & indexed['aligned'].keys()
        for recipe, indexed_rows in indexed.items():
            for population in ('full', 'matched'):
                selected = indexed_rows if population == 'full' else {
                    key: indexed_rows[key] for key in common}
                rows = external.capped(selected)
                parcels = len({r['partition_group'] for r in rows})
                output.append({'test_cohort': cohort, 'window': recipe, 'population': population,
                               'usable_parcels': parcels, 'usable_pixels': len(rows),
                               'parcel_shortfall': max(0, 20 - parcels),
                               'pixel_shortfall': max(0, 400 - len(rows)),
                               'target_met': parcels >= 20 and len(rows) >= 400})
    return output


def run(repository):
    """Write a new diagnostic run; never change primary preprocessing pointers."""
    _, report, output = paths(repository)
    before = collection_signature(report)
    cohorts = discover(report)
    # Preserve the standard requirement that development batches finish first.
    development = test_cohort.completed_sources(report)
    occupied = list(test_cohort.source_footprints(development))
    output = output.parent / 'cabbage_test_checks'
    output.mkdir(parents=True, exist_ok=True)
    run_folder = Path(mkdtemp(prefix='check_', dir=output))
    save_json(run_folder / 'collection_inventory.json', cohorts)
    all_rows, parcel_rows = {}, []
    for cohort in cohorts:
        name = cohort['campaign']
        all_rows[name] = {'fixed': [], 'aligned': []}
        if not cohort['sources']:
            continue
        footprints = [f for f in test_cohort.source_footprints(cohort['sources'])
                      if f['properties']['crop'] == 'cabbage']
        eligible, excluded = test_cohort.exclude_overlap(footprints, occupied)
        # First registered cohort wins: JECAM, then supplements. Prevent double counting.
        occupied.extend(eligible)
        folder = run_folder / name
        folder.mkdir()
        save_json(folder / 'overlap_exclusions.json', {'excluded_parcels': sorted(excluded)})
        if not footprints:
            continue
        analysis = WindowAnalysis(cohort['sources'], [], folder, report, adaptive.COLLECTION_NAME)
        analysis.audit()
        regions = [{**r, 'properties': {**r['properties'], 'partition': 'test'}}
                   for r in analysis.regions if r['properties']['crop'] == 'cabbage'
                   and r['properties']['partition_group'] not in excluded]
        ids = {r['properties']['fragment_id'] for r in regions}
        quality = folder / 'quality.json'
        save_json(quality, {
            'regions': regions, 'sources': cohort['sources'],
            'quality_rows': [r for r in analysis.quality_rows if r['region_id'] in ids],
            'composite_rows': [r for r in analysis.composite_rows if r['region_id'] in ids],
            'pixel_paths': {key: str(value) for key, value in analysis.pixel_rows.paths.items() if key in ids},
        })
        if regions:
            windows = external.select(quality, report)
            external.export_features(windows)
            data = json.loads(windows.read_text())
            all_rows[name] = {recipe: data[recipe] for recipe in ('fixed', 'aligned')}
        for footprint in footprints:
            group = footprint['properties']['partition_group']
            counts = {recipe: len(external.capped({external.pixel_key(r): r
                      for r in all_rows[name][recipe] if r['partition_group'] == group}))
                      for recipe in ('fixed', 'aligned')}
            parcel_rows.append({'campaign': name, 'test_cohort': cohort['cohort'],
                                'parcel_group': group, 'overlap_excluded': group in excluded,
                                'fixed_pixels': counts['fixed'], 'aligned_pixels': counts['aligned']})
    if before != collection_signature(report):
        raise RuntimeError('Collection changed during the check; rerun. This partial attempt is preserved.')
    counts = summarize(all_rows)
    labels = {c['campaign']: c['cohort'] for c in cohorts}
    for row in counts:
        row['campaign'] = row['test_cohort']
        row['test_cohort'] = labels.get(row['test_cohort'], row['test_cohort'])
    seasons.write_records(run_folder / 'window_counts.csv', counts)
    seasons.write_records(run_folder / 'parcel_outcomes.csv', parcel_rows)
    save_json(run_folder / 'check_receipt.json', {
        'status': 'completed', 'campaign_metadata': before,
        'scope': 'Completed local cabbage test imagery only; pending batches are unassessed',
        'fixed_start_month': 5, 'minimum_valid_fraction': .8, 'pixels_per_parcel': 20,
        'files': {str(p.relative_to(run_folder)): seasons.file_sha256(p)
                  for p in sorted(run_folder.rglob('*')) if p.is_file()},
    })
    return run_folder
