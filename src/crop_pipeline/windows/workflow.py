"""Saved-imagery window preparation; scientific rules shared with the original notebook.

No acquisition or model fitting occurs here. Methods retain complete quality and
cycle audits; notebook layout is independent of collection/archive identity.
"""

from pathlib import Path
import json
from collections import Counter
from crop_pipeline.sources import annual, adaptive, pilot_sources
from crop_pipeline.imagery import quality as crop_workflow
from crop_pipeline.windows import annual_archive, seasons as season_workflow
from crop_pipeline.windows import timing_quality, cycle_windows
from crop_pipeline.sources import sampling as pilot_sampling


class WindowAnalysis:
    def __init__(self, sources, source_manifest_paths, analysis_folder, report_folder,
                 source_run_name, minimum_valid_fraction=0.8, run_development_experiments=False):
        if isinstance(minimum_valid_fraction, bool) or not 0.8 <= float(minimum_valid_fraction) <= 1:
            raise ValueError("Imagery coverage threshold must be between 0.8 and 1")
        self.sources = sources
        self.source_manifest_paths = source_manifest_paths
        self.analysis_folder = Path(analysis_folder)
        self.report_folder = Path(report_folder)
        self.source_run_name = source_run_name
        self.minimum_valid_fraction = minimum_valid_fraction
        self.run_development_experiments = run_development_experiments


    def audit(self):
        if self.source_run_name in {annual.COLLECTION_NAME, adaptive.COLLECTION_NAME}:
            self.regions, self.quality_rows, self.pixel_rows = annual_archive.audit_sources(self.sources, self.analysis_folder, self.minimum_valid_fraction)
        else:
            self.regions, self.quality_rows, self.pixel_rows = pilot_sources.audit_sources(self.sources, self.analysis_folder, self.minimum_valid_fraction)
            crop_workflow.write_table(self.analysis_folder / 'quality/selected_pixel_bands.csv', self.pixel_rows, ['region_id', 'crop', 'partition_group', 'name', 'x', 'y', 'source_crs', 'valid', 'clear_count'] + crop_workflow.STUDY_BANDS)
        crop_workflow.write_table(self.analysis_folder / 'quality/composite_quality.csv', self.quality_rows, ['region_id', 'crop', 'partition_group', 'name', 'start', 'end', 'selected_pixels', 'valid_pixels', 'valid_fraction', 'status', 'image_path', 'error'])
        print('Regions audited:', len(self.regions))
        print('Composite outcomes:', dict(Counter((row['status'] for row in self.quality_rows))))
        self.timeline_folder = self.analysis_folder / 'timelines'
        self.composite_rows = crop_workflow.composite_timelines(self.quality_rows, self.pixel_rows, self.timeline_folder)
        self.example_regions = {next((row['region_id'] for row in self.composite_rows if row['crop'] == crop)) for crop in sorted({row['crop'] for row in self.composite_rows})}
        self.example_composite_rows = [row for row in self.composite_rows if row['region_id'] in self.example_regions]
        crop_workflow.plot_timelines(self.example_composite_rows, self.timeline_folder / 'composite_plots', title='Semi-monthly composite summaries (bin midpoints)')
        print('Composite timeline rows:', len(self.composite_rows))
        self.plot_files = sorted((self.timeline_folder / 'composite_plots').glob('*.png'))


    def select(self):
        self.detector_settings = dict(crop_workflow.DEFAULT_DETECTOR)
        self.detector_settings['relative_threshold'] = 0.2
        self.observed_path = self.timeline_folder / 'observed_timelines.json'
        self.observed_rows = json.loads(self.observed_path.read_text(encoding='utf-8')) if self.observed_path.exists() else []
        self.observed_regions = {row['region_id'] for row in self.observed_rows}
        self.detector_rows = self.observed_rows + [row for row in self.composite_rows if row['region_id'] not in self.observed_regions]
        self.greenup_results = crop_workflow.detect_greenup(self.detector_rows, self.detector_settings)
        self.greenup_folder = self.analysis_folder / 'greenup'
        crop_workflow.save_json(self.greenup_folder / 'candidate_greenup.json', self.greenup_results)
        crop_workflow.save_json(self.greenup_folder / 'detector_settings.json', {'parameters': self.detector_settings, 'validated': False, 'label_relevance_verified': False})
        self.columns = ['region_id', 'partition_group', 'status', 'reason', 'source_kind', 'onset_interval_start', 'onset_interval_end', 'peak_date', 'baseline', 'amplitude', 'ndvi_threshold', 'gap_days', 'confirmations', 'month_boundary_uncertain', 'label_relevance', 'validated']
        crop_workflow.write_table(self.greenup_folder / 'candidate_greenup.csv', self.greenup_results, self.columns)
        print('Detector outcomes:', dict(Counter((row['status'] for row in self.greenup_results))))
        print('Results:', self.greenup_folder)
        self.sensitivity = {}
        self.sensitivity_rows = self.detector_rows
        if self.source_run_name == adaptive.COLLECTION_NAME:
            self.locked = adaptive.assignments(self.report_folder)
            self.sensitivity_rows = [row for row in self.detector_rows if self.locked.get(row.get('partition_group')) != 'test']
        for fraction in [0.15, 0.2, 0.25]:
            self.results = crop_workflow.detect_greenup(self.sensitivity_rows, {**self.detector_settings, 'relative_threshold': fraction})
            self.sensitivity[str(fraction)] = self.results
        crop_workflow.save_json(self.greenup_folder / 'threshold_sensitivity.json', self.sensitivity)
        self.plot_region_ids = self.example_regions
        crop_workflow.plot_greenup([row for row in self.detector_rows if row['region_id'] in self.plot_region_ids], [row for row in self.greenup_results if row['region_id'] in self.plot_region_ids], self.greenup_folder / 'plots')
        self.season_folder = self.analysis_folder / 'seasons'
        self.experiment_artifacts = []
        self.review_folder = self.season_folder / 'review'
        season_workflow.prepare_review_files(self.greenup_results, self.regions, self.detector_settings, self.review_folder)
        self.configuration_path = self.review_folder / 'frozen_configuration.json'
        self.reviews_path = self.review_folder / 'season_reviews.csv'
        self.registry_path = self.review_folder / 'field_registry.csv'
        self.configuration = json.loads((self.configuration_path if self.configuration_path.exists() else self.review_folder / 'frozen_configuration.template.json').read_text(encoding='utf-8'))
        threshold = float(self.configuration['minimum_valid_fraction'])
        if not 0.8 <= threshold <= 1 or threshold != self.minimum_valid_fraction:
            raise ValueError('Window coverage threshold differs from the imagery audit or is below 0.8; reconcile the review configuration before exporting')
        if self.configuration['detector_parameters'] != self.detector_settings:
            raise ValueError('The review configuration uses different detector settings. Review or rerun detection before selecting windows.')
        self.configuration['cycle_selection_policy'] = 'earliest_complete'
        self.configuration['usable_pixels_per_crop'] = None if self.source_run_name in {annual.COLLECTION_NAME, adaptive.COLLECTION_NAME} else pilot_sampling.USABLE_PIXELS_PER_CROP
        self.reviews = season_workflow.read_table(self.reviews_path) if self.reviews_path.exists() else []
        self.configuration['automatic_timing_policy'] = {**timing_quality.DEFAULT_TIMING_POLICY, **self.configuration.get('automatic_timing_policy', {})}
        self.accepted_seasons, self.estimated_proposals, self.season_audit = season_workflow.select_season_windows(self.greenup_results, self.reviews, self.regions, self.configuration)
        self.estimated_windows, self.automatic_timing_rows, self.season_audit = season_workflow.qualify_estimated_windows(self.estimated_proposals, self.season_audit, self.detector_rows, self.configuration)
        self.estimated_windows = cycle_windows.flag_four_month_context(self.estimated_windows, self.greenup_results)
        season_workflow.write_records(self.season_folder / 'automatic_timing_audit.csv', self.automatic_timing_rows)
        season_workflow.save_json(self.season_folder / 'automatic_timing_policy.json', {'quality_policy': self.configuration['automatic_timing_policy'], 'cycle_selection_policy': self.configuration['cycle_selection_policy'], 'detector_parameters': self.detector_settings, 'independently_validated': False})
        season_workflow.write_records(self.season_folder / 'season_review_audit.csv', self.season_audit)
        print('Reference-supported field-seasons:', len(self.accepted_seasons))
        print('Initial estimated proposals:', len(self.estimated_proposals))
        print('Automatically qualified estimates:', sum((row['timing_review_status'] == 'automated_pass' for row in self.estimated_windows)))
        print('Total qualified estimated windows:', len(self.estimated_windows))
        print('Unresolved outcomes:', sum((row['window_status'] == 'unresolved' for row in self.season_audit)))
        print('Automatic timing failures:', sum((row['automatic_timing_status'] == 'failed' for row in self.automatic_timing_rows)))
        print('Review templates:', self.review_folder)
        season_workflow.write_records(self.season_folder / 'estimated_proposals.csv', self.estimated_proposals)
        season_workflow.save_json(self.season_folder / 'estimated_proposals.json', self.estimated_proposals)
        season_workflow.write_records(self.season_folder / 'accepted_seasons.csv', self.accepted_seasons)
        season_workflow.save_json(self.season_folder / 'accepted_seasons.json', self.accepted_seasons)
        season_workflow.write_records(self.season_folder / 'estimated_windows.csv', self.estimated_windows)
        season_workflow.save_json(self.season_folder / 'estimated_windows.json', self.estimated_windows)
        print('Reference-supported windows saved:', len(self.accepted_seasons))
        print('Estimated observation windows saved:', len(self.estimated_windows))
        print('Audit:', self.season_folder / 'season_review_audit.csv')
        self.feature_rows, self.feature_exclusions, self.sample_losses, self.bin_coverage = season_workflow.build_features(self.accepted_seasons, self.pixel_rows, self.quality_rows, minimum_fraction=self.configuration['minimum_valid_fraction'])
        self.estimated_feature_rows, self.estimated_exclusions, self.estimated_losses, self.estimated_bin_coverage = season_workflow.build_features(self.estimated_windows, self.pixel_rows, self.quality_rows, minimum_fraction=self.configuration['minimum_valid_fraction'])
        self.cycle_candidate_folder = self.season_folder / 'cycle_candidates'
        season_workflow.write_records(self.cycle_candidate_folder / 'complete_features.csv', self.estimated_feature_rows)
        season_workflow.write_records(self.cycle_candidate_folder / 'feature_exclusions.csv', self.estimated_exclusions)
        season_workflow.write_records(self.cycle_candidate_folder / 'sample_losses.csv', self.estimated_losses)
        season_workflow.write_records(self.cycle_candidate_folder / 'bin_coverage.csv', self.estimated_bin_coverage)
        self.estimated_windows, self.estimated_feature_rows, self.estimated_exclusions, self.estimated_losses, self.estimated_bin_coverage, self.cycle_selection_audit = cycle_windows.choose_complete_cycles(self.estimated_windows, self.estimated_feature_rows, self.estimated_exclusions, self.estimated_losses, self.estimated_bin_coverage)
        season_workflow.write_records(self.season_folder / 'cycle_selection_audit.csv', self.cycle_selection_audit)
        self.cycle_outcomes = {row['candidate_id']: row for row in self.cycle_selection_audit}
        self.season_audit = [{**row, **{key: value for key, value in self.cycle_outcomes.get(row['candidate_id'], {}).items() if key.startswith('cycle_selection_') or key == 'complete_usable_pixels'}} for row in self.season_audit]
        season_workflow.write_records(self.season_folder / 'season_review_audit.csv', self.season_audit)
        season_workflow.write_records(self.season_folder / 'estimated_windows.csv', self.estimated_windows)
        season_workflow.save_json(self.season_folder / 'estimated_windows.json', self.estimated_windows)
        print('Cycle selection:', dict(Counter((row['cycle_selection_status'] for row in self.cycle_selection_audit))))
        print('Selected estimated windows:', len(self.estimated_windows))
        self.feature_folder = self.season_folder / 'features'
        self.estimated_feature_folder = self.season_folder / 'estimated_features'
        self.reserved_groups = {region['properties']['partition_group'] for region in self.regions if region['properties'].get('source_split', 'train') != 'train'}
        self.reserved_features = [row for row in self.feature_rows + self.estimated_feature_rows if row['partition_group'] in self.reserved_groups]
        season_workflow.write_records(self.season_folder / 'reserved_upstream_features.csv', self.reserved_features)
        self.feature_rows = [row for row in self.feature_rows if row['partition_group'] not in self.reserved_groups]
        self.estimated_feature_rows = [row for row in self.estimated_feature_rows if row['partition_group'] not in self.reserved_groups]
        season_workflow.write_records(self.feature_folder / 'all_complete_features.csv', self.feature_rows)
        season_workflow.write_records(self.estimated_feature_folder / 'all_complete_features.csv', self.estimated_feature_rows)
        self.pilot_registry = season_workflow.read_table(self.registry_path) if self.registry_path.exists() else []
        if self.source_run_name == adaptive.COLLECTION_NAME:
            self.feature_rows, self.estimated_feature_rows, self.usable_selection_summary = adaptive.select_usable(self.feature_rows, self.estimated_feature_rows, self.report_folder)
        else:
            self.feature_rows, self.estimated_feature_rows, self.usable_selection_summary = pilot_sampling.trim_external_pilot(self.feature_rows, self.estimated_feature_rows, self.pilot_registry, target_per_crop=self.configuration['usable_pixels_per_crop'], seed=self.configuration['random_seed'], crops=('carrot', 'cabbage', 'potato'))
        season_workflow.write_records(self.season_folder / 'usable_pixel_selection.csv', self.usable_selection_summary)
        print('Usable selection by crop:', self.usable_selection_summary)
        season_workflow.write_records(self.feature_folder / 'complete_features.csv', self.feature_rows)
        season_workflow.write_records(self.estimated_feature_folder / 'complete_features.csv', self.estimated_feature_rows)
        print('Reference-supported complete pixel windows:', len(self.feature_rows))
        print('Estimated complete pixel windows:', len(self.estimated_feature_rows))
        print('Features per complete pixel:', len(season_workflow.feature_columns()))
        season_workflow.write_records(self.feature_folder / 'feature_exclusions.csv', self.feature_exclusions)
        season_workflow.write_records(self.feature_folder / 'sample_losses.csv', self.sample_losses)
        season_workflow.write_records(self.feature_folder / 'bin_coverage.csv', self.bin_coverage)
        season_workflow.write_records(self.estimated_feature_folder / 'feature_exclusions.csv', self.estimated_exclusions)
        season_workflow.write_records(self.estimated_feature_folder / 'sample_losses.csv', self.estimated_losses)
        season_workflow.write_records(self.estimated_feature_folder / 'bin_coverage.csv', self.estimated_bin_coverage)
        print('Reference-supported excluded pixel windows:', sum((row['excluded_pixels'] for row in self.sample_losses)))
        print('Estimated excluded pixel windows:', sum((row['excluded_pixels'] for row in self.estimated_losses)))
        self.field_registry = season_workflow.read_table(self.registry_path) if self.registry_path.exists() else []
        self.partitioned_rows, self.exploratory_rows = ([], [])
        self.reviewed_estimated_features = [row for row in self.estimated_feature_rows if season_workflow.timing_is_qualified(row)]
        if self.field_registry:
            self.partitioned_rows = season_workflow.assign_partitions(self.feature_rows, self.field_registry)
            if self.reviewed_estimated_features:
                self.exploratory_rows = season_workflow.prepare_exploratory_rows(self.partitioned_rows, self.reviewed_estimated_features, self.field_registry)
        else:
            print('Field assignment pending: complete field_registry.csv before preparing model inputs.')
        season_workflow.write_records(self.season_folder / 'partitioned_features.csv', self.partitioned_rows)
        season_workflow.write_records(self.season_folder / 'sample_counts.csv', season_workflow.sample_counts(self.partitioned_rows))
        season_workflow.write_records(self.season_folder / 'exploratory_partitioned_features.csv', self.exploratory_rows)
        print('Primary partitioned pixel windows:', len(self.partitioned_rows))
        print('Exploratory partitioned pixel windows:', len(self.exploratory_rows))


    def export_formal(self):
        self.coverage()
        if self.run_development_experiments:
            self.coverage_experiment_folder = self.season_folder / 'experiments' / 'coverage'
            self.development_regions = season_workflow.development_region_ids(self.regions, self.field_registry)
            self.development_timelines = [row for row in self.detector_rows if row['region_id'] in self.development_regions]
            self.coverage_decisions, self.coverage_summary_rows = season_workflow.summarize_coverage_thresholds(self.development_timelines, thresholds=(0.7, 0.8, 0.9))
            season_workflow.write_records(self.coverage_experiment_folder / 'decisions.csv', self.coverage_decisions)
            season_workflow.write_records(self.coverage_experiment_folder / 'summary.csv', self.coverage_summary_rows)
            print('Coverage comparison:', self.coverage_experiment_folder)
            self.experiment_artifacts.extend([self.coverage_experiment_folder / 'decisions.csv', self.coverage_experiment_folder / 'summary.csv'])
        else:
            print('Optional sensitivity export disabled; the primary timing and quality checks still run.')
        if self.run_development_experiments:
            self.window_experiment_folder = self.season_folder / 'experiments' / 'windows'
            self.development_regions = season_workflow.development_region_ids(self.regions, self.field_registry)
            self.development_seasons = [season for season in self.accepted_seasons + self.estimated_windows if season['region_id'] in self.development_regions]
            for bin_shift in (-1, 0, 1):
                self.variant_folder = self.window_experiment_folder / f'shift_{bin_shift:+d}'
                self.shifted_features, self.shifted_exclusions, self.shifted_losses, self.shifted_coverage = season_workflow.build_features(self.development_seasons, self.pixel_rows, self.quality_rows, minimum_fraction=self.configuration['minimum_valid_fraction'], bin_shift=bin_shift)
                season_workflow.write_records(self.variant_folder / 'features.csv', self.shifted_features)
                season_workflow.write_records(self.variant_folder / 'exclusions.csv', self.shifted_exclusions)
                season_workflow.write_records(self.variant_folder / 'losses.csv', self.shifted_losses)
                season_workflow.write_records(self.variant_folder / 'coverage.csv', self.shifted_coverage)
                self.experiment_artifacts.extend([self.variant_folder / filename for filename in ('features.csv', 'exclusions.csv', 'losses.csv', 'coverage.csv')])
            print('Window comparison:', self.window_experiment_folder)
        else:
            print('Optional sensitivity export disabled; the primary timing and quality checks still run.')
        self.training_input_folder = self.season_folder / 'training_input'
        self.reference_configuration = {**self.configuration, 'window_policy': 'reference_supported_only'}
        self.handoff_path = season_workflow.export_training_handoff(self.training_input_folder, self.partitioned_rows, self.reference_configuration, self.coverage_summary)
        self.exploratory_configuration = {**self.configuration, 'window_policy': 'exploratory_external_estimates'}
        self.exploratory_input_folder = self.season_folder / 'exploratory_training_input'
        self.exploratory_handoff_path = season_workflow.export_training_handoff(self.exploratory_input_folder, self.exploratory_rows, self.exploratory_configuration, self.coverage_summary)
        self.report_artifacts = [self.season_folder / 'cycle_selection_audit.csv', *sorted(self.cycle_candidate_folder.glob('*.csv')), self.season_folder / 'usable_pixel_selection.csv', self.feature_folder / 'all_complete_features.csv', self.estimated_feature_folder / 'all_complete_features.csv', self.season_folder / 'season_review_audit.csv', self.season_folder / 'accepted_seasons.csv', self.season_folder / 'accepted_seasons.json', self.season_folder / 'estimated_windows.csv', self.season_folder / 'estimated_windows.json', self.season_folder / 'estimated_proposals.csv', self.season_folder / 'estimated_proposals.json', self.season_folder / 'automatic_timing_audit.csv', self.season_folder / 'automatic_timing_policy.json', self.season_folder / 'exploratory_partitioned_features.csv', self.estimated_feature_folder / 'complete_features.csv', self.estimated_feature_folder / 'feature_exclusions.csv', self.estimated_feature_folder / 'sample_losses.csv', self.estimated_feature_folder / 'bin_coverage.csv', self.exploratory_handoff_path, self.exploratory_input_folder / 'partitioned_features.csv', self.exploratory_input_folder / 'configuration.json', self.exploratory_input_folder / 'coverage.json', self.feature_folder / 'complete_features.csv', self.feature_folder / 'feature_exclusions.csv', self.feature_folder / 'sample_losses.csv', self.feature_folder / 'bin_coverage.csv', self.season_folder / 'partitioned_features.csv', self.season_folder / 'sample_counts.csv', self.handoff_path, self.training_input_folder / 'partitioned_features.csv', self.training_input_folder / 'configuration.json', self.training_input_folder / 'coverage.json']
        self.preprocessing_manifest = season_workflow.package_run(self.season_folder / 'preprocessing_package', self.configuration, input_paths=[self.configuration_path, self.reviews_path, self.registry_path, self.greenup_folder / 'candidate_greenup.json', self.greenup_folder / 'detector_settings.json', self.analysis_folder / 'quality' / ('selected_pixel_bands.index.json' if self.source_run_name in {annual.COLLECTION_NAME, adaptive.COLLECTION_NAME} else 'selected_pixel_bands.csv'), self.analysis_folder / 'quality' / 'composite_quality.csv', self.analysis_folder / 'source_inputs.json', Path('src/crop_pipeline/sources/sampling.py'), Path('src/crop_pipeline/sources/pilot_sources.py'), Path('src/crop_pipeline/sources/francecrops.py'), Path('src/crop_pipeline/imagery/retrieval.py'), Path('notebooks/02_francecrops_preparation.ipynb'), Path('src/crop_pipeline/common.py'), Path('src/crop_pipeline/imagery/quality.py'), Path('src/crop_pipeline/windows/seasons.py'), Path('src/crop_pipeline/windows/timing_quality.py'), Path('src/crop_pipeline/windows/cycle_windows.py'), Path('src/crop_pipeline/windows/detector.py'), Path('requirements-notebooks.txt'), Path('notebooks/01_sen4agrinet_preparation.ipynb'), Path('notebooks/04_four_month_windows.ipynb')], artifact_paths=self.report_artifacts + self.experiment_artifacts, coverage=self.coverage_summary)
        print('Primary training handoff:', self.handoff_path)
        print('Exploratory training handoff:', self.exploratory_handoff_path)
        print('Preparation status:', self.coverage_summary['preparation_status'])
        print('Next: open 05_feature_engineering.ipynb to derive the three experiment designs.')


    def coverage(self):
        self.imagery_versions = {source['request_version'] for source in self.sources}
        self.imagery_resampling = {json.dumps(source['resampling'], sort_keys=True) for source in self.sources}
        self.coverage_summary = {'imagery_request_version': next(iter(self.imagery_versions)) if len(self.imagery_versions) == 1 else 'mixed', 'resampling': json.loads(next(iter(self.imagery_resampling))) if len(self.imagery_resampling) == 1 else 'mixed', 'usable_pixel_selection': self.usable_selection_summary, 'cycle_selection_policy': self.configuration.get('cycle_selection_policy', 'single_supported_cycle'), 'cycle_detector': self.detector_settings.get('cycle_detector', 'legacy_local_peaks'), 'cycle_status_counts': dict(Counter((row.get('cycle_status', row['status']) for row in self.greenup_results))), 'windows_overlapping_next_cycle': sum((bool(row.get('window_overlaps_next_cycle')) for row in self.estimated_windows)), 'cycle_selection_counts': dict(Counter((row['cycle_selection_status'] for row in self.cycle_selection_audit))), 'source_manifests': [str(path) for path in self.source_manifest_paths], 'detector_regions': len({row['region_id'] for row in self.greenup_results}), 'detector_outcomes': dict(Counter((row['status'] for row in self.greenup_results))), 'review_outcomes': dict(Counter((row['decision'] for row in self.season_audit))), 'accepted_seasons': len(self.accepted_seasons), 'estimated_windows': len(self.estimated_windows), 'estimated_proposals': len(self.estimated_proposals), 'automatic_timing_outcomes': dict(Counter((row['automatic_timing_status'] for row in self.automatic_timing_rows))), 'estimated_timing_reviews': dict(Counter((row['timing_review_status'] for row in self.estimated_windows))), 'unresolved_outcomes': sum((row['window_status'] == 'unresolved' for row in self.season_audit)), 'estimated_complete_pixel_windows': len(self.estimated_feature_rows), 'estimated_sample_losses': self.estimated_losses, 'exploratory_sample_counts': season_workflow.sample_counts(self.exploratory_rows), 'complete_pixel_windows': len(self.feature_rows), 'partitioned_pixel_windows': len(self.partitioned_rows), 'excluded_pixel_windows': sum((row['excluded_pixels'] for row in self.sample_losses)), 'sample_losses': self.sample_losses, 'sample_counts': season_workflow.sample_counts(self.partitioned_rows), 'preparation_status': 'prepared' if self.partitioned_rows else 'pending_inputs', 'mapping_coverage': 'not_measured'}
        return self.coverage_summary
