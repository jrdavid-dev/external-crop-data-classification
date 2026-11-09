"""Readable summaries retain processing records and work in notebooks or terminals."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from crop_pipeline import notebook_display as presentation


class NotebookDisplayTests(unittest.TestCase):
    def test_partition_summary_is_readable_and_does_not_change_input(self):
        rows = [
            {"crop": "carrot", "partition": "train", "pixels": 1203, "parcel_groups": 6},
            {"crop": "carrot", "partition": "validation", "pixels": 34, "parcel_groups": 2},
            {"crop": "carrot", "partition": "test", "pixels": 20, "parcel_groups": 1},
        ]
        original = deepcopy(rows)
        output = io.StringIO()
        with patch('IPython.get_ipython', return_value=None), redirect_stdout(output):
            presentation.show_table(rows, title="Usable samples")
        text = output.getvalue()
        for expected in ("Usable samples", "Crop", "Partition", "Parcels", "Carrot", "Training", "Validation", "Reserved test", "1,203"):
            self.assertIn(expected, text)
        self.assertNotIn("{'crop'", text)
        self.assertEqual(rows, original)

    def test_notebook_table_has_caption_and_safe_html(self):
        with patch('IPython.get_ipython', return_value=object()), patch('IPython.display.display') as display:
            presentation.show_table([{"crop": "potato", "partition": "test", "pixels": 10, "note": "<tag>"}], title="Held-out samples")
        html = display.call_args.args[0].data
        self.assertIn('<caption>Held-out samples</caption>', html)
        self.assertIn('Reserved test', html)
        self.assertIn('&lt;tag&gt;', html)
        self.assertNotIn('>test<', html)

    def test_dataframe_crop_index_and_nested_audits_preserve_values(self):
        frame = pd.DataFrame({"passed": [35], "excluded": [2]}, index=pd.Index(['carrot'], name='crop'))
        original = frame.copy(deep=True)
        values = {"valid_footprints": 35, "excluded_footprints": [{"parcel_id": "p1", "reason": "Invalid geometry"}],
                  "reference": Path('reference.parquet'), "targets_met": False}
        original_values = deepcopy(values)
        output = io.StringIO()
        with patch('IPython.get_ipython', return_value=None), redirect_stdout(output):
            presentation.show_table(frame, title="Label screening")
            presentation.show_mapping(values, title="Reference audit")
        text = output.getvalue()
        for expected in ('Crop', 'Carrot', 'Invalid geometry', 'reference.parquet', 'No'):
            self.assertIn(expected, text)
        self.assertNotIn('{', text)
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(values, original_values)

    def test_empty_records_and_years_are_clear(self):
        output = io.StringIO()
        with redirect_stdout(output):
            presentation.show_table([], title="Exclusions")
            presentation.show_items(['2019', '2020'], title="Years")
        self.assertIn('Exclusions: no records.', output.getvalue())
        self.assertIn('Years: 2019, 2020', output.getvalue())
