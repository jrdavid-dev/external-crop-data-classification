import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from crop_pipeline import share
from crop_pipeline.sources import adaptive, test_cohort
from test_jecam_transfer import completed_batch


class ShareTests(TestCase):
    def test_complete_package_works_without_sender_or_network_and_preserves_partitions(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sender, receiver = root/'sender', root/'receiver'
            report = sender/'data/pipeline_work/inventory'
            for source in ('sen4agrinet', 'francecrops', 'cropharvest'):
                completed_batch(report, source, {source: (1,1,2,2)})
            package = share.export(sender, root/'package')
            shutil.rmtree(sender)
            with patch('requests.sessions.Session.request', side_effect=AssertionError('No network')):
                imported = share.import_package(receiver, package)
                self.assertEqual(share.import_package(receiver, package), imported)
            self.assertEqual(len(imported), 3)
            for folder in imported:
                selection = json.loads((folder/'selection.json').read_text())
                expected = 'test' if selection['source']=='cropharvest' else 'train'
                self.assertEqual(set(selection['assignments'].values()), {expected})
                self.assertTrue(list(folder.rglob('tiffs/*.tif')))
            image = next(package.rglob('tiffs/*.tif'))
            image.write_bytes(b'damaged')
            with self.assertRaisesRegex(ValueError,'checksum'):
                share.import_package(root/'other', package)
            self.assertFalse((root/'other').exists())

    def test_policy_mismatch_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); report=root/'sender/data/pipeline_work/inventory'
            completed_batch(report, 'cropharvest', {'test':(1,1,2,2)})
            package=share.export(root/'sender', root/'package')
            data=json.loads((package/'package.json').read_text())
            data['bundles'][0]['cohort']='cabbage_supplement'
            (package/'package.json').write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'policy mismatch'):
                share.verify(package)

    def test_supplement_stays_in_separate_test_policy(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); report=root/'sender/data/pipeline_work/inventory'
            source='california_dwr'
            batch=completed_batch(report, source, {'carrot':(1,1,2,2)})
            # Move only the receipt/selection registration; handoff paths remain
            # valid in the synthetic sender until export relocates them.
            selection=json.loads((batch/'selection.json').read_text())
            selection['assignments']={'carrot':'test'}
            (batch/'selection.json').write_text(json.dumps(selection))
            target=adaptive.root(report, share.cohorts()[source][1])/'rounds/0001'/source
            target.mkdir(parents=True)
            for name in ('selection.json','state.json','completed.json'):
                shutil.move(batch/name,target/name)
            package=share.export(root/'sender', root/'package')
            [imported]=share.import_package(root/'receiver',package)
            self.assertIn(adaptive.policy_key(share.cohorts()[source][1]), str(imported))
            self.assertEqual(json.loads((imported/'selection.json').read_text())['assignments'], {'carrot':'test'})
