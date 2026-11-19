from contextlib import chdir
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from crop_pipeline import storage_migration
from crop_pipeline.sources import annual


class StorageMigrationTests(unittest.TestCase):
    def test_merge_preserves_bytes_and_historical_paths(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            first, second=[root/p for p in storage_migration.LEGACY_ROOTS]
            for folder in (first,second):
                (folder/'annual_tifs').mkdir(parents=True)
            (first/'annual_tifs/a.tif').write_bytes(b'original-imagery')
            (second/'annual_tifs/b.tif').write_bytes(b'supplement-imagery')
            receipt=json.dumps({'path':str(second/'annual_tifs/b.tif'), 'locked':'train'})
            (first/'receipt.json').write_text(receipt)
            manifest=storage_migration.migrate(root)
            self.assertEqual(json.loads(manifest.read_text())['status'],'verified')
            target=root/storage_migration.DESTINATION
            self.assertEqual((target/'receipt.json').read_text(),receipt)
            self.assertEqual((first/'annual_tifs/b.tif').read_bytes(), b'supplement-imagery')
            self.assertEqual((second/'annual_tifs/b.tif').read_bytes(), b'supplement-imagery')
            self.assertTrue(first.is_symlink() and second.is_symlink())
            with self.assertRaises(FileExistsError):storage_migration.migrate(root)

    def test_collision_stops_before_moving_any_data(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in storage_migration.LEGACY_ROOTS:
                folder=root/name;folder.mkdir(parents=True)
                (folder/'state.json').write_text(name)
            with self.assertRaisesRegex(ValueError,'collision'):storage_migration.migrate(root)
            for name in storage_migration.LEGACY_ROOTS:
                folder=root/name
                self.assertFalse(folder.is_symlink())
                self.assertEqual((folder/'state.json').read_text(),name)
            self.assertFalse((root/storage_migration.DESTINATION).exists())

    def test_archive_setting_is_anchored_to_repository_from_nested_notebook(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();nested=root/'notebooks/data_collection';nested.mkdir(parents=True)
            settings={'annual_archive_root':'./data/pipeline_work/annual_archive'}
            with chdir(nested):
                self.assertEqual(annual.archive_root(settings,root/'data/pipeline_work/inventory',repository_root=root),
                                 root/'data/pipeline_work/annual_archive')
                self.assertEqual(annual.archive_root({'annual_archive_root':str(root/'external')},root/'inventory',repository_root=root),root/'external')
