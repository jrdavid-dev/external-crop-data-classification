from io import BytesIO
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from crop_pipeline import path_relocation as relocation


def sha(value):
    return hashlib.sha256(value).hexdigest()


class PathRelocationTests(unittest.TestCase):
    def test_stream_boundaries_and_original_transfer_hashes(self):
        old=sha(b'old');new=sha(b'new')
        raw=(b'x'*65530+b'/old/work/a '+old.encode()+b' "original_handoff_hashes": ["'+old.encode()+b'"]')
        output=BytesIO()
        with patch.object(relocation,'CHUNK',77):
            result=relocation.tokens(BytesIO(raw),{b'/old/work':b'/new/data'},{old.encode():new.encode()},output)
        expected=raw.replace(b'/old/work',b'/new/data').replace(old.encode(),new.encode(),1)
        self.assertEqual(output.getvalue(),expected)
        self.assertEqual(result[0],sha(raw));self.assertEqual(result[1],sha(expected))

    def test_dependency_checksums_and_originals_are_preserved(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();work=root/'data/pipeline_work';work.mkdir(parents=True)
            for name in relocation.LEGACY_ROOTS:
                alias=root/name;alias.parent.mkdir(parents=True,exist_ok=True);alias.symlink_to(work)
            image=work/'image.tif';image.write_bytes(b'unchanged-imagery')
            child=work/'child.json'
            original=json.dumps({'image':str(root/'sen4agrinet_work/image.tif')}).encode();child.write_bytes(original)
            parent=work/'completed.json'
            parent.write_text(json.dumps({'path':str(root/'sen4agrinet_work/child.json'),'sha256':sha(original),'bytes':len(original),
                                         'original_handoff_hashes':[sha(original)],'assignments':{'a':'train'}}))
            grandparent=work/'receipt.json';grandparent.write_text(json.dumps({'sha256':sha(parent.read_bytes())}))
            run=relocation.relocate(root)
            self.assertEqual(json.loads(child.read_text())['image'],str(image))
            data=json.loads(parent.read_text())
            self.assertEqual(data['sha256'],sha(child.read_bytes()))
            self.assertEqual(data['bytes'],child.stat().st_size)
            self.assertEqual(data['original_handoff_hashes'],[sha(original)])
            self.assertEqual(data['assignments'],{'a':'train'})
            self.assertEqual(json.loads(grandparent.read_text())['sha256'],sha(parent.read_bytes()))
            self.assertTrue((run/'original_metadata.tar.gz').is_file())
            self.assertEqual(image.read_bytes(),b'unchanged-imagery')
            rebased=child.read_bytes()
            child.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError,"Rebased metadata changed"):
                relocation.finalize(run)
            self.assertTrue((root/relocation.LEGACY_ROOTS[0]).is_symlink())
            child.write_bytes(rebased)
            relocation.finalize(run)
            for name in relocation.LEGACY_ROOTS:
                self.assertFalse((root/name).exists())
                self.assertFalse((root/name).is_symlink())
            self.assertEqual(json.loads(child.read_text())['image'],str(image))

    def test_digest_in_content_addressed_filename_is_not_rewritten(self):
        old=sha(b'old');new=sha(b'new')
        raw=json.dumps({'path':'/old/work/'+old+'.json','sha256':old}).encode()
        output=BytesIO()
        relocation.tokens(BytesIO(raw),{b'/old/work':b'/new/data'},{old.encode():new.encode()},output)
        data=json.loads(output.getvalue())
        self.assertEqual(data['path'],'/new/data/'+old+'.json')
        self.assertEqual(data['sha256'],new)

    def test_rollback_preserves_concurrent_edits_and_restores_only_our_writes(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();work=root/'data/pipeline_work';work.mkdir(parents=True)
            for name in relocation.LEGACY_ROOTS:
                alias=root/name;alias.parent.mkdir(parents=True,exist_ok=True);alias.symlink_to(work)
            child=work/'child.json'
            original=json.dumps({'path':str(root/'sen4agrinet_work/image.tif')}).encode()
            child.write_bytes(original)
            parent=work/'parent.json';parent.write_text(json.dumps({'sha256':sha(original)}))
            replace=Path.replace
            def concurrent_write(path,target):
                result=replace(path,target)
                if Path(target)==child:parent.write_text('{"external_edit":true}')
                return result
            with patch.object(Path,'replace',concurrent_write):
                with self.assertRaisesRegex(ValueError,'Metadata changed during migration'):
                    relocation.relocate(root)
            self.assertEqual(child.read_bytes(),original)
            self.assertEqual(json.loads(parent.read_text()),{'external_edit':True})
            self.assertTrue((root/relocation.LEGACY_ROOTS[0]).is_symlink())
