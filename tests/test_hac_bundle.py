"""Portable HAC rejects incomplete, mismatched, or modified codec artifacts."""
import json
from pathlib import Path
import tempfile
import unittest
from src.hac.bundle import EXACT,FORMAT,pack,read_bundle,write_json

class HacBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name);self.raw=self.root/'raw';self.raw.mkdir()
        for name in EXACT:
            (self.raw/name).write_bytes(b'controlled-test-payload')
        for family in ('feat','scaling','offsets'):
            (self.raw/(family+'_0_0.b')).write_bytes(b'arithmetic-bytes')
        self.config=dict(format=FORMAT,backend='HAC',patched_infos=[12,7,3000],
            architecture=dict(decoded_version=True,ste_binary=True),
            white_background=False,all_views_train_test=True,
            shared_decoder_parameter_bytes=1234,native_mlp_parameter_bytes=1234)
        write_json(self.raw/'decoder_config.json',self.config)
        write_json(self.raw/'patched_infos.json',[12,7,3000])
    def tearDown(self):self.temp.cleanup()
    def pack(self):return pack(self.raw,self.root/'bundle','test')
    def test_manifest_total_includes_itself_and_decoder(self):
        m=self.pack();size=sum(p.stat().st_size for p in (self.root/'bundle').rglob('*') if p.is_file())
        self.assertEqual(m['storage']['artifact_bytes'],size)
        self.assertTrue(m['codec_only']);self.assertFalse(m['hash_in_shared_decoder'])
    def test_missing_side_information_rejected(self):
        (self.raw/'patched_infos.json').unlink()
        with self.assertRaises(ValueError):self.pack()
    def test_wrong_backend_rejected(self):
        self.config['backend']='HAC++';write_json(self.raw/'decoder_config.json',self.config)
        with self.assertRaises(ValueError):self.pack()
    def test_changed_stream_rejected(self):
        self.pack();(self.root/'bundle/hac/hash.b').write_bytes(b'corrupted')
        with self.assertRaises(ValueError):read_bundle(self.root/'bundle')
    def test_missing_arithmetic_batch_rejected(self):
        self.config['patched_infos']=[9000,6001,3000]
        write_json(self.raw/'decoder_config.json',self.config)
        write_json(self.raw/'patched_infos.json',self.config['patched_infos'])
        with self.assertRaises(ValueError):self.pack()
    def test_extra_file_rejected(self):
        self.pack();(self.root/'bundle/hac/hidden.pth').write_bytes(b'checkpoint')
        with self.assertRaises(ValueError):read_bundle(self.root/'bundle')
    def test_existing_bundle_preserved(self):
        self.pack()
        with self.assertRaises(FileExistsError):self.pack()
    def test_batch_metadata_disagreement_rejected(self):
        write_json(self.raw/'patched_infos.json',[13,7,3000])
        with self.assertRaises(ValueError):self.pack()

if __name__=='__main__':unittest.main()
