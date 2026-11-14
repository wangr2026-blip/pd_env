"""Offline checks for asset integrity and checkpoint download recovery."""
import hashlib
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    'pretrained_download', Path(__file__).resolve().parents[1] / 'scripts/download_pretrained_models.py')
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


class DownloadIntegrityTests(unittest.TestCase):
    def test_public_only_plan_needs_neither_credentials_nor_network(self):
        asset = {'name': 'model.safetensors', 'bytes': 1, 'sha256': None, 'git_blob': None}
        lock = {'models': {
            'public_fixture': {'repo_id': 'synthetic/public', 'revision': '0' * 40,
                               'gated': False, 'files': [asset]},
            'gated_fixture': {'repo_id': 'synthetic/gated', 'revision': '1' * 40,
                              'gated': 'auto', 'files': [asset]}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path = root / 'lock.json'
            lock_path.write_text(json.dumps(lock))
            args = ['download_pretrained_models.py', '--lock', str(lock_path),
                    '--output-root', str(root / 'models'), '--public-only', '--plan-only']
            output = io.StringIO()
            with patch.object(downloader.sys, 'argv', args), \
                    patch.object(downloader, 'token', side_effect=AssertionError('No credentials required')), \
                    patch.object(downloader, 'fetch_json', side_effect=AssertionError('No network required')), \
                    redirect_stdout(output):
                self.assertEqual(downloader.main(), 0)
            self.assertIn('public_fixture', output.getvalue())
            self.assertNotIn('gated_fixture', output.getvalue())

    def test_rejects_same_size_corruption(self):
        original = b'synthetic checkpoint bytes'
        asset = {'name': 'model.safetensors', 'bytes': len(original),
                 'sha256': hashlib.sha256(original).hexdigest(), 'git_blob': None}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / asset['name']
            path.write_bytes(original)
            self.assertEqual(downloader.check_file(path, asset), asset['sha256'])
            path.write_bytes(b'!' + original[1:])
            self.assertIsNone(downloader.check_file(path, asset))

    def test_support_files_checked_as_git_blobs(self):
        data = b'{"sampling_rate":16000}\n'
        asset = {'bytes': len(data), 'sha256': None,
                 'git_blob': hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'preprocessor_config.json'
            path.write_bytes(data)
            self.assertIsNotNone(downloader.check_file(path, asset))
            path.write_bytes(data.replace(b'16000', b'48000'))
            self.assertIsNone(downloader.check_file(path, asset))

    def test_avoids_duplicate_framework_weights_and_remote_code(self):
        siblings = [{'rfilename': name, 'size': 1, 'blobId': '0' * 40}
                    for name in ['model.safetensors', 'pytorch_model.bin', 'flax_model.msgpack',
                                 'tf_model.h5', 'config.json', 'preprocessor_config.json',
                                 'README.md', 'modeling_custom.py', 'onnx/model.onnx']]
        names = {item['name'] for item in downloader.select_files(siblings)}
        self.assertEqual(names, {'model.safetensors', 'config.json', 'preprocessor_config.json', 'README.md'})

    def test_verified_download_reusable_without_network(self):
        data = b'synthetic checkpoint'
        asset = {'name': 'model.safetensors', 'bytes': len(data),
                 'sha256': hashlib.sha256(data).hexdigest(), 'git_blob': None}
        entry = {'repo_id': 'synthetic/model', 'revision': '0' * 40, 'gated': False, 'files': [asset]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / 'checkpoints' / 'fixture'
            destination.mkdir(parents=True)
            (destination / asset['name']).write_bytes(data)
            with patch.object(downloader, 'curl', side_effect=AssertionError('Network must not be used')):
                result = downloader.download_model('fixture', entry, root, None, verify_only=True)
            self.assertEqual(result['status'], 'verified')

    def test_invalid_response_never_promoted_to_completed_asset(self):
        data = b'valid'
        asset = {'name': 'model.safetensors', 'bytes': len(data),
                 'sha256': hashlib.sha256(data).hexdigest(), 'git_blob': None}
        entry = {'repo_id': 'synthetic/model', 'revision': '0' * 40, 'gated': False, 'files': [asset]}
        def corrupt_response(url, output, auth, resume):
            output.write_bytes(b'wrong')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(downloader, 'curl', side_effect=corrupt_response):
                with self.assertRaisesRegex(RuntimeError, 'Integrity check failed'):
                    downloader.download_model('fixture', entry, root, None, verify_only=False)
            self.assertFalse((root / 'checkpoints/fixture/model.safetensors').exists())
            self.assertFalse((root / 'checkpoints/fixture/download_provenance.json').exists())

    def test_interrupted_transfer_keeps_bytes_before_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            partial = Path(directory) / 'model.safetensors.part'
            calls = []
            def interrupted_then_complete(command, **kwargs):
                calls.append(command)
                if len(calls) == 1:
                    partial.write_bytes(b'first half')
                    return SimpleNamespace(returncode=18, stderr='partial transfer')
                self.assertEqual(partial.read_bytes(), b'first half')
                with partial.open('ab') as stream:
                    stream.write(b' second half')
                return SimpleNamespace(returncode=0, stderr='')
            with patch.object(downloader.subprocess, 'run', side_effect=interrupted_then_complete), \
                    patch.object(downloader.time, 'sleep'):
                downloader.curl('https://huggingface.co/synthetic', partial, resume=True)
            self.assertEqual(len(calls), 2)
            self.assertTrue(all('--continue-at' in command for command in calls))
            self.assertEqual(partial.read_bytes(), b'first half second half')

    def test_full_size_bad_partial_is_restarted(self):
        data = b'valid'
        asset = {'name': 'model.safetensors', 'bytes': len(data),
                 'sha256': hashlib.sha256(data).hexdigest(), 'git_blob': None}
        entry = {'repo_id': 'synthetic/model', 'revision': '0' * 40, 'gated': False, 'files': [asset]}
        def valid_response(url, output, auth, resume):
            self.assertEqual(output.stat().st_size, 0)
            output.write_bytes(data)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / 'checkpoints/fixture'
            destination.mkdir(parents=True)
            (destination / 'model.safetensors.part').write_bytes(b'wrong')
            with patch.object(downloader, 'curl', side_effect=valid_response):
                result = downloader.download_model('fixture', entry, root, None, verify_only=False)
            self.assertEqual(result['status'], 'verified')


if __name__ == '__main__':
    unittest.main()
