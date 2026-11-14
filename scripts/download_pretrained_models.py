#!/usr/bin/env python3
"""Download pinned public model assets with curl; never read participant data.

No Python dependencies are required. The default local staging directory is
models/ (Git-ignored). Set MPOWER_MODEL_ROOT on HPC or pass --output-root.
Only one PyTorch weight format is downloaded, preferring safetensors.
Tokens, if available, go to curl's stdin, never its command line or manifests.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import quote


REPO = Path(__file__).resolve().parents[1]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def token():
    if os.environ.get('HF_TOKEN'):
        return os.environ['HF_TOKEN'].strip()
    home = Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface'))
    path = Path(os.environ.get('HF_TOKEN_PATH', home / 'token'))
    return path.read_text().strip() if path.is_file() else None


def curl(url, output, auth=None, resume=False):
    command = ['curl', '--fail', '--location', '--silent', '--show-error',
               '--retry', '0', '--connect-timeout', '30',
               '--speed-limit', '1024', '--speed-time', '120',
               '--output', str(output)]
    if resume:
        command += ['--continue-at', '-']
    # curl does not forward Authorization to a different redirect host by default.
    options = ''
    if auth:
        if any(c in auth for c in '\r\n"\\'):
            raise ValueError('Invalid Hugging Face token characters')
        command += ['--config', '-']
        options = f'header = "Authorization: Bearer {auth}"\n'
    for attempt in range(4):
        # A fresh curl invocation recalculates the resume offset after an error.
        # curl's built-in retry can rewind a multi-GB download to its initial
        # offset, losing bytes already received during that invocation.
        result = subprocess.run(command + [url], input=options, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not result.returncode:
            return
        denied = any(f'error: {status}' in result.stderr for status in (401, 403, 404))
        transient = result.returncode in {6, 7, 16, 18, 28, 35, 52, 55, 56, 92}
        if attempt == 3 or denied or not transient:
            # Signed redirect URLs and credentials are never persisted.
            raise RuntimeError(f'curl failed ({result.returncode}): {result.stderr.strip()[:400]}')
        print(f'Retrying {output.name} after transfer error {result.returncode}; keeping partial bytes', flush=True)
        time.sleep(2 ** attempt)


def fetch_json(url, auth):
    with tempfile.TemporaryDirectory(prefix='mpower-model-api-') as directory:
        path = Path(directory) / 'response.json'
        curl(url, path, auth)
        return json.loads(path.read_text())


def select_files(siblings):
    available = {item['rfilename']: item for item in siblings}
    names = set(available)
    # Standard HF single-file or sharded PyTorch releases. Keep root assets only;
    # alternate precision/framework exports and training data are not needed.
    if 'model.safetensors' in names:
        weights = {'model.safetensors'}
    elif 'model.safetensors.index.json' in names:
        weights = {n for n in names if n.startswith('model-') and n.endswith('.safetensors')}
        weights.add('model.safetensors.index.json')
    elif 'pytorch_model.bin' in names:
        weights = {'pytorch_model.bin'}
    elif 'pytorch_model.bin.index.json' in names:
        weights = {n for n in names if n.startswith('pytorch_model-') and n.endswith('.bin')}
        weights.add('pytorch_model.bin.index.json')
    else:
        raise ValueError('No standard Hugging Face PyTorch weights found')
    support = {n for n in names if '/' not in n and (
        n.endswith(('.json', '.txt', '.md', '.model', '.tiktoken'))
        or n.upper().startswith(('LICENSE', 'NOTICE', 'COPYING')))}
    selected = []
    for name in sorted(weights | support):
        item = available[name]
        if '/' in name or name in {'.', '..'}:
            raise ValueError('Unsafe model asset path')
        selected.append({'name': name, 'bytes': item['size'],
                         'sha256': item.get('lfs', {}).get('sha256'),
                         'git_blob': None if item.get('lfs') else item.get('blobId')})
    return selected


def resolve_inventory(inventory, auth, workers):
    def resolve(pair):
        alias, repo_id = pair
        metadata = fetch_json(f'https://huggingface.co/api/models/{quote(repo_id, safe="/")}?blobs=true', auth)
        return alias, {'repo_id': repo_id, 'revision': metadata['sha'],
                       'gated': metadata.get('gated', False),
                       'files': select_files(metadata['siblings'])}
    models = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for future in as_completed([executor.submit(resolve, pair) for pair in inventory['models'].items()]):
            alias, entry = future.result()
            models[alias] = entry
            print(f'Resolved {alias}: {entry["revision"]}', flush=True)
    return {'resolved_at': datetime.now(timezone.utc).isoformat(),
            'models': dict(sorted(models.items()))}


def check_file(path, asset):
    if not path.is_file() or path.stat().st_size != asset['bytes']:
        return None
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(f'blob {asset["bytes"]}\0'.encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha256.update(block)
            if asset.get('git_blob'):
                blob.update(block)
    digest = sha256.hexdigest()
    if asset.get('sha256') and digest != asset['sha256']:
        return None
    if asset.get('git_blob') and blob.hexdigest() != asset['git_blob']:
        return None
    return digest


def download_model(alias, entry, root, auth, verify_only):
    destination = root / 'checkpoints' / alias
    destination.mkdir(parents=True, exist_ok=True)
    verified = []
    for asset in entry['files']:
        path = destination / asset['name']
        digest = check_file(path, asset)
        if not digest:
            if verify_only:
                raise RuntimeError(f'Missing or invalid file: {alias}/{asset["name"]}')
            print(f'Downloading {alias}/{asset["name"]} ({asset["bytes"] / 1e9:.3f} GB)', flush=True)
            partial = path.with_name(path.name + '.part')
            if partial.exists() and partial.stat().st_size == asset['bytes']:
                digest = check_file(partial, asset)
            if not digest:
                if partial.exists() and partial.stat().st_size >= asset['bytes']:
                    # A full-size but invalid partial cannot be repaired by an
                    # EOF range request. Restart only this known-bad artifact.
                    partial.write_bytes(b'')
                # Cache-bust redirects to avoid reusing expired signed CDN URLs.
                url = (f'https://huggingface.co/{quote(entry["repo_id"], safe="/")}'
                       f'/resolve/{entry["revision"]}/{quote(asset["name"], safe="")}'
                       f'?download=true&request={time.time_ns()}')
                curl(url, partial, auth, resume=True)
                digest = check_file(partial, asset)
            if not digest:
                raise RuntimeError(f'Integrity check failed: {alias}/{asset["name"]}')
            partial.replace(path)
        verified.append({'name': asset['name'], 'bytes': asset['bytes'], 'sha256': digest})
    provenance = {**entry, 'files': verified, 'verified_at': datetime.now(timezone.utc).isoformat()}
    save_json(destination / 'download_provenance.json', provenance)
    print(f'VERIFIED {alias} ({sum(f["bytes"] for f in verified) / 1e9:.3f} GB)', flush=True)
    return {'status': 'verified', 'path': str(destination.relative_to(root)),
            'repo_id': entry['repo_id'], 'revision': entry['revision'],
            'bytes': sum(f['bytes'] for f in verified), 'file_count': len(verified)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, default=REPO / 'configs/pretrained_models.json')
    parser.add_argument('--lock', type=Path, default=REPO / 'configs/pretrained_models.lock.json')
    parser.add_argument('--output-root', type=Path,
                        default=Path(os.environ.get('MPOWER_MODEL_ROOT', REPO / 'models')))
    parser.add_argument('--models', nargs='+', help='Download/verify only these inventory aliases')
    parser.add_argument('--public-only', action='store_true',
                        help='Select ungated checkpoints and download without Hugging Face credentials')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--refresh-lock', action='store_true', help='Explicitly resolve current upstream revisions again')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    auth = None if args.public_only else token()
    if args.lock.exists() and not args.refresh_lock:
        lock = json.loads(args.lock.read_text())
    else:
        if args.verify_only:
            parser.error('--verify-only requires an existing lock file')
        lock = resolve_inventory(json.loads(args.inventory.read_text()), auth, args.workers)
        save_json(args.lock, lock)
    if args.models and set(args.models) - set(lock['models']):
        parser.error('Unknown model alias')
    entries = {k: v for k, v in lock['models'].items()
               if (not args.models or k in args.models) and (not args.public_only or not v.get('gated'))}
    if not entries:
        parser.error('No checkpoints selected; --public-only excludes gated models')
    required = sum(f['bytes'] for entry in entries.values() for f in entry['files'])
    print(f'Pinned assets: {required / 1e9:.3f} GB; destination: {root}', flush=True)
    if args.plan_only:
        for alias, entry in entries.items():
            print(f'{alias}: {sum(f["bytes"] for f in entry["files"]) / 1e9:.3f} GB; gated={entry["gated"]}', flush=True)
        return 0
    remaining = sum(max(0, f['bytes'] - (root / 'checkpoints' / alias / f['name']).stat().st_size)
                    if (root / 'checkpoints' / alias / f['name']).is_file() else f['bytes']
                    for alias, entry in entries.items() for f in entry['files'])
    if not args.verify_only and remaining + 1024 ** 3 > shutil.disk_usage(root).free:
        raise RuntimeError('Insufficient free space for model assets plus a 1 GiB margin')
    status_path = root / 'download_status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {'models': {}}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_model, alias, entry, root, auth, args.verify_only): alias
                   for alias, entry in entries.items()}
        for future in as_completed(futures):
            alias = futures[future]
            try:
                status['models'][alias] = future.result()
            except Exception as error:
                status['models'][alias] = {'status': 'incomplete', 'error': str(error),
                                           'repo_id': entries[alias]['repo_id'],
                                           'revision': entries[alias]['revision']}
                print(f'INCOMPLETE {alias}: {error}', flush=True)
            status['updated_at'] = datetime.now(timezone.utc).isoformat()
            save_json(status_path, status)
    return int(any(status['models'][alias]['status'] != 'verified' for alias in entries))


if __name__ == '__main__':
    sys.exit(main())
