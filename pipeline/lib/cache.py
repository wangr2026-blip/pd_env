"""Content-addressed stage directories with a completion manifest.

A stage directory is complete only once ``manifest.json`` exists; it is
written last, atomically, so an interrupted run is never mistaken for a hit.
"""
import datetime as dt
import json
import shutil
from pathlib import Path

from mpower_common import save_json, write_table  # noqa: F401  (re-exported)

KEY_LENGTH = 16


class StageCache:
    def __init__(self, cfg, stage, key, payload=None):
        self.stage = stage
        self.key = key
        self.payload = payload
        self.dir = Path(cfg["output_root"]) / cfg["run_name"] / stage / key[:KEY_LENGTH]
        self.manifest_path = self.dir / "manifest.json"

    def complete(self):
        return self.manifest_path.exists()

    def manifest(self):
        return json.loads(self.manifest_path.read_text())

    def path(self, name):
        return self.dir / name

    def prepare(self, force=False):
        """Start a fresh attempt; partial outputs from earlier attempts are removed."""
        if self.dir.exists() and (force or not self.complete()):
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def finalize(self, outputs, summary=None):
        missing = [o for o in outputs if not (self.dir / o).exists()]
        if missing:
            raise RuntimeError(f"{self.stage}: declared outputs not written: {missing}")
        save_json(self.manifest_path, {
            "stage": self.stage,
            "key": self.key,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "outputs": sorted(outputs),
            "inputs": self.payload,
            "summary": summary or {},
        })
        latest = self.dir.parent / "LATEST"
        latest.write_text(self.key[:KEY_LENGTH] + "\n")
